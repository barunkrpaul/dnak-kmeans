"""
Benchmark v2: Lloyd vs Hamerly vs Elkan vs DNAK.
Adds: Elkan baseline, real datasets (image color quantization d=3,
digits PCA d=3), 3 initialization seeds per config, mean/std stats.
Exactness asserted on every single run.
"""
import numpy as np, time, csv, json
from dnak import (make_blobs, kmeanspp_init, update_centroids, sse_of,
                  lloyd, hamerly, dnak)

# ---------------------------------------------------------------- Elkan
def elkan(X, C0, max_iter=300):
    n, d = X.shape
    k = C0.shape[0]
    C = C0.copy()
    comps = 0
    t0 = time.perf_counter()

    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    comps += n * k
    a = np.argmin(D, axis=1)
    u = D[np.arange(n), a]
    L = D.copy()  # n x k exact lower bounds initially

    for it in range(1, max_iter + 1):
        Cn = update_centroids(X, a, C, k)
        drift = np.linalg.norm(Cn - C, axis=1)
        C = Cn
        CC = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
        np.fill_diagonal(CC, np.inf)
        s = 0.5 * CC.min(axis=1)

        u = u + drift[a]
        L = np.maximum(L - drift[None, :], 0.0)

        active = u > s[a]
        # tighten upper bounds for active points
        idx = np.where(active)[0]
        if len(idx):
            u[idx] = np.linalg.norm(X[idx] - C[a[idx]], axis=1)
            L[idx, a[idx]] = u[idx]
            comps += len(idx)
            active = np.zeros(n, dtype=bool)
            active[idx] = u[idx] > s[a[idx]]

        changed = 0
        for j in range(k):
            cond = active & (a != j) & (u > L[:, j]) & (u > 0.5 * CC[a, j])
            ii = np.where(cond)[0]
            if len(ii) == 0:
                continue
            dj = np.linalg.norm(X[ii] - C[j], axis=1)
            comps += len(ii)
            L[ii, j] = dj
            sw = dj < u[ii]
            if sw.any():
                iw = ii[sw]
                a[iw] = j
                u[iw] = dj[sw]
                changed += len(iw)
        if changed == 0:
            Ct = update_centroids(X, a, C, k)
            if np.allclose(Ct, C):
                break
    t = time.perf_counter() - t0
    return dict(name="Elkan", labels=a, C=C, iters=it, comps=comps,
                time=t, sse=sse_of(X, a, C))

# ---------------------------------------------------------------- real data
def load_image_pixels(name, n_sub, seed):
    from sklearn.datasets import load_sample_image
    img = load_sample_image(name).astype(np.float64) / 255.0
    P = img.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=min(n_sub, len(P)), replace=False)
    P = P[idx]
    # general-position jitter: 8-bit pixels are heavily tied; add
    # negligible noise (1e-7) so nearest centroids are unique (declared
    # in the paper; SSE effect < 1e-6 relative)
    return P + rng.uniform(-1e-7, 1e-7, size=P.shape)

def load_digits_pca(dims=3):
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    X = load_digits().data
    return PCA(n_components=dims, random_state=0).fit_transform(X)

# ---------------------------------------------------------------- driver
def run(X, k, init_seed):
    C0 = kmeanspp_init(X, k, seed=init_seed)
    rL = lloyd(X, C0)
    out = [rL]
    for algo in (hamerly, elkan, dnak):
        r = algo(X, C0)
        assert abs(r["sse"] - rL["sse"]) / rL["sse"] < 1e-10, \
            f"SSE VIOLATION: {r['name']}"
        mm = np.where(r["labels"] != rL["labels"])[0]
        for i in mm:  # any label mismatch must be an exact distance tie
            d1 = np.linalg.norm(X[i] - r["C"][r["labels"][i]])
            d2 = np.linalg.norm(X[i] - rL["C"][rL["labels"][i]])
            assert abs(d1 - d2) < 1e-9, \
                f"EXACTNESS VIOLATION: {r['name']} point {i}: {d1} vs {d2}"
        out.append(r)
    return out

if __name__ == "__main__":
    SEEDS = [101, 202, 303]
    configs = []
    # synthetic
    for (n, d, k, ds) in [(10000, 2, 32, 7), (10000, 2, 100, 11),
                          (20000, 3, 64, 13), (20000, 4, 64, 17),
                          (30000, 2, 128, 19)]:
        X = make_blobs(n, d, k_true=k, seed=ds)
        configs.append((f"synth n={n} d={d} k={k}", X, k))
    # real
    configs.append(("china-RGB n=30000 d=3 k=64",
                    load_image_pixels("china.jpg", 30000, 1), 64))
    configs.append(("flower-RGB n=30000 d=3 k=32",
                    load_image_pixels("flower.jpg", 30000, 2), 32))
    configs.append(("digits-PCA3 n=1797 d=3 k=10",
                    load_digits_pca(3), 10))

    rows = []
    for cname, X, k in configs:
        agg = {}
        for seed in SEEDS:
            for r in run(X, k, seed):
                agg.setdefault(r["name"], []).append(r["comps"])
        lm = np.mean(agg["Lloyd"])
        print(f"\n=== {cname} (3 seeds, all exact) ===")
        for name in ("Lloyd", "Hamerly", "Elkan", "DNAK"):
            c = np.array(agg[name], dtype=float)
            red = np.mean(agg["Lloyd"]) / c  # per-seed pairing approx
            reds = np.array(agg["Lloyd"]) / c
            print(f"  {name:8s} comps={c.mean():>14,.0f} ± {c.std():>12,.0f}"
                  f"   reduction={reds.mean():6.2f}x ± {reds.std():4.2f}")
            rows.append(dict(config=cname, algo=name,
                             comps_mean=int(c.mean()), comps_std=int(c.std()),
                             red_mean=round(float(reds.mean()), 2),
                             red_std=round(float(reds.std()), 2)))
    with open("/home/claude/results_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nALL RUNS EXACT. Saved results_v2.csv")
