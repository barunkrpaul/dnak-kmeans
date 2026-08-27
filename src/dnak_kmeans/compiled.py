"""
Compiled (Numba) implementations of Lloyd, Hamerly, Elkan and DNAK,
with a wall-clock benchmark. All algorithms share the tie rule
"switch only if strictly closer"; exactness (identical SSE) asserted.
JIT compilation is excluded from timing via a warm-up pass.
"""
import numpy as np, time, csv
from numba import njit
from scipy.spatial import Delaunay
from dnak import make_blobs, kmeanspp_init

# ------------------------------------------------------------ kernels
@njit(cache=True)
def _dist(X, i, C, j):
    s = 0.0
    for t in range(X.shape[1]):
        dv = X[i, t] - C[j, t]
        s += dv * dv
    return np.sqrt(s)

@njit(cache=True)
def full_scan_init(X, C, a, u, l):
    n, k = X.shape[0], C.shape[0]
    comps = 0
    for i in range(n):
        b1 = 1e300; b2 = 1e300; bj = -1
        for j in range(k):
            d = _dist(X, i, C, j); comps += 1
            if d < b1:
                b2 = b1; b1 = d; bj = j
            elif d < b2:
                b2 = d
        a[i] = bj; u[i] = b1; l[i] = b2
    return comps

@njit(cache=True)
def centroid_update(X, a, C_old, k):
    n, d = X.shape
    C = np.zeros((k, d)); cnt = np.zeros(k)
    for i in range(n):
        cnt[a[i]] += 1.0
        for t in range(d):
            C[a[i], t] += X[i, t]
    for j in range(k):
        if cnt[j] > 0:
            for t in range(d):
                C[j, t] /= cnt[j]
        else:
            for t in range(d):
                C[j, t] = C_old[j, t]
    return C

@njit(cache=True)
def lloyd_iter(X, C, a):
    n, k = X.shape[0], C.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        cur = a[i]
        best = _dist(X, i, C, cur); bj = cur; comps += 1
        for j in range(k):
            if j == cur:
                continue
            d = _dist(X, i, C, j); comps += 1
            if d < best:            # strictly closer only (tie rule)
                best = d; bj = j
        if bj != cur:
            a[i] = bj; changed += 1
    return changed, comps

@njit(cache=True)
def hamerly_iter(X, C, a, u, l, drift, dmax, s):
    n, k = X.shape[0], C.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        u[i] += drift[a[i]]; l[i] -= dmax
        m = s[a[i]] if s[a[i]] > l[i] else l[i]
        if u[i] <= m:
            continue
        u[i] = _dist(X, i, C, a[i]); comps += 1
        if u[i] <= m:
            continue
        cur = a[i]
        b1 = u[i]; bj = cur; b2 = 1e300
        for j in range(k):
            if j == cur:
                continue
            d = _dist(X, i, C, j); comps += 1
            if d < b1:              # strictly closer (tie rule)
                b2 = b1; b1 = d; bj = j
            elif d < b2:
                b2 = d
        if bj != cur:
            a[i] = bj; changed += 1
        u[i] = b1; l[i] = b2
    return changed, comps

@njit(cache=True)
def elkan_iter(X, C, a, u, L, CC, s, drift):
    n, k = X.shape[0], C.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        u[i] += drift[a[i]]
        for j in range(k):
            L[i, j] -= drift[j]
            if L[i, j] < 0.0:
                L[i, j] = 0.0
        if u[i] <= s[a[i]]:
            continue
        tight = False
        old = a[i]
        for j in range(k):
            if j == a[i]:
                continue
            if u[i] > L[i, j] and u[i] > 0.5 * CC[a[i], j]:
                if not tight:
                    u[i] = _dist(X, i, C, a[i]); comps += 1
                    L[i, a[i]] = u[i]; tight = True
                    if u[i] <= L[i, j] or u[i] <= 0.5 * CC[a[i], j]:
                        continue
                d = _dist(X, i, C, j); comps += 1
                L[i, j] = d
                if d < u[i]:        # strictly closer (tie rule)
                    a[i] = j; u[i] = d
        if a[i] != old:
            changed += 1
    return changed, comps

@njit(cache=True)
def dnak_iter(X, C, a, u, l, drift, dmax, indptr, indices):
    n = X.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        u[i] += drift[a[i]]; l[i] -= dmax
        if u[i] <= l[i]:
            continue
        u[i] = _dist(X, i, C, a[i]); comps += 1
        if u[i] <= l[i]:
            continue
        cur = a[i]; curd = u[i]; second = 1e300
        while True:
            b = 1e300; bj = -1
            for p in range(indptr[cur], indptr[cur + 1]):
                j = indices[p]
                d = _dist(X, i, C, j); comps += 1
                if d < b:
                    b = d; bj = j
            if b < curd:            # strictly closer: hop
                cur = bj; curd = b
            else:                   # local min = global nearest
                second = b
                break
        if cur != a[i]:
            a[i] = cur; changed += 1
        u[i] = curd; l[i] = second
    return changed, comps

# ------------------------------------------------------------ drivers
def _cc_s(C):
    CC = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(CC, np.inf)
    return CC, 0.5 * CC.min(axis=1)

def _adjacency_csr(C):
    try:
        tri = Delaunay(C)
    except Exception:
        tri = Delaunay(C, qhull_options="QJ")
    k = C.shape[0]
    adj = [set() for _ in range(k)]
    for sx in tri.simplices:
        for ii in range(len(sx)):
            for jj in range(ii + 1, len(sx)):
                adj[sx[ii]].add(sx[jj]); adj[sx[jj]].add(sx[ii])
    indptr = np.zeros(k + 1, np.int64)
    for j in range(k):
        indptr[j + 1] = indptr[j] + len(adj[j])
    indices = np.empty(indptr[-1], np.int64)
    for j in range(k):
        indices[indptr[j]:indptr[j + 1]] = sorted(adj[j])
    return indptr, indices

def _sse(X, a, C):
    return float(((X - C[a]) ** 2).sum())

def run_algo(name, X, C0, max_iter=300):
    n, d = X.shape; k = C0.shape[0]
    C = C0.copy()
    a = np.zeros(n, np.int64); u = np.zeros(n); l = np.zeros(n)
    t0 = time.perf_counter()
    comps = full_scan_init(X, C, a, u, l)
    L = None
    if name == "Elkan":
        L = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    for it in range(1, max_iter + 1):
        Cn = centroid_update(X, a, C, k)
        drift = np.sqrt(((Cn - C) ** 2).sum(-1)); dmax = drift.max()
        C = Cn
        if name == "Lloyd":
            ch, cp = lloyd_iter(X, C, a)
        elif name == "Hamerly":
            CC, s = _cc_s(C)
            ch, cp = hamerly_iter(X, C, a, u, l, drift, dmax, s)
        elif name == "Elkan":
            CC, s = _cc_s(C)
            ch, cp = elkan_iter(X, C, a, u, L, CC, s, drift)
        else:  # DNAK
            indptr, indices = _adjacency_csr(C)
            ch, cp = dnak_iter(X, C, a, u, l, drift, dmax, indptr, indices)
        comps += cp
        if ch == 0:
            Ct = centroid_update(X, a, C, k)
            if np.allclose(Ct, C):
                break
    t = time.perf_counter() - t0
    return dict(name=name, labels=a, C=C, iters=it, comps=comps,
                time=t, sse=_sse(X, a, C))

def warmup():
    X = make_blobs(500, 2, 4, seed=0)
    C0 = kmeanspp_init(X, 4, seed=0)
    for nm in ("Lloyd", "Hamerly", "Elkan", "DNAK"):
        run_algo(nm, X, C0, max_iter=5)

def load_full_image(name):
    from sklearn.datasets import load_sample_image
    rng = np.random.default_rng(0)
    P = load_sample_image(name).astype(np.float64).reshape(-1, 3) / 255.0
    return P + rng.uniform(-1e-7, 1e-7, size=P.shape)

if __name__ == "__main__":
    warmup()
    SEEDS = [101, 202, 303]
    configs = [("synth n=100000 d=2 k=64", make_blobs(100000, 2, 64, 21), 64),
               ("synth n=100000 d=2 k=128", make_blobs(100000, 2, 128, 23), 128),
               ("synth n=100000 d=3 k=64", make_blobs(100000, 3, 64, 25), 64),
               ("china-RGB n=273280 d=3 k=64", load_full_image("china.jpg"), 64)]
    rows = []
    for cname, X, k in configs:
        agg = {}
        for seed in SEEDS:
            C0 = kmeanspp_init(X, k, seed)
            rL = run_algo("Lloyd", X, C0)
            res = [rL]
            for nm in ("Hamerly", "Elkan", "DNAK"):
                r = run_algo(nm, X, C0)
                assert abs(r["sse"] - rL["sse"]) / rL["sse"] < 1e-10, \
                    f"SSE mismatch {nm} {cname} seed {seed}"
                res.append(r)
            for r in res:
                agg.setdefault(r["name"], []).append(r["time"])
        print(f"\n=== {cname} (3 seeds, wall-clock, all exact) ===")
        lm = np.mean(agg["Lloyd"])
        for nm in ("Lloyd", "Hamerly", "Elkan", "DNAK"):
            ts = np.array(agg[nm]); sp = np.array(agg["Lloyd"]) / ts
            print(f"  {nm:8s} time={ts.mean():7.3f}s ± {ts.std():5.3f}"
                  f"   speedup={sp.mean():5.2f}x ± {sp.std():4.2f}")
            rows.append(dict(config=cname, algo=nm,
                             time_mean=round(float(ts.mean()), 3),
                             time_std=round(float(ts.std()), 3),
                             speedup_mean=round(float(sp.mean()), 2),
                             speedup_std=round(float(sp.std()), 2)))
    with open("/home/claude/wallclock.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nALL RUNS EXACT. Saved wallclock.csv")
