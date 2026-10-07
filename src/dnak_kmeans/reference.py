"""
Readable pure NumPy/SciPy reference implementations of Lloyd's algorithm,
Hamerly's algorithm and DNAK. These are written for clarity, not speed;
all timings in the paper come from the compiled kernels in kernels.py.
"""
import numpy as np
import time
from scipy.spatial import Delaunay
try:
    from scipy.spatial import QhullError
except ImportError:  # older SciPy
    from scipy.spatial.qhull import QhullError


# ---------------------------------------------------------------- data / init

def make_blobs(n, d, k_true, seed, spread=0.6, box=10.0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-box, box, size=(k_true, d))
    sizes = rng.multinomial(n, np.ones(k_true) / k_true)
    parts = [rng.normal(c, spread, size=(s, d)) for c, s in zip(centers, sizes)]
    X = np.vstack(parts)
    rng.shuffle(X)
    return X.astype(np.float64)

def kmeanspp_init(X, k, seed):
    """Standard k-means++ seeding (shared identically by all algorithms)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(n)]
    d2 = np.sum((X - C[0]) ** 2, axis=1)
    for j in range(1, k):
        probs = d2 / d2.sum()
        C[j] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((X - C[j]) ** 2, axis=1))
    return C

def sse_of(X, labels, C):
    return float(np.sum((X - C[labels]) ** 2))

def update_centroids(X, labels, C_old, k):
    C = C_old.copy()
    for j in range(k):
        pts = X[labels == j]
        if len(pts) > 0:
            C[j] = pts.mean(axis=0)
    return C

# ---------------------------------------------------------------- Lloyd's

def lloyd(X, C0, max_iter=300):
    n, d = X.shape
    k = C0.shape[0]
    C = C0.copy()
    labels = np.full(n, -1)
    comps = 0
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)  # n x k
        comps += n * k
        new_labels = np.argmin(D, axis=1)
        if it > 1:  # tie rule: keep current assignment unless strictly closer
            dmin = D[np.arange(n), new_labels]
            keep = D[np.arange(n), labels] <= dmin
            new_labels[keep] = labels[keep]
        changed = int(np.sum(new_labels != labels))
        labels = new_labels
        C = update_centroids(X, labels, C, k)
        if changed == 0:
            break
    t = time.perf_counter() - t0
    return dict(name="Lloyd", labels=labels, C=C, iters=it,
                comps=comps, time=t, sse=sse_of(X, labels, C))

# ---------------------------------------------------------------- Hamerly

def hamerly(X, C0, max_iter=300):
    n, d = X.shape
    k = C0.shape[0]
    C = C0.copy()
    comps = 0
    cc_comps = 0  # centroid-centroid distances (overhead, O(k^2) per iter)
    t0 = time.perf_counter()

    # initial full scan
    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    comps += n * k
    order = np.argsort(D, axis=1)
    a = order[:, 0]
    u = D[np.arange(n), a]
    l = D[np.arange(n), order[:, 1]]

    for it in range(1, max_iter + 1):
        Cn = update_centroids(X, a, C, k)
        drift = np.linalg.norm(Cn - C, axis=1)
        C = Cn
        dmax = drift.max()
        # s(j) = 0.5 * dist to nearest other centroid
        CC = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
        cc_comps += k * k
        np.fill_diagonal(CC, np.inf)
        s = 0.5 * CC.min(axis=1)

        u = u + drift[a]
        l = l - dmax

        m = np.maximum(s[a], l)
        cand = np.where(u > m)[0]
        # tighten u for candidates
        if len(cand) > 0:
            u[cand] = np.linalg.norm(X[cand] - C[a[cand]], axis=1)
            comps += len(cand)
            cand2 = cand[u[cand] > m[cand]]
        else:
            cand2 = cand
        changed = 0
        if len(cand2) > 0:
            D2 = np.linalg.norm(X[cand2, None, :] - C[None, :, :], axis=2)
            comps += len(cand2) * k
            order2 = np.argsort(D2, axis=1)
            na = order2[:, 0]
            dmin2 = D2[np.arange(len(cand2)), na]
            keep = D2[np.arange(len(cand2)), a[cand2]] <= dmin2
            na[keep] = a[cand2][keep]  # tie rule: keep current
            changed = int(np.sum(na != a[cand2]))
            a[cand2] = na
            u[cand2] = D2[np.arange(len(cand2)), na]
            # lower bound: min distance among centroids != assigned
            D2m = D2.copy()
            D2m[np.arange(len(cand2)), na] = np.inf
            l[cand2] = D2m.min(axis=1)
        if changed == 0 and dmax == 0.0:
            break
        if changed == 0:
            # one more centroid update pass to confirm convergence
            Ct = update_centroids(X, a, C, k)
            if np.allclose(Ct, C):
                break
    t = time.perf_counter() - t0
    return dict(name="Hamerly", labels=a, C=C, iters=it,
                comps=comps, cc_comps=cc_comps, time=t, sse=sse_of(X, a, C))

# ---------------------------------------------------------------- DNAK

def delaunay_adjacency(C):
    """Adjacency lists of the Delaunay graph over centroids."""
    k = C.shape[0]
    try:
        tri = Delaunay(C)
    except QhullError:
        tri = Delaunay(C, qhull_options="QJ")
    adj = [set() for _ in range(k)]
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                adj[simplex[i]].add(simplex[j])
                adj[simplex[j]].add(simplex[i])
    return [np.fromiter(s, dtype=np.int64) for s in adj]

def dnak(X, C0, max_iter=300):
    n, d = X.shape
    k = C0.shape[0]
    C = C0.copy()
    comps = 0
    graph_time = 0.0
    skipped_total = 0
    walk_hops_total = 0
    t0 = time.perf_counter()

    # initial full scan (exact u and l)
    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    comps += n * k
    order = np.argsort(D, axis=1)
    a = order[:, 0]
    u = D[np.arange(n), a]
    l = D[np.arange(n), order[:, 1]]

    for it in range(1, max_iter + 1):
        Cn = update_centroids(X, a, C, k)
        drift = np.linalg.norm(Cn - C, axis=1)
        C = Cn
        dmax = drift.max()

        tg = time.perf_counter()
        adj = delaunay_adjacency(C)
        graph_time += time.perf_counter() - tg

        u = u + drift[a]
        l = l - dmax

        changed = 0
        for i in range(n):
            if u[i] <= l[i]:
                skipped_total += 1
                continue
            # tighten upper bound
            du = np.linalg.norm(X[i] - C[a[i]])
            comps += 1
            u[i] = du
            if u[i] <= l[i]:
                skipped_total += 1
                continue
            # Delaunay greedy walk
            cur = a[i]
            cur_d = du
            hops = 0
            while True:
                nb = adj[cur]
                nd = np.linalg.norm(X[i] - C[nb], axis=1)
                comps += len(nb)
                jmin = int(np.argmin(nd))
                if nd[jmin] < cur_d:
                    cur = int(nb[jmin])
                    cur_d = float(nd[jmin])
                    hops += 1
                else:
                    # local min on Delaunay graph = global nearest (exact)
                    second = float(nd[jmin])  # min over neighbors of final site
                    break
            walk_hops_total += hops
            if cur != a[i]:
                changed += 1
                a[i] = cur
            u[i] = cur_d
            l[i] = second  # exact 2nd-nearest via order-2 Voronoi lemma
        if changed == 0 and dmax < 1e-12:
            break
        if changed == 0:
            Ct = update_centroids(X, a, C, k)
            if np.allclose(Ct, C):
                break
    t = time.perf_counter() - t0
    return dict(name="DNAK", labels=a, C=C, iters=it, comps=comps,
                time=t, graph_time=graph_time, sse=sse_of(X, a, C),
                skipped=skipped_total, hops=walk_hops_total)

# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    X = make_blobs(5000, 2, 32, seed=7)
    C0 = kmeanspp_init(X, 32, seed=8)
    rL, rH, rD = lloyd(X, C0), hamerly(X, C0), dnak(X, C0)
    for r in (rL, rH, rD):
        print(f"{r['name']:8s} iters={r['iters']:3d} comps={r['comps']:>10,d} "
              f"sse={r['sse']:.6f}")
    assert np.array_equal(rL["labels"], rD["labels"]), "DNAK differs from Lloyd"
    assert np.array_equal(rL["labels"], rH["labels"]), "Hamerly differs from Lloyd"
    print("reference implementations agree with Lloyd's algorithm")
