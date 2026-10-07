"""
Common driver for all compiled methods. Every method runs the same outer
loop (update centroids, compute drifts, assignment pass, convergence test)
from the same initial centroids, so the only difference between methods
is the assignment pass. Timings include all per-iteration overheads
(centroid-centroid distances, sorting, Delaunay construction).
"""
import time
import numpy as np
from scipy.spatial import Delaunay

from . import kernels as K

ALGOS = ["Lloyd", "Hamerly", "Elkan", "Yinyang", "Exponion", "Ball", "DNAK"]
ALL_ALGOS = ALGOS + ["DNAK-Lazy"]
ABLATION = ["Hamerly", "Hamerly-noS", "DNAK", "DNAK+s"]
LAZY_REBUILD_FRACTION = 0.10   # rebuild when >10% of examined points fail


def centroid_distances(C):
    CC = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(CC, np.inf)
    return CC


def complete_csr(k):
    """Complete graph on k centroids. Every graph that contains all Delaunay
    edges keeps the walk and the second-nearest bound exact, so the
    complete graph is a safe (if slow) fallback."""
    indptr = np.arange(0, k * (k - 1) + 1, k - 1, dtype=np.int64)
    indices = np.array([j for i in range(k) for j in range(k) if j != i],
                       dtype=np.int64)
    return indptr, indices


def delaunay_csr(C, stats=None):
    """Delaunay graph of the centroids in CSR form (indptr, indices).

    Qhull triangulates co-spherical configurations ('Qt'), which only adds
    edges to the Delaunay graph and is therefore safe. If Qhull cannot
    build a full-dimensional triangulation (fewer than d+2 centroids, or
    all centroids in a lower-dimensional flat), the complete graph is used
    instead of a joggled triangulation, because a joggled triangulation
    is not guaranteed to contain every Delaunay edge."""
    k, d = C.shape
    if k <= d + 1:
        out = complete_csr(k)
    else:
        try:
            tri = Delaunay(C)
            indptr, indices = tri.vertex_neighbor_vertices
            out = indptr.astype(np.int64), indices.astype(np.int64)
            if np.any(np.diff(out[0]) == 0):   # a centroid left out (duplicate)
                out = complete_csr(k)
        except Exception:
            out = complete_csr(k)
            if stats is not None:
                stats["fallbacks"] = stats.get("fallbacks", 0) + 1
    if stats is not None:
        deg = np.diff(out[0])
        stats["deg_sum"] = stats.get("deg_sum", 0.0) + float(deg.mean())
        stats["deg_max"] = max(stats.get("deg_max", 0), int(deg.max()))
        stats["graphs"] = stats.get("graphs", 0) + 1
    return out


def yinyang_groups(C0, t, seed=0, iters=5):
    """Static centroid groups from a short k-means on the initial
    centroids, as in Ding et al. (2015)."""
    rng = np.random.default_rng(seed)
    k = C0.shape[0]
    G = C0[rng.choice(k, size=t, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmin(((C0[:, None, :] - G[None, :, :]) ** 2).sum(-1), 1)
        for g in range(t):
            if np.any(lab == g):
                G[g] = C0[lab == g].mean(0)
    lab = np.argmin(((C0[:, None, :] - G[None, :, :]) ** 2).sum(-1), 1)
    order = np.argsort(lab, kind="stable")
    gptr = np.zeros(t + 1, np.int64)
    for g in range(t):
        gptr[g + 1] = gptr[g] + int((lab == g).sum())
    return lab.astype(np.int64), gptr, order.astype(np.int64)


def ball_neighbours(C, r):
    """Neighbour clusters of each cluster j: all h with ||c_j - c_h|| < 2 r_j,
    sorted by centroid distance (CSR arrays)."""
    CC = centroid_distances(C)
    k = C.shape[0]
    ptr = np.zeros(k + 1, np.int64); idx_l = []; dst_l = []
    for j in range(k):
        cand = np.nonzero(CC[j] < 2.0 * r[j])[0]
        cand = cand[np.argsort(CC[j, cand], kind="stable")]
        idx_l.append(cand); dst_l.append(CC[j, cand])
        ptr[j + 1] = ptr[j] + len(cand)
    idx = np.concatenate(idx_l).astype(np.int64) if k else np.empty(0, np.int64)
    dst = np.concatenate(dst_l) if k else np.empty(0)
    return ptr, idx, dst


def delaunay_full(C):
    """Triangulation of the centroids with the pieces needed for the
    certificate check: CSR graph, simplices, neighbours and orientation."""
    k, d = C.shape
    if k <= d + 1:
        ip, ix = complete_csr(k)
        return ip, ix, None, None, None
    try:
        tri = Delaunay(C)
    except Exception:
        ip, ix = complete_csr(k)
        return ip, ix, None, None, None
    ip, ix = tri.vertex_neighbor_vertices
    if np.any(np.diff(ip) == 0) or len(tri.coplanar):
        ip, ix = complete_csr(k)
        return ip, ix, None, None, None
    S = tri.simplices.astype(np.int64); N = tri.neighbors.astype(np.int64)
    V = C[S]
    E = V[:, 1:, :] - V[:, :1, :]
    sign = np.sign(np.linalg.det(E)).astype(np.float64)
    return ip.astype(np.int64), ix.astype(np.int64), S, N, sign


def sse(X, a, C):
    return float(((X - C[a]) ** 2).sum())


def run_algo(name, X, C0, max_iter=500):
    """Run one method from initial centroids C0. Returns labels, final
    centroids, iterations, distance computations, wall-clock time, SSE,
    and method-specific counters."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, d = X.shape; k = C0.shape[0]
    C = C0.astype(np.float64).copy()
    a = np.zeros(n, np.int64); u = np.zeros(n); l = np.zeros(n)
    extra = dict(graph_time=0.0, walks=0, hops=0, rebuilds=0,
                 examined=0, failed=0, sort_time=0.0)
    gstats = {}
    t0 = time.perf_counter()

    if name == "Elkan":
        L = np.empty((n, k))
        comps = K.elkan_init(X, C, a, u, L)
    elif name == "Yinyang":
        t = max(1, k // 10)
        g_of, gptr, gidx = yinyang_groups(C0, t)
        LB = np.empty((n, t))
        comps = K.yy_init(X, C, g_of, t, a, u, LB)
    elif name in ("DNAK-5", "DNAK-6"):
        pptr, pidx = delaunay_csr(C)
        pnb, pel, ps = K.sort_adjacency(C, pptr, pidx)
        order = K.morton_order(X) if d >= 2 else np.arange(n)
        comps = K.walk_init(X, order, C, pptr, pnb, pel, a, u, l)
    else:
        comps = K.full_scan_init(X, C, a, u, l)
    if name == "Ball":
        dcur = np.empty(n)
    if name == "DNAK-Lazy":
        Cs = None; indptr = indices = None; rebuild = True

    it = 0
    for it in range(1, max_iter + 1):
        Cn = K.centroid_update(X, a, C, k)
        drift = np.sqrt(((Cn - C) ** 2).sum(-1)); dmax = float(drift.max())
        C = Cn
        if name == "Lloyd":
            ch, cp = K.lloyd_iter(X, C, a)
        elif name == "Hamerly":
            s = 0.5 * centroid_distances(C).min(axis=1)
            ch, cp = K.hamerly_iter(X, C, a, u, l, drift, dmax, s)
        elif name == "Hamerly-noS":                 # ablation
            ch, cp = K.hamerly_iter(X, C, a, u, l, drift, dmax, np.zeros(k))
        elif name == "Elkan":
            CC = centroid_distances(C)
            s = 0.5 * CC.min(axis=1)
            ch, cp = K.elkan_iter(X, C, a, u, L, CC, s, drift)
        elif name == "Yinyang":
            gd = np.zeros(len(gptr) - 1)
            for g in range(len(gd)):
                if gptr[g + 1] > gptr[g]:
                    gd[g] = drift[gidx[gptr[g]:gptr[g + 1]]].max()
            ch, cp = K.yy_iter(X, C, a, u, LB, drift, gd, g_of, gptr, gidx,
                               len(gd))
        elif name == "Exponion-annuli":
            ts = time.perf_counter()
            CC = K.centroid_dist_matrix(C)
            order, inner, end = K.exp_annuli(CC, 40)
            extra["sort_time"] += time.perf_counter() - ts
            ch, cp = K.exp_annuli_iter(X, C, a, u, l, drift, dmax,
                                       order, inner, end)
        elif name == "Exponion":   # full sort of centroid distances (main results)
            ts = time.perf_counter()
            CC = centroid_distances(C)
            sidx = np.argsort(CC, axis=1)[:, :k - 1].astype(np.int64)
            sval = np.ascontiguousarray(np.take_along_axis(CC, sidx, axis=1))
            extra["sort_time"] += time.perf_counter() - ts
            ch, cp = K.exp_iter(X, C, a, u, l, drift, dmax, sidx, sval)
        elif name == "Ball":
            r, cp0 = K.ball_radius(X, C, a, dcur, k)
            nptr, nidx, ndst = ball_neighbours(C, r)
            ch, cp = K.ball_assign(X, C, a, dcur, nptr, nidx, ndst)
            cp += cp0
        elif name in ("DNAK", "DNAK+s"):
            tg = time.perf_counter()
            indptr, indices = delaunay_csr(C, gstats)
            extra["graph_time"] += time.perf_counter() - tg
            if name == "DNAK":
                ch, cp, w, h = K.dnak_iter(X, C, a, u, l, drift, dmax,
                                           indptr, indices)
            else:                                    # ablation
                s = 0.5 * centroid_distances(C).min(axis=1)
                ch, cp, w, h = K.dnak_s_iter(X, C, a, u, l, drift, dmax,
                                             indptr, indices, s)
            extra["walks"] += w; extra["hops"] += h
        elif name == "DNAK-6":
            tg = time.perf_counter()
            if it == 1:
                S6 = N6 = sg6 = None
            reuse = (S6 is not None and
                     K.delaunay_still_valid(C, S6, N6, sg6, 1e-12, drift > 0.0))
            if reuse:
                indptr, indices = pptr, pidx
            else:
                indptr, indices, S6, N6, sg6 = delaunay_full(C)
                extra["rebuilds"] += 1
            nb, el, s2 = K.sort_adjacency(C, indptr, indices)
            dloc, enew = K.local_drift(pptr, pidx, indptr, nb, el, drift)
            pptr, pidx = indptr, indices
            extra["graph_time"] += time.perf_counter() - tg
            ch, cp = K.dnak5_iter(X, C, a, u, l, drift, dloc, enew, indptr, nb, el, s2)
        elif name == "DNAK-5":
            tg = time.perf_counter()
            indptr, indices = delaunay_csr(C, gstats)
            nb, el, s2 = K.sort_adjacency(C, indptr, indices)
            dloc, enew = K.local_drift(pptr, pidx, indptr, nb, el, drift)
            pptr, pidx = indptr, indices
            extra["graph_time"] += time.perf_counter() - tg
            ch, cp = K.dnak5_iter(X, C, a, u, l, drift, dloc, enew, indptr, nb, el, s2)
        elif name == "DNAK-4":
            tg = time.perf_counter()
            if it == 1:
                pptr, pidx = delaunay_csr(C0.astype(np.float64))
                H = np.empty((max_iter + 1, k, d)); H[0] = C0
                u0 = u.copy(); l0 = l.copy(); tl = np.zeros(n, np.int64)
            H[it] = C
            M = K.max_displacement(H, it)
            indptr, indices = delaunay_csr(C, gstats)
            nb, el, s2 = K.sort_adjacency(C, indptr, indices)
            st = K.stable_clusters(pptr, pidx, indptr, indices, drift)
            pptr, pidx = indptr, indices
            extra["graph_time"] += time.perf_counter() - tg
            ch, cp = K.dnak4_iter(X, H, it, M, a, u0, l0, tl, st, indptr, nb, el, s2)
        elif name == "DNAK-3":
            tg = time.perf_counter()
            if it == 1:
                pptr, pidx = delaunay_csr(C0.astype(np.float64))
                cum = 0.0; lastcum = np.zeros(n)
            indptr, indices = delaunay_csr(C, gstats)
            nb, el, s2 = K.sort_adjacency(C, indptr, indices)
            st = K.stable_clusters(pptr, pidx, indptr, indices, drift)
            pptr, pidx = indptr, indices
            extra["graph_time"] += time.perf_counter() - tg
            cum += dmax
            ch, cp = K.dnak3_iter(X, C, a, u, l, drift, cum, lastcum, st,
                                  indptr, nb, el, s2)
            extra["walks"] += int(st.sum())
        elif name == "DNAK-2":
            tg = time.perf_counter()
            indptr, indices = delaunay_csr(C, gstats)
            nb, el, s2 = K.sort_adjacency(C, indptr, indices)
            extra["graph_time"] += time.perf_counter() - tg
            ch, cp = K.dnak2_iter(X, C, a, u, l, drift, dmax, indptr, nb, el, s2)
        elif name == "DNAK-Lazy":
            if rebuild:
                tg = time.perf_counter()
                indptr, indices = delaunay_csr(C)
                extra["graph_time"] += time.perf_counter() - tg
                Cs = C.copy(); extra["rebuilds"] += 1
            eps = float(np.sqrt(((C - Cs) ** 2).sum(-1)).max())
            ch, cp, ex, fa = K.dnak_lazy_iter(X, C, Cs, a, u, l, drift, dmax,
                                              eps, indptr, indices)
            extra["examined"] += ex; extra["failed"] += fa
            rebuild = ex > 0 and fa > LAZY_REBUILD_FRACTION * ex
        else:
            raise ValueError(name)
        comps += cp
        if ch == 0:
            Ct = K.centroid_update(X, a, C, k)
            if np.allclose(Ct, C):
                break
    tm = time.perf_counter() - t0
    out = dict(name=name, labels=a, C=C, iters=it, comps=int(comps),
               time=tm, sse=sse(X, a, C))
    out.update(extra)
    if gstats.get("graphs"):
        out["deg_mean"] = gstats["deg_sum"] / gstats["graphs"]
        out["deg_max"] = gstats["deg_max"]
        out["graph_fallbacks"] = gstats.get("fallbacks", 0)
    return out


def is_exact(r, ref):
    """Identical assignments and centroids as the reference (Lloyd) run."""
    return (r["iters"] == ref["iters"]
            and np.array_equal(r["labels"], ref["labels"])
            and np.array_equal(r["C"], ref["C"]))


def warmup():
    """Compile every kernel once so that JIT time is excluded from timing."""
    from .data import make_blobs, kmeanspp_init
    for d in (2, 3):
        X = make_blobs(800, d, 6, seed=0)
        C0 = kmeanspp_init(X, 12, seed=0)
        for nm in ALL_ALGOS + ["Hamerly-noS", "DNAK+s", "Exponion-annuli", "DNAK-2", "DNAK-3", "DNAK-4", "DNAK-5", "DNAK-6"]:
            run_algo(nm, X, C0, max_iter=6)
