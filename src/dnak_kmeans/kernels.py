"""
Numba-compiled kernels for the exact k-means methods compared in the paper:
Lloyd, Hamerly, Elkan, Yinyang, Exponion, Ball k-means, DNAK and DNAK-Lazy.

Every kernel uses the same tie rule: a point changes cluster only if a
centroid is strictly closer than its current one. Every kernel counts the
point-to-centroid distances it evaluates (centroid-to-centroid distances,
which all bound-based methods need, are not counted for any method).
"""
import numpy as np
from numba import njit

BIG = 1e300


@njit(cache=True)
def _dist(X, i, C, j):
    s = 0.0
    for t in range(X.shape[1]):
        dv = X[i, t] - C[j, t]
        s += dv * dv
    return np.sqrt(s)


# ----------------------------------------------------------------- shared
@njit(cache=True)
def full_scan_init(X, C, a, u, l):
    """Initial assignment with exact nearest (u) and second-nearest (l)."""
    n, k = X.shape[0], C.shape[0]
    comps = 0
    for i in range(n):
        b1 = BIG; b2 = BIG; bj = -1
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
        else:                       # empty cluster keeps its old centroid
            for t in range(d):
                C[j, t] = C_old[j, t]
    return C


# ----------------------------------------------------------------- Lloyd
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
            if d < best:
                best = d; bj = j
        if bj != cur:
            a[i] = bj; changed += 1
    return changed, comps


# ----------------------------------------------------------------- Hamerly
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
        b1 = u[i]; bj = cur; b2 = BIG
        for j in range(k):
            if j == cur:
                continue
            d = _dist(X, i, C, j); comps += 1
            if d < b1:
                b2 = b1; b1 = d; bj = j
            elif d < b2:
                b2 = d
        if bj != cur:
            a[i] = bj; changed += 1
        u[i] = b1; l[i] = b2
    return changed, comps


# ----------------------------------------------------------------- Elkan
@njit(cache=True)
def elkan_init(X, C, a, u, L):
    n, k = X.shape[0], C.shape[0]
    comps = 0
    for i in range(n):
        b1 = BIG; bj = -1
        for j in range(k):
            d = _dist(X, i, C, j); comps += 1
            L[i, j] = d
            if d < b1:
                b1 = d; bj = j
        a[i] = bj; u[i] = b1
    return comps


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
                if d < u[i]:
                    a[i] = j; u[i] = d
        if a[i] != old:
            changed += 1
    return changed, comps


# ----------------------------------------------------------------- Yinyang
@njit(cache=True)
def yy_init(X, C, g_of, t, a, u, LB):
    """One distance per (point, centroid): nearest centroid and, for every
    group, the smallest distance to a non-assigned centroid in the group."""
    n, k = X.shape[0], C.shape[0]
    comps = 0
    dd = np.empty(k)
    for i in range(n):
        b1 = BIG; bj = -1
        for j in range(k):
            d = _dist(X, i, C, j); comps += 1
            dd[j] = d
            if d < b1:
                b1 = d; bj = j
        a[i] = bj; u[i] = b1
        for gg in range(t):
            LB[i, gg] = BIG
        for j in range(k):
            if j != bj and dd[j] < LB[i, g_of[j]]:
                LB[i, g_of[j]] = dd[j]
    return comps


@njit(cache=True)
def yy_iter(X, C, a, u, LB, drift, gd, g_of, gptr, gidx, t):
    """Yinyang with the global filter and the group filter
    (Ding et al., 2015); the optional local filter is not used."""
    n = X.shape[0]
    changed = 0; comps = 0
    gm1 = np.empty(t); gm2 = np.empty(t)
    gm1j = np.empty(t, np.int64); scanned = np.empty(t, np.bool_)
    for i in range(n):
        u[i] += drift[a[i]]
        gmin = BIG
        for gg in range(t):
            LB[i, gg] -= gd[gg]
            if LB[i, gg] < gmin:
                gmin = LB[i, gg]
        if u[i] <= gmin:
            continue
        old = a[i]
        ut = _dist(X, i, C, old); comps += 1
        u[i] = ut
        if u[i] <= gmin:
            continue
        b1 = ut; b1j = old
        for gg in range(t):
            scanned[gg] = False
            if ut <= LB[i, gg] and gg != g_of[old]:
                continue
            scanned[gg] = True
            m1 = BIG; m1j = -1; m2 = BIG
            if gg == g_of[old]:
                m1 = ut; m1j = old
            for p in range(gptr[gg], gptr[gg + 1]):
                j = gidx[p]
                if j == old:
                    continue
                d = _dist(X, i, C, j); comps += 1
                if d < m1:
                    m2 = m1; m1 = d; m1j = j
                elif d < m2:
                    m2 = d
                if d < b1:
                    b1 = d; b1j = j
            gm1[gg] = m1; gm2[gg] = m2; gm1j[gg] = m1j
        for gg in range(t):
            if not scanned[gg]:
                continue
            LB[i, gg] = gm2[gg] if gm1j[gg] == b1j else gm1[gg]
        if b1j != old:
            go = g_of[old]
            if ut < LB[i, go]:
                LB[i, go] = ut
            a[i] = b1j; changed += 1
        u[i] = b1
    return changed, comps


# ----------------------------------------------------------------- Exponion
@njit(cache=True)
def exp_iter(X, C, a, u, l, drift, dmax, sidx, sval):
    """Exponion-style search (Newling and Fleuret, 2016): candidates are the
    centroids within distance 2u of the assigned centroid, read from that
    centroid's sorted list of centroid-centroid distances."""
    n = X.shape[0]; km1 = sidx.shape[1]
    changed = 0; comps = 0
    for i in range(n):
        u[i] += drift[a[i]]; l[i] -= dmax
        if u[i] <= l[i]:
            continue
        old = a[i]
        ut = _dist(X, i, C, old); comps += 1
        u[i] = ut
        if u[i] <= l[i]:
            continue
        r = 2.0 * ut
        b1 = ut; b1j = old; b2 = BIG
        p = 0
        while p < km1 and sval[old, p] <= r:
            j = sidx[old, p]
            d = _dist(X, i, C, j); comps += 1
            if d < b1:
                b2 = b1; b1 = d; b1j = j
            elif d < b2:
                b2 = d
            p += 1
        outer = (sval[old, p] - ut) if p < km1 else BIG
        if b1j != old:
            a[i] = b1j; changed += 1
        u[i] = b1
        l[i] = b2 if b2 < outer else outer
    return changed, comps


# ----------------------------------------------------------------- Ball k-means
@njit(cache=True)
def ball_radius(X, C, a, dcur, k):
    """Distance of every point to its own centroid, and cluster radii."""
    n = X.shape[0]
    r = np.zeros(k)
    for i in range(n):
        d = _dist(X, i, C, a[i])
        dcur[i] = d
        if d > r[a[i]]:
            r[a[i]] = d
    return r, n


@njit(cache=True)
def ball_assign(X, C, a, dcur, nptr, nidx, ndist):
    """Neighbour-cluster and annular-region filtering of Ball k-means
    (Xia et al., 2022). For a point at distance d from its centroid c_j,
    only neighbours c_h with ||c_j - c_h|| < 2d can be strictly closer; the
    neighbour list of c_j is sorted by ||c_j - c_h|| so the scan stops at
    the first neighbour outside the point's annulus."""
    n = X.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        cur = a[i]; d0 = dcur[i]
        best = d0; bj = cur
        lim = 2.0 * d0
        for p in range(nptr[cur], nptr[cur + 1]):
            if ndist[p] >= lim:
                break
            h = nidx[p]
            d = _dist(X, i, C, h); comps += 1
            if d < best:
                best = d; bj = h
        if bj != cur:
            a[i] = bj; changed += 1
    return changed, comps


# ----------------------------------------------------------------- DNAK
@njit(cache=True)
def dnak_iter(X, C, a, u, l, drift, dmax, indptr, indices):
    """One DNAK assignment pass (Algorithm 1 of the paper)."""
    n = X.shape[0]
    changed = 0; comps = 0; walks = 0; hops = 0
    for i in range(n):
        u[i] += drift[a[i]]; l[i] -= dmax
        if u[i] <= l[i]:
            continue
        u[i] = _dist(X, i, C, a[i]); comps += 1
        if u[i] <= l[i]:
            continue
        walks += 1
        cur = a[i]; curd = u[i]; second = BIG
        while True:
            b = BIG; bj = -1
            for p in range(indptr[cur], indptr[cur + 1]):
                j = indices[p]
                d = _dist(X, i, C, j); comps += 1
                if d < b:
                    b = d; bj = j
            if b < curd:            # a strictly closer neighbour: hop
                cur = bj; curd = b; hops += 1
            else:                   # local minimum = global nearest (Lemma 1)
                second = b          # exact second-nearest (Lemma 2)
                break
        if cur != a[i] and second == curd:
            # exact tie among nearest centroids: Lloyd's scan keeps the
            # lowest index, so do the same among the tied neighbours
            t = cur
            comps += indptr[t + 1] - indptr[t]
            for p in range(indptr[t], indptr[t + 1]):
                j = indices[p]
                if j < cur and _dist(X, i, C, j) == curd:
                    cur = j
        if cur != a[i]:
            a[i] = cur; changed += 1
        u[i] = curd; l[i] = second
    return changed, comps, walks, hops


@njit(cache=True)
def dnak_s_iter(X, C, a, u, l, drift, dmax, indptr, indices, s):
    """Ablation variant: DNAK with Hamerly's additional s-test."""
    n = X.shape[0]
    changed = 0; comps = 0; walks = 0; hops = 0
    for i in range(n):
        u[i] += drift[a[i]]; l[i] -= dmax
        m = s[a[i]] if s[a[i]] > l[i] else l[i]
        if u[i] <= m:
            continue
        u[i] = _dist(X, i, C, a[i]); comps += 1
        if u[i] <= m:
            continue
        walks += 1
        cur = a[i]; curd = u[i]; second = BIG
        while True:
            b = BIG; bj = -1
            for p in range(indptr[cur], indptr[cur + 1]):
                j = indices[p]
                d = _dist(X, i, C, j); comps += 1
                if d < b:
                    b = d; bj = j
            if b < curd:            # a strictly closer neighbour: hop
                cur = bj; curd = b; hops += 1
            else:                   # local minimum = global nearest (Lemma 1)
                second = b          # exact second-nearest (Lemma 2)
                break
        if cur != a[i] and second == curd:
            # exact tie among nearest centroids: Lloyd's scan keeps the
            # lowest index, so do the same among the tied neighbours
            t = cur
            comps += indptr[t + 1] - indptr[t]
            for p in range(indptr[t], indptr[t + 1]):
                j = indices[p]
                if j < cur and _dist(X, i, C, j) == curd:
                    cur = j
        if cur != a[i]:
            a[i] = cur; changed += 1
        u[i] = curd; l[i] = second
    return changed, comps, walks, hops


@njit(cache=True)
def dnak_lazy_iter(X, C, Cs, a, u, l, drift, dmax, eps, indptr, indices):
    """DNAK with a stale Delaunay graph (Lemma 4 of the paper).

    Cs is the centroid snapshot on which the graph (indptr, indices) was
    built and eps bounds how far any centroid has moved since. The walk is
    run on the snapshot; its answer is accepted only if the certificate
    ||x - c_n0|| < d2_0 - eps holds, otherwise the point falls back to an
    exact full scan. Returns the number of examined points and of
    certificate failures so the driver can decide when to rebuild."""
    n, k = X.shape[0], C.shape[0]
    changed = 0; comps = 0; examined = 0; failed = 0
    for i in range(n):
        u[i] += drift[a[i]]; l[i] -= dmax
        if u[i] <= l[i]:
            continue
        u[i] = _dist(X, i, C, a[i]); comps += 1
        if u[i] <= l[i]:
            continue
        examined += 1
        # greedy walk on the snapshot graph with snapshot coordinates
        cur = a[i]
        if eps == 0.0:              # snapshot is current: reuse u[i]
            curd = u[i]
        else:
            curd = _dist(X, i, Cs, cur); comps += 1
        second = BIG
        while True:
            b = BIG; bj = -1
            for p in range(indptr[cur], indptr[cur + 1]):
                j = indices[p]
                d = _dist(X, i, Cs, j); comps += 1
                if d < b:
                    b = d; bj = j
            if b < curd:
                cur = bj; curd = b
            else:
                second = b
                break
        if cur == a[i]:
            dnow = u[i]
        else:
            dnow = _dist(X, i, C, cur); comps += 1
        if dnow < second - eps:     # certificate holds: cur is the unique nearest
            if cur != a[i]:
                a[i] = cur; changed += 1
            u[i] = dnow; l[i] = second - eps
        else:                       # certificate fails: exact full scan
            failed += 1
            old = a[i]
            b1 = u[i]; bj = old; b2 = BIG
            for j in range(k):
                if j == old:
                    continue
                d = _dist(X, i, C, j); comps += 1
                if d < b1:
                    b2 = b1; b1 = d; bj = j
                elif d < b2:
                    b2 = d
            if bj != old:
                a[i] = bj; changed += 1
            u[i] = b1; l[i] = b2
    return changed, comps, examined, failed


# ----------------------------------------------------------------- Exponion (annuli)
@njit(cache=True)
def centroid_dist_matrix(C):
    k, d = C.shape
    CC = np.empty((k, k))
    for i in range(k):
        CC[i, i] = BIG
        for j in range(i + 1, k):
            s = 0.0
            for t in range(d):
                dv = C[i, t] - C[j, t]
                s += dv * dv
            CC[i, j] = CC[j, i] = np.sqrt(s)
    return CC


@njit(cache=True)
def exp_annuli(CC, nb):
    """Exponion's partial ordering (Newling and Fleuret, 2016): for every
    centroid j the other centroids are grouped into annuli whose outer
    radii grow as m_j * 2^t (m_j = distance to the nearest other centroid),
    by a counting sort in O(k) per row instead of a full sort.
    Returns order (k x k-1), inner radius of every annulus (k x nb) and the
    end offset of every annulus in the order (k x nb)."""
    k = CC.shape[0]
    order = np.empty((k, k - 1), np.int64)
    inner = np.empty((k, nb)); end = np.zeros((k, nb), np.int64)
    bucket = np.empty(k, np.int64); cnt = np.zeros(nb, np.int64)
    for j in range(k):
        m = BIG
        for h in range(k):
            if h != j and CC[j, h] < m:
                m = CC[j, h]
        # only annuli that can hold centroids are needed
        nbj = nb
        if m <= 0.0:
            m = 1e-300
        pw = 0.5 * m
        for t in range(nb):
            cnt[t] = 0
            inner[j, t] = 0.0 if t == 0 else pw
            pw *= 2.0
        for h in range(k):
            if h == j:
                continue
            r = CC[j, h] / m
            if r <= 1.0:
                t = 0
            else:
                t = int(np.ceil(np.log2(r)))
                if t > nb - 1:
                    t = nb - 1
                # guard against rounding in log2: enforce m*2^(t-1) < CC <= m*2^t
                while t > 0 and r <= 2.0 ** (t - 1):
                    t -= 1
                while t < nb - 1 and r > 2.0 ** t:
                    t += 1
            bucket[h] = t; cnt[t] += 1
        pos = 0
        for t in range(nb):
            pos += cnt[t]; end[j, t] = pos
        fill = np.zeros(nb, np.int64)
        for h in range(k):
            if h == j:
                continue
            t = bucket[h]
            start = end[j, t] - cnt[t]
            order[j, start + fill[t]] = h; fill[t] += 1
    return order, inner, end


@njit(cache=True)
def exp_annuli_iter(X, C, a, u, l, drift, dmax, order, inner, end):
    """Exponion assignment pass: scan whole annuli whose inner radius is at
    most 2u; the next unscanned inner radius minus u bounds the rest."""
    n = X.shape[0]; nb = inner.shape[1]
    changed = 0; comps = 0
    for i in range(n):
        u[i] += drift[a[i]]; l[i] -= dmax
        if u[i] <= l[i]:
            continue
        old = a[i]
        ut = _dist(X, i, C, old); comps += 1
        u[i] = ut
        if u[i] <= l[i]:
            continue
        r = 2.0 * ut
        b1 = ut; b1j = old; b2 = BIG
        t = 0; p = 0
        while t < nb and inner[old, t] <= r:
            while p < end[old, t]:
                j = order[old, p]
                d = _dist(X, i, C, j); comps += 1
                if d < b1:
                    b2 = b1; b1 = d; b1j = j
                elif d < b2:
                    b2 = d
                elif d == b1 and j < b1j and b1j != old:
                    b1j = j                     # Lloyd's lowest-index tie rule
                p += 1
            t += 1
        outer = (inner[old, t] - ut) if t < nb else BIG
        if b1j != old:
            a[i] = b1j; changed += 1
        u[i] = b1
        l[i] = b2 if b2 < outer else outer
    return changed, comps



# ----------------------------------------------------------------- DNAK-2
@njit(cache=True)
def sort_adjacency(C, indptr, indices):
    """Edge lengths of the Delaunay graph and, for every centroid, its
    neighbour list sorted by edge length; s[j] = half the shortest edge at j,
    i.e. half the distance from c_j to its nearest other centroid (the
    nearest-neighbour graph is a subgraph of the Delaunay graph)."""
    k = C.shape[0]
    nb = indices.copy()
    el = np.empty(indices.shape[0])
    s = np.empty(k)
    for j in range(k):
        lo = indptr[j]; hi = indptr[j + 1]
        for p in range(lo, hi):
            el[p] = _dist(C, j, C, indices[p])
        order = np.argsort(el[lo:hi])
        tmp_n = nb[lo:hi].copy(); tmp_e = el[lo:hi].copy()
        for q in range(hi - lo):
            nb[lo + q] = tmp_n[order[q]]; el[lo + q] = tmp_e[order[q]]
        s[j] = 0.5 * el[lo] if hi > lo else BIG
    return nb, el, s


@njit(cache=True)
def dnak2_iter(X, C, a, u, l, drift, dmax, indptr, nb, el, s):
    """DNAK-2: DNAK with two filters that need only the Delaunay edges.
    (1) Hamerly's half-separation test, with s(j) taken from the shortest
        Delaunay edge at c_j instead of an O(k^2) distance table.
    (2) Annulus filter on the walk: at a centroid c at distance D from x, a
        neighbour h with ||c - c_h|| > 2D satisfies ||x - c_h|| > D, so it
        can be neither strictly closer nor tied; neighbours are sorted by
        edge length and the scan stops at the first such edge, whose value
        ||c - c_h|| - D bounds all remaining neighbours from below. The new
        lower bound is the minimum of the computed and the bounded
        distances, which is valid because the second-nearest centroid is a
        Delaunay neighbour of the nearest one (Lemma 3)."""
    n = X.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        ai = a[i]
        u[i] += drift[ai]; l[i] -= dmax
        m = l[i] if l[i] > s[ai] else s[ai]
        if u[i] <= m:
            continue
        u[i] = _dist(X, i, C, ai); comps += 1
        if u[i] <= m:
            continue
        cur = ai; D = u[i]
        while True:
            best = BIG; bj = -1; rest = BIG
            lim = 2.0 * D
            for p in range(indptr[cur], indptr[cur + 1]):
                if el[p] > lim:
                    rest = el[p] - D
                    break
                j = nb[p]
                d = _dist(X, i, C, j); comps += 1
                if d < best:
                    best = d; bj = j
            if best < D:
                cur = bj; D = best
            else:
                break
        if cur != ai and best == D:
            # exact tie among nearest centroids: lowest index, as Lloyd's scan
            t = cur
            for p in range(indptr[t], indptr[t + 1]):
                if el[p] > 2.0 * D:
                    break
                j = nb[p]
                comps += 1
                if j < cur and _dist(X, i, C, j) == D:
                    cur = j
        if cur != ai:
            a[i] = cur; changed += 1
        u[i] = D
        l[i] = best if best < rest else rest
    return changed, comps


@njit(cache=True)
def stable_clusters(prev_ptr, prev_idx, ptr, idx, drift):
    """stable[j] is True when c_j and all its Delaunay neighbours did not
    move and the neighbour set of c_j is the same in the previous and the
    current graph. The Voronoi cell of c_j is the intersection of the
    half-spaces defined by its Delaunay neighbours, so the cell is then
    unchanged and no point assigned to c_j can find a strictly closer
    centroid."""
    k = drift.shape[0]
    st = np.zeros(k, np.bool_)
    for j in range(k):
        if drift[j] != 0.0:
            continue
        lo0 = prev_ptr[j]; hi0 = prev_ptr[j + 1]
        lo1 = ptr[j]; hi1 = ptr[j + 1]
        if hi0 - lo0 != hi1 - lo1:
            continue
        ok = True
        for p in range(lo1, hi1):
            if drift[idx[p]] != 0.0:
                ok = False
                break
        if not ok:
            continue
        A = np.sort(prev_idx[lo0:hi0]); B = np.sort(idx[lo1:hi1])
        for q in range(A.shape[0]):
            if A[q] != B[q]:
                ok = False
                break
        st[j] = ok
    return st


@njit(cache=True)
def dnak3_iter(X, C, a, u, l, drift, cum, lastcum, stable,
               indptr, nb, el, s):
    """DNAK-3 = DNAK-2 plus skipping of stable clusters (see
    stable_clusters). The lower bound of a skipped point is corrected
    lazily: cum is the running sum of the maximum drift over iterations,
    and lastcum[i] its value when point i was last processed."""
    n = X.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        ai = a[i]
        if stable[ai]:
            continue
        u[i] += drift[ai]; l[i] -= cum - lastcum[i]; lastcum[i] = cum
        m = l[i] if l[i] > s[ai] else s[ai]
        if u[i] <= m:
            continue
        u[i] = _dist(X, i, C, ai); comps += 1
        if u[i] <= m:
            continue
        cur = ai; D = u[i]
        while True:
            best = BIG; bj = -1; rest = BIG
            lim = 2.0 * D
            for p in range(indptr[cur], indptr[cur + 1]):
                if el[p] > lim:
                    rest = el[p] - D
                    break
                j = nb[p]
                d = _dist(X, i, C, j); comps += 1
                if d < best:
                    best = d; bj = j
            if best < D:
                cur = bj; D = best
            else:
                break
        if cur != ai and best == D:
            t = cur
            for p in range(indptr[t], indptr[t + 1]):
                if el[p] > 2.0 * D:
                    break
                j = nb[p]
                comps += 1
                if j < cur and _dist(X, i, C, j) == D:
                    cur = j
        if cur != ai:
            a[i] = cur; changed += 1
        u[i] = D
        l[i] = best if best < rest else rest
    return changed, comps


@njit(cache=True)
def max_displacement(H, t):
    """M[T] = max_j ||c_j(t) - c_j(T)|| for every earlier iteration T < t,
    from the stored centroid history H[T] (bounds of Newling and Fleuret,
    2016, that do not accumulate one maximum drift per iteration)."""
    k = H.shape[1]; d = H.shape[2]
    M = np.zeros(t + 1)
    for T in range(t):
        mx = 0.0
        for j in range(k):
            s2 = 0.0
            for q in range(d):
                dv = H[t, j, q] - H[T, j, q]
                s2 += dv * dv
            if s2 > mx:
                mx = s2
        M[T] = np.sqrt(mx)
    return M


@njit(cache=True)
def dnak4_iter(X, H, t, M, a, u0, l0, t0, stable, indptr, nb, el, s):
    """DNAK-4 = DNAK-3 with history-based bounds: u0, l0 are the exact
    bounds of point i at iteration t0[i]; at iteration t the bounds are
    u0 + ||c_a(t) - c_a(t0)|| and l0 - M[t0]."""
    n = X.shape[0]; d = X.shape[1]
    C = H[t]
    changed = 0; comps = 0
    for i in range(n):
        ai = a[i]
        if stable[ai]:
            continue
        T = t0[i]
        if T == t:
            continue
        s2 = 0.0
        for q in range(d):
            dv = C[ai, q] - H[T, ai, q]
            s2 += dv * dv
        ub = u0[i] + np.sqrt(s2)
        lb = l0[i] - M[T]
        m = lb if lb > s[ai] else s[ai]
        if ub <= m:
            continue
        ub = _dist(X, i, C, ai); comps += 1
        if ub <= m:
            # tighter u is kept relative to the current iteration
            u0[i] = ub; l0[i] = lb; t0[i] = t
            continue
        cur = ai; D = ub
        while True:
            best = BIG; bj = -1; rest = BIG
            lim = 2.0 * D
            for p in range(indptr[cur], indptr[cur + 1]):
                if el[p] > lim:
                    rest = el[p] - D
                    break
                j = nb[p]
                dd = _dist(X, i, C, j); comps += 1
                if dd < best:
                    best = dd; bj = j
            if best < D:
                cur = bj; D = best
            else:
                break
        if cur != ai and best == D:
            tt = cur
            for p in range(indptr[tt], indptr[tt + 1]):
                if el[p] > 2.0 * D:
                    break
                j = nb[p]
                comps += 1
                if j < cur and _dist(X, i, C, j) == D:
                    cur = j
        if cur != ai:
            a[i] = cur; changed += 1
        u0[i] = D
        l0[i] = best if best < rest else rest
        t0[i] = t
    return changed, comps


# ----------------------------------------------------------------- DNAK-5
@njit(cache=True)
def local_drift(prev_ptr, prev_idx, ptr, nb, el, drift):
    """Per cluster j: dloc[j] = max drift over the neighbours that c_j kept
    from the previous graph, and enew[j] = shortest edge to a neighbour
    that is new in the current graph (BIG if none)."""
    k = drift.shape[0]
    dloc = np.zeros(k); enew = np.full(k, BIG)
    for j in range(k):
        old = np.sort(prev_idx[prev_ptr[j]:prev_ptr[j + 1]])
        mx = 0.0; mn = BIG
        for p in range(ptr[j], ptr[j + 1]):
            h = nb[p]
            # binary search h in old
            lo = 0; hi = old.shape[0] - 1; found = False
            while lo <= hi:
                mid = (lo + hi) // 2
                if old[mid] == h:
                    found = True; break
                elif old[mid] < h:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if found:
                if drift[h] > mx:
                    mx = drift[h]
            else:
                if el[p] < mn:
                    mn = el[p]
        dloc[j] = mx; enew[j] = mn
    return dloc, enew


@njit(cache=True)
def morton_order(X):
    """Point indices sorted along a Z-order curve (first two coordinates),
    so that consecutive points are spatially close."""
    n = X.shape[0]
    mn0 = X[:, 0].min(); mx0 = X[:, 0].max()
    mn1 = X[:, 1].min(); mx1 = X[:, 1].max()
    key = np.empty(n, np.int64)
    for i in range(n):
        xi = int((X[i, 0] - mn0) / (mx0 - mn0 + 1e-300) * 65535.0)
        yi = int((X[i, 1] - mn1) / (mx1 - mn1 + 1e-300) * 65535.0)
        z = 0
        for b in range(16):
            z |= ((xi >> b) & 1) << (2 * b)
            z |= ((yi >> b) & 1) << (2 * b + 1)
        key[i] = z
    return np.argsort(key)


@njit(cache=True)
def walk_init(X, order, C, indptr, nb, el, a, U, L):
    """Initial exact assignment by Delaunay walks instead of a full scan.
    Each point starts its walk at the centroid found for the previous point
    in spatial order. Returns the number of distance computations."""
    comps = 0
    cur0 = 0
    for q in range(order.shape[0]):
        i = order[q]
        cur = cur0; D = _dist(X, i, C, cur); comps += 1
        while True:
            best = BIG; bj = -1; rest = BIG; lim = 2.0 * D
            for p in range(indptr[cur], indptr[cur + 1]):
                if el[p] > lim:
                    rest = el[p] - D; break
                j = nb[p]
                d = _dist(X, i, C, j); comps += 1
                if d < best:
                    best = d; bj = j
            if best < D:
                cur = bj; D = best
            else:
                break
        # ties: lowest index among equally near neighbours (Lloyd's scan)
        if best == D:
            t = cur
            for p in range(indptr[t], indptr[t + 1]):
                if el[p] > 2.0 * D:
                    break
                j = nb[p]; comps += 1
                if j < cur and _dist(X, i, C, j) == D:
                    cur = j
        a[i] = cur; U[i] = D
        L[i] = best if best < rest else rest
        cur0 = cur
    return comps


@njit(cache=True)
def dnak5_iter(X, C, a, U, L, drift, dloc, enew, indptr, nb, el, s):
    """DNAK-5: Delaunay-local bounds.
    Invariant: U >= ||x - c_a|| and L <= min over the current Delaunay
    neighbours h of c_a of ||x - c_h||. By Lemma 1 a point can only have a
    strictly closer centroid if one of these neighbours is strictly closer,
    so U <= L (or U <= s(a)) certifies the assignment. Bounds are corrected
    with the drift of the neighbours of a only (dloc), and a neighbour that
    is new in this graph is bounded through its edge length (enew)."""
    n = X.shape[0]
    changed = 0; comps = 0
    for i in range(n):
        ai = a[i]
        U[i] += drift[ai]
        Li = L[i] - dloc[ai]
        if enew[ai] < BIG:
            e = enew[ai] - U[i]
            if e < Li:
                Li = e
        L[i] = Li
        m = Li if Li > s[ai] else s[ai]
        if U[i] <= m:
            continue
        U[i] = _dist(X, i, C, ai); comps += 1
        if U[i] <= m:
            continue
        cur = ai; D = U[i]
        while True:
            best = BIG; bj = -1; rest = BIG; lim = 2.0 * D
            for p in range(indptr[cur], indptr[cur + 1]):
                if el[p] > lim:
                    rest = el[p] - D; break
                j = nb[p]
                d = _dist(X, i, C, j); comps += 1
                if d < best:
                    best = d; bj = j
            if best < D:
                cur = bj; D = best
            else:
                break
        if cur != ai and best == D:
            t = cur
            for p in range(indptr[t], indptr[t + 1]):
                if el[p] > 2.0 * D:
                    break
                j = nb[p]; comps += 1
                if j < cur and _dist(X, i, C, j) == D:
                    cur = j
        if cur != ai:
            a[i] = cur; changed += 1
        U[i] = D
        L[i] = best if best < rest else rest
    return changed, comps


# ----------------------------------------------------------------- Delaunay certificate
@njit(cache=True)
def _det(Mx, m):
    """Determinant of the leading m x m block of Mx by Gaussian elimination."""
    A = Mx[:m, :m].copy()
    det = 1.0
    for c in range(m):
        piv = c
        for r in range(c + 1, m):
            if abs(A[r, c]) > abs(A[piv, c]):
                piv = r
        if A[piv, c] == 0.0:
            return 0.0
        if piv != c:
            for q in range(m):
                tmp = A[c, q]; A[c, q] = A[piv, q]; A[piv, q] = tmp
            det = -det
        det *= A[c, c]
        for r in range(c + 1, m):
            f = A[r, c] / A[c, c]
            for q in range(c, m):
                A[r, q] -= f * A[c, q]
    return det


@njit(cache=True)
def delaunay_still_valid(C, simplices, neighbors, orient_sign, rel_eps, moved):
    """Exact (up to rel_eps) check that a stored triangulation of the
    previous centroids is still a Delaunay triangulation of the current
    centroids C. Conditions (the Delaunay lemma): every simplex keeps its
    orientation; every interior facet is locally Delaunay, i.e. the vertex
    opposite the facet in the neighbouring simplex lies outside the
    circumsphere; every convex-hull facet still has all centroids on its
    inner side. Any test within rel_eps of zero counts as failed, so the
    graph is rebuilt in doubtful cases."""
    m, dp1 = simplices.shape
    d = dp1 - 1
    k = C.shape[0]
    Mx = np.empty((dp1, dp1))
    # sign of the lifted determinant for a point inside the circumsphere
    # (the barycentre of simplex 0), relative to its orientation sign
    bary = np.zeros(d)
    for r in range(dp1):
        for q in range(d):
            bary[q] += C[simplices[0, r], q] / dp1
    for r in range(dp1):
        s2 = 0.0
        for q in range(d):
            dv = C[simplices[0, r], q] - bary[q]
            Mx[r, q] = dv; s2 += dv * dv
        Mx[r, d] = s2
    inside_sign = 1.0 if _det(Mx, dp1) * orient_sign[0] > 0 else -1.0
    smoved = np.zeros(m, np.bool_)
    for si in range(m):
        for r in range(dp1):
            if moved[simplices[si, r]]:
                smoved[si] = True
    for si in range(m):
        if not smoved[si]:
            # unchanged simplex: only facets towards a changed neighbour,
            # and hull facets, still need testing
            any_nb = False
            for f in range(dp1):
                nbs = neighbors[si, f]
                if nbs < 0 or smoved[nbs]:
                    any_nb = True
            if not any_nb:
                continue
        # orientation
        for r in range(d):
            for q in range(d):
                Mx[r, q] = C[simplices[si, r + 1], q] - C[simplices[si, 0], q]
        o = _det(Mx, d)
        scale = 0.0
        for r in range(d):
            for q in range(d):
                scale += abs(Mx[r, q])
        if o * orient_sign[si] <= rel_eps * (scale ** d):
            return False
        # facets
        for f in range(dp1):
            nbs = neighbors[si, f]
            if nbs >= 0:
                if not (smoved[si] or smoved[nbs]):
                    continue
                if nbs < si and smoved[nbs]:
                    continue                 # tested from the other side
                # opposite vertex in the neighbour: the one not in simplex si
                opp = -1
                for v in range(dp1):
                    cand = simplices[nbs, v]
                    present = False
                    for w in range(dp1):
                        if simplices[si, w] == cand:
                            present = True; break
                    if not present:
                        opp = cand; break
                # in-sphere test: rows (p_v - q, |p_v - q|^2)
                for r in range(dp1):
                    s2 = 0.0
                    for q in range(d):
                        dv = C[simplices[si, r], q] - C[opp, q]
                        Mx[r, q] = dv; s2 += dv * dv
                    Mx[r, d] = s2
                ins = _det(Mx, dp1)
                scale = 0.0
                for r in range(dp1):
                    for q in range(dp1):
                        scale += abs(Mx[r, q])
                # opp must be strictly outside the circumsphere
                if ins * orient_sign[si] * inside_sign >= -rel_eps * (scale ** dp1):
                    return False
            else:
                # hull facet: all centroids on the same side as the opposite
                # vertex simplices[si, f]
                vs = np.empty(d, np.int64); c0 = 0
                for v in range(dp1):
                    if v != f:
                        vs[c0] = simplices[si, v]; c0 += 1
                # side of the vertex opposite the hull facet
                ov = simplices[si, f]
                for r in range(d - 1):
                    for q in range(d):
                        Mx[r, q] = C[vs[r + 1], q] - C[vs[0], q]
                for q in range(d):
                    Mx[d - 1, q] = C[ov, q] - C[vs[0], q]
                def_side = _det(Mx, d)
                if def_side == 0.0:
                    return False
                for v in range(k):
                    if not (moved[v] or smoved[si]):
                        continue
                    skip = (v == ov)
                    for r in range(d):
                        if vs[r] == v:
                            skip = True
                    if skip:
                        continue
                    for q in range(d):
                        Mx[d - 1, q] = C[v, q] - C[vs[0], q]
                    side = _det(Mx, d)
                    if side * def_side <= 0.0:
                        return False
    return True
