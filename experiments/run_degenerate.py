"""Exactness on degenerate and pathological inputs, without jitter.

Every method is compared with Lloyd's algorithm (identical iterations,
assignments and centroids). Writes results/degenerate.csv. A mismatch is
recorded, not raised, because exact ties are where tie-breaking between
methods can legitimately differ."""
import csv, os
import numpy as np
from _setup import RESULTS
from dnak_kmeans import ALL_ALGOS, run_algo, is_exact, warmup, make_blobs, kmeanspp_init

rng = np.random.default_rng(0)
g = np.stack(np.meshgrid(np.arange(60), np.arange(60)), -1).reshape(-1, 2).astype(float)
cases = []
# 1. integer grid: massive exact ties between points and centroids
cases.append(("integer grid 60x60 (exact ties)", g, 40))
# 2. every point duplicated three times
B = make_blobs(3000, 2, 20, 1); cases.append(("duplicated points (x3)", np.repeat(B, 3, 0), 20))
# 3. all points collinear in the plane (centroids collinear -> no 2-D triangulation)
t = rng.uniform(0, 1, 5000); cases.append(("collinear points in R^2", np.c_[t, 2 * t + 1], 16))
# 4. points on a circle (co-circular configurations)
th = rng.uniform(0, 2 * np.pi, 5000); cases.append(("points on a circle", np.c_[np.cos(th), np.sin(th)], 24))
# 5. coordinate scale 1e6 and 1e-6
cases.append(("scale 1e6", make_blobs(5000, 2, 30, 2) * 1e6, 30))
cases.append(("scale 1e-6", make_blobs(5000, 2, 30, 3) * 1e-6, 30))
# 6. k close to n
cases.append(("k close to n (n=60, k=50)", make_blobs(60, 2, 5, 4), 50))
# 7. k <= d+1 (no Delaunay triangulation possible)
cases.append(("k = 3 in d = 3", make_blobs(3000, 3, 3, 5), 3))
# 8. heavily overlapping clusters
cases.append(("overlapping clusters (spread 5)", make_blobs(5000, 2, 30, 6, spread=5.0), 30))
# 9. 8-bit RGB values without jitter
from sklearn.datasets import load_sample_image
P = load_sample_image("flower.jpg").reshape(-1, 3)[::20].astype(float) / 255.0
cases.append(("8-bit RGB, no jitter", P, 32))

warmup()
rows = []
for name, X, k in cases:
    for seed in (0, 1, 2):
        C0 = kmeanspp_init(X, k, seed)
        ref = run_algo("Lloyd", X, C0)
        for m in ALL_ALGOS[1:] + ["DNAK-6"]:
            r = run_algo(m, X, C0)
            rows.append(dict(case=name, seed=seed, algo=m, exact=is_exact(r, ref),
                             graph_fallbacks=r.get("graph_fallbacks", "")))
with open(os.path.join(RESULTS, "degenerate.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows)
print(d.pivot_table(index="case", columns="algo", values="exact", aggfunc="sum").to_string())
f = d[d.algo == "DNAK"].copy(); f["graph_fallbacks"] = pd.to_numeric(f.graph_fallbacks)
print("complete-graph fallbacks per case:", f.groupby("case").graph_fallbacks.sum().to_dict())
