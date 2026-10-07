"""Exponion implementation check: full sort of the centroid-distance table
(the variant used in the main benchmark) against the annulus ordering of
Newling and Fleuret (2016), with the time spent on centroid-side work
(centroid distances and ordering) recorded separately. DNAK is run on
the same seeds. Writes results/raw_exponion_check.csv."""
import csv, os
from _setup import RESULTS
from dnak_kmeans import run_algo, is_exact, warmup, make_blobs, kmeanspp_init, load_uber

CONFIGS = [("synth-2D-k128", lambda: make_blobs(100000, 2, 128, 23), 128),
           ("uber-k256", lambda: load_uber(200000, seed=0), 256),
           ("sweep-2D-k512", lambda: make_blobs(100000, 2, 512, 2512), 512),
           ("sweep-3D-k512", lambda: make_blobs(100000, 3, 512, 3512), 512)]
METHODS = ["Exponion", "Exponion-annuli", "DNAK"]
warmup()
rows = []
for name, loader, k in CONFIGS:
    X = loader()
    for seed in (101, 202, 303):
        C0 = kmeanspp_init(X, k, seed)
        rs = {m: run_algo(m, X, C0) for m in METHODS}
        for m in METHODS:
            assert is_exact(rs[m], rs["DNAK"])
            rows.append(dict(config=name, k=k, seed=seed, algo=m, iters=rs[m]["iters"],
                             comps=rs[m]["comps"], time_s=round(rs[m]["time"], 4),
                             centroid_side_s=round(rs[m]["sort_time"] + rs[m]["graph_time"], 4)))
        print(name, seed, {m: round(rs[m]["time"], 2) for m in METHODS}, flush=True)
with open(os.path.join(RESULTS, "raw_exponion_check.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
