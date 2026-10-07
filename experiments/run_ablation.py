"""Ablation and Delaunay degree (Exponion here is the full-sort variant).

Methods: Hamerly, Hamerly-noS (Hamerly without its s-test), DNAK,
DNAK+s (DNAK with Hamerly's s-test), Exponion. Hamerly is the exactness
reference (it is itself exact by full scans and was verified against
Lloyd in the main benchmark). Writes results/raw_ablation.csv; resumable."""
import csv, os, sys
import numpy as np
from _setup import RESULTS
from dnak_kmeans import run_algo, is_exact, warmup, make_blobs, kmeanspp_init, load_uber, load_road

METHODS = ["Hamerly", "Hamerly-noS", "DNAK", "DNAK+s", "Exponion"]
CONFIGS = [
    ("synth-2D-k128", lambda: make_blobs(100000, 2, 128, 23), 128),
    ("synth-3D-k64", lambda: make_blobs(100000, 3, 64, 25), 64),
    ("uber-k256", lambda: load_uber(200000, seed=0), 256),
    ("road-k64", load_road, 64),
    ("sweep-2D-k512", lambda: make_blobs(100000, 2, 512, 2512), 512),
    ("sweep-3D-k512", lambda: make_blobs(100000, 3, 512, 3512), 512),
]
SEEDS = [101, 202, 303]
FIELDS = ["config", "k", "d", "seed", "algo", "iters", "comps", "time_s",
          "graph_time_s", "sort_time_s", "deg_mean", "deg_max", "exact"]
path = os.path.join(RESULTS, "raw_ablation.csv")
done = set()
if os.path.exists(path):
    import pandas as pd
    p = pd.read_csv(path)
    for key, g in p.groupby(["config", "seed"]):
        if len(g) == len(METHODS):
            done.add(key)
    p[p.apply(lambda r: (r.config, r.seed) in done, axis=1)].to_csv(path, index=False)
warmup()
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    for name, loader, k in CONFIGS:
        if all((name, s) in done for s in SEEDS):
            continue
        X = loader()
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            C0 = kmeanspp_init(X, k, seed)
            rs = {m: run_algo(m, X, C0) for m in METHODS}
            for m, r in rs.items():
                assert is_exact(r, rs["Hamerly"]), (m, name, seed)
                w.writerow(dict(config=name, k=k, d=X.shape[1], seed=seed, algo=m,
                                iters=r["iters"], comps=r["comps"],
                                time_s=f"{r['time']:.4f}",
                                graph_time_s=f"{r['graph_time']:.4f}",
                                sort_time_s=f"{r['sort_time']:.4f}",
                                deg_mean=f"{r.get('deg_mean', float('nan')):.3f}",
                                deg_max=r.get("deg_max", ""), exact=True))
            f.flush()
            print(name, seed, {m: round(rs[m]["time"], 2) for m in METHODS}, flush=True)
print("ablation finished")
