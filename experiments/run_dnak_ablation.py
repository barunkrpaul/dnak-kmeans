"""Ablation of the mechanisms of DNAK (Section 4.1 of the paper), one
session, same seeds. Variants (all exact, checked against DNAK-B):
DNAK-B   global drift bounds, full-scan initialisation (Algorithm 1)
DNAK-2   + Delaunay s-test and annulus stop in the walk
DNAK-5   + Delaunay-local bounds and walk-based initialisation
DNAK-6   + certificate-gated reuse of the triangulation (= DNAK)
Writes results/raw_dnak_ablation.csv."""
import csv, os
from _setup import RESULTS
from dnak_kmeans import run_algo, is_exact, warmup, make_blobs, kmeanspp_init, load_uber
warmup()
CONF = [("sweep-2D-k512", lambda: make_blobs(100000, 2, 512, 2512), 512),
        ("sweep-3D-k512", lambda: make_blobs(100000, 3, 512, 3512), 512),
        ("uber-k256", lambda: load_uber(200000, seed=0), 256)]
VAR = ["DNAK", "DNAK-2", "DNAK-5", "DNAK-6"]
path = os.path.join(RESULTS, "raw_dnak_ablation.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "seed", "variant", "iters", "comps", "time_s", "graph_time_s", "rebuilds", "exact"])
    w.writeheader()
    for name, loader, k in CONF:
        X = loader()
        for seed in (101, 202, 303):
            C0 = kmeanspp_init(X, k, seed)
            ref = None
            for v in VAR:
                r = run_algo(v, X, C0)
                if ref is None: ref = r
                assert is_exact(r, ref), (v, name, seed)
                w.writerow(dict(config=name, seed=seed, variant=v, iters=r["iters"], comps=r["comps"],
                                time_s=f"{r['time']:.4f}", graph_time_s=f"{r['graph_time']:.4f}",
                                rebuilds=r["rebuilds"], exact=True))
            f.flush()
            print(name, seed, flush=True)
