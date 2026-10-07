"""DNAK-6 against the original Exponion (eakmeans) and Ball k-means code,
interleaved and repeated to reduce timing noise. Writes
results/raw_dnak6_original.csv."""
import csv, os, subprocess, re, tempfile, sys
import numpy as np
from _setup import RESULTS
from dnak_kmeans import run_algo, is_exact, warmup, make_blobs, kmeanspp_init, load_uber, load_road
from run_original_code import run_eak, run_ball, same_partition

CONFIGS = [("synth-2D-k128", lambda: make_blobs(100000, 2, 128, 23), 128),
           ("sweep-2D-k512", lambda: make_blobs(100000, 2, 512, 2512), 512),
           ("sweep-3D-k512", lambda: make_blobs(100000, 3, 512, 3512), 512),
           ("uber-k256", lambda: load_uber(200000, seed=0), 256),
           ("road-k64", load_road, 64)]
REPEATS = int(os.environ.get("DNAK_REPEATS", "2"))
SEEDS = [int(x) for x in os.environ.get("DNAK_SEEDS", "101,202,303").split(",")]
only = sys.argv[1:]
warmup()
path = os.path.join(RESULTS, "raw_dnak6_original.csv")
new = not os.path.exists(path)
with open(path, "a", newline="") as f, tempfile.TemporaryDirectory() as tmp:
    w = csv.DictWriter(f, fieldnames=["config", "seed", "rep", "method", "iters", "comps", "time_s", "same_as_dnak6"])
    if new:
        w.writeheader()
    for name, loader, k in CONFIGS:
        if only and name not in only:
            continue
        X = loader(); n, d = X.shape
        xtxt = os.path.join(tmp, "X.txt"); xcsv = os.path.join(tmp, "X.csv")
        np.savetxt(xtxt, X, fmt="%.17g", header=f"{n} {d}", comments="")
        np.savetxt(xcsv, X, fmt="%.17g", delimiter=",")
        for seed in SEEDS:
            C0 = kmeanspp_init(X, k, seed)
            ctxt = os.path.join(tmp, "C0.txt"); ccsv = os.path.join(tmp, "C0.csv")
            np.savetxt(ctxt, C0, fmt="%.17g"); np.savetxt(ccsv, C0, fmt="%.17g", delimiter=",")
            ref = None
            for rep in range(REPEATS):
                r6 = run_algo("DNAK-6", X, C0)
                if ref is None:
                    ref = r6
                it, t, lab = run_eak("expNS", xtxt, ctxt, k, tmp)
                itb, tb, labb = run_ball(0, xcsv, ccsv, tmp)
                rows = [("DNAK-6", r6["iters"], r6["comps"], r6["time"], True),
                        ("Exponion (eakmeans)", it, -1, t, same_partition(lab, ref["labels"]) and it == ref["iters"]),
                        ("Ball k-means (original)", itb, -1, tb, same_partition(labb, ref["labels"]) and itb == ref["iters"])]
                for m, i_, c_, t_, ok in rows:
                    w.writerow(dict(config=name, seed=seed, rep=rep, method=m, iters=i_, comps=c_, time_s=f"{t_:.4f}", same_as_dnak6=ok))
                f.flush()
                print(name, seed, rep, {m: round(t_, 2) for m, _, _, t_, _ in rows}, flush=True)
