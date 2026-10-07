"""k-sweep (Figure 3 of the paper): speed-up over Lloyd's algorithm as k
grows, at n = 100,000, d = 2 and d = 3. Writes results/raw_ksweep.csv."""
import os, sys
from _setup import RESULTS
from dnak_kmeans import warmup, make_blobs
from dnak_kmeans.bench import run_config

SEEDS = [101, 202, 303]
ALGOS = ["Lloyd", "Hamerly", "Yinyang", "Exponion", "Ball", "DNAK"]
SWEEP = [(2, k) for k in (64, 96, 128, 192, 256, 384, 512)] + \
        [(3, k) for k in (64, 128, 256, 512)]

if __name__ == "__main__":
    warmup()
    out = os.path.join(RESULTS, "raw_ksweep.csv")
    for d, k in SWEEP:
        X = make_blobs(100000, d, k, seed=1000 * d + k)
        run_config(f"sweep-{d}D-k{k}", "Gaussian mixture", X, k, SEEDS,
                   ALGOS, out, log=lambda m: print(m, flush=True))
    print("k-sweep finished:", out)
