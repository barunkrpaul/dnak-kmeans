"""Adds DNAK-6 (and re-runs DNAK) on every (configuration, seed) of the
main benchmark and the k-sweep, appending rows to the raw files.
Exactness: DNAK-6 is compared with DNAK run on the same seed, and DNAK
itself was verified against Lloyd's algorithm on these seeds."""
import csv, os, sys
import pandas as pd
from _setup import RESULTS
from dnak_kmeans import run_algo, is_exact, warmup, make_blobs, kmeanspp_init
from dnak_kmeans.bench import FIELDS
from run_benchmark import CONFIGS
from run_ksweep import SWEEP, SEEDS as KSEEDS

warmup()
for raw, which in (("raw_main.csv", "main"), ("raw_ksweep.csv", "sweep")):
    path = os.path.join(RESULTS, raw)
    df = pd.read_csv(path)
    done = set(df[df.algo == "DNAK-6"][["config", "seed"]].itertuples(index=False, name=None))
    jobs = []
    if which == "main":
        for name, ds, loader, k, seeds in CONFIGS:
            jobs.append((name, ds, loader, k, seeds))
    else:
        for d, k in SWEEP:
            jobs.append((f"sweep-{d}D-k{k}", "Gaussian mixture",
                         (lambda d=d, k=k: make_blobs(100000, d, k, seed=1000 * d + k)), k, KSEEDS))
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for name, ds, loader, k, seeds in jobs:
            todo = [s for s in seeds if (name, s) not in done]
            if not todo:
                continue
            X = loader(); n, d = X.shape
            for seed in todo:
                C0 = kmeanspp_init(X, k, seed)
                ref = run_algo("DNAK", X, C0)
                r = run_algo("DNAK-6", X, C0)
                assert is_exact(r, ref), (name, seed)
                row = df[(df.config == name) & (df.seed == seed) & (df.algo == "DNAK")].iloc[0]
                assert int(row.iters) == r["iters"] and int(row.comps) == ref["comps"], "DNAK run differs from logged run"
                w.writerow(dict(config=name, dataset=ds, n=n, d=d, k=k, seed=seed,
                                algo="DNAK-6", iters=r["iters"], comps=r["comps"],
                                time_s=f"{r['time']:.4f}", graph_time_s=f"{r['graph_time']:.4f}",
                                sse=repr(r["sse"]), exact=True, walks=0, hops=0,
                                rebuilds=r["rebuilds"], examined=0, failed=0))
                f.flush()
                print(name, seed, "DNAK %.2f  DNAK-6 %.2f  rebuilds %d/%d" % (ref["time"], r["time"], r["rebuilds"], r["iters"]), flush=True)
