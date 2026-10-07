"""Same-session paired timing of DNAK-B (global drift bounds) and DNAK
(DNAK-6, Delaunay-local bounds and certificate reuse) on every
(configuration, seed) of the main benchmark. Writes results/raw_paired_b6.csv."""
import csv, os
from _setup import RESULTS
from dnak_kmeans import run_algo, is_exact, warmup, kmeanspp_init
from run_benchmark import CONFIGS
warmup()
path = os.path.join(RESULTS, "raw_paired_b6.csv")
done = set()
if os.path.exists(path):
    done = {(r["config"], int(r["seed"])) for r in csv.DictReader(open(path))}
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "seed", "iters", "time_dnak_b", "time_dnak6",
                                      "comps_dnak_b", "comps_dnak6", "graph_dnak_b", "graph_dnak6", "exact"])
    if new:
        w.writeheader()
    for name, ds, loader, k, seeds in CONFIGS:
        todo = [s for s in seeds if (name, s) not in done]
        if not todo:
            continue
        X = loader()
        for seed in todo:
            C0 = kmeanspp_init(X, k, seed)
            rb = run_algo("DNAK", X, C0); r6 = run_algo("DNAK-6", X, C0)
            assert is_exact(r6, rb)
            w.writerow(dict(config=name, seed=seed, iters=rb["iters"], time_dnak_b=f"{rb['time']:.4f}",
                            time_dnak6=f"{r6['time']:.4f}", comps_dnak_b=rb["comps"], comps_dnak6=r6["comps"],
                            graph_dnak_b=f"{rb['graph_time']:.4f}", graph_dnak6=f"{r6['graph_time']:.4f}", exact=True))
            f.flush()
            print(name, seed, round(rb["time"], 2), round(r6["time"], 2), flush=True)
