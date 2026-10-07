"""Scaling with n (d = 2, k = 128): n = 1e5, 3e5, 1e6, three seeds.
Lloyd's algorithm is not run at these sizes; instead every method is
checked to produce output identical to DNAK (iterations, assignments and
centroids), and DNAK itself is checked against Lloyd's algorithm in the
main benchmark. Resumable; writes results/raw_nscale.csv."""
import csv, os
from _setup import RESULTS
from dnak_kmeans import run_algo, is_exact, warmup, make_blobs, kmeanspp_init

METHODS = ["DNAK", "Hamerly", "Yinyang", "Exponion", "Ball"]
SIZES = [100000, 300000, 1000000]
SEEDS = [101, 202, 303]
path = os.path.join(RESULTS, "raw_nscale.csv")
done = set()
if os.path.exists(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cnt = {}
    for r in rows:
        cnt[(int(r["n"]), int(r["seed"]))] = cnt.get((int(r["n"]), int(r["seed"])), 0) + 1
    done = {key for key, c in cnt.items() if c == len(METHODS)}
warmup()
new = not os.path.exists(path)
with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n", "d", "k", "seed", "algo", "iters",
                                      "comps", "time_s", "exact_vs_dnak"])
    if new:
        w.writeheader()
    for n in SIZES:
        X = make_blobs(n, 2, 128, seed=23)
        for seed in SEEDS:
            if (n, seed) in done:
                continue
            C0 = kmeanspp_init(X, 128, seed)
            ref = run_algo("DNAK", X, C0)
            res = [ref] + [run_algo(m, X, C0) for m in METHODS[1:]]
            for r in res:
                assert is_exact(r, ref), (r["name"], n, seed)
                w.writerow(dict(n=n, d=2, k=128, seed=seed, algo=r["name"],
                                iters=r["iters"], comps=r["comps"],
                                time_s=f"{r['time']:.4f}", exact_vs_dnak=True))
            f.flush()
            print(n, seed, {r["name"]: round(r["time"], 2) for r in res}, flush=True)
