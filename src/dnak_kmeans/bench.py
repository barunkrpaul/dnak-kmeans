"""Benchmark harness: runs every method from the same k-means++ seeds,
checks exactness against Lloyd's algorithm on every run, and appends one
row per (configuration, seed, method) to a raw CSV file. Already finished
(configuration, seed) pairs are skipped, so an interrupted run resumes."""
import csv
import os
import platform
import numpy as np

from .data import kmeanspp_init
from .runner import run_algo, is_exact

FIELDS = ["config", "dataset", "n", "d", "k", "seed", "algo", "iters",
          "comps", "time_s", "graph_time_s", "sse", "exact", "walks",
          "hops", "rebuilds", "examined", "failed"]


def _done(path, n_algos):
    """(config, seed) pairs with a complete set of rows. Incomplete groups
    (from an interrupted write) are removed from the file."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("failed") not in (None, "")]
    groups = {}
    for r in rows:
        groups.setdefault((r["config"], int(r["seed"])), []).append(r)
    done = {key for key, g in groups.items() if len(g) >= n_algos}
    keep = [r for r in rows if (r["config"], int(r["seed"])) in done]
    if len(keep) != len(rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(keep)
    return done


def run_config(config, dataset, X, k, seeds, algos, path, log=print):
    done = _done(path, len(algos))
    new = not os.path.exists(path)
    n, d = X.shape
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for seed in seeds:
            if (config, seed) in done:
                continue
            C0 = kmeanspp_init(X, k, seed)
            ref = run_algo("Lloyd", X, C0)
            rows = [ref]
            for nm in algos:
                if nm == "Lloyd":
                    continue
                r = run_algo(nm, X, C0)
                if not is_exact(r, ref):
                    raise AssertionError(
                        f"EXACTNESS FAILURE: {nm} on {config}, seed {seed}")
                rows.append(r)
            for r in rows:
                w.writerow(dict(config=config, dataset=dataset, n=n, d=d, k=k,
                                seed=seed, algo=r["name"], iters=r["iters"],
                                comps=r["comps"], time_s=f"{r['time']:.4f}",
                                graph_time_s=f"{r['graph_time']:.4f}",
                                sse=repr(r["sse"]), exact=True,
                                walks=r["walks"], hops=r["hops"],
                                rebuilds=r["rebuilds"],
                                examined=r["examined"], failed=r["failed"]))
            f.flush()
            msg = " ".join(f"{r['name']}={ref['time'] / r['time']:.1f}x"
                           for r in rows[1:])
            log(f"[{config}] seed {seed}: Lloyd {ref['time']:.2f}s, "
                f"{ref['iters']} it | {msg}")


def machine_info():
    import numba, scipy
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, numba=numba.__version__,
                processor=platform.processor() or platform.machine(),
                system=platform.platform())
