"""Main benchmark (Table 2 and Figure 2 of the paper).
Writes one row per (configuration, seed, method) to results/raw_main.csv."""
import json, os, sys
from _setup import RESULTS
from dnak_kmeans import (ALL_ALGOS, warmup, make_blobs, load_image_rgb,
                         load_uber, load_road)
from dnak_kmeans.bench import run_config, machine_info

SYN_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
REAL_SEEDS = [101, 202, 303, 404, 505]

CONFIGS = [
    # name, dataset, loader, k, seeds
    ("synth-2D-k64", "Gaussian mixture", lambda: make_blobs(100000, 2, 64, 21), 64, SYN_SEEDS),
    ("synth-2D-k128", "Gaussian mixture", lambda: make_blobs(100000, 2, 128, 23), 128, SYN_SEEDS),
    ("synth-3D-k64", "Gaussian mixture", lambda: make_blobs(100000, 3, 64, 25), 64, SYN_SEEDS),
    ("synth-4D-k64", "Gaussian mixture", lambda: make_blobs(50000, 4, 64, 17), 64, SYN_SEEDS),
    ("uber-k64", "Uber NYC pick-ups", lambda: load_uber(200000, seed=0), 64, REAL_SEEDS),
    ("uber-k256", "Uber NYC pick-ups", lambda: load_uber(200000, seed=0), 256, REAL_SEEDS),
    ("road-k64", "3D Road Network", load_road, 64, REAL_SEEDS),
    ("china-k64", "china.jpg RGB", lambda: load_image_rgb("china.jpg"), 64, REAL_SEEDS),
    ("flower-k32", "flower.jpg RGB", lambda: load_image_rgb("flower.jpg"), 32, REAL_SEEDS),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    with open(os.path.join(RESULTS, "machine.json"), "w") as f:
        json.dump(machine_info(), f, indent=2)
    warmup()
    out = os.path.join(RESULTS, "raw_main.csv")
    for name, ds, loader, k, seeds in CONFIGS:
        if only and name not in only:
            continue
        run_config(name, ds, loader(), k, seeds, ALL_ALGOS, out,
                   log=lambda m: print(m, flush=True))
    print("main benchmark finished:", out)
