import sys, csv, numpy as np
from bench2 import run, load_image_pixels, load_digits_pca
from dnak import make_blobs

batch = int(sys.argv[1])
SEEDS = [101, 202, 303]
if batch == 1:
    cfgs = [("synth n=10000 d=2 k=32", make_blobs(10000,2,32,7), 32),
            ("synth n=10000 d=2 k=100", make_blobs(10000,2,100,11), 100),
            ("synth n=20000 d=3 k=64", make_blobs(20000,3,64,13), 64)]
elif batch == 2:
    cfgs = [("synth n=20000 d=4 k=64", make_blobs(20000,4,64,17), 64),
            ("synth n=30000 d=2 k=128", make_blobs(30000,2,128,19), 128)]
else:
    cfgs = [("china-RGB n=30000 d=3 k=64", load_image_pixels("china.jpg",30000,1), 64),
            ("flower-RGB n=30000 d=3 k=32", load_image_pixels("flower.jpg",30000,2), 32),
            ("digits-PCA3 n=1797 d=3 k=10", load_digits_pca(3), 10)]

rows = []
for cname, X, k in cfgs:
    agg = {}
    for seed in SEEDS:
        for r in run(X, k, seed):
            agg.setdefault(r["name"], []).append(r["comps"])
    print(f"\n=== {cname} (3 seeds, all runs exact) ===")
    for name in ("Lloyd","Hamerly","Elkan","DNAK"):
        c = np.array(agg[name], float)
        reds = np.array(agg["Lloyd"]) / c
        print(f"  {name:8s} comps={c.mean():>14,.0f}   reduction={reds.mean():6.2f}x ± {reds.std():4.2f}")
        rows.append(dict(config=cname, algo=name, comps_mean=int(c.mean()),
                         comps_std=int(c.std()), red_mean=round(float(reds.mean()),2),
                         red_std=round(float(reds.std()),2)))
mode = "w" if batch == 1 else "a"
with open("results_v2.csv", mode, newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if batch == 1: w.writeheader()
    w.writerows(rows)
print("batch", batch, "done")
