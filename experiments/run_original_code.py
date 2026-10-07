"""Comparison with the original authors' implementations.

Exponion, Yinyang and Hamerly: eakmeans (Newling and Fleuret; C++).
Ball k-means: the authors' C++ release (ring and no-ring versions), patched
only to use double precision and a command-line entry point.
Build them first with external/build_external.sh.

Every external run starts from the same k-means++ centroids as DNAK and its
labels are compared with those of DNAK (which is checked against Lloyd's
algorithm in the main benchmark); the number of iterations is compared too.
External times are those reported by the programs themselves (file input
and output excluded). Writes results/raw_original.csv (resumable).

usage: python run_original_code.py [main|large]"""
import csv, os, re, subprocess, sys, tempfile
import numpy as np
from _setup import RESULTS, ROOT
from dnak_kmeans import (run_algo, warmup, make_blobs, kmeanspp_init,
                         load_uber, load_road)

EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external")
EAK = os.path.join(EXT, "eakmeans", "bin", "blaslesskmeans")
BALL = os.path.join(EXT, "ballkm")
SEEDS = [101, 202, 303]

MAIN = [("synth-2D-k128", lambda: make_blobs(100000, 2, 128, 23), 128),
        ("sweep-2D-k256", lambda: make_blobs(100000, 2, 256, 2256), 256),
        ("sweep-2D-k512", lambda: make_blobs(100000, 2, 512, 2512), 512),
        ("sweep-3D-k512", lambda: make_blobs(100000, 3, 512, 3512), 512),
        ("uber-k256", lambda: load_uber(200000, seed=0), 256),
        ("road-k64", load_road, 64)]
LARGE = [("uber-full-k1024", lambda: load_uber(None), 1024),
         ("uber-full-k512", lambda: load_uber(None), 512)]

FIELDS = ["config", "n", "d", "k", "seed", "method", "implementation",
          "iters", "time_s", "same_labels_as_dnak", "same_iters_as_dnak"]


def run_eak(alg, xtxt, ctxt, k, tmp):
    lab = os.path.join(tmp, "eak_lab.txt")
    out = subprocess.run([EAK, "-din", xtxt, "-cin", ctxt, "-nc", str(k),
                          "-alg", alg, "-lou", lab, "-cver", "1", "-nth", "1"],
                         capture_output=True, text=True, check=True).stdout
    # Yinyang first runs a small k-means to group the centroids; the last
    # summary line belongs to the main run.
    m = re.findall(r"rounds: (\d+).*?time : (\d+) ms", out)[-1]
    return int(m[0]), int(m[1]) / 1000.0, np.loadtxt(lab, dtype=int)


def same_partition(a, b):
    """True if labels a and b define the same clusters. eakmeans' Yinyang
    reorders the centroids by group, so labels may differ by a relabeling."""
    if a.shape != b.shape:
        return False
    pairs = np.unique(np.stack([a, b], 1), axis=0)
    return len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1]))


def run_ball(ring, xcsv, ccsv, tmp):
    lab = os.path.join(tmp, "ball_lab.txt")
    out = subprocess.run([BALL, xcsv, ccsv, str(ring), lab],
                         capture_output=True, text=True, check=True).stdout
    it = int(re.search(r"iterations\s*:\s*\|\|(\d+)", out).group(1))
    per = float(re.search(r"Time per round:\s*\|\|([\d.eE+-]+)", out).group(1))
    # Ball k-means counts the final, unchanged pass as an iteration
    return it - 1, per * it / 1000.0, np.loadtxt(lab, dtype=int)


def main(which):
    configs = MAIN if which == "main" else LARGE
    path = os.path.join(RESULTS, f"raw_original_{which}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        cnt = {}
        for r in rows:
            key = (r["config"], int(r["seed"]))
            cnt[key] = cnt.get(key, 0) + 1
        done = {key for key, c in cnt.items() if c == (7 if which == "main" else 6)}
    warmup()
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f, tempfile.TemporaryDirectory() as tmp:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        cache = {}
        for name, loader, k in configs:
            todo = [s for s in SEEDS if (name, s) not in done]
            if not todo:
                continue
            key = loader
            X = loader()
            n, d = X.shape
            xtxt = os.path.join(tmp, "X.txt"); xcsv = os.path.join(tmp, "X.csv")
            np.savetxt(xtxt, X, fmt="%.17g", header=f"{n} {d}", comments="")
            np.savetxt(xcsv, X, fmt="%.17g", delimiter=",")
            for seed in todo:
                C0 = kmeanspp_init(X, k, seed)
                ctxt = os.path.join(tmp, "C0.txt"); ccsv = os.path.join(tmp, "C0.csv")
                np.savetxt(ctxt, C0, fmt="%.17g")
                np.savetxt(ccsv, C0, fmt="%.17g", delimiter=",")
                ref = run_algo("DNAK", X, C0)
                exp = run_algo("Exponion", X, C0)
                res = [("DNAK", "ours (Numba)", ref["iters"], ref["time"], ref["labels"]),
                       ("Exponion", "ours (Numba)", exp["iters"], exp["time"], exp["labels"])]
                eak_algs = (("expNS", "Exponion"), ("syinNS", "Yinyang"),
                            ("ham", "Hamerly"))
                if which == "large":      # Hamerly's full scans are too slow here
                    eak_algs = eak_algs[:2]
                for alg, meth in eak_algs:
                    it, t, lab = run_eak(alg, xtxt, ctxt, k, tmp)
                    res.append((meth, "eakmeans (C++)", it, t, lab))
                for ring, meth in ((0, "Ball k-means"), (1, "Ball k-means (ring)")):
                    try:
                        it, t, lab = run_ball(ring, xcsv, ccsv, tmp)
                    except subprocess.CalledProcessError:   # e.g. out of memory
                        it, t, lab = -1, float("nan"), None
                    res.append((meth, "original (C++)", it, t, lab))
                for meth, impl, it, t, lab in res:
                    w.writerow(dict(config=name, n=n, d=d, k=k, seed=seed,
                                    method=meth, implementation=impl, iters=it,
                                    time_s=f"{t:.4f}",
                                    same_labels_as_dnak=(lab is not None and bool(same_partition(lab, ref["labels"]))),
                                    same_iters_as_dnak=it == ref["iters"]))
                f.flush()
                print(name, seed, {f"{m}/{i[:4]}": round(t, 2) for m, i, _, t, _ in res},
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
