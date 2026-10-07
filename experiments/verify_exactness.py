"""Quick exactness check: every method must reproduce Lloyd's algorithm
(identical iteration count, assignments and centroids) on small synthetic
problems in d = 2, 3, 4 and on a subsample of a real image."""
import _setup  # noqa: F401
from dnak_kmeans import (ALL_ALGOS, run_algo, is_exact, warmup, make_blobs,
                         kmeanspp_init, load_image_rgb)

warmup()
checks = 0
for d in (2, 3, 4):
    for seed in range(5):
        X = make_blobs(5000, d, 30, seed)
        C0 = kmeanspp_init(X, 40, seed + 9)
        ref = run_algo("Lloyd", X, C0)
        for nm in ALL_ALGOS[1:]:
            assert is_exact(run_algo(nm, X, C0), ref), (nm, d, seed)
            checks += 1
X = load_image_rgb("china.jpg")[::9]
for seed in range(3):
    C0 = kmeanspp_init(X, 32, seed)
    ref = run_algo("Lloyd", X, C0)
    for nm in ALL_ALGOS[1:]:
        assert is_exact(run_algo(nm, X, C0), ref), (nm, "china", seed)
        checks += 1
print(f"PASS: {checks} runs, every method identical to Lloyd's algorithm")
