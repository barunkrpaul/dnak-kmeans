"""DNAK: Delaunay-Neighbor Accelerated k-means (exact acceleration of
Lloyd's algorithm), with compiled baselines used in the paper."""
from .data import (make_blobs, kmeanspp_init, load_image_rgb, load_uber,
                   load_road)
from .runner import run_algo, is_exact, warmup, ALGOS, ALL_ALGOS


def dnak(X, C0, max_iter=500):
    """Exact k-means with DNAK from initial centroids C0 (compiled)."""
    return run_algo("DNAK", X, C0, max_iter)


def lloyd(X, C0, max_iter=500):
    """Lloyd's algorithm from initial centroids C0 (compiled)."""
    return run_algo("Lloyd", X, C0, max_iter)


__all__ = ["dnak", "lloyd", "run_algo", "is_exact", "warmup", "ALGOS",
           "ALL_ALGOS", "make_blobs", "kmeanspp_init", "load_image_rgb",
           "load_uber", "load_road"]
__version__ = "1.0.0"
