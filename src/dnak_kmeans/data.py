"""Datasets and k-means++ seeding shared by every experiment."""
import os
import numpy as np

JITTER = 1e-7          # general-position jitter for quantised real data


def make_blobs(n, d, k_true, seed, spread=0.6, box=10.0):
    """Gaussian mixture: k_true centres uniform in [-box, box]^d."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-box, box, size=(k_true, d))
    sizes = rng.multinomial(n, np.ones(k_true) / k_true)
    parts = [rng.normal(c, spread, size=(s, d)) for c, s in zip(centers, sizes)]
    X = np.vstack(parts)
    rng.shuffle(X)
    return X.astype(np.float64)


def kmeanspp_init(X, k, seed):
    """Standard k-means++ seeding (Arthur and Vassilvitskii, 2007)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(n)]
    d2 = np.sum((X - C[0]) ** 2, axis=1)
    for j in range(1, k):
        C[j] = X[rng.choice(n, p=d2 / d2.sum())]
        d2 = np.minimum(d2, np.sum((X - C[j]) ** 2, axis=1))
    return C


def _jitter(X, seed=0):
    rng = np.random.default_rng(seed)
    return X + rng.uniform(-JITTER, JITTER, size=X.shape)


def load_image_rgb(name):
    """All pixels of a scikit-learn sample image ('china.jpg' or
    'flower.jpg') as RGB values in [0, 1], with general-position jitter."""
    from sklearn.datasets import load_sample_image
    P = load_sample_image(name).astype(np.float64).reshape(-1, 3) / 255.0
    return _jitter(P)


UBER_URL = ("https://raw.githubusercontent.com/plotly/datasets/master/"
            "uber-rides-data1.csv")


def load_uber(n=None, seed=0, path=None):
    """Uber pick-up locations in New York City, April 2014 (latitude,
    longitude), from the plotly/datasets repository. Coordinates are
    min-max scaled to [0, 1] per axis and jittered. If n is given, a fixed
    random subsample of n points is returned."""
    import pandas as pd
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                            "uber-rides-data1.csv")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        print(f"downloading {UBER_URL} -> {path}")
        urllib.request.urlretrieve(UBER_URL, path)
    df = pd.read_csv(path)
    X = df[["Lat", "Lon"]].to_numpy(np.float64)
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    if n is not None and n < X.shape[0]:
        idx = np.random.default_rng(seed).choice(X.shape[0], n, replace=False)
        X = X[np.sort(idx)]
    return _jitter(X, seed=1)


ROAD_URL = ("https://raw.githubusercontent.com/prathmachowksey/"
            "Multivariate-Linear-Regression/master/3D_spatial_network.txt")
ROAD_MD5 = "f0d7afa0eaf736cdda22904a183bc16c"


def load_road(path=None):
    """UCI 3D Road Network, North Jutland (n = 434,874; longitude,
    latitude, altitude). The original source is
    https://archive.ics.uci.edu/dataset/246; because the UCI server blocks
    scripted downloads, the identical file (MD5 checked) is fetched from a
    public GitHub mirror. Each column is standardised and jittered."""
    import hashlib
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                            "3D_spatial_network.txt")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        print(f"downloading {ROAD_URL} -> {path}")
        urllib.request.urlretrieve(ROAD_URL, path)
    with open(path, "rb") as f:
        if hashlib.md5(f.read()).hexdigest() != ROAD_MD5:
            raise ValueError("3D_spatial_network.txt does not match the UCI file")
    R = np.loadtxt(path, delimiter=",")[:, 1:4]
    X = (R - R.mean(0)) / R.std(0)
    return _jitter(X, seed=2)
