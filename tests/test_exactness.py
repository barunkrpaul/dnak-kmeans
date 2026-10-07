"""Every method must reproduce Lloyd's algorithm exactly."""
import numpy as np
import pytest
from dnak_kmeans import (ALL_ALGOS, run_algo, is_exact, make_blobs,
                         kmeanspp_init)
from dnak_kmeans.reference import (lloyd as ref_lloyd, hamerly as ref_hamerly,
                                   dnak as ref_dnak)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_compiled_methods_match_lloyd(d, seed):
    X = make_blobs(3000, d, 20, seed)
    C0 = kmeanspp_init(X, 25, seed + 7)
    ref = run_algo("Lloyd", X, C0)
    for name in ALL_ALGOS[1:]:
        assert is_exact(run_algo(name, X, C0), ref), name


def test_reference_dnak_matches_reference_lloyd():
    X = make_blobs(2000, 2, 16, 3)
    C0 = kmeanspp_init(X, 16, 4)
    rL = ref_lloyd(X, C0)
    assert np.array_equal(ref_dnak(X, C0)["labels"], rL["labels"])
    assert np.array_equal(ref_hamerly(X, C0)["labels"], rL["labels"])
