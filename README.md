# DNAK — Delaunay-Neighbor Accelerated K-means

**Exact acceleration of Lloyd's k-means algorithm using computational geometry.**
DNAK produces *bit-identical* clusterings to standard k-means while doing far
less work, by exploiting the Delaunay graph of the evolving centroid set.

> Research code accompanying the manuscript
> *"DNAK: Exact Acceleration of Lloyd's k-Means in Linear Memory via
> Delaunay-Neighbor Pruning"* (Barun Kr Paul, NITTTR Kolkata).
> Status: manuscript in preparation; see `paper/dnak_paper.tex`.

## The idea in one paragraph is given here

In late k-means iterations almost no point changes cluster, yet Lloyd's
algorithm still computes all `n × k` distances every round. DNAK maintains
just three scalars per point (assignment, upper bound, lower bound — the same
O(n) state as Hamerly's algorithm) and uses two geometric facts:
(1) a point can only be reassigned to a **Delaunay neighbor** of its current
centroid, and a greedy walk on the Delaunay graph provably finds the exact
nearest centroid while touching only ~6 candidates instead of k;
(2) the second-nearest centroid is always a Delaunay neighbor of the nearest
(order-2 Voronoi property), so DNAK's lower bounds are *exact*, making its
skip test maximally tight. The result is proved to reproduce Lloyd's
assignments at every iteration.

## Measured results

All results below are from this repository's scripts; exactness against
Lloyd's algorithm was asserted programmatically on **every run** (3 seeds per
configuration, deterministic tie-breaking, general-position jitter of 1e-7 on
quantized image data — all disclosed in the paper).

### Wall-clock (compiled Numba kernels, identical structure for all methods)

| Configuration | Lloyd | Hamerly | Elkan | **DNAK** |
|---|---|---|---|---|
| synth n=100k, d=2, k=64  | 4.56 s | 8.2× | 4.1× | **16.5×** |
| synth n=100k, d=2, k=128 | 14.09 s | 8.8× | 5.3× | **26.2×** |
| synth n=100k, d=3, k=64  | 3.81 s | 9.3× | 4.6× | **11.1×** |
| china image, n=273,280, d=3, k=64 | 24.98 s | 8.4× | 5.0× | **15.9×** |

DNAK is the fastest method in every configuration. Elkan computes the fewest
raw distances but its Θ(nk) per-iteration bound maintenance makes it slower
than Hamerly in wall-clock time at these dimensionalities — an honest
trade-off the paper reports in full (see `results/` for distance-computation
and total-operation tables across 8 configurations including real datasets).

### Honest scope

- Claims are restricted to **low dimension (d ≤ 4)**: explicit Delaunay
  construction is impractical for high d. High-d extensions (Gabriel-graph
  walk with bound certification) are future work.
- Exactness on tie-degenerate data holds relative to a declared
  tie-breaking rule ("keep current assignment unless strictly closer").
- Remaining before journal submission: Yinyang/Exponion baselines,
  additional large real datasets, 10+ seeds.

## Repository layout

```
src/dnak_kmeans/reference.py   Pure NumPy/SciPy reference: Lloyd, Hamerly,
                               DNAK, k-means++ init, distance-comp counters
src/dnak_kmeans/compiled.py    Numba-compiled kernels for all four
                               algorithms + wall-clock benchmark
experiments/benchmark_counts.py    Elkan baseline, real datasets,
                                   tie-aware exactness checks
experiments/run_count_batches.py   Batched multi-seed count benchmark
results/results_v2.csv         Distance computations, mean±std, 8 configs
results/totals.csv             Total-operation model (distances + bounds)
results/wallclock.csv          Compiled wall-clock times
paper/dnak_paper.tex           Manuscript (Elsevier elsarticle) + figure
```

## Quick start

```bash
pip install -r requirements.txt

# distance-computation benchmark with exactness verification (batches 1-3)
cd experiments
python run_count_batches.py 1
python run_count_batches.py 2
python run_count_batches.py 3

# compiled wall-clock benchmark
python ../src/dnak_kmeans/compiled.py
```

Minimal usage:

```python
from dnak_kmeans import dnak, lloyd, kmeanspp_init
import numpy as np

X = np.random.default_rng(0).normal(size=(10000, 2))
C0 = kmeanspp_init(X, k=32, seed=1)
result = dnak(X, C0)          # identical output to lloyd(X, C0)
print(result["sse"], result["comps"])
```

## Reproducibility

Every number in the manuscript's tables is generated programmatically from
the CSVs in `results/`; no result is hand-typed. Random seeds are fixed in
the scripts. A strict no-fabrication protocol applies: any figure or claim
must trace to a script in this repository.

**Reproduce in your browser:** open `DNAK_reproduce_colab.ipynb` in
[Google Colab](https://colab.research.google.com/) (Runtime → Run all, ~10–15 min).
It verifies exactness for all six methods and reproduces the DNAK/Exponion
crossover on reduced problem sizes.

## License

MIT — see `LICENSE`.

## Citation

@unpublished{paul2026dnak,
  author = {Paul, Barun Kr},
  title  = {{DNAK}: Exact Acceleration of {Lloyd's} $k$-Means in Linear
            Memory via {Delaunay}-Neighbor Pruning},
  note   = {Manuscript in preparation. Code:
            \url{https://github.com/barunkrpaul/dnak-kmeans}},
  year   = {2026}
}
