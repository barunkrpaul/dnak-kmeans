raw_kdtree_colab.csv: one row per run of the tree-based comparison (Section 5.8,
Table 7 of the Pattern Recognition revision). Columns: config, n, d, k,
k-means++ seed, repeat, method (DNAK = DNAK-6 of this package; the others are
mlpack 4.8 algorithms called through its Python bindings), iterations as
reported by each program (mlpack counts the initial assignment; -1 = mlpack
printed no convergence message), wall-clock time in seconds, whether the
final labels and centroids were identical to DNAK's, and the number of
differing labels. Session: Google Colab, AMD EPYC 7B12 (1 core, 2 threads),
12 GB, Python 3.13.16, numpy 2.1.3, scipy 1.16.3, numba 0.61.2, mlpack 4.8.0,
11 October 2026. Produced by experiments/dnak_kdtree_filtering_colab.ipynb.
summary_kdtree_colab.csv: per-configuration means, geometric-mean time ratios
(method / DNAK), runs on which DNAK was faster, and runs with identical output.
