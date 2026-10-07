#!/usr/bin/env bash
# Reproduces every number, table and figure in the paper.
set -e
cd "$(dirname "$0")"
python verify_exactness.py       # quick exactness check
python run_benchmark.py          # main benchmark        -> raw_main.csv
python run_ksweep.py             # k-sweep               -> raw_ksweep.csv
python run_nscale.py             # scaling with n        -> raw_nscale.csv
python run_ablation.py           # ablation and degrees  -> raw_ablation.csv
python run_exponion_check.py     # Exponion variants     -> raw_exponion_check.csv
python run_degenerate.py         # degenerate inputs     -> degenerate.csv
python make_tables_figures.py    # all tables and figures
