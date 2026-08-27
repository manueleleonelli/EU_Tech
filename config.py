"""Configuration for the definition-sensitivity paper pipeline."""
from pathlib import Path

DATA_DIR = Path('.')                     # folder containing the four raw CSVs
OUT_DIR = Path('output'); OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_BOOT = 400          # firm-level bootstrap replications per specification
N_PERM = 500          # permutation replications per specification
N_JOBS = -1           # parallelism for sklearn/econml internals

# Forest hyperparameters (thesis baseline)
FOREST = dict(n_estimators=200, min_samples_leaf=30)
CV_FOLDS = 5

# HE decimal-comma corruption heuristic: HE rows above this threshold whose
# value is not a multiple of 100 are treated as x100-inflated and divided by 100.
# Set EC_FIX = False once Iisakki re-extracts contributions with decimal=','.
EC_FIX = True
EC_SUSPECT_THRESHOLD = 3.5e6   # ~ H2020 99th percentile

MIN_SUBGROUP_N = 100
MIN_CELL_N = 30
