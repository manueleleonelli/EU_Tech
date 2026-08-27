#  Horizon definition-sensitivity paper

 Requires: `pandas`, `numpy`, `scikit-learn`,
`scipy`, `econml` (tested on 0.16.0). Put the four raw CSVs in this folder (or
edit `DATA_DIR` in `config.py`).

## Run order

```
python build_data.py                    # ~2 min: corrected analysis_ready_v2.csv
python descriptives_and_robustness.py   # ~30 min: Tables 1-3, IPW attrition check,
                                         #   hyperparameter/winsorisation/clean-cohort sensitivity
python run_dml.py                        # several hours: all specs with firm-level
                                         #   bootstrap (B=400) and permutation test (P=500)
python run_dml.py --smoke                # 2 min plumbing test first, if you like
```

`run_dml.py` checkpoints each completed specification to `output/dml_results.csv`
and resumes automatically if interrupted. Adjust `N_BOOT`, `N_PERM` in
`config.py`; B=200 halves the runtime at modest cost in CI precision.

## Outputs

- `output/analysis_ready_v2.csv` — corrected firm-level dataset
- `output/dml_results.csv` — one row per spec: ATE (+SE, p), premium,
  bootstrap SE and 95% CI, permutation p, sample sizes, flags
- `output/descriptives.txt`, `output/attrition_reweighted.csv`,
  `output/sensitivity.csv`


