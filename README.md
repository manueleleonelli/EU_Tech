# Corrected analysis pipeline — Horizon definition-sensitivity paper

Replaces the two thesis notebooks. Requires: `pandas`, `numpy`, `scikit-learn`,
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

## What changed relative to the thesis pipeline

1. **FIX 1 (treatment):** Horizon Europe EC contributions with stripped decimal
   commas (×100 inflated) repaired by heuristic; repaired firms carry
   `ec_was_fixed = 1`. Once Iisakki re-extracts the contributions with
   `decimal=','` parsing, set `EC_FIX = False` in `config.py` and rebuild.
2. **FIX 2 (instruments):** EIC Pathfinder/Transition → Early-stage (via the
   topics field). The thesis classification is kept in
   `primary_instrument_thesis`.
3. **FIX 3 (definitions):** `ht_oecd` is now the strict Galindo-Rueda & Verger
   top tier (21, 26, 30.3, 58.2, 72 at 4-digit precision); the thesis coding
   survives as `ht_oecd_thesis` ("OECD ICT-inclusive"), run as a fifth
   definition throughout.
4. **Inference:** the premium is reported with a firm-level bootstrap of the
   whole pipeline (percentile CIs) plus a permutation test of the high-tech
   label. The independence-assumption SE is gone. The ATE keeps econml's
   within-forest inference, with a bootstrap SE alongside.
5. **Reporting:** descriptives are produced for both the full and estimation
   samples; the "Other" instrument category is shown; early-stage cells carry
   a `flag_low_power` marker (report in appendix with a power caveat).

## Outputs

- `output/analysis_ready_v2.csv` — corrected firm-level dataset
- `output/dml_results.csv` — one row per spec: ATE (+SE, p), premium,
  bootstrap SE and 95% CI, permutation p, sample sizes, flags
- `output/descriptives.txt`, `output/attrition_reweighted.csv`,
  `output/sensitivity.csv`

## Suggested reporting

Specification curve of premiums with bootstrap CIs; permutation p-values in
the same table. For the paper's results section: exploratory analysis
(descriptives on both samples), then model results (ATEs, premiums with honest
CIs), then the robustness block (IPW, sensitivity, clean-cohort check).
