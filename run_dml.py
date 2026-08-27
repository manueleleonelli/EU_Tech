"""
Run all DML specifications with honest inference.

Grid: {4 definitions} x {revenue, employment} x {overall, SME, collaborative,
early-stage, independent, subsidiary}. Early-stage is estimated but flagged:
with n<200 and min_samples_leaf=30 the forest cannot detect heterogeneity, so
its premiums are reported in an appendix with an explicit power caveat rather
than as evidence of a zero differential.

Checkpointing: each completed specification is appended to output/dml_results.csv,
so the script can be interrupted and resumed. Bootstrap and permutation are the
expensive parts; with N_BOOT=400 and N_PERM=500 expect several hours on a
laptop for the full grid (the overall specs dominate; subgroups are fast).

Usage:
    python run_dml.py             # full grid
    python run_dml.py --smoke     # tiny B/P on two specs, to test the plumbing
"""
import sys
import numpy as np
import pandas as pd
from config import OUT_DIR, N_BOOT, N_PERM, SEED, MIN_SUBGROUP_N, MIN_CELL_N
from dml_core import make_controls, point, bootstrap, permutation

SMOKE = '--smoke' in sys.argv

DEFS = [('ht_oecd', 'OECD (strict GRV)'),
        ('ht_calvino', 'Calvino'),
        ('ht_pavitt', 'Pavitt narrow'),
        ('ht_pavitt_broad', 'Pavitt broad')]
# The thesis's ICT-inclusive OECD coding runs as a fifth robustness definition:
DEFS_ROBUST = DEFS + [('ht_oecd_thesis', 'OECD (ICT-inclusive)')]

SUBGROUPS = [('Overall', None, None),
             ('SME', 'primary_instrument', 'SME'),
             ('Collaborative', 'primary_instrument', 'Collaborative'),
             ('Early-stage', 'primary_instrument', 'Early-stage'),
             ('Independent', 'independence', 'Independent'),
             ('Subsidiary', 'independence', 'Subsidiary')]

OUTCOMES = [('Revenue', 'rev_growth', 'keep_rev'),
            ('Employment', 'emp_growth', 'keep_emp')]


def main():
    df = pd.read_csv(OUT_DIR / 'analysis_ready_v2.csv')
    X_all = make_controls(df)
    res_path = OUT_DIR / ('dml_results_smoke.csv' if SMOKE else 'dml_results.csv')
    done = set()
    if res_path.exists():
        prev = pd.read_csv(res_path)
        done = set(zip(prev['subgroup'], prev['outcome'], prev['definition']))
        print(f'Resuming: {len(done)} specifications already complete.')

    n_boot = 20 if SMOKE else N_BOOT
    n_perm = 30 if SMOKE else N_PERM
    defs = DEFS if SMOKE else DEFS_ROBUST
    grid = [(sg, oc) for sg in SUBGROUPS for oc in OUTCOMES]
    if SMOKE:
        grid = [(SUBGROUPS[1], OUTCOMES[0])]   # SME/Revenue: small n, fast
        defs = DEFS[:1]

    rows = []
    for (sg_name, fcol, fval), (out_name, out_col, keep_col) in grid:
        mask = df[keep_col].astype(bool)
        if fcol is not None:
            mask &= df[fcol] == fval
        n = int(mask.sum())
        if n < MIN_SUBGROUP_N:
            print(f'skip {sg_name}/{out_name}: n={n}')
            continue
        Y = df.loc[mask, out_col].values
        T = df.loc[mask, 'log_ec'].values
        Xb = X_all.loc[mask].values.astype(float)

        for ht_col, ht_label in defs:
            if (sg_name, out_name, ht_label) in done:
                continue
            ht = df.loc[mask, ht_col].values
            if ht.sum() < MIN_CELL_N or (1 - ht).sum() < MIN_CELL_N:
                print(f'skip {sg_name}/{out_name}/{ht_label}: thin cell')
                continue

            print(f'{sg_name} | {out_name} | {ht_label} (n={n}) ...', flush=True)
            r = point(Y, T, Xb, ht)
            b = bootstrap(Y, T, Xb, ht, n_boot=n_boot, seed=SEED)
            p = permutation(Y, T, Xb, ht, n_perm=n_perm,
                            observed_premium=r['premium'], seed=SEED)
            row = dict(subgroup=sg_name, outcome=out_name, definition=ht_label,
                       n=r['n'], n_ht=r['n_ht'], n_non_ht=r['n_non_ht'],
                       ate=r['ate'], ate_se=r['ate_se'], ate_pval=r['ate_pval'],
                       ate_se_boot=b['ate_se_boot'],
                       premium=r['premium'],
                       premium_se_boot=b['premium_se'],
                       premium_ci_lo=b['premium_ci'][0],
                       premium_ci_hi=b['premium_ci'][1],
                       perm_pval=p['perm_pval'],
                       perm_null_sd=p['perm_null_sd'],
                       flag_low_power=int(n < 300))
            rows.append(row)
            pd.DataFrame([row]).to_csv(res_path, mode='a', index=False,
                                       header=not res_path.exists())
            print(f'   premium={row["premium"]:+.4f}  boot SE={row["premium_se_boot"]:.4f}  '
                  f'boot 95% CI=[{row["premium_ci_lo"]:+.4f},{row["premium_ci_hi"]:+.4f}]  '
                  f'perm p={row["perm_pval"]:.3f}')

    print(f'\nDone. Results in {res_path}')


if __name__ == '__main__':
    main()
