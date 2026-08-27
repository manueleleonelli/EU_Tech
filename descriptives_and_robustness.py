"""
Descriptive tables and the two robustness exercises.

Part A  Descriptive tables (Tables 1-3 of the paper) on BOTH the full sample
        and the estimation sample, so the two halves of the paper describe the
        same firms. Chi-squared tests as before; the 'Other' instrument
        category is reported explicitly instead of dropped silently.

Part B  Attrition reweighting. P(in estimation sample | observables that exist
        for all 14,139 firms) via gradient boosting; inverse-probability
        weights (trimmed at the 99th percentile); headline DML specs re-run
        weighted. If the reweighted premiums match the unweighted ones, the
        73% attrition is ignorable on observables.

Part C  Sensitivity: forest hyperparameters, treatment winsorisation, and a
        clean-cohort check (drop 2022-23 cohorts entirely, whose treatment
        needed the HE decimal repair) on the headline specifications.

Run after build_data.py. Parts B and C use point estimates only (no bootstrap)
to stay fast; promote any spec that moves to the full bootstrap in run_dml.py.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from config import OUT_DIR, SEED
from dml_core import make_controls, point

DEFS = [('ht_oecd', 'OECD (strict GRV)'), ('ht_oecd_thesis', 'OECD (ICT-incl.)'),
        ('ht_calvino', 'Calvino'), ('ht_pavitt', 'Pavitt narrow'),
        ('ht_pavitt_broad', 'Pavitt broad')]

HEADLINE = [('Overall', None, None, 'Revenue'),
            ('Overall', None, None, 'Employment'),
            ('SME', 'primary_instrument', 'SME', 'Revenue'),
            ('Collaborative', 'primary_instrument', 'Collaborative', 'Revenue')]


def part_a(df):
    lines = []
    for label, mask in [('Full sample', pd.Series(True, index=df.index)),
                        ('Estimation sample (rev)', df['keep_rev'].astype(bool))]:
        lines.append(f'\n=== {label} (N = {int(mask.sum()):,}) ===')
        lines.append('High-tech shares:')
        for c, l in DEFS:
            lines.append(f'  {l:20s} {df.loc[mask, c].mean():6.1%}')
        sub = df.loc[mask]
        mi = sub['primary_instrument'].isin(['SME', 'Collaborative', 'Early-stage', 'Other'])
        lines.append('\nBy instrument (incl. Other, reported not hidden):')
        for c, l in DEFS:
            sh = sub.loc[mi].groupby('primary_instrument')[c].mean()
            ct = pd.crosstab(sub.loc[mi & (sub['primary_instrument'] != 'Other'),
                                     'primary_instrument'],
                             sub.loc[mi & (sub['primary_instrument'] != 'Other'), c])
            chi2, p, _, _ = chi2_contingency(ct)
            lines.append('  {:20s} '.format(l) + '  '.join(
                f'{k}: {v:.1%}' for k, v in sh.items()) + f'  | chi2={chi2:.1f} p={p:.1e}')
        mo = sub['independence'].isin(['Independent', 'Subsidiary'])
        lines.append('\nBy ownership:')
        for c, l in DEFS:
            sh = sub.loc[mo].groupby('independence')[c].mean()
            ct = pd.crosstab(sub.loc[mo, 'independence'], sub.loc[mo, c])
            chi2, p, _, _ = chi2_contingency(ct)
            lines.append(f'  {l:20s} Indep {sh["Independent"]:.1%}  '
                         f'Subs {sh["Subsidiary"]:.1%}  | chi2={chi2:.1f} p={p:.1e}')
    txt = '\n'.join(lines)
    print(txt)
    (OUT_DIR / 'descriptives.txt').write_text(txt)


def part_b(df, X_all):
    print('\n=== Part B: attrition reweighting ===')
    Z = pd.concat([pd.get_dummies(df[c], prefix=c) for c in
                   ['nace_section', 'country', 'primary_instrument',
                    'independence', 'first_grant_year']] +
                  [np.log(df[['ec_predetermined']].clip(lower=1))], axis=1)
    y = df['keep_rev'].astype(int).values
    clf = HistGradientBoostingClassifier(random_state=SEED)
    ps = cross_val_predict(clf, Z.values, y, cv=5, method='predict_proba')[:, 1]
    df['ipw'] = np.where(y == 1, 1.0 / np.clip(ps, 0.01, 1), 0.0)
    cap = np.percentile(df.loc[y == 1, 'ipw'], 99)
    df['ipw'] = df['ipw'].clip(upper=cap)
    print(f'Selection model AUC-ish check: mean p(selected|selected)='
          f'{ps[y==1].mean():.3f} vs p(selected|not)={ps[y==0].mean():.3f}')

    rows = []
    for sg, fcol, fval, out in HEADLINE:
        keep = 'keep_rev' if out == 'Revenue' else 'keep_emp'
        ocol = 'rev_growth' if out == 'Revenue' else 'emp_growth'
        mask = df[keep].astype(bool)
        if fcol: mask &= df[fcol] == fval
        Y, T = df.loc[mask, ocol].values, df.loc[mask, 'log_ec'].values
        Xb = X_all.loc[mask].values.astype(float)
        w = df.loc[mask, 'ipw'].values
        for c, l in DEFS:
            ht = df.loc[mask, c].values
            if ht.sum() < 30 or (1 - ht).sum() < 30: continue
            r0 = point(Y, T, Xb, ht)
            r1 = point(Y, T, Xb, ht, weights=w)
            rows.append(dict(subgroup=sg, outcome=out, definition=l,
                             premium_unweighted=r0['premium'],
                             premium_ipw=r1['premium'],
                             ate_unweighted=r0['ate'], ate_ipw=r1['ate']))
            print(f'{sg}/{out}/{l}: premium {r0["premium"]:+.4f} -> '
                  f'{r1["premium"]:+.4f} (IPW) | ATE {r0["ate"]:+.4f} -> {r1["ate"]:+.4f}')
    pd.DataFrame(rows).to_csv(OUT_DIR / 'attrition_reweighted.csv', index=False)


def part_c(df, X_all):
    print('\n=== Part C: sensitivity ===')
    rows = []
    mask0 = df['keep_rev'].astype(bool)
    Y = df.loc[mask0, 'rev_growth'].values
    T = df.loc[mask0, 'log_ec'].values
    Xb = X_all.loc[mask0].values.astype(float)
    ht = df.loc[mask0, 'ht_oecd'].values

    # C1: hyperparameters
    for leaf in [10, 30, 50]:
        for ntree in [200, 500]:
            r = point(Y, T, Xb, ht, forest_kw=dict(min_samples_leaf=leaf,
                                                   n_estimators=ntree))
            rows.append(dict(check='hyperparams', detail=f'leaf={leaf},trees={ntree}',
                             premium=r['premium'], ate=r['ate']))
            print(f'leaf={leaf} trees={ntree}: ATE={r["ate"]:+.4f} '
                  f'premium={r["premium"]:+.4f}')

    # C2: winsorise treatment at 1st/99th pct
    Tw = np.clip(T, np.percentile(T, 1), np.percentile(T, 99))
    r = point(Y, Tw, Xb, ht)
    rows.append(dict(check='winsorised_T', detail='p1/p99',
                     premium=r['premium'], ate=r['ate']))
    print(f'winsorised T: ATE={r["ate"]:+.4f} premium={r["premium"]:+.4f}')

    # C3: drop cohorts needing the HE decimal repair (2022-23)
    clean = mask0 & (df['first_grant_year'] < 2022)
    Yc = df.loc[clean, 'rev_growth'].values
    Tc = df.loc[clean, 'log_ec'].values
    Xc = X_all.loc[clean].values.astype(float)
    hc = df.loc[clean, 'ht_oecd'].values
    r = point(Yc, Tc, Xc, hc)
    rows.append(dict(check='pre2022_cohorts', detail=f'n={int(clean.sum())}',
                     premium=r['premium'], ate=r['ate']))
    print(f'pre-2022 cohorts only (n={int(clean.sum())}): '
          f'ATE={r["ate"]:+.4f} premium={r["premium"]:+.4f}')
    pd.DataFrame(rows).to_csv(OUT_DIR / 'sensitivity.csv', index=False)


if __name__ == '__main__':
    df = pd.read_csv(OUT_DIR / 'analysis_ready_v2.csv')
    X_all = make_controls(df)
    part_a(df)
    part_b(df, X_all)
    part_c(df, X_all)
