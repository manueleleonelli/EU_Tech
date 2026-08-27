"""
Core estimation machinery shared by all analysis scripts.

Three inference layers for the high-tech premium, replacing the thesis's
independence-assumption SE (which treated two group means from the same
forest as independent and produced CIs an order of magnitude too tight):

  1. point()      -- point estimates: ATE (with econml's honest within-forest
                     inference) and the premium (difference in mean CATEs).
  2. bootstrap()  -- firm-level bootstrap of the ENTIRE pipeline (refit both
                     nuisance models and the forest on each resample). This is
                     the paper's headline uncertainty for the premium.
  3. permutation()-- randomisation test: shuffle the high-tech label, refit,
                     rebuild the null distribution of the premium. Answers
                     "is the premium distinguishable from label noise?"
                     directly, which is the referee's question.
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
from config import FOREST, CV_FOLDS, SEED

warnings.filterwarnings('ignore')


def make_controls(df):
    """53-feature control matrix as in the thesis (plus stable column order)."""
    parts = [df[['log_rev_pre', 'log_emp_pre', 'log_assets_pre', 'pm_pre', 'offset']]]
    for col, pref in [('country', 'ctry'), ('nace_section', 'sec'),
                      ('independence', 'indep'), ('first_grant_year', 'cohort')]:
        parts.append(pd.get_dummies(df[col], prefix=pref, drop_first=True))
    X = pd.concat(parts, axis=1)
    X.columns = X.columns.astype(str)
    return X


def _fit(Y, T, X, seed, weights=None, forest_kw=None):
    kw = dict(FOREST)
    if forest_kw:
        kw.update(forest_kw)
    est = CausalForestDML(
        model_y=RandomForestRegressor(**kw, n_jobs=-1, random_state=seed),
        model_t=RandomForestRegressor(**kw, n_jobs=-1, random_state=seed),
        n_estimators=kw['n_estimators'],
        min_samples_leaf=kw['min_samples_leaf'],
        random_state=seed, cv=CV_FOLDS)
    est.fit(Y, T, X=X, sample_weight=weights)
    return est


def _premium(est, X, ht, weights=None):
    cate = est.effect(X).flatten()
    m = ht == 1
    if weights is None:
        return float(cate[m].mean() - cate[~m].mean())
    w = np.asarray(weights, dtype=float)
    return float(np.average(cate[m], weights=w[m])
                 - np.average(cate[~m], weights=w[~m]))


def point(Y, T, Xb, ht, seed=SEED, weights=None, forest_kw=None):
    X = np.column_stack([Xb, ht.reshape(-1, 1)])
    est = _fit(Y, T, X, seed, weights, forest_kw)
    inf = est.ate_inference(X=X)
    s = lambda x: float(np.asarray(x).flat[0])
    ate, ate_se = s(inf.mean_point), s(inf.stderr_mean)
    return dict(ate=ate, ate_se=ate_se,
                ate_pval=s(inf.pvalue()) if callable(getattr(inf, 'pvalue', None))
                else float('nan'),
                premium=_premium(est, X, ht, weights),
                n=len(Y), n_ht=int(ht.sum()), n_non_ht=int((1 - ht).sum()))


def bootstrap(Y, T, Xb, ht, n_boot, seed=SEED, weights=None):
    """Firm-level bootstrap of premium and ATE. Returns replicate arrays."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    prem, ates = [], []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        hb = ht[idx]
        if hb.sum() < 5 or (1 - hb).sum() < 5:
            continue
        X = np.column_stack([Xb[idx], hb.reshape(-1, 1)])
        wb = None if weights is None else weights[idx]
        est = _fit(Y[idx], T[idx], X, seed=int(rng.integers(1e9)),
                   weights=wb)
        prem.append(_premium(est, X, hb, wb))
        ates.append(float(est.effect(X).flatten().mean() if wb is None
                          else np.average(est.effect(X).flatten(), weights=wb)))
    prem, ates = np.array(prem), np.array(ates)
    return dict(premium_reps=prem, ate_reps=ates,
                premium_se=float(prem.std(ddof=1)),
                premium_ci=(float(np.percentile(prem, 2.5)),
                            float(np.percentile(prem, 97.5))),
                ate_se_boot=float(ates.std(ddof=1)))


def permutation(Y, T, Xb, ht, n_perm, observed_premium, seed=SEED):
    """Permute the HT label within the subgroup; two-sided p-value."""
    rng = np.random.default_rng(seed + 1)
    null = []
    for p in range(n_perm):
        hp = rng.permutation(ht)
        X = np.column_stack([Xb, hp.reshape(-1, 1)])
        est = _fit(Y, T, X, seed=int(rng.integers(1e9)))
        null.append(_premium(est, X, hp))
    null = np.array(null)
    pval = float((np.abs(null) >= abs(observed_premium)).mean())
    # add-one correction to avoid p = 0
    pval = max(pval, 1.0 / (n_perm + 1))
    return dict(perm_null=null, perm_pval=pval,
                perm_null_sd=float(null.std(ddof=1)))
