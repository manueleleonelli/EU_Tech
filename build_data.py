"""
Build the analysis-ready dataset (v2), correcting three issues found in the
thesis pipeline:

  FIX 1  Horizon Europe EC contributions: HE rows whose euro amounts contained
         cents were inflated x100 upstream (thousand separators and the decimal
         comma were stripped: "2.063.988,75" -> 206398875). Verified against
         CORDIS (project 101091934: recorded 206,398,875 for one participant
         vs a total project contribution of 10,111,515.75). Heuristic repair
         here; permanent fix is re-extraction with locale-aware parsing.

  FIX 2  Instrument classification: EIC Pathfinder and Transition participations
         (identified via the topics field) are early-stage instruments
         (TRL 1-6), not SME instruments. 149 participations, 64 firms move.

  FIX 3  OECD definition: the top tier of Galindo-Rueda & Verger (2016) is
         ISIC 21, 26, 30.3, 58.2, 72 -- not {21,26,59,60,61,62,63,72}. The
         strict definition is implemented at 4-digit precision (3030-3039,
         5820-5829 for the two 3-digit classes). The thesis coding is kept as
         `ht_oecd_thesis` (an ICT-inclusive variant) for comparison.

Also made explicit: the yr_minus_0 duplicate-column unit trap. The unsuffixed
`operating_revenue__turnover__yr_minus_0` is in EUR while every other revenue
column (including the `.1` duplicate) is in kEUR; the `.1` column must be used.
An assertion now guards this instead of a silent preference.
"""
import pandas as pd
import numpy as np
import warnings
from config import DATA_DIR, OUT_DIR, EC_FIX, EC_SUSPECT_THRESHOLD

warnings.filterwarnings('ignore')


def load_raw():
    df = pd.read_csv(DATA_DIR / 'company_master_wide.csv', low_memory=False)
    cp = pd.read_csv(DATA_DIR / 'company_project.csv')
    calvino = pd.read_csv(DATA_DIR / 'calvino_digital_intensity.csv')
    pavitt = pd.read_csv(DATA_DIR / 'pavitt_taxonomy_nace2.csv')
    return df, cp, calvino, pavitt


def expand_range_map(table, key_col, val_col):
    out = {}
    for _, row in table.iterrows():
        k = str(row[key_col])
        if '-' in k:
            lo, hi = k.split('-')
            for c in range(int(lo), int(hi) + 1):
                out[c] = row[val_col]
        else:
            out[int(k)] = row[val_col]
    return out


def add_taxonomies(df, calvino, pavitt):
    df['nace_4d'] = pd.to_numeric(df['NACE Rev. 2, core code (4 digits)'], errors='coerce')
    df['nace_2d'] = (df['nace_4d'] // 100).astype('Int64')

    sections = [(1,3,'A'),(5,9,'B'),(10,33,'C'),(35,35,'D'),(36,39,'E'),(41,43,'F'),
                (45,47,'G'),(49,53,'H'),(55,56,'I'),(58,63,'J'),(64,66,'K'),(68,68,'L'),
                (69,75,'M'),(77,82,'N'),(84,84,'O'),(85,85,'P'),(86,88,'Q'),(90,93,'R'),
                (94,96,'S'),(97,98,'T'),(99,99,'U')]
    def to_section(c):
        if pd.isna(c): return 'Unknown'
        for lo, hi, s in sections:
            if lo <= int(c) <= hi: return s
        return 'Unknown'
    df['nace_section'] = df['nace_2d'].apply(to_section)

    # FIX 3 -- strict Galindo-Rueda & Verger (2016) top tier, at 4-digit precision
    n4, n2 = df['nace_4d'], df['nace_2d']
    df['ht_oecd'] = (
        n2.isin([21, 26, 72])
        | ((n4 >= 3030) & (n4 < 3040))     # 30.3 air and spacecraft
        | ((n4 >= 5820) & (n4 < 5830))     # 58.2 software publishing
    ).astype(int)
    # Thesis coding retained as an explicit ICT-inclusive variant
    df['ht_oecd_thesis'] = n2.isin([21, 26, 59, 60, 61, 62, 63, 72]).astype(int)

    cmap = expand_range_map(calvino, 'isic_rev4', 'digital_intensity_2013_15')
    df['ht_calvino'] = (df['nace_2d'].map(cmap) == 'High').astype(int)

    pmap = expand_range_map(pavitt, 'nace_rev2', 'pavitt_category')
    df['pavitt_category'] = df['nace_2d'].map(pmap)
    df['ht_pavitt'] = (df['pavitt_category'] == 'Science based').astype(int)
    df['ht_pavitt_broad'] = df['pavitt_category'].isin(
        ['Science based', 'Specialised suppliers']).astype(int)
    return df


def fix_ec_contributions(cp):
    """FIX 1 -- repair x100-inflated Horizon Europe contributions."""
    cp['ec_raw'] = pd.to_numeric(cp['ecContribution_cordis'], errors='coerce')
    if EC_FIX:
        suspect = ((cp['_programme'] == 'HE')
                   & (cp['ec_raw'] > EC_SUSPECT_THRESHOLD)
                   & (cp['ec_raw'] % 100 != 0))
        cp['ec'] = np.where(suspect, cp['ec_raw'] / 100, cp['ec_raw'])
        cp['ec_fixed_flag'] = suspect.astype(int)
        n = int(suspect.sum())
        print(f'FIX 1: repaired {n} HE participations '
              f'(median before {cp.loc[suspect,"ec_raw"].median():,.0f}, '
              f'after {cp.loc[suspect,"ec"].median():,.0f})')
    else:
        cp['ec'] = cp['ec_raw']
        cp['ec_fixed_flag'] = 0
    return cp


def classify_instruments(cp):
    """FIX 2 -- Pathfinder/Transition are early-stage; everything else as thesis."""
    def base(scheme):
        s = str(scheme).upper()
        if any(x in s for x in ['SME-1', 'SME-2', 'EIC', 'FTI']): return 'SME'
        if any(x in s for x in ['ERC', 'MSCA']): return 'Early-stage'
        if any(x in s for x in ['RIA', 'CSA', 'ECSEL', 'JTI', 'BBI', 'IMI',
                                'FCH', 'SESAR', 'S2R', 'CLEAN SKY']): return 'Collaborative'
        if s == 'IA' or s.startswith('IA-') or '-IA' in s: return 'Collaborative'
        return 'Other'

    def corrected(scheme, topic):
        s, t = str(scheme).upper(), str(topic).upper()
        if 'EIC' in s and ('PATHFINDER' in t or 'TRANSITION' in t):
            return 'Early-stage'
        return base(scheme)

    cp['instrument'] = [corrected(s, t) for s, t
                        in zip(cp['project_fundingScheme'], cp['topics'])]
    cp['instrument_thesis'] = cp['project_fundingScheme'].apply(base)
    n = int((cp['instrument'] != cp['instrument_thesis']).sum())
    print(f'FIX 2: reclassified {n} Pathfinder/Transition participations to Early-stage')
    return cp


def primary_instrument(cp, col):
    g = (cp.groupby(['organisationID', col])['ec'].sum().reset_index()
           .sort_values('ec', ascending=False)
           .drop_duplicates('organisationID'))
    return g.set_index('organisationID')[col]


def build_treatment(cp):
    cp['startDate'] = pd.to_datetime(cp['startDate'], errors='coerce')
    cp['grant_year'] = cp['startDate'].dt.year
    fg = cp.groupby('organisationID')['grant_year'].min().rename('fgy')
    cp = cp.join(fg, on='organisationID')
    pre = cp[cp['grant_year'] <= cp['fgy']]
    ec = pre.groupby('organisationID')['ec'].sum().rename('ec_predetermined')
    flagged = pre.groupby('organisationID')['ec_fixed_flag'].max().rename('ec_was_fixed')
    return pd.concat([ec, flagged], axis=1)


def build_financials(df):
    """Grant-anchored outcome and control construction, with the unit guard."""
    # Unit guard for the duplicate year-0 revenue column
    a = pd.to_numeric(df['operating_revenue__turnover__yr_minus_0'], errors='coerce')
    b = pd.to_numeric(df['operating_revenue__turnover__yr_minus_0.1'], errors='coerce')
    r1 = pd.to_numeric(df['operating_revenue__turnover__yr_minus_1'], errors='coerce')
    m = (a > 0) & (b > 0) & (r1 > 0)
    assert np.abs(np.log(b[m] / r1[m])).median() < np.abs(np.log(a[m] / r1[m])).median(), \
        'Unit check failed: the .1 revenue column is no longer the kEUR one.'

    def matrix(prefix, n=30):
        M = np.full((len(df), n), np.nan)
        for j in range(n):
            col = f'{prefix}yr_minus_{j}'
            if j == 0 and f'{col}.1' in df.columns:
                col = f'{col}.1'
            if col in df.columns:
                M[:, j] = pd.to_numeric(df[col], errors='coerce').values
        return M

    rev, emp = matrix('operating_revenue__turnover__'), matrix('number_of_employees__')
    ass, pm = matrix('total_assets__'), matrix('profit_margin__')

    df['last_year'] = pd.to_numeric(df['Last avail. year'], errors='coerce')
    df['first_grant_year'] = pd.to_numeric(df['first_grant_year'], errors='coerce')
    df['offset'] = df['last_year'] - df['first_grant_year']
    off = df['offset'].values

    def pre_mean(M):
        out = np.full(len(off), np.nan)
        for i, o in enumerate(off):
            if np.isnan(o): continue
            idx = [int(o) + 3, int(o) + 4, int(o) + 5]
            if all(0 <= k < 30 for k in idx):
                v = [M[i, k] for k in idx if not np.isnan(M[i, k])]
                if v: out[i] = np.mean(v)
        return out

    rp, ep = rev[:, 0], emp[:, 0]
    rpre, epre = pre_mean(rev), pre_mean(emp)
    df['rev_growth'] = np.where((rp > 0) & (rpre > 0), np.log(rp) - np.log(rpre), np.nan)
    df['emp_growth'] = np.where((ep > 0) & (epre > 0), np.log(ep) - np.log(epre), np.nan)
    df['log_rev_pre'] = np.where(rpre > 0, np.log(rpre), np.nan)
    df['log_emp_pre'] = np.where(epre > 0, np.log(epre), np.nan)
    apre = pre_mean(ass)
    df['log_assets_pre'] = np.where(apre > 0, np.log(apre), np.nan)
    df['pm_pre'] = pre_mean(pm)
    return df


def apply_filters(df):
    timing = (df['offset'] >= 1) & (df['offset'] + 5 <= 29) & df['offset'].notna()
    fin = ['rev_growth', 'emp_growth', 'log_rev_pre', 'log_emp_pre',
           'log_assets_pre', 'pm_pre']
    df.loc[~timing, fin] = np.nan
    controls = (df[['log_rev_pre', 'log_emp_pre', 'log_assets_pre', 'pm_pre',
                    'offset']].notna().all(axis=1)
                & df['nace_2d'].notna() & df['Country ISO code'].notna()
                & df['first_grant_year'].notna())
    df['keep_rev'] = df['rev_growth'].notna() & df['log_ec'].notna() & controls
    df['keep_emp'] = df['emp_growth'].notna() & df['log_ec'].notna() & controls
    print('\nAttrition: full {:,} | timing {:,} | final rev {:,} | final emp {:,}'
          .format(len(df), int(timing.sum()),
                  int(df['keep_rev'].sum()), int(df['keep_emp'].sum())))
    return df


def main():
    df, cp, calvino, pavitt = load_raw()
    df = add_taxonomies(df, calvino, pavitt)
    cp = fix_ec_contributions(cp)
    cp = classify_instruments(cp)

    df = df.join(primary_instrument(cp, 'instrument').rename('primary_instrument'),
                 on='organisationID')
    df = df.join(primary_instrument(cp, 'instrument_thesis')
                 .rename('primary_instrument_thesis'), on='organisationID')
    df = df.join(build_treatment(cp), on='organisationID')
    df['log_ec'] = np.log(df['ec_predetermined'].where(df['ec_predetermined'] > 0))

    def indep(v):
        v = str(v).strip().upper()
        if v.startswith('A'): return 'Independent'
        if v.startswith('B'): return 'Partial'
        if v.startswith(('C', 'D')): return 'Subsidiary'
        return 'Unknown'
    df['independence'] = df['bvd_independence'].apply(indep)

    df = build_financials(df)
    df = apply_filters(df)

    top = df['Country ISO code'].value_counts().head(15).index
    df['country'] = df['Country ISO code'].where(df['Country ISO code'].isin(top), 'Other')

    keep = ['organisationID', 'country', 'nace_2d', 'nace_4d', 'nace_section',
            'ht_oecd', 'ht_oecd_thesis', 'ht_calvino', 'ht_pavitt', 'ht_pavitt_broad',
            'primary_instrument', 'primary_instrument_thesis', 'independence',
            'first_grant_year', 'last_year', 'offset',
            'ec_predetermined', 'ec_was_fixed', 'log_ec',
            'rev_growth', 'emp_growth',
            'log_rev_pre', 'log_emp_pre', 'log_assets_pre', 'pm_pre',
            'keep_rev', 'keep_emp']
    df[keep].to_csv(OUT_DIR / 'analysis_ready_v2.csv', index=False)
    print(f"\nSaved {OUT_DIR/'analysis_ready_v2.csv'}")

    print('\nHigh-tech shares (full sample):')
    for c in ['ht_oecd', 'ht_oecd_thesis', 'ht_calvino', 'ht_pavitt', 'ht_pavitt_broad']:
        print(f'  {c:16s} {df[c].mean():6.1%}  ({int(df[c].sum()):,})')
    print('\nPrimary instrument (corrected):')
    print(df['primary_instrument'].value_counts().to_string())


if __name__ == '__main__':
    main()
