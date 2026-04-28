"""
DSA 210 - Tourism Demand Under Shock
Phase 3: Data Pipeline — Merge & Engineer Features
Author: Eren Sean Harley | 36054

Data Sources:
  - TÜİK: Visitor counts by nationality (2003-2025)
  - IMF DataMapper API: GDP growth, inflation, GDP per capita  (data/imf_macro_data.csv)
  - IMF DataMapper API: Turkey macro + PPP exchange rate        (data/imf_turkey_data.csv)
  - World Bank Data360 WGI: Political Stability Index           (data/wb_political_stability.csv)

Run fetch_macro_data.py first to refresh the three CSV files above.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

# ============================================================
# 1. LOAD TÜİK VISITOR DATA
# ============================================================

df_raw = pd.read_excel(
    DATA_DIR / 'Çıkış Yapan Yabancı ve Vatandaşlar.xls',
    engine="xlrd", header=None
)

years = [int(x) for x in df_raw.iloc[3, 1:].dropna().tolist()]

country_map = {
    'Almanya-Germany':                              'Germany',
    'İngiltere (Birleşik Krallık)-United Kingdom':  'United Kingdom',
    'Fransa- France':                               'France',
    'Hollanda-Netherlands':                         'Netherlands',
    'Rusya Federasyonu-Russian Federation':         'Russia',
    'İran-Iran':                                    'Iran',
    'Azerbaycan-Azerbaijan':                        'Azerbaijan',
    'Gürcistan-Georgia':                            'Georgia',
    'Bulgaristan-Bulgaria':                         'Bulgaria',
    'Yunanistan-Greece':                            'Greece',
    'Ukrayna-Ukraine':                              'Ukraine',
    'A.B.D. - U.S.A.':                             'USA',
    'İsrail- Israel':                               'Israel',
    'Irak- Iraq':                                   'Iraq',
    'Suriye- Syria':                                'Syria',
    'B.A.E.-U.A.E.':                               'United Arab Emirates',
    'Katar- Qatar':                                 'Qatar',
}

records = []
for _, row in df_raw.iloc[4:122].iterrows():
    raw = str(row.iloc[0]).strip()
    if pd.isna(row.iloc[0]) or raw in ['nan', '']:
        continue
    country = country_map.get(raw)
    if not country:
        continue
    for i, year in enumerate(years):
        val = row.iloc[i + 1]
        if pd.notna(val):
            records.append({'country': country, 'year': year, 'visitors': float(val)})

df_visitors = pd.DataFrame(records)
df_visitors = df_visitors[df_visitors['year'].between(2003, 2025)]
print(f"Visitors loaded: {df_visitors.shape} | Countries: {df_visitors['country'].nunique()}")

# ============================================================
# 2. LOAD IMF SOURCE MARKET DATA (from API-generated CSV)
# ============================================================

df_imf = pd.read_csv(DATA_DIR / 'imf_macro_data.csv')
df_imf = df_imf[df_imf['year'].between(2003, 2025)].reset_index(drop=True)
print(f"IMF source data loaded: {df_imf.shape}")

# ============================================================
# 3. LOAD TURKEY MACRO DATA (from API-generated CSV)
# ============================================================

df_turkey = pd.read_csv(DATA_DIR / 'imf_turkey_data.csv')
df_turkey = df_turkey[df_turkey['year'].between(2003, 2025)].reset_index(drop=True)
# Rename to match the column name used downstream
df_turkey = df_turkey.rename(columns={'tur_ppp_exchange_rate': 'tur_ppp_rate'})
print(f"Turkey macro loaded: {df_turkey.shape}")

# ============================================================
# 4. LOAD WORLD BANK POLITICAL STABILITY (from API-generated CSV)
# ============================================================

df_wgi_all = pd.read_csv(DATA_DIR / 'wb_political_stability.csv')
print(f"WGI loaded: {df_wgi_all.shape}")

# ============================================================
# 5. MERGE INTO PANEL DATASET
# ============================================================

df = df_visitors.merge(df_imf, on=['country', 'year'], how='left')
df = df.merge(df_wgi_all[df_wgi_all['country'] != 'Turkey'], on=['country', 'year'], how='left')
df = df.merge(df_turkey, on='year', how='left')
df = df.merge(
    df_wgi_all[df_wgi_all['country'] == 'Turkey']
    .rename(columns={'political_stability': 'tur_political_stability'})
    [['year', 'tur_political_stability']],
    on='year', how='left'
)

df = df.sort_values(['country', 'year']).reset_index(drop=True)

# ============================================================
# 6. ENGINEER FEATURES
# ============================================================

# Shock dummies
df['covid']                = df['year'].between(2020, 2021).astype(int)
df['coup_2016']            = (df['year'] == 2016).astype(int)
df['syria_conflict']       = df['year'].between(2011, 2015).astype(int)
df['russia_turkey_crisis'] = (df['year'] == 2015).astype(int)
df['russia_ukraine_war']   = (df['year'] >= 2022).astype(int)
# mena_tension_recent covers ONLY the post-2023 period.
# The old combined dummy (2011-2015 OR 2023+) was scrapped — it merged two unrelated
# shock regimes (Syria war vs post-Oct-2023 MENA tensions) into one meaningless coefficient.
# syria_conflict already covers 2011-2015; keep them separate for ML in Phase 4.
df['mena_tension_recent']  = (df['year'] >= 2023).astype(int)

# YoY % change in visitors
df['visitors_yoy'] = df.groupby('country')['visitors'].pct_change() * 100

# Pre-shock baseline (2017-2019 average = pre-COVID, post-coup recovery)
baseline = (df[df['year'].between(2017, 2019)]
            .groupby('country')['visitors'].mean()
            .rename('baseline_visitors'))
df = df.merge(baseline.reset_index(), on='country', how='left')
df['visitors_vs_baseline'] = (df['visitors'] / df['baseline_visitors'] - 1) * 100

# Log visitors
df['log_visitors'] = np.log(df['visitors'])

# Turkey currency weakness proxy: % change in PPP rate (higher = weaker lira)
# Compute year-over-year change from unique Turkey-level data, then merge back.
# (Using df.pct_change() directly would compare across country boundaries since the
#  panel is sorted by country→year, producing garbage values at each country's first row.)
tur_ppp_yoy = (df[['year', 'tur_ppp_rate']].drop_duplicates()
               .sort_values('year')
               .assign(tur_currency_weakness=lambda x: x['tur_ppp_rate'].pct_change() * 100)
               [['year', 'tur_currency_weakness']])
df = df.merge(tur_ppp_yoy, on='year', how='left')

# Market classification
western_eu    = ['Germany', 'United Kingdom', 'France', 'Netherlands']
eastern_eu    = ['Bulgaria', 'Greece', 'Ukraine']
former_soviet = ['Russia', 'Azerbaijan', 'Georgia']
mena_markets  = ['Iran', 'Iraq', 'Syria', 'Israel', 'United Arab Emirates', 'Qatar']


def classify(c):
    if c in western_eu:    return 'Western Europe'
    if c in eastern_eu:    return 'Eastern Europe'
    if c in former_soviet: return 'Former Soviet'
    if c in mena_markets:  return 'MENA'
    return 'Other'


df['market_group'] = df['country'].map(classify)

# MENA subgroup flags for H6 within-MENA descriptive split.
# MENA-stable: UAE and Qatar — no major bilateral shocks with Turkey 2003-2025.
# MENA-conflict-affected: Iran, Iraq, Israel, Syria — home-country instability or sanctions.
# These flags are used in 02_eda_and_hypothesis.py for narrative comparison only
# (n=2 in stable subgroup is too small for a formal test).
df['is_mena_stable']   = df['country'].isin(['United Arab Emirates', 'Qatar']).astype(int)
df['is_mena_conflict'] = df['country'].isin(['Iran', 'Iraq', 'Israel', 'Syria']).astype(int)

print(f"\nFinal panel: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

out_path = DATA_DIR / 'panel_dataset.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
