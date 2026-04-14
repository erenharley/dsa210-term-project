"""
DSA 210 - Tourism Demand Under Shock
Macro Data Fetcher
Author: Eren Sean Harley | 36054

Sources:
  - IMF DataMapper API: https://www.imf.org/external/datamapper/api/v1/
  - World Bank Data360 API: https://data360api.worldbank.org/data360/data
"""

import requests
import pandas as pd
import time

# Source market countries
IMF_COUNTRIES = {
    'DEU': 'Germany',
    'GBR': 'United Kingdom',
    'FRA': 'France',
    'NLD': 'Netherlands',
    'RUS': 'Russia',
    'IRN': 'Iran',
    'AZE': 'Azerbaijan',
    'GEO': 'Georgia',
    'BGR': 'Bulgaria',
    'GRC': 'Greece',
    'UKR': 'Ukraine',
    'USA': 'USA',
    'ISR': 'Israel',
    'IRQ': 'Iraq',
    'SYR': 'Syria',
}

WB_COUNTRIES = {
    'DEU': 'Germany',
    'GBR': 'United Kingdom',
    'FRA': 'France',
    'NLD': 'Netherlands',
    'RUS': 'Russia',
    'IRN': 'Iran',
    'AZE': 'Azerbaijan',
    'GEO': 'Georgia',
    'BGR': 'Bulgaria',
    'GRC': 'Greece',
    'UKR': 'Ukraine',
    'USA': 'USA',
    'ISR': 'Israel',
    'IRQ': 'Iraq',
    'SYR': 'Syria',
}

# Turkey — destination country, fetched separately
TURKEY_IMF_CODE = 'TUR'
TURKEY_WB_CODE  = 'TUR'

YEARS     = list(range(2003, 2026))
years_str = ",".join(str(y) for y in YEARS)

# ============================================================
# PART 1: IMF DataMapper — Source market macro data
# ============================================================

IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"
IMF_INDICATORS = {
    'NGDP_RPCH': 'gdp_growth',
    'PCPIPCH':   'inflation',
    'NGDPDPC':   'gdp_per_capita',
}

country_str = "/".join(IMF_COUNTRIES.keys())

print("=" * 60)
print("PART 1: Fetching IMF data — source markets")
print("=" * 60)

imf_records = []

for imf_code, col_name in IMF_INDICATORS.items():
    url = f"{IMF_BASE}/{imf_code}/{country_str}?periods={years_str}"
    print(f"  GET {imf_code}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    values = r.json().get('values', {}).get(imf_code, {})

    matched = 0
    for iso3, year_dict in values.items():
        country_name = IMF_COUNTRIES.get(iso3)
        if not country_name:
            continue
        for year_str, value in year_dict.items():
            imf_records.append({
                'country':   country_name,
                'year':      int(year_str),
                'indicator': col_name,
                'value':     float(value) if value is not None else None
            })
            matched += 1
    print(f"    -> {matched} records")
    time.sleep(0.5)

df_imf_long = pd.DataFrame(imf_records)
df_imf = df_imf_long.pivot_table(
    index=['country', 'year'],
    columns='indicator',
    values='value',
    aggfunc='first'
).reset_index()
df_imf.columns.name = None
df_imf = df_imf[df_imf['year'].between(2003, 2025)].reset_index(drop=True)

print(f"\nSource market IMF shape: {df_imf.shape}")
df_imf.to_csv('imf_macro_data.csv', index=False)
print("✓ Saved: imf_macro_data.csv")

# ============================================================
# PART 2: IMF DataMapper — Turkey destination data
# Extra indicator: NGDP_RPCH, PCPIPCH, NGDPDPC + exchange rate proxy
# ============================================================

print("\n" + "=" * 60)
print("PART 2: Fetching IMF data — Turkey (destination)")
print("=" * 60)

# Also fetch PPPEX = implied PPP conversion rate (proxy for currency weakness)
TURKEY_INDICATORS = {
    'NGDP_RPCH': 'tur_gdp_growth',
    'PCPIPCH':   'tur_inflation',
    'NGDPDPC':   'tur_gdp_per_capita',
    'PPPEX':     'tur_ppp_exchange_rate',   # higher = weaker lira in real terms
}

turkey_records = []

for imf_code, col_name in TURKEY_INDICATORS.items():
    url = f"{IMF_BASE}/{imf_code}/{TURKEY_IMF_CODE}?periods={years_str}"
    print(f"  GET {imf_code} for Turkey...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    values = r.json().get('values', {}).get(imf_code, {})
    year_dict = values.get(TURKEY_IMF_CODE, {})

    for year_str, value in year_dict.items():
        turkey_records.append({
            'year':      int(year_str),
            'indicator': col_name,
            'value':     float(value) if value is not None else None
        })
    print(f"    -> {len(year_dict)} records")
    time.sleep(0.5)

df_turkey_long = pd.DataFrame(turkey_records)
df_turkey = df_turkey_long.pivot_table(
    index='year',
    columns='indicator',
    values='value',
    aggfunc='first'
).reset_index()
df_turkey.columns.name = None
df_turkey = df_turkey[df_turkey['year'].between(2003, 2025)].reset_index(drop=True)

print(f"\nTurkey IMF shape: {df_turkey.shape}")
print(df_turkey.to_string())
df_turkey.to_csv('imf_turkey_data.csv', index=False)
print("✓ Saved: imf_turkey_data.csv")

# ============================================================
# PART 3: World Bank WGI — Political Stability
# Source markets + Turkey
# ============================================================

WB_BASE      = "https://data360api.worldbank.org/data360/data"
ALL_WB_CODES = dict(WB_COUNTRIES)
ALL_WB_CODES[TURKEY_WB_CODE] = 'Turkey'

print("\n" + "=" * 60)
print("PART 3: Fetching World Bank WGI Political Stability...")
print("=" * 60)

# Detect correct COMP_BREAKDOWN_1 value
r_check = requests.get(
    WB_BASE,
    params={'DATABASE_ID': 'WB_WGI', 'INDICATOR': 'GOV_WGI_PV', 'REF_AREA': 'DEU'},
    timeout=30
)
breakdowns = set(rec.get('COMP_BREAKDOWN_1') for rec in r_check.json().get('value', []))
print(f"  Available breakdowns: {breakdowns}")
target_breakdown = 'WGI_EST' if 'WGI_EST' in breakdowns else sorted(breakdowns)[0]
print(f"  Using: {target_breakdown}")

wb_records = []

for iso3, country_name in ALL_WB_CODES.items():
    params = {
        'DATABASE_ID':      'WB_WGI',
        'INDICATOR':        'GOV_WGI_PV',
        'REF_AREA':         iso3,
        'COMP_BREAKDOWN_1': target_breakdown,
        'timePeriodFrom':   '2003',
        'timePeriodTo':     '2024',
    }
    print(f"  {country_name} ({iso3})...")
    r = requests.get(WB_BASE, params=params, timeout=30)
    r.raise_for_status()
    records = r.json().get('value', [])
    print(f"    Records: {len(records)}")

    for record in records:
        try:
            wb_records.append({
                'country':             country_name,
                'year':                int(record['TIME_PERIOD']),
                'political_stability': float(record['OBS_VALUE']) if record.get('OBS_VALUE') else None
            })
        except (KeyError, ValueError) as e:
            print(f"    Skipping: {e}")

    time.sleep(0.3)

df_wb = pd.DataFrame(wb_records)
print(f"\nWGI Political Stability shape: {df_wb.shape}")
print(df_wb[df_wb['country'] == 'Turkey'].to_string())
df_wb.to_csv('wb_political_stability.csv', index=False)
print("\n✓ Saved: wb_political_stability.csv")

print("\n" + "=" * 60)
print("ALL DONE. Upload these 3 CSVs to Claude:")
print("  - imf_macro_data.csv       (source market macro)")
print("  - imf_turkey_data.csv      (Turkey macro + exchange rate)")
print("  - wb_political_stability.csv (political stability all countries incl. Turkey)")
print("=" * 60)