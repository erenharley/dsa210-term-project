"""
DSA 210 - Tourism Demand Under Shock
Phase 3: EDA + Hypothesis Tests — Market Sensitivity & Recovery
Author: Eren Sean Harley | 36054
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR.parent / 'images'
FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(BASE_DIR.parent / 'data' / 'panel_dataset.csv')

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f5f5f5',
    'axes.grid': True, 'grid.color': 'white', 'grid.linewidth': 1.3,
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11, 'legend.fontsize': 9
})

GROUP_COLORS = {
    'Western Europe': '#2196F3',
    'Eastern Europe': '#4CAF50',
    'Former Soviet':  '#FF5722',
    'MENA':           '#9C27B0',
    'Other':          '#607D8B',
}

SHOCKS = [
    (2011, 2015, '#FFE0E0', 'Syria Conflict'),
    (2015, 2016, '#FFD580', 'Russia-TUR Crisis'),
    (2016, 2017, '#FFCF6B', '2016 Coup'),
    (2020, 2022, '#BBDEFB', 'COVID-19'),
    (2022, 2026, '#E8D5FF', 'Russia-UKR War / MENA'),
]

# ============================================================
# FIG 1: Visitor trends by market group with shocks
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
groups = ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA']

for ax, group in zip(axes.flatten(), groups):
    countries = df[df['market_group'] == group]['country'].unique()
    for s_start, s_end, s_color, s_label in SHOCKS:
        ax.axvspan(s_start, s_end, alpha=0.25, color=s_color, zorder=0)
    for country in countries:
        d = df[df['country'] == country].sort_values('year')
        ax.plot(d['year'], d['visitors']/1e6, marker='o', ms=4, lw=2.2, label=country)
    ax.set_title(f'{group} Markets', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Visitors (millions)')
    ax.legend(loc='upper left')
    ax.set_xlim(2003, 2025)

# Shock legend
patches = [mpatches.Patch(color=c, alpha=0.5, label=l) for _, _, c, l in SHOCKS]
fig.legend(handles=patches, loc='lower center', ncol=5,
           title='Shock Periods', fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Turkey Inbound Tourism by Market Group (2003–2025)\nIMF + TÜİK Data',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1_trends_by_group.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved")

# ============================================================
# FIG 2: Shock sensitivity — % drop per market per shock
# ============================================================

def shock_impact(country, shock_year, pre_window=2):
    """Mean visitors in [shock_year-pre_window, shock_year-1] vs shock_year."""
    d = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    shock = d[d['year'] == shock_year]['visitors'].values
    if pre == 0 or len(shock) == 0 or np.isnan(pre):
        return np.nan
    return (shock[0] / pre - 1) * 100

shocks_to_test = {
    'COVID\n(2020 vs 2018-19)':       2020,
    '2016 Coup\n(2016 vs 2014-15)':   2016,
    'Syria Onset\n(2012 vs 2010-11)': 2012,
    'Russia-TUR\n(2015 vs 2013-14)':  2015,
}

all_countries = sorted(df['country'].unique())

impact_df = pd.DataFrame(index=all_countries)
for shock_label, shock_yr in shocks_to_test.items():
    impact_df[shock_label] = [shock_impact(c, shock_yr) for c in all_countries]

fig, axes = plt.subplots(1, 4, figsize=(20, 8), sharey=True)

for ax, col in zip(axes, impact_df.columns):
    vals = impact_df[col].dropna().sort_values()
    colors = [GROUP_COLORS[df[df['country']==c]['market_group'].values[0]] for c in vals.index]
    bars = ax.barh(vals.index, vals.values, color=colors, alpha=0.85, edgecolor='white', lw=1.2)
    ax.axvline(0, color='black', lw=1.2)
    ax.set_title(col, fontweight='bold', fontsize=11)
    ax.set_xlabel('% Change vs Pre-Shock')
    for bar, val in zip(bars, vals.values):
        ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=8)

# Group legend
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items() if g != 'Other']
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           title='Market Group', bbox_to_anchor=(0.5, -0.04))
fig.suptitle('Shock Sensitivity by Market: % Change vs Pre-Shock Baseline\n(Source: TÜİK, IMF)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_shock_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved")

# ============================================================
# FIG 3: Recovery speed — years to return to pre-shock level
# ============================================================

def years_to_recover(country, shock_year, pre_window=2, max_years=8):
    """How many years after shock_year to exceed pre-shock mean."""
    d = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    if np.isnan(pre) or pre == 0:
        return np.nan
    post = d[d['year'] > shock_year]
    for _, row in post.iterrows():
        if row['visitors'] >= pre:
            return row['year'] - shock_year
    return np.nan  # not recovered within data

recovery_shocks = {
    'COVID (pre=2018-19)':       2020,
    '2016 Coup (pre=2014-15)':  2016,
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (shock_label, shock_yr) in zip(axes, recovery_shocks.items()):
    rec = {}
    for c in all_countries:
        yrs = years_to_recover(c, shock_yr)
        rec[c] = yrs

    rec_s = pd.Series(rec).dropna().sort_values()
    colors = [GROUP_COLORS[df[df['country']==c]['market_group'].values[0]] for c in rec_s.index]
    bars = ax.barh(rec_s.index, rec_s.values, color=colors, alpha=0.85, edgecolor='white', lw=1.2)
    ax.axvline(1, color='gray', lw=1, ls='--', alpha=0.7, label='1 year')
    ax.axvline(2, color='orange', lw=1, ls='--', alpha=0.7, label='2 years')
    ax.axvline(3, color='red', lw=1, ls='--', alpha=0.7, label='3 years')
    ax.set_title(f'Recovery Speed: {shock_label}', fontweight='bold')
    ax.set_xlabel('Years to Return to Pre-Shock Level')
    ax.legend(fontsize=9)
    for bar, val in zip(bars, rec_s.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}y', va='center', fontsize=9, fontweight='bold')

    not_recovered = [c for c in all_countries if pd.isna(rec.get(c)) and
                     not df[(df['country']==c) & (df['year']==shock_yr)]['visitors'].empty]
    if not_recovered:
        ax.text(0.98, 0.02, f'Not recovered by 2025:\n{", ".join(not_recovered)}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                color='darkred', bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.7))

legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items() if g != 'Other']
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           title='Market Group', bbox_to_anchor=(0.5, -0.06))
fig.suptitle('Recovery Speed by Market After Major Shocks\n(Source: TÜİK)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig3_recovery_speed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")

# ============================================================
# FIG 4: Turkey lira weakness vs total visitors — does cheap Turkey attract?
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

total_by_year = df.groupby('year')['visitors'].sum().reset_index()
turkey_macro  = df[['year','tur_ppp_rate','tur_inflation','tur_political_stability']].drop_duplicates()
merged = total_by_year.merge(turkey_macro, on='year')

# PPP rate vs total visitors
ax = axes[0]
sc = ax.scatter(merged['tur_ppp_rate'], merged['visitors']/1e6,
                c=merged['year'], cmap='plasma', s=80, zorder=3)
for _, row in merged.iterrows():
    ax.annotate(str(int(row['year'])),
                (row['tur_ppp_rate'], row['visitors']/1e6),
                fontsize=7, xytext=(4, 2), textcoords='offset points', alpha=0.8)
plt.colorbar(sc, ax=ax, label='Year')
x, y = merged['tur_ppp_rate'], merged['visitors']/1e6
m, b = np.polyfit(x, y, 1)
ax.plot(np.sort(x), m*np.sort(x)+b, 'r--', lw=2,
        label=f'OLS slope={m:.1f}M per unit PPP')
r, p = stats.pearsonr(x, y)
ax.set_title(f'Turkey PPP Rate vs Total Visitors\n(Pearson r={r:.2f}, p={p:.3f})',
             fontweight='bold')
ax.set_xlabel('Turkey PPP Exchange Rate (higher = weaker lira)')
ax.set_ylabel('Total Visitors (millions)')
ax.legend()

# Turkey political stability vs visitors
ax = axes[1]
sc2 = ax.scatter(merged['tur_political_stability'], merged['visitors']/1e6,
                 c=merged['year'], cmap='plasma', s=80, zorder=3)
for _, row in merged.iterrows():
    ax.annotate(str(int(row['year'])),
                (row['tur_political_stability'], row['visitors']/1e6),
                fontsize=7, xytext=(4, 2), textcoords='offset points', alpha=0.8)
plt.colorbar(sc2, ax=ax, label='Year')
x2, y2 = merged['tur_political_stability'], merged['visitors']/1e6
m2, b2 = np.polyfit(x2.dropna(), y2[x2.notna()], 1)
ax.plot(np.sort(x2.dropna()), m2*np.sort(x2.dropna())+b2, 'r--', lw=2)
r2, p2 = stats.pearsonr(x2.dropna(), y2[x2.notna()])
ax.set_title(f'Turkey Political Stability vs Total Visitors\n(Pearson r={r2:.2f}, p={p2:.3f})',
             fontweight='bold')
ax.set_xlabel('Turkey WGI Political Stability Score')
ax.set_ylabel('Total Visitors (millions)')

fig.suptitle("Turkey's Own Conditions as Pull/Push Factors\n(Source: IMF, World Bank WGI, TÜİK)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig4_turkey_conditions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")

# ============================================================
# FIG 5: Market resilience heatmap — % vs 2017-19 baseline each year
# ============================================================

pivot = df.pivot_table(index='country', columns='year', values='visitors_vs_baseline')
pivot = pivot.loc[:, 2010:]  # focus on 2010-2025

# Sort by market group then country
group_order = df[['country','market_group']].drop_duplicates().sort_values('market_group')
ordered_countries = group_order['country'].tolist()
pivot = pivot.reindex(ordered_countries)

fig, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(pivot, cmap='RdYlGn', center=0, vmin=-100, vmax=150,
            annot=True, fmt='.0f', linewidths=0.4, linecolor='white',
            annot_kws={'size': 8}, ax=ax,
            cbar_kws={'label': '% vs 2017-19 Baseline', 'shrink': 0.7})

# Vertical lines for shock years
shock_yrs = [2016, 2020, 2022]
years_list = list(pivot.columns)
for sy in shock_yrs:
    if sy in years_list:
        ax.axvline(years_list.index(sy), color='black', lw=2.5, zorder=5)

# Horizontal lines between groups
group_boundaries = []
current_group = None
for i, c in enumerate(ordered_countries):
    g = df[df['country']==c]['market_group'].values[0]
    if g != current_group:
        if current_group is not None:
            group_boundaries.append(i)
        current_group = g
for b in group_boundaries:
    ax.axhline(b, color='black', lw=2, zorder=5)

ax.set_title('Market Resilience Heatmap: % vs 2017-19 Baseline\n'
             '[Bold vertical lines = 2016 Coup, 2020 COVID, 2022 Russia-Ukraine War]',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig5_resilience_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved")

# ============================================================
# HYPOTHESIS TESTS
# ============================================================

SEP = "=" * 65

def print_h(num, title):
    print(f"\n{SEP}\nH{num}: {title}\n{SEP}")

# -----------------------------------------------------------
# H1: Market sensitivity differs significantly by market group
# (Western EU vs MENA vs Former Soviet during 2016 coup)
# -----------------------------------------------------------

print_h(1, "Market groups respond differently to Turkey's domestic shocks (2016 coup)")

coup_impact = {}
for c in all_countries:
    imp = shock_impact(c, 2016, pre_window=2)
    if not np.isnan(imp):
        coup_impact[c] = imp

coup_df = pd.DataFrame.from_dict(coup_impact, orient='index', columns=['impact'])
coup_df['group'] = coup_df.index.map(
    lambda c: df[df['country']==c]['market_group'].values[0])

groups_data = {g: coup_df[coup_df['group']==g]['impact'].values
               for g in ['Western Europe','Former Soviet','MENA','Eastern Europe']}

print("\nMean % impact by group (2016 coup):")
for g, vals in groups_data.items():
    if len(vals) > 0:
        print(f"  {g:20s}: {vals.mean():+.1f}%  (n={len(vals)})")

# One-way ANOVA across groups
valid_groups = [v for v in groups_data.values() if len(v) >= 2]
f_stat, p_anova = stats.f_oneway(*valid_groups)
print(f"\nOne-way ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")
print(f"{'✓ REJECT H0' if p_anova < 0.05 else '✗ FAIL TO REJECT H0'} at α=0.05")
print("Interpretation: Market groups" +
      (" DO differ significantly" if p_anova < 0.05 else " do NOT differ significantly") +
      " in their sensitivity to Turkey's 2016 coup.")

# -----------------------------------------------------------
# H2: MENA tensions (2011-2015) reduced MENA-origin visitors
#     more than other market groups
# -----------------------------------------------------------

print_h(2, "MENA-origin markets more sensitive to regional conflict (Syria 2011-2015)")

pre_syria  = df[df['year'].between(2008,2010)].groupby(['country','market_group'])['visitors'].mean()
dur_syria  = df[df['year'].between(2011,2015)].groupby(['country','market_group'])['visitors'].mean()
syria_chg  = ((dur_syria - pre_syria) / pre_syria * 100).reset_index()
syria_chg.columns = ['country','market_group','pct_change']

mena_chg   = syria_chg[syria_chg['market_group']=='MENA']['pct_change'].values
other_chg  = syria_chg[syria_chg['market_group']!='MENA']['pct_change'].values

print("\n% change in visitors 2011-15 vs 2008-10 baseline:")
print(syria_chg.sort_values('market_group').to_string(index=False))

t_stat, p_val = stats.ttest_ind(mena_chg, other_chg, equal_var=False)
print(f"\nMENA mean change:  {mena_chg.mean():+.1f}%")
print(f"Other mean change: {other_chg.mean():+.1f}%")
print(f"Welch t-test: t={t_stat:.3f}, p={p_val:.4f}")
print(f"{'✓ REJECT H0' if p_val < 0.05 else '✗ FAIL TO REJECT H0'} at α=0.05")
print("Interpretation: MENA markets" +
      (" were significantly MORE affected" if p_val < 0.05 else " were NOT significantly more affected") +
      " by Syria conflict than other markets.")

# -----------------------------------------------------------
# H3: Recovery speed after COVID differs by market group
# -----------------------------------------------------------

print_h(3, "Recovery speed after COVID differs significantly by market group")

rec_years = {}
for c in all_countries:
    y = years_to_recover(c, 2020, pre_window=2)
    rec_years[c] = y

rec_df = pd.DataFrame.from_dict(rec_years, orient='index', columns=['years_to_recover'])
rec_df['group'] = rec_df.index.map(
    lambda c: df[df['country']==c]['market_group'].values[0])

print("\nRecovery years after COVID by country:")
print(rec_df.sort_values('group').to_string())

rec_groups = {g: rec_df[rec_df['group']==g]['years_to_recover'].dropna().values
              for g in rec_df['group'].unique()}

print("\nMean recovery years by group:")
for g, vals in rec_groups.items():
    if len(vals) > 0:
        print(f"  {g:20s}: {vals.mean():.1f} years  (n={len(vals)}, recovered={len(vals)})")

not_rec = rec_df[rec_df['years_to_recover'].isna()].index.tolist()
print(f"\nNot recovered by 2025: {not_rec}")

valid_rec = [v for v in rec_groups.values() if len(v) >= 2]
if len(valid_rec) >= 2:
    f2, p2 = stats.f_oneway(*valid_rec)
    print(f"\nOne-way ANOVA: F={f2:.3f}, p={p2:.4f}")
    print(f"{'✓ REJECT H0' if p2 < 0.05 else '✗ FAIL TO REJECT H0'} at α=0.05")

# -----------------------------------------------------------
# H4: Turkey's currency weakness moderates shock impact
#     (weaker lira → smaller visitor drop during shocks)
# -----------------------------------------------------------

print_h(4, "Turkey currency weakness moderates visitor shock impact")

shock_years_list = [2016, 2020]
results = []
for c in all_countries:
    for sy in shock_years_list:
        imp = shock_impact(c, sy, pre_window=2)
        tur_row = df[(df['year']==sy)][['tur_ppp_rate','tur_inflation']].drop_duplicates()
        if not np.isnan(imp) and len(tur_row) > 0:
            results.append({
                'country': c, 'shock_year': sy,
                'impact': imp,
                'tur_ppp_rate': tur_row['tur_ppp_rate'].values[0],
                'tur_inflation': tur_row['tur_inflation'].values[0],
            })

res_df = pd.DataFrame(results)
r_ppp, p_ppp = stats.pearsonr(res_df['tur_ppp_rate'], res_df['impact'])
print(f"\nCorrelation: Turkey PPP rate vs visitor shock impact")
print(f"  Pearson r = {r_ppp:.3f}, p = {p_ppp:.4f}")
print(f"  Direction: {'Higher PPP (weaker lira) → SMALLER drop' if r_ppp > 0 else 'Higher PPP (weaker lira) → LARGER drop'}")
print(f"  {'✓ SIGNIFICANT' if p_ppp < 0.05 else '✗ NOT SIGNIFICANT'} at α=0.05")

# -----------------------------------------------------------
# H5: Source country GDP per capita predicts visitor volume
#     (wealthier source markets → more tourists)
# -----------------------------------------------------------

print_h(5, "Source country GDP per capita positively predicts visitor volume")

df_h5 = df.dropna(subset=['gdp_per_capita','log_visitors'])
r5, p5 = stats.pearsonr(df_h5['gdp_per_capita'], df_h5['log_visitors'])
r5s, p5s = stats.spearmanr(df_h5['gdp_per_capita'], df_h5['log_visitors'])

print(f"\n  n = {len(df_h5)} country-year observations")
print(f"  Pearson r  = {r5:.4f}, p = {p5:.6f}")
print(f"  Spearman r = {r5s:.4f}, p = {p5s:.6f}")
print(f"  {'✓ REJECT H0' if p5s < 0.05 else '✗ FAIL TO REJECT H0'} at α=0.05 (Spearman)")
print(f"  Interpretation: {'Wealthier source countries send significantly MORE tourists to Turkey.' if r5s > 0 and p5s < 0.05 else 'No significant relationship.'}")

# ============================================================
# SUMMARY TABLE
# ============================================================

print(f"\n{SEP}")
print("HYPOTHESIS TEST SUMMARY")
print(SEP)
print("""
  H1: Market sensitivity to domestic shock (coup)
      → One-way ANOVA across market groups
      → Tests whether sensitivity differs by market geography

  H2: MENA markets more sensitive to regional conflict (Syria)
      → Welch t-test: MENA vs non-MENA during 2011-15
      → Key for MENA tension forecasting

  H3: Recovery speed differs by market group (post-COVID)
      → One-way ANOVA on years-to-recover
      → Actionable for tourism firm targeting

  H4: Turkey currency weakness moderates shock impact
      → Pearson correlation: PPP rate vs % visitor drop
      → Tests if cheap Turkey cushions political shocks

  H5: Source country GDP per capita → visitor volume
      → Pearson + Spearman correlation
      → Controls for wealth in demand modelling
""")
