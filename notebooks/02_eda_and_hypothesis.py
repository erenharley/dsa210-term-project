"""
DSA 210 - Tourism Demand Under Shock
Phase 3 (Redesign): EDA + Hypothesis Tests H1-H6
Author: Eren Sean Harley | 36054

Lecture-aligned hypothesis tests (Weeks 4a, 4b, 5a):
  H1, H3  — One-way ANOVA  +  Kruskal-Wallis robustness check
             Tukey HSD deliberately excluded — not in lecture scope.
  H2, H6  — Levene's test for equal variances
             -> independent t-test (equal_var=True) if Levene p > 0.05
             -> Mann-Whitney U fallback if Levene p <= 0.05
             Welch t-test deliberately excluded — not in lecture scope.
  H4      — Pearson correlation on tur_currency_weakness (YoY %, NOT PPP level)
  H5      — Pearson + Spearman aggregate, plus per-group Spearman breakdown

Figure design rules (per CLAUDE.md):
  - Finding-driven titles: a sentence stating what the figure proves.
  - Test statistic + p-value annotated directly on each figure.
  - Omnibus language only — no directional pairwise claims without pairwise tests.
  - Honest titles when tests are non-significant.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent / 'data'
FIGURES_DIR = BASE_DIR.parent / 'images'
FIGURES_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_DIR / 'panel_dataset.csv')
all_countries = sorted(df['country'].unique())

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.color':       '#e0e0e0',
    'grid.linewidth':   0.8,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   10,
    'legend.fontsize':  9,
})

GROUP_COLORS = {
    'Western Europe': '#2196F3',
    'Eastern Europe': '#4CAF50',
    'Former Soviet':  '#FF5722',
    'MENA':           '#9C27B0',
    'Other':          '#607D8B',
}

# ── Helpers ────────────────────────────────────────────────────────────────

def shock_impact(country, shock_year, pre_window=2):
    """% change in visitors in shock_year vs the pre_window years before it."""
    d   = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    shock_val = d[d['year'] == shock_year]['visitors'].values
    if pre == 0 or len(shock_val) == 0 or np.isnan(pre):
        return np.nan
    return (shock_val[0] / pre - 1) * 100


def years_to_recover(country, shock_year, pre_window=2):
    """Years after shock_year until visitors first exceed the pre-shock mean."""
    d   = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    if np.isnan(pre) or pre == 0:
        return np.nan
    for _, row in d[d['year'] > shock_year].iterrows():
        if row['visitors'] >= pre:
            return row['year'] - shock_year
    return np.nan


SEP = "=" * 65

def print_h(num, title):
    print(f"\n{SEP}\nH{num}: {title}\n{SEP}")


def levene_then_test(group_a, group_b, label_a='Group A', label_b='Group B'):
    """
    Lecture-taught pattern for two-group comparison:
      1. Levene's test for equal variances.
      2. If Levene p > 0.05  -> independent t-test (equal_var=True).
         If Levene p <= 0.05 -> Mann-Whitney U (non-parametric fallback).
    Returns dict with all stats for printing and figure annotation.
    """
    lev_stat, lev_p = stats.levene(group_a, group_b)
    equal_var = lev_p > 0.05
    print(f"  Levene's test: W={lev_stat:.3f}, p={lev_p:.4f}  "
          f"-> {'Equal variances (use t-test)' if equal_var else 'Unequal variances (use Mann-Whitney U)'}")

    if equal_var:
        t_stat, t_p = stats.ttest_ind(group_a, group_b, equal_var=True)
        test_name   = "Independent t-test (equal variances)"
        stat_label  = f"t={t_stat:.3f}"
        main_stat, main_p = t_stat, t_p
    else:
        u_stat, u_p = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        test_name   = "Mann-Whitney U (unequal variances)"
        stat_label  = f"U={u_stat:.1f}"
        main_stat, main_p = u_stat, u_p

    print(f"  {test_name}: {stat_label}, p={main_p:.4f}  "
          f"-> {'SIGNIFICANT' if main_p < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")
    print(f"  {label_a} mean: {group_a.mean():+.1f}%  (n={len(group_a)})")
    print(f"  {label_b} mean: {group_b.mean():+.1f}%  (n={len(group_b)})")

    return {
        'lev_stat':  lev_stat,  'lev_p':    lev_p,
        'equal_var': equal_var,
        'test_name': test_name, 'stat_label': stat_label,
        'main_stat': main_stat, 'main_p':    main_p,
        'significant': main_p < 0.05,
    }


# ============================================================
# H1: Market groups respond differently to Turkey's 2016 coup
# Tests: One-way ANOVA (primary) + Kruskal-Wallis (robustness)
# Figure 1: Strip/dot plot — % change in 2016 per country, grouped
# 'Other' (USA) has n=1 and is excluded from both tests.
# ============================================================

print_h(1, "Market groups respond differently to Turkey's 2016 coup")

coup_impact = {c: shock_impact(c, 2016, pre_window=2) for c in all_countries}
coup_df = pd.DataFrame.from_dict(coup_impact, orient='index', columns=['impact'])
coup_df.dropna(inplace=True)
coup_df['group'] = coup_df.index.map(
    lambda c: df[df['country'] == c]['market_group'].values[0]
)

group_order_h1 = ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA']
groups_data_h1 = {g: coup_df[coup_df['group'] == g]['impact'].values for g in group_order_h1}

print("\nMean % impact by group (2016 coup):")
for g, vals in groups_data_h1.items():
    if len(vals):
        print(f"  {g:20s}: {vals.mean():+.1f}%  (n={len(vals)})")
print("  Other (USA):          excluded — n=1, cannot be in ANOVA/Kruskal-Wallis")

valid_h1 = [v for v in groups_data_h1.values() if len(v) >= 2]
f1_stat, p1_anova = stats.f_oneway(*valid_h1)
h1_kw_stat, p1_kw = stats.kruskal(*valid_h1)

print(f"\nOne-way ANOVA:      F={f1_stat:.3f},  p={p1_anova:.4f}  "
      f"-> {'SIGNIFICANT' if p1_anova < 0.05 else 'not significant'}")
print(f"Kruskal-Wallis:     H={h1_kw_stat:.3f},  p={p1_kw:.4f}  "
      f"-> {'SIGNIFICANT' if p1_kw < 0.05 else 'not significant'}")

# --- Fig 1 ---
np.random.seed(42)
fig, ax = plt.subplots(figsize=(11, 6))

group_means_h1 = coup_df.groupby('group')['impact'].mean()

for i, group in enumerate(group_order_h1):
    sub = coup_df[coup_df['group'] == group]
    jitter = np.random.uniform(-0.18, 0.18, size=len(sub))
    ax.scatter(
        sub['impact'], [i] * len(sub) + jitter,
        color=GROUP_COLORS[group], s=130, zorder=3,
        alpha=0.85, edgecolors='white', lw=0.8
    )
    for (country, row), jit in zip(sub.iterrows(), jitter):
        ax.annotate(country, (row['impact'], i + jit),
                    fontsize=7.5, xytext=(5, 0), textcoords='offset points',
                    va='center', alpha=0.85)
    mn = group_means_h1.get(group, np.nan)
    if not np.isnan(mn):
        ax.hlines(i, mn - 3, mn + 3, color=GROUP_COLORS[group], lw=4, zorder=4)
        ax.annotate(f'mean: {mn:+.0f}%', (mn, i + 0.30),
                    ha='center', fontsize=8.5,
                    color=GROUP_COLORS[group], fontweight='bold')

ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.5)
ax.set_yticks(range(len(group_order_h1)))
ax.set_yticklabels(group_order_h1)
ax.set_xlabel('% Change in Visitors vs 2014-15 Baseline')

annot1 = (
    f"ANOVA: F={f1_stat:.2f}, p={p1_anova:.3f}  "
    f"({'significant' if p1_anova < 0.05 else 'not significant'})\n"
    f"Kruskal-Wallis: H={h1_kw_stat:.2f}, p={p1_kw:.3f}  "
    f"({'significant' if p1_kw < 0.05 else 'not significant'})\n"
    f"'Other' (USA, n=1) excluded from both tests"
)
ax.text(0.02, 0.97, annot1, transform=ax.transAxes,
        fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# Finding-driven title — omnibus only; no pairwise/directional group claims
if p1_anova < 0.05 and p1_kw < 0.05:
    fig1_title = (
        f"Market Groups Differ Significantly in 2016 Coup Sensitivity "
        f"(ANOVA p={p1_anova:.3f}, Kruskal-Wallis p={p1_kw:.3f})"
    )
elif p1_anova < 0.05 or p1_kw < 0.05:
    fig1_title = (
        f"2016 Coup Sensitivity: Mixed Evidence Across Tests "
        f"(ANOVA p={p1_anova:.3f}, Kruskal-Wallis p={p1_kw:.3f}) — Interpret With Caution"
    )
else:
    fig1_title = (
        f"No Significant Difference in 2016 Coup Sensitivity Across Market Groups "
        f"(ANOVA p={p1_anova:.3f}, Kruskal-Wallis p={p1_kw:.3f})"
    )

ax.set_title(fig1_title, fontweight='bold', pad=12, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1_h1_coup_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved")


# ============================================================
# H2 (Reformulated): Proximity to Syria conflict predicts visitor
#     change during the war period (2011-15 vs 2008-10).
# "Syria-bordering" markets: Iraq, Iran, Israel (geographic
#     neighbors / close regional neighbors of Syria).
#     Syria itself is excluded — it IS the conflict country.
# Tests: Levene's -> independent t-test or Mann-Whitney U
# Figure 2: Slope chart + distribution comparison
# Note: 2008-10 was a Turkey-tourism trough, so most countries
#       show absolute increases over 2011-15. The test asks about
#       *relative* differences between groups, not absolute drops.
# ============================================================

print_h(2, "Proximity to Syria conflict predicts visitor change (2011-15 vs 2008-10)")

SYRIA_BORDERING = ['Iraq', 'Iran', 'Israel']
# Syria itself is excluded: it IS the conflict country; its mechanism is different.
test_countries_h2 = [c for c in all_countries if c != 'Syria']

pre_syria = (
    df[(df['year'].between(2008, 2010)) & (df['country'].isin(test_countries_h2))]
    .groupby(['country', 'market_group'])['visitors'].mean()
)
dur_syria = (
    df[(df['year'].between(2011, 2015)) & (df['country'].isin(test_countries_h2))]
    .groupby(['country', 'market_group'])['visitors'].mean()
)
syria_chg = ((dur_syria - pre_syria) / pre_syria * 100).reset_index()
syria_chg.columns = ['country', 'market_group', 'pct_change']
syria_chg['proximity'] = syria_chg['country'].apply(
    lambda c: 'Syria-bordering' if c in SYRIA_BORDERING else 'Non-bordering'
)
syria_chg = syria_chg.dropna(subset=['pct_change'])

bordering_chg     = syria_chg[syria_chg['proximity'] == 'Syria-bordering']['pct_change'].values
non_bordering_chg = syria_chg[syria_chg['proximity'] == 'Non-bordering']['pct_change'].values

print("\n% change 2011-15 vs 2008-10 baseline (Syria excluded from test):")
print(syria_chg.sort_values('proximity').to_string(index=False))
print(f"\nSyria-bordering mean: {bordering_chg.mean():+.1f}%  (n={len(bordering_chg)})")
print(f"Non-bordering mean:   {non_bordering_chg.mean():+.1f}%  (n={len(non_bordering_chg)})")
print("\nNote: 2008-10 was a Turkey-tourism trough. Most countries show absolute "
      "increases over 2011-15. The test asks about *relative* differences between groups.")

h2_result = levene_then_test(
    bordering_chg, non_bordering_chg,
    label_a='Syria-bordering', label_b='Non-bordering'
)

# Per-country means for slope chart (all countries including Syria shown visually)
pre_means_h2 = df[df['year'].between(2008, 2010)].groupby('country')['visitors'].mean()
dur_means_h2 = df[df['year'].between(2011, 2015)].groupby('country')['visitors'].mean()

# --- Fig 2 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2, 1]})

ax = axes[0]
for country in all_countries:
    if country not in pre_means_h2.index or country not in dur_means_h2.index:
        continue
    pre_v = pre_means_h2[country] / 1e6
    dur_v = dur_means_h2[country] / 1e6
    group = df[df['country'] == country]['market_group'].values[0]
    color = GROUP_COLORS[group]
    is_bordering   = country in SYRIA_BORDERING
    is_syria_itself = (country == 'Syria')

    ax.plot(
        [0, 1], [pre_v, dur_v],
        color=color,
        alpha=0.95 if is_bordering else (0.65 if is_syria_itself else 0.35),
        lw=3.0  if is_bordering else (2.0  if is_syria_itself else 1.2),
        ls='-'  if is_bordering else ('--' if is_syria_itself else ':')
    )
    ax.scatter([0, 1], [pre_v, dur_v], color=color,
               s=70 if is_bordering else 25,
               alpha=0.9 if is_bordering else 0.5, zorder=3)
    label = country + ('*' if is_bordering else ' (excl.)' if is_syria_itself else '')
    ax.annotate(label, (1.03, dur_v),
                fontsize=7.5 if is_bordering else 6.5, va='center', color=color,
                fontweight='bold' if is_bordering else 'normal')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-war\n(2008-10 mean)', 'War period\n(2011-15 mean)'], fontsize=10)
ax.set_ylabel('Mean Annual Visitors (millions)')
ax.set_xlim(-0.12, 1.48)
ax.set_title('* = Syria-bordering (in test)   Syria (conflict origin) shown dashed, excluded', fontsize=9)

legend_patches = [mpatches.Patch(color=c, label=g)
                  for g, c in GROUP_COLORS.items() if g != 'Other']
ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

stat2_txt = (
    f"Levene: W={h2_result['lev_stat']:.2f}, p={h2_result['lev_p']:.3f}  "
    f"-> {'equal var' if h2_result['equal_var'] else 'unequal var'}\n"
    f"{h2_result['stat_label']}, p={h2_result['main_p']:.3f}  "
    f"({'significant' if h2_result['significant'] else 'not significant'})\n"
    f"Syria-bordering: {bordering_chg.mean():+.0f}%   Non-bordering: {non_bordering_chg.mean():+.0f}%\n"
    f"Note: 2008-10 trough -> most markets show absolute gains over 2011-15"
)
ax.text(0.02, 0.03, stat2_txt, transform=ax.transAxes,
        ha='left', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# Right panel: distribution comparison
ax2 = axes[1]
ax2.hist(non_bordering_chg, bins=8, alpha=0.55, color='#607D8B',
         label=f'Non-bordering (mean {non_bordering_chg.mean():+.0f}%)',
         orientation='horizontal', density=True)
ax2.hist(bordering_chg, bins=4, alpha=0.80, color=GROUP_COLORS['MENA'],
         label=f'Syria-bordering (mean {bordering_chg.mean():+.0f}%)',
         orientation='horizontal', density=True)
ax2.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
ax2.set_xlabel('Density')
ax2.set_ylabel('% Change (2011-15 vs 2008-10)')
ax2.legend(fontsize=8)
ax2.set_title('Distribution\nComparison', fontsize=9)

# Finding-driven title
if h2_result['significant']:
    if bordering_chg.mean() < non_bordering_chg.mean():
        fig2_title = (
            f"Syria-Bordering Markets Grew Less Than Non-Bordering During War Period "
            f"({h2_result['stat_label']}, p={h2_result['main_p']:.3f})"
        )
    else:
        fig2_title = (
            f"Syria-Bordering and Non-Bordering Markets Diverged During War Period "
            f"({h2_result['stat_label']}, p={h2_result['main_p']:.3f})"
        )
else:
    fig2_title = (
        f"No Significant Difference: Syria-Bordering vs Non-Bordering During War Period "
        f"({h2_result['stat_label']}, p={h2_result['main_p']:.3f})"
    )

fig.suptitle(fig2_title, fontweight='bold', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_h2_syria_mena.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved")


# ============================================================
# H3: COVID recovery speed differs significantly by market group
# Tests: One-way ANOVA (primary) + Kruskal-Wallis (robustness)
# Figure 3: Small multiples — recovery trajectories 2019-2025
# Countries not recovered by 2025 are excluded from tests but noted.
# ============================================================

print_h(3, "COVID recovery speed differs significantly by market group")

rec_years_h3 = {c: years_to_recover(c, 2020, pre_window=2) for c in all_countries}
rec_df = pd.DataFrame.from_dict(rec_years_h3, orient='index', columns=['years_to_recover'])
rec_df['group'] = rec_df.index.map(
    lambda c: df[df['country'] == c]['market_group'].values[0]
)

print("\nYears to recover post-COVID by country:")
print(rec_df.sort_values('group').to_string())

not_rec_h3 = rec_df[rec_df['years_to_recover'].isna()].index.tolist()
print(f"\nNot recovered by 2025 (excluded from test): {not_rec_h3}")

rec_groups_h3 = {
    g: rec_df[rec_df['group'] == g]['years_to_recover'].dropna().values
    for g in ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA']
}
print("\nMean recovery years by group (recovered countries only):")
for g, vals in rec_groups_h3.items():
    if len(vals):
        print(f"  {g:20s}: {vals.mean():.1f} years  (n={len(vals)})")
    else:
        print(f"  {g:20s}: no countries recovered by 2025")

valid_rec_h3 = [v for v in rec_groups_h3.values() if len(v) >= 2]
if len(valid_rec_h3) >= 2:
    f3_stat, p3_anova = stats.f_oneway(*valid_rec_h3)
    h3_kw_stat, p3_kw = stats.kruskal(*valid_rec_h3)
    print(f"\nOne-way ANOVA:   F={f3_stat:.3f},  p={p3_anova:.4f}  "
          f"-> {'SIGNIFICANT' if p3_anova < 0.05 else 'not significant'}")
    print(f"Kruskal-Wallis:  H={h3_kw_stat:.3f},  p={p3_kw:.4f}  "
          f"-> {'SIGNIFICANT' if p3_kw < 0.05 else 'not significant'}")
    print("Note: effective n per group is small — tests may have low power")
else:
    f3_stat, p3_anova = np.nan, np.nan
    h3_kw_stat, p3_kw = np.nan, np.nan
    print("Insufficient recovered countries for ANOVA/Kruskal-Wallis")

# Per-group mean trajectory (% of 2018-19 baseline) for Figure 3
baseline_covid = (
    df[df['year'].between(2018, 2019)]
    .groupby('country')['visitors'].mean()
    .rename('baseline_covid')
)
df_fig3 = df[df['year'].between(2019, 2025)].copy()
df_fig3 = df_fig3.merge(baseline_covid.reset_index(), on='country', how='left')
df_fig3['pct_of_baseline'] = df_fig3['visitors'] / df_fig3['baseline_covid'] * 100

# --- Fig 3 ---
group_order_fig3 = ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA']
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

for ax, group in zip(axes.flatten(), group_order_fig3):
    countries_g = df[df['market_group'] == group]['country'].unique()
    ax.axhline(100, color='black', lw=1.5, ls='--', alpha=0.65, label='Pre-COVID baseline')

    for country in countries_g:
        sub = df_fig3[df_fig3['country'] == country].sort_values('year')
        ax.plot(sub['year'], sub['pct_of_baseline'],
                color=GROUP_COLORS[group], alpha=0.45, lw=1.5, marker='o', ms=3.5)
        cross = sub[sub['pct_of_baseline'] >= 100]
        if not cross.empty:
            fc = cross.iloc[0]
            ax.scatter(fc['year'], fc['pct_of_baseline'],
                       color=GROUP_COLORS[group], s=140, zorder=5,
                       marker='*', edgecolors='black', lw=0.5)

    group_mean_traj = (
        df_fig3[df_fig3['market_group'] == group]
        .groupby('year')['pct_of_baseline'].mean()
    )
    ax.plot(group_mean_traj.index, group_mean_traj.values,
            color=GROUP_COLORS[group], lw=3.0, alpha=1.0, zorder=4,
            label=f'{group} mean')

    ax.set_xlim(2018.8, 2025.2)
    ax.set_xticks(range(2019, 2026))
    ax.set_title(group, fontweight='bold', color=GROUP_COLORS[group])
    ax.set_xlabel('Year')
    ax.set_ylabel('Visitors as % of 2018-19 Baseline')
    ax.legend(fontsize=8)

stat3_parts = []
if not np.isnan(f3_stat):
    stat3_parts.append(
        f"ANOVA: F={f3_stat:.2f}, p={p3_anova:.3f} "
        f"({'significant' if p3_anova < 0.05 else 'not significant'})"
    )
    stat3_parts.append(
        f"Kruskal-Wallis: H={h3_kw_stat:.2f}, p={p3_kw:.3f} "
        f"({'significant' if p3_kw < 0.05 else 'not significant'})"
    )
if not_rec_h3:
    stat3_parts.append(f"Not recovered by 2025 (excluded): {', '.join(not_rec_h3)}")
if stat3_parts:
    fig.text(0.5, -0.01, '  |  '.join(stat3_parts), ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# Finding-driven title (omnibus)
if not np.isnan(p3_anova) and p3_anova < 0.05 and p3_kw < 0.05:
    fig3_title = (
        f"COVID Recovery Speed Differs Significantly Across Market Groups "
        f"(ANOVA p={p3_anova:.3f}, Kruskal-Wallis p={p3_kw:.3f})  *=Year of Full Recovery"
    )
elif not np.isnan(p3_anova) and (p3_anova < 0.05 or p3_kw < 0.05):
    fig3_title = (
        f"COVID Recovery Speed: Tests Disagree "
        f"(ANOVA p={p3_anova:.3f}, Kruskal-Wallis p={p3_kw:.3f}) — Interpret With Caution  *=Full Recovery"
    )
elif not np.isnan(p3_anova):
    fig3_title = (
        f"No Significant Difference in COVID Recovery Speed Across Groups "
        f"(ANOVA p={p3_anova:.3f}, Kruskal-Wallis p={p3_kw:.3f})  *=Year of Full Recovery"
    )
else:
    fig3_title = "COVID Recovery Trajectories by Market Group  *=Year of Full Recovery"

fig.suptitle(fig3_title, fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig3_h3_covid_recovery.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")


# ============================================================
# H4: Turkey lira weakness (YoY %) correlates with larger visitor
#     drops during shock years.
# Variable: tur_currency_weakness — YoY % change in PPP rate.
#     NOT tur_ppp_rate level, which rises monotonically and would
#     conflate "weaker lira" with "later in time."
# Test: Pearson correlation (country × shock-year observations)
# Figure 4: Scatter with OLS line, colored by market group
# ============================================================

print_h(4, "Turkey currency weakness (YoY %) correlates with visitor drops during shocks")

shock_years_h4 = [2009, 2016, 2020]
results_h4 = []
for c in all_countries:
    for sy in shock_years_h4:
        imp = shock_impact(c, sy, pre_window=2)
        tur_row = (
            df[df['year'] == sy][['tur_ppp_rate', 'tur_currency_weakness']]
            .drop_duplicates()
        )
        if not np.isnan(imp) and len(tur_row) > 0:
            cw = tur_row['tur_currency_weakness'].values[0]
            if not np.isnan(cw):
                group = df[df['country'] == c]['market_group'].values[0]
                results_h4.append({
                    'country':               c,
                    'shock_year':            sy,
                    'impact':                imp,
                    'tur_currency_weakness': cw,
                    'group':                 group,
                })

res_h4 = pd.DataFrame(results_h4)
print(f"\n  n = {len(res_h4)} country x shock-year observations")
print("  Turkey YoY currency weakness by shock year:")
print(res_h4.groupby('shock_year')['tur_currency_weakness'].first()
      .rename('YoY % change in PPP rate').to_string())

r4, p4 = stats.pearsonr(res_h4['tur_currency_weakness'], res_h4['impact'])
print(f"\nPearson r (tur_currency_weakness vs visitor impact) = {r4:.3f},  p = {p4:.4f}")
print(f"Direction: {'Higher YoY weakness -> LARGER visitor drop (negative r)' if r4 < 0 else 'Higher YoY weakness -> smaller drop (positive r)'}")
print(f"{'SIGNIFICANT' if p4 < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

# --- Fig 4 ---
fig, ax = plt.subplots(figsize=(11, 7))

for group in GROUP_COLORS:
    sub = res_h4[res_h4['group'] == group]
    if sub.empty:
        continue
    ax.scatter(sub['tur_currency_weakness'], sub['impact'],
               color=GROUP_COLORS[group], s=90, alpha=0.75, zorder=3,
               edgecolors='white', lw=0.6, label=group)

for _, row in res_h4.iterrows():
    ax.annotate(
        f"{row['country'][:3]}\n'{str(row['shock_year'])[2:]}",
        (row['tur_currency_weakness'], row['impact']),
        fontsize=6.5, xytext=(4, 3), textcoords='offset points', alpha=0.70
    )

x4 = res_h4['tur_currency_weakness']
y4 = res_h4['impact']
m4, b4 = np.polyfit(x4, y4, 1)
x4_range = np.linspace(x4.min(), x4.max(), 100)
ax.plot(x4_range, m4 * x4_range + b4, 'r--', lw=2,
        label=f'OLS (slope={m4:.2f}% per 1% currency weakness)')

ax.axhline(0, color='black', lw=1, alpha=0.4)
ax.set_xlabel('Turkey YoY Currency Weakness  (% change in PPP rate; higher = weaker lira)')
ax.set_ylabel('% Change in Visitors vs Pre-Shock Baseline')
ax.legend(fontsize=8, loc='upper right')

stat4_txt = (
    f"Pearson r = {r4:.3f},  p = {p4:.3f}\n"
    f"n = {len(res_h4)} obs. (17 countries x 3 shock years)\n"
    f"Caveat: only 3 shock years sampled;\n"
    f"cumulative vs instantaneous lira weakness not separated."
)
ax.text(0.02, 0.04, stat4_txt, transform=ax.transAxes,
        fontsize=8, va='bottom', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

if r4 < 0 and p4 < 0.05:
    fig4_title = (
        f"Weaker Lira (YoY) Accompanies Larger Visitor Drops During Shocks "
        f"(Pearson r={r4:.2f}, p={p4:.3f})"
    )
elif r4 < 0:
    fig4_title = (
        f"Negative Lira-Visitor Relationship — Not Statistically Significant "
        f"(Pearson r={r4:.2f}, p={p4:.3f})"
    )
else:
    fig4_title = (
        f"No Negative Lira-Visitor Relationship Detected in Shock Years "
        f"(Pearson r={r4:.2f}, p={p4:.3f})"
    )

ax.set_title(fig4_title, fontweight='bold', pad=10, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig4_h4_lira_weakness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")


# ============================================================
# H5: Source country GDP per capita positively predicts visitor volume
# Tests: Pearson + Spearman (aggregate) + per-group Spearman breakdown
# Figure 5: Scatter — log(GDP per capita) vs log(visitors)
#           with per-group Spearman annotation box
# Simpson's paradox note: within-group relationships may differ in
# sign from the aggregate (e.g. wealthy MENA countries send fewer
# tourists than their GDP level would predict).
# ============================================================

print_h(5, "Source country GDP per capita predicts visitor volume")

df_h5 = df.dropna(subset=['gdp_per_capita', 'log_visitors'])
r5,  p5  = stats.pearsonr(df_h5['gdp_per_capita'], df_h5['log_visitors'])
r5s, p5s = stats.spearmanr(df_h5['gdp_per_capita'], df_h5['log_visitors'])

print(f"\n  AGGREGATE (all {len(df_h5)} country-year obs.):")
print(f"  Pearson r  = {r5:.4f},  p = {p5:.6f}")
print(f"  Spearman r = {r5s:.4f},  p = {p5s:.6f}")
print(f"  {'SIGNIFICANT' if p5s < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05 (Spearman)")

print("\n  PER-GROUP Spearman r (* = significant at alpha=0.05):")
group_spearman_h5 = {}
for g in ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA', 'Other']:
    sub = df_h5[df_h5['market_group'] == g]
    if len(sub) >= 5:
        rg, pg = stats.spearmanr(sub['gdp_per_capita'], sub['log_visitors'])
        group_spearman_h5[g] = (rg, pg, len(sub))
        sig_str = 'SIGNIFICANT' if pg < 0.05 else 'not significant'
        print(f"    {g:20s}: r={rg:+.3f}, p={pg:.4f}  ({sig_str}, n={len(sub)})")
    else:
        group_spearman_h5[g] = (np.nan, np.nan, len(sub))
        print(f"    {g:20s}: n={len(sub)} — too small for reliable correlation")

# --- Fig 5 ---
fig, ax = plt.subplots(figsize=(11, 7))

for group in GROUP_COLORS:
    sub = df_h5[df_h5['market_group'] == group]
    ax.scatter(np.log(sub['gdp_per_capita']), sub['log_visitors'],
               color=GROUP_COLORS[group], alpha=0.40, s=40,
               edgecolors='none', label=group)

x5 = np.log(df_h5['gdp_per_capita'])
y5 = df_h5['log_visitors']
m5, b5 = np.polyfit(x5, y5, 1)
x5_range = np.linspace(x5.min(), x5.max(), 100)
ax.plot(x5_range, m5 * x5_range + b5, 'r--', lw=2, label='OLS trend (aggregate)')

ax.set_xlabel('log(Source Country GDP per Capita, USD)')
ax.set_ylabel('log(Annual Visitors to Turkey)')

per_group_lines = []
for g in ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA', 'Other']:
    rg, pg, ng = group_spearman_h5[g]
    if not np.isnan(rg):
        sig_mark = '*' if pg < 0.05 else ''
        per_group_lines.append(f"  {g[:12]:12s}: r={rg:+.2f}{sig_mark}")
    else:
        per_group_lines.append(f"  {g[:12]:12s}: n={ng} (small)")

stat5_txt = (
    f"Aggregate  Pearson r  = {r5:.3f},  p = {p5:.4f}\n"
    f"           Spearman r = {r5s:.3f},  p = {p5s:.4f}\n"
    f"           n = {len(df_h5)} country-year obs.\n"
    f"Per-group Spearman (* = p<0.05):\n"
    + "\n".join(per_group_lines)
)
ax.text(0.02, 0.97, stat5_txt, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

ax.legend(fontsize=8, loc='lower right')

if p5s < 0.05 and r5s > 0:
    strength5 = "Weakly positive" if abs(r5s) < 0.3 else "Moderately positive"
    fig5_title = (
        f"GDP per Capita {strength5} Relates to Visitor Volume in Aggregate "
        f"(Spearman r={r5s:.2f}, p={p5s:.3f}); Within-Group Patterns Diverge"
    )
elif p5s < 0.05:
    fig5_title = (
        f"GDP per Capita Significantly Relates to Visitor Volume "
        f"(Spearman r={r5s:.2f}, p={p5s:.3f}); Within-Group Direction Varies"
    )
else:
    fig5_title = (
        f"GDP per Capita Does Not Significantly Predict Visitor Volume in Aggregate "
        f"(Spearman r={r5s:.2f}, p={p5s:.3f}); Within-Group Patterns Diverge"
    )

ax.set_title(fig5_title, fontweight='bold', pad=10, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig5_h5_gdp_visitors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved")


# ============================================================
# H6: Post-2023 MENA tensions reduced MENA-origin tourism more
#     than non-MENA markets (2023-25 mean vs 2017-19 baseline)
# Tests: Levene's -> independent t-test or Mann-Whitney U
# Figure 6: Two-panel
#   Left  — Horizontal bar chart: all countries % vs 2017-19 baseline
#   Right — Within-MENA descriptive split (conflict-affected vs stable)
#           DESCRIPTIVE ONLY: n=2 in stable subgroup is too small to test
# ============================================================

print_h(6, "Post-2023 MENA tensions suppressed MENA-origin tourism more than non-MENA")

post_2023 = (
    df[df['year'].between(2023, 2025)]
    .groupby(['country', 'market_group'])['visitors'].mean()
)
baseline_17_19 = (
    df[df['year'].between(2017, 2019)]
    .groupby('country')['visitors'].mean()
)
h6_df = ((post_2023 - baseline_17_19) / baseline_17_19 * 100).reset_index()
h6_df.columns = ['country', 'market_group', 'pct_vs_baseline']
h6_df = h6_df.dropna(subset=['pct_vs_baseline'])

mena_h6    = h6_df[h6_df['market_group'] == 'MENA']['pct_vs_baseline'].values
nonmena_h6 = h6_df[h6_df['market_group'] != 'MENA']['pct_vs_baseline'].values

print("\n% vs 2017-19 baseline, 2023-25 mean (all countries):")
print(h6_df.sort_values('market_group').to_string(index=False))

print(f"\nFormal test: MENA (n={len(mena_h6)}) vs Non-MENA (n={len(nonmena_h6)})")
h6_result = levene_then_test(mena_h6, nonmena_h6, label_a='MENA', label_b='Non-MENA')

# Within-MENA descriptive split (NOT a formal test)
MENA_CONFLICT = ['Iran', 'Iraq', 'Israel', 'Syria']
MENA_STABLE   = ['United Arab Emirates', 'Qatar']

mena_conflict_vals = h6_df[h6_df['country'].isin(MENA_CONFLICT)]['pct_vs_baseline']
mena_stable_vals   = h6_df[h6_df['country'].isin(MENA_STABLE)]['pct_vs_baseline']

print("\nWithin-MENA descriptive split (NOT a formal test — n=2 in stable subgroup):")
print("  MENA-conflict-affected (Iran, Iraq, Israel, Syria):")
for _, row in h6_df[h6_df['country'].isin(MENA_CONFLICT)].iterrows():
    print(f"    {row['country']:25s}: {row['pct_vs_baseline']:+.1f}%")
print(f"  Subgroup mean: {mena_conflict_vals.mean():+.1f}%  (n={len(mena_conflict_vals)})")
print("  MENA-stable (UAE, Qatar):")
for _, row in h6_df[h6_df['country'].isin(MENA_STABLE)].iterrows():
    print(f"    {row['country']:25s}: {row['pct_vs_baseline']:+.1f}%")
print(f"  Subgroup mean: {mena_stable_vals.mean():+.1f}%  (n={len(mena_stable_vals)})")
print(
    "\n  Interpretation: if both subgroups declined similarly, regional spillover"
    "\n  is the more plausible driver. If only conflict-affected declined, the H6"
    "\n  result likely reflects home-country damage rather than regional sentiment."
)

# --- Fig 6 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})

# Left panel: full horizontal bar chart (all 17 countries)
ax = axes[0]
h6_sorted  = h6_df.sort_values('pct_vs_baseline')
bar_colors = [GROUP_COLORS[g] for g in h6_sorted['market_group']]
alphas     = [1.0 if g == 'MENA' else 0.50 for g in h6_sorted['market_group']]

bars = ax.barh(h6_sorted['country'], h6_sorted['pct_vs_baseline'],
               color=bar_colors, edgecolor='white', lw=0.8)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)

ax.axvline(0, color='black', lw=1.5)
ax.set_xlabel('% Change in Mean Visitors vs 2017-19 Baseline  (2023-25 average)')

for bar, val in zip(bars, h6_sorted['pct_vs_baseline']):
    ax.text(val + (2 if val >= 0 else -2),
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.0f}%', va='center',
            ha='left' if val >= 0 else 'right', fontsize=8.5)

stat6_txt = (
    f"Levene: W={h6_result['lev_stat']:.2f}, p={h6_result['lev_p']:.3f}  "
    f"-> {'equal var' if h6_result['equal_var'] else 'unequal var'}\n"
    f"{h6_result['stat_label']}, p={h6_result['main_p']:.3f}  "
    f"({'significant' if h6_result['significant'] else 'not significant'})\n"
    f"MENA mean: {mena_h6.mean():+.0f}%    Non-MENA mean: {nonmena_h6.mean():+.0f}%"
)
ax.text(0.02, 0.97, stat6_txt, transform=ax.transAxes,
        fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

mena_patch  = mpatches.Patch(color=GROUP_COLORS['MENA'], label='MENA markets')
other_patch = mpatches.Patch(color='#aaaaaa', alpha=0.5,  label='Non-MENA (faded)')
ax.legend(handles=[mena_patch, other_patch], fontsize=9, loc='lower right')

# Right panel: within-MENA descriptive split
ax2 = axes[1]
within_mena = (
    h6_df[h6_df['country'].isin(MENA_CONFLICT + MENA_STABLE)]
    .sort_values('pct_vs_baseline')
)
sub_colors2 = ['#E91E63' if c in MENA_CONFLICT else '#00BCD4'
                for c in within_mena['country']]

bars2 = ax2.barh(within_mena['country'], within_mena['pct_vs_baseline'],
                 color=sub_colors2, edgecolor='white', lw=0.8, alpha=0.85)
ax2.axvline(0, color='black', lw=1.2)
ax2.set_xlabel('% vs 2017-19 Baseline')

for bar, val in zip(bars2, within_mena['pct_vs_baseline']):
    ax2.text(val + (1.5 if val >= 0 else -1.5),
             bar.get_y() + bar.get_height() / 2,
             f'{val:+.0f}%', va='center',
             ha='left' if val >= 0 else 'right', fontsize=9)

conflict_patch = mpatches.Patch(color='#E91E63', label=f'Conflict-affected (mean {mena_conflict_vals.mean():+.0f}%)')
stable_patch   = mpatches.Patch(color='#00BCD4', label=f'Stable: UAE, Qatar (mean {mena_stable_vals.mean():+.0f}%)')
ax2.legend(handles=[conflict_patch, stable_patch], fontsize=8, loc='lower right')
ax2.set_title(
    "Within-MENA — DESCRIPTIVE ONLY\n"
    f"n=2 in stable subgroup (no formal test)",
    fontsize=9
)

# Finding-driven title (omnibus on the formal test)
if h6_result['significant']:
    if mena_h6.mean() < nonmena_h6.mean():
        fig6_title = (
            f"Post-2023 MENA Tensions Associated With Larger Visitor Declines "
            f"in MENA-Origin Markets ({h6_result['stat_label']}, p={h6_result['main_p']:.3f})"
        )
    else:
        fig6_title = (
            f"MENA and Non-MENA Post-2023 Visitor Changes Differ Significantly "
            f"({h6_result['stat_label']}, p={h6_result['main_p']:.3f})"
        )
else:
    fig6_title = (
        f"No Significant Difference in Post-2023 Visitor Changes: MENA vs Non-MENA "
        f"({h6_result['stat_label']}, p={h6_result['main_p']:.3f})"
    )

fig.suptitle(fig6_title, fontweight='bold', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig6_h6_mena_tension_recent.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6 saved")


# ============================================================
# HYPOTHESIS TEST SUMMARY
# Copy-pastable for Stage 3 README update.
# ============================================================

print(f"\n{SEP}")
print("HYPOTHESIS TEST SUMMARY  (alpha = 0.05)")
print(SEP)

# H1
h1_anova_sig = 'SIGNIFICANT' if p1_anova < 0.05 else 'not significant'
h1_kw_sig    = 'SIGNIFICANT' if p1_kw    < 0.05 else 'not significant'
if p1_anova < 0.05 and p1_kw < 0.05:
    h1_interp = "Both tests agree: market groups differ significantly in 2016 coup sensitivity."
elif p1_anova >= 0.05 and p1_kw >= 0.05:
    h1_interp = "No group-level difference detected in 2016 coup sensitivity."
else:
    h1_interp = "Tests disagree — interpret with caution (small group sizes may inflate power for one test)."

print(f"""
H1 — 2016 coup sensitivity differs by market group
     ANOVA:          F={f1_stat:.3f}, p={p1_anova:.4f}  [{h1_anova_sig}]
     Kruskal-Wallis: H={h1_kw_stat:.3f}, p={p1_kw:.4f}  [{h1_kw_sig}]
     'Other' (USA, n=1) excluded from both tests.
     Interpretation: {h1_interp}
""")

# H2
h2_sig    = 'SIGNIFICANT' if h2_result['significant'] else 'not significant'
h2_interp = (
    "Countries geographically close to Syria showed different visitor changes "
    "than non-bordering markets during the war period."
    if h2_result['significant']
    else "No significant difference in visitor change between Syria-bordering and non-bordering markets."
)
print(f"""H2 — Proximity to Syria conflict predicts visitor change (2011-15 vs 2008-10)
     Levene:          W={h2_result['lev_stat']:.3f}, p={h2_result['lev_p']:.4f}  -> {'equal variances' if h2_result['equal_var'] else 'unequal variances'}
     {h2_result['test_name']}: {h2_result['stat_label']}, p={h2_result['main_p']:.4f}  [{h2_sig}]
     Syria-bordering (Iraq, Iran, Israel): mean {bordering_chg.mean():+.1f}%  (n={len(bordering_chg)})
     Non-bordering (all others excl. Syria):  mean {non_bordering_chg.mean():+.1f}%  (n={len(non_bordering_chg)})
     Note: Syria (conflict origin) excluded from test.
           2008-10 trough means most markets show gains over war period — test is about relative differences.
     Interpretation: {h2_interp}
""")

# H3
if not np.isnan(p3_anova):
    h3_anova_sig = 'SIGNIFICANT' if p3_anova < 0.05 else 'not significant'
    h3_kw_sig    = 'SIGNIFICANT' if p3_kw    < 0.05 else 'not significant'
    if p3_anova < 0.05 and p3_kw < 0.05:
        h3_interp = "Both tests agree: COVID recovery speed differs across market groups."
    elif p3_anova >= 0.05 and p3_kw >= 0.05:
        h3_interp = "No significant group-level difference in recovery speed. Note: small effective n reduces power."
    else:
        h3_interp = "Tests disagree — likely driven by very small group sizes after excluding unrecovered countries."
    print(f"""H3 — COVID recovery speed differs by market group
     ANOVA:          F={f3_stat:.3f}, p={p3_anova:.4f}  [{h3_anova_sig}]
     Kruskal-Wallis: H={h3_kw_stat:.3f}, p={p3_kw:.4f}  [{h3_kw_sig}]
     Not recovered by 2025 (excluded from test): {not_rec_h3}
     Interpretation: {h3_interp}
""")
else:
    print("H3 — COVID recovery speed: insufficient data for ANOVA/Kruskal-Wallis\n")

# H4
h4_sig = 'SIGNIFICANT' if p4 < 0.05 else 'not significant'
if r4 < 0 and p4 < 0.05:
    h4_interp = "Weaker lira (YoY) accompanies larger visitor drops during shock years."
elif r4 < 0:
    h4_interp = "Negative lira-visitor relationship present but not statistically significant."
else:
    h4_interp = "No negative lira-visitor relationship detected — positive slope observed."
print(f"""H4 — Turkey lira weakness (YoY %) and visitor drops during shocks
     Variable: tur_currency_weakness (YoY % change in PPP rate; NOT level)
     Pearson r = {r4:.3f},  p = {p4:.4f}  [{h4_sig}]
     n = {len(res_h4)} (17 countries x 3 shock years: 2009, 2016, 2020)
     Caveat: only 3 shock years; cumulative vs instantaneous lira weakness not separated.
     Interpretation: {h4_interp}
""")

# H5
h5_sig = 'SIGNIFICANT' if p5s < 0.05 else 'not significant'
if p5s < 0.05 and r5s > 0:
    h5_interp = "Aggregate positive and significant; within-group relationships diverge in sign (potential Simpson's paradox)."
elif p5s >= 0.05:
    h5_interp = "Aggregate correlation non-significant; see per-group breakdown for within-group patterns."
else:
    h5_interp = "See per-group breakdown for interpretation."
print(f"""H5 — Source GDP per capita and visitor volume
     Pearson r  = {r5:.3f},  p = {p5:.4f}
     Spearman r = {r5s:.3f},  p = {p5s:.4f}  [{h5_sig}]
     n = {len(df_h5)} country-year obs.
     Per-group Spearman:""")
for g in ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA', 'Other']:
    rg, pg, ng = group_spearman_h5[g]
    if not np.isnan(rg):
        sig_mark = '[SIGNIFICANT]' if pg < 0.05 else '[not significant]'
        print(f"       {g:20s}: r={rg:+.3f}, p={pg:.4f}  {sig_mark}")
    else:
        print(f"       {g:20s}: n={ng} — too small for reliable correlation")
print(f"     Interpretation: {h5_interp}\n")

# H6
h6_sig = 'SIGNIFICANT' if h6_result['significant'] else 'not significant'
if h6_result['significant'] and mena_h6.mean() < nonmena_h6.mean():
    h6_interp = (
        "Post-2023 MENA tensions associated with significantly larger visitor declines "
        "in MENA-origin markets vs non-MENA. Within-MENA descriptive split shows whether "
        "both conflict-affected and stable subgroups declined (regional spillover) or only "
        "conflict-affected (home-country damage)."
    )
elif not h6_result['significant']:
    h6_interp = "No significant difference in post-2023 visitor changes between MENA and non-MENA markets."
else:
    h6_interp = "MENA and non-MENA post-2023 changes differ significantly — see figure for direction."
print(f"""H6 — Post-2023 MENA tensions and MENA-origin tourism (vs 2017-19 baseline)
     Levene:          W={h6_result['lev_stat']:.3f}, p={h6_result['lev_p']:.4f}  -> {'equal variances' if h6_result['equal_var'] else 'unequal variances'}
     {h6_result['test_name']}: {h6_result['stat_label']}, p={h6_result['main_p']:.4f}  [{h6_sig}]
     MENA mean: {mena_h6.mean():+.1f}%    Non-MENA mean: {nonmena_h6.mean():+.1f}%
     Within-MENA descriptive (NOT a formal test — n=2 in stable subgroup):
       Conflict-affected (Iran, Iraq, Israel, Syria): mean {mena_conflict_vals.mean():+.1f}%  (n={len(mena_conflict_vals)})
       Stable (UAE, Qatar):                           mean {mena_stable_vals.mean():+.1f}%  (n={len(mena_stable_vals)})
     Interpretation: {h6_interp}
""")
