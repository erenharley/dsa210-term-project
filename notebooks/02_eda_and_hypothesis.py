"""
DSA 210 - Tourism Demand Under Shock
Phase 3 (Redesign): EDA + Hypothesis Tests H1-H6
Author: Eren Sean Harley | 36054

Six finding-driven figures, each paired 1:1 with a hypothesis test.
ANOVA results for H1 and H3 are followed by Tukey HSD post-hoc tests.
H6 is new: tests whether post-2023 MENA tensions suppressed MENA-origin
tourism more than non-MENA markets.

Figure design rules (per CLAUDE.md):
  - Titles state the finding, not a description of what is plotted.
  - Test statistic + p-value annotated directly on each figure.
  - Titles soften if Tukey HSD or t-test does not support the directional claim.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
    """
    Return % change in visitors in shock_year vs the pre_window years before it.
    e.g. shock_impact('Germany', 2016, pre_window=2) computes
         (visitors_2016 / mean(visitors_2014-2015) - 1) * 100
    """
    d   = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    shock_val = d[d['year'] == shock_year]['visitors'].values
    if pre == 0 or len(shock_val) == 0 or np.isnan(pre):
        return np.nan
    return (shock_val[0] / pre - 1) * 100


def years_to_recover(country, shock_year, pre_window=2):
    """
    Return the number of years after shock_year until visitors first exceed
    the pre-shock mean. Returns NaN if not recovered within the dataset.
    """
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


# ============================================================
# H1: Market groups respond differently to Turkey's 2016 coup
# Test: One-way ANOVA + Tukey HSD post-hoc
# Figure 1: Strip/dot plot — % drop in 2016 per country, grouped
# ============================================================

print_h(1, "Market groups respond differently to Turkey's 2016 coup")

coup_impact = {
    c: shock_impact(c, 2016, pre_window=2)
    for c in all_countries
}
coup_df = pd.DataFrame.from_dict(coup_impact, orient='index', columns=['impact'])
coup_df.dropna(inplace=True)
coup_df['group'] = coup_df.index.map(
    lambda c: df[df['country'] == c]['market_group'].values[0]
)

group_order_h1 = ['Western Europe', 'Eastern Europe', 'Former Soviet', 'MENA']
groups_data_h1 = {
    g: coup_df[coup_df['group'] == g]['impact'].values
    for g in group_order_h1
}

print("\nMean % impact by group (2016 coup):")
for g, vals in groups_data_h1.items():
    if len(vals):
        print(f"  {g:20s}: {vals.mean():+.1f}%  (n={len(vals)})")

valid_h1 = [v for v in groups_data_h1.values() if len(v) >= 2]
f1_stat, p1_anova = stats.f_oneway(*valid_h1)
print(f"\nOne-way ANOVA: F={f1_stat:.3f}, p={p1_anova:.4f}")
print(f"{'REJECT H0' if p1_anova < 0.05 else 'FAIL TO REJECT H0'} at alpha=0.05")

tukey1 = pairwise_tukeyhsd(coup_df['impact'], coup_df['group'], alpha=0.05)
print("\nTukey HSD pairwise comparisons (H1):")
print(tukey1.summary())

# Extract significant Tukey pairs for annotation
tukey1_rows = tukey1._results_table.data[1:]
sig_pairs_h1 = [(r[0], r[1]) for r in tukey1_rows if r[6]]  # reject=True

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
        ax.annotate(
            country, (row['impact'], i + jit),
            fontsize=7.5, xytext=(5, 0), textcoords='offset points',
            va='center', alpha=0.85
        )
    mn = group_means_h1.get(group, np.nan)
    if not np.isnan(mn):
        ax.hlines(i, mn - 3, mn + 3, color=GROUP_COLORS[group], lw=4, zorder=4)
        ax.annotate(
            f'mean: {mn:+.0f}%', (mn, i + 0.30),
            ha='center', fontsize=8.5,
            color=GROUP_COLORS[group], fontweight='bold'
        )

ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.5)
ax.set_yticks(range(len(group_order_h1)))
ax.set_yticklabels(group_order_h1)
ax.set_xlabel('% Change in Visitors vs 2014-15 Baseline')

annot1 = f"ANOVA: F={f1_stat:.2f}, p={p1_anova:.3f}"
if sig_pairs_h1:
    pairs_str = '; '.join([f"{a} vs {b}" for a, b in sig_pairs_h1])
    annot1 += f"\nTukey HSD significant pairs: {pairs_str}"
else:
    annot1 += "\nTukey HSD: no significant pairwise differences at alpha=0.05"
ax.text(
    0.02, 0.97, annot1, transform=ax.transAxes,
    fontsize=8, va='top', ha='left',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
)

# Finding-driven title — soften if ANOVA or Tukey does not support directional claim
most_sensitive_h1 = group_means_h1.idxmin()
if p1_anova < 0.05 and sig_pairs_h1:
    fig1_title = (
        f"{most_sensitive_h1} Dropped Hardest in Turkey's 2016 Coup "
        f"— Groups Differ Significantly (F={f1_stat:.2f}, p={p1_anova:.3f})"
    )
elif p1_anova < 0.05:
    fig1_title = (
        f"Groups Differ Significantly in 2016 Coup Sensitivity "
        f"(F={f1_stat:.2f}, p={p1_anova:.3f}) — No Pairwise Pair Survives Tukey HSD"
    )
else:
    fig1_title = (
        f"No Significant Difference in 2016 Coup Sensitivity Across Market Groups "
        f"(ANOVA F={f1_stat:.2f}, p={p1_anova:.3f})"
    )

ax.set_title(fig1_title, fontweight='bold', pad=12, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig1_h1_coup_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved")


# ============================================================
# H2: MENA markets fell harder than non-MENA during Syria war
# Test: Welch t-test (MENA vs non-MENA, mean visitors 2011-15 vs 2008-10)
# Figure 2: Slope chart — pre-war vs war-period mean visitors
# Note: Russia-Turkey 2015 crisis is excluded from this figure;
#       it is a one-country/one-year event, not a market-group phenomenon.
# ============================================================

print_h(2, "MENA-origin markets fell harder than non-MENA during Syria war (2011-15)")

pre_syria = (
    df[df['year'].between(2008, 2010)]
    .groupby(['country', 'market_group'])['visitors'].mean()
)
dur_syria = (
    df[df['year'].between(2011, 2015)]
    .groupby(['country', 'market_group'])['visitors'].mean()
)
syria_chg = ((dur_syria - pre_syria) / pre_syria * 100).reset_index()
syria_chg.columns = ['country', 'market_group', 'pct_change']

mena_chg  = syria_chg[syria_chg['market_group'] == 'MENA']['pct_change'].values
other_chg = syria_chg[syria_chg['market_group'] != 'MENA']['pct_change'].values

print("\n% change 2011-15 vs 2008-10 baseline:")
print(syria_chg.sort_values('market_group').to_string(index=False))

t2_stat, p2_val = stats.ttest_ind(mena_chg, other_chg, equal_var=False)
print(f"\nMENA mean change:  {mena_chg.mean():+.1f}%")
print(f"Other mean change: {other_chg.mean():+.1f}%")
print(f"Welch t-test: t={t2_stat:.3f}, p={p2_val:.4f}")
print(f"{'REJECT H0' if p2_val < 0.05 else 'FAIL TO REJECT H0'} at alpha=0.05")

# Per-country means for slope chart
pre_means_h2 = (
    df[df['year'].between(2008, 2010)].groupby('country')['visitors'].mean()
)
dur_means_h2 = (
    df[df['year'].between(2011, 2015)].groupby('country')['visitors'].mean()
)

# --- Fig 2 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                         gridspec_kw={'width_ratios': [2, 1]})

ax = axes[0]
for country in all_countries:
    if country not in pre_means_h2.index or country not in dur_means_h2.index:
        continue
    pre_v = pre_means_h2[country] / 1e6
    dur_v = dur_means_h2[country] / 1e6
    group = df[df['country'] == country]['market_group'].values[0]
    color = GROUP_COLORS[group]
    is_mena = (group == 'MENA')
    ax.plot(
        [0, 1], [pre_v, dur_v],
        color=color,
        alpha=0.95 if is_mena else 0.50,
        lw=2.5 if is_mena else 1.5,
        ls='-' if is_mena else '--'
    )
    ax.scatter([0, 1], [pre_v, dur_v], color=color,
               s=50 if is_mena else 25, alpha=0.9 if is_mena else 0.5, zorder=3)
    ax.annotate(
        country, (1.03, dur_v), fontsize=7.5, va='center', color=color
    )

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-war\n(2008-10 mean)', 'War period\n(2011-15 mean)'], fontsize=10)
ax.set_ylabel('Mean Annual Visitors (millions)')
ax.set_xlim(-0.12, 1.30)

legend_patches = [
    mpatches.Patch(color=c, label=g)
    for g, c in GROUP_COLORS.items() if g != 'Other'
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

stat2_txt = (
    f"Welch t = {t2_stat:.2f},  p = {p2_val:.3f}\n"
    f"MENA mean change: {mena_chg.mean():+.0f}%\n"
    f"Non-MENA mean change: {other_chg.mean():+.0f}%"
)
ax.text(
    0.50, 0.03, stat2_txt, transform=ax.transAxes,
    ha='center', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
)

# Right panel: distribution comparison
ax2 = axes[1]
ax2.hist(
    other_chg, bins=8, alpha=0.55, color='#607D8B',
    label=f'Non-MENA (mean {other_chg.mean():+.0f}%)',
    orientation='horizontal', density=True
)
ax2.hist(
    mena_chg, bins=4, alpha=0.80, color=GROUP_COLORS['MENA'],
    label=f'MENA (mean {mena_chg.mean():+.0f}%)',
    orientation='horizontal', density=True
)
ax2.axhline(0, color='black', lw=1, ls='--', alpha=0.5)
ax2.set_xlabel('Density')
ax2.set_ylabel('% Change in Visitors (2011-15 vs 2008-10)')
ax2.legend(fontsize=8)
ax2.set_title('Distribution\nComparison', fontsize=9)

# Finding-driven title
if p2_val < 0.05 and mena_chg.mean() < other_chg.mean():
    fig2_title = (
        f"MENA Markets Fell Harder Than Non-MENA During Syria War "
        f"(Welch t={t2_stat:.2f}, p={p2_val:.3f})"
    )
elif p2_val < 0.05:
    fig2_title = (
        f"MENA and Non-MENA Diverged Significantly During Syria War "
        f"(Welch t={t2_stat:.2f}, p={p2_val:.3f})"
    )
else:
    fig2_title = (
        f"No Significant Difference: MENA vs Non-MENA During Syria War "
        f"(Welch t={t2_stat:.2f}, p={p2_val:.3f})"
    )

fig.suptitle(fig2_title, fontweight='bold', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_h2_syria_mena.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved")


# ============================================================
# H3: COVID recovery speed differs significantly by market group
# Test: One-way ANOVA + Tukey HSD post-hoc on years-to-recover
# Figure 3: Small multiples — recovery trajectory 2019-2025 per group
# ============================================================

print_h(3, "COVID recovery speed differs significantly by market group")

rec_years_h3 = {c: years_to_recover(c, 2020, pre_window=2) for c in all_countries}
rec_df = pd.DataFrame.from_dict(rec_years_h3, orient='index', columns=['years_to_recover'])
rec_df['group'] = rec_df.index.map(
    lambda c: df[df['country'] == c]['market_group'].values[0]
)

print("\nYears to recover post-COVID by country:")
print(rec_df.sort_values('group').to_string())

rec_groups_h3 = {
    g: rec_df[rec_df['group'] == g]['years_to_recover'].dropna().values
    for g in rec_df['group'].unique()
}
print("\nMean recovery years by group (recovered only):")
for g, vals in rec_groups_h3.items():
    if len(vals):
        print(f"  {g:20s}: {vals.mean():.1f} years  (n={len(vals)})")

not_rec_h3 = rec_df[rec_df['years_to_recover'].isna()].index.tolist()
print(f"Not recovered by 2025: {not_rec_h3}")

valid_rec_h3 = [v for v in rec_groups_h3.values() if len(v) >= 2]
if len(valid_rec_h3) >= 2:
    f3_stat, p3_anova = stats.f_oneway(*valid_rec_h3)
    print(f"\nOne-way ANOVA: F={f3_stat:.3f}, p={p3_anova:.4f}")
    print(f"{'REJECT H0' if p3_anova < 0.05 else 'FAIL TO REJECT H0'} at alpha=0.05")
    rec_valid_df = rec_df.dropna(subset=['years_to_recover'])
    tukey3 = pairwise_tukeyhsd(
        rec_valid_df['years_to_recover'], rec_valid_df['group'], alpha=0.05
    )
    print("\nTukey HSD pairwise comparisons (H3):")
    print(tukey3.summary())
    tukey3_rows  = tukey3._results_table.data[1:]
    sig_pairs_h3 = [(r[0], r[1]) for r in tukey3_rows if r[6]]
else:
    f3_stat, p3_anova = np.nan, np.nan
    sig_pairs_h3 = []

# Compute per-group mean trajectory (% of 2018-19 baseline) for Figure 3
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
        ax.plot(
            sub['year'], sub['pct_of_baseline'],
            color=GROUP_COLORS[group], alpha=0.45, lw=1.5, marker='o', ms=3.5
        )
        # Star at first year visitor count exceeds 2018-19 mean
        cross = sub[sub['pct_of_baseline'] >= 100]
        if not cross.empty:
            fc = cross.iloc[0]
            ax.scatter(
                fc['year'], fc['pct_of_baseline'],
                color=GROUP_COLORS[group], s=140, zorder=5,
                marker='*', edgecolors='black', lw=0.5
            )

    # Group mean trajectory (bold)
    group_mean_traj = (
        df_fig3[df_fig3['market_group'] == group]
        .groupby('year')['pct_of_baseline'].mean()
    )
    ax.plot(
        group_mean_traj.index, group_mean_traj.values,
        color=GROUP_COLORS[group], lw=3.0, alpha=1.0, zorder=4,
        label=f'{group} mean'
    )

    ax.set_xlim(2018.8, 2025.2)
    ax.set_xticks(range(2019, 2026))
    ax.set_title(group, fontweight='bold', color=GROUP_COLORS[group])
    ax.set_xlabel('Year')
    ax.set_ylabel('Visitors as % of 2018-19 Baseline')
    ax.legend(fontsize=8)

stat3_txt = (
    f"ANOVA F={f3_stat:.2f}, p={p3_anova:.3f}"
    if not np.isnan(f3_stat) else ""
)
if not_rec_h3:
    stat3_txt += f"  |  Not recovered by 2025: {', '.join(not_rec_h3)}"
if stat3_txt:
    fig.text(
        0.5, -0.01, stat3_txt, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
    )

# Finding-driven title
slowest_group_h3 = (
    rec_df.groupby('group')['years_to_recover'].mean().idxmax()
    if not rec_df['years_to_recover'].isna().all() else ''
)
if not np.isnan(p3_anova) and p3_anova < 0.05 and sig_pairs_h3:
    fig3_title = (
        f"COVID Recovery Speed Differs Significantly Across Groups "
        f"— {slowest_group_h3} Slowest (F={f3_stat:.2f}, p={p3_anova:.3f}) "
        f"  *=Year of Full Recovery"
    )
elif not np.isnan(p3_anova) and p3_anova < 0.05:
    fig3_title = (
        f"COVID Recovery Speed Differs Significantly by Group "
        f"(F={f3_stat:.2f}, p={p3_anova:.3f}) — No Pairwise Pair Survives Tukey HSD  *=Full Recovery"
    )
else:
    fig3_title = (
        f"COVID Recovery Trajectories by Market Group "
        f"(ANOVA F={f3_stat:.2f}, p={p3_anova:.3f})  *=Year of Full Recovery"
    )

fig.suptitle(fig3_title, fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig3_h3_covid_recovery.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")


# ============================================================
# H4: Turkey lira weakness correlates with larger visitor drops
#     during shock years (crisis effect dominates price effect)
# Test: Pearson correlation — Turkey YoY currency weakness vs
#       % visitor change across shock year × country observations
# Figure 4: Scatter with OLS line, colored by market group
# ============================================================

print_h(4, "Turkey currency weakness correlates with larger visitor drops during shocks")

# Use three major shock years: 2009 (financial), 2016 (coup), 2020 (COVID)
# Each gives one (country, shock_year) observation.
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
            group = df[df['country'] == c]['market_group'].values[0]
            results_h4.append({
                'country':              c,
                'shock_year':           sy,
                'impact':               imp,
                'tur_ppp_rate':         tur_row['tur_ppp_rate'].values[0],
                'tur_currency_weakness': tur_row['tur_currency_weakness'].values[0],
                'group':                group,
            })

res_h4 = pd.DataFrame(results_h4)

# Primary correlation: PPP rate level vs visitor impact
r4, p4 = stats.pearsonr(res_h4['tur_ppp_rate'], res_h4['impact'])
print(f"\nPearson r (PPP rate vs impact) = {r4:.3f},  p = {p4:.4f}")
print(f"Direction: {'Higher PPP (weaker lira) -> LARGER drop' if r4 < 0 else 'Higher PPP -> smaller drop'}")
print(f"{'SIGNIFICANT' if p4 < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

# --- Fig 4 ---
fig, ax = plt.subplots(figsize=(11, 7))

for group in GROUP_COLORS:
    sub = res_h4[res_h4['group'] == group]
    if sub.empty:
        continue
    ax.scatter(
        sub['tur_ppp_rate'], sub['impact'],
        color=GROUP_COLORS[group], s=90, alpha=0.75, zorder=3,
        edgecolors='white', lw=0.6, label=group
    )

# Label each point with abbreviated country + shock year
for _, row in res_h4.iterrows():
    ax.annotate(
        f"{row['country'][:3]}\n'{str(row['shock_year'])[2:]}",
        (row['tur_ppp_rate'], row['impact']),
        fontsize=6.5, xytext=(4, 3), textcoords='offset points', alpha=0.70
    )

# OLS line
x4 = res_h4['tur_ppp_rate']
y4 = res_h4['impact']
m4, b4 = np.polyfit(x4, y4, 1)
x4_range = np.linspace(x4.min(), x4.max(), 100)
ax.plot(x4_range, m4 * x4_range + b4, 'r--', lw=2,
        label=f'OLS (slope={m4:.1f}% / unit PPP)')

ax.axhline(0, color='black', lw=1, alpha=0.4)
ax.set_xlabel('Turkey PPP Conversion Rate  (higher = weaker lira)')
ax.set_ylabel('% Change in Visitors vs Pre-Shock Baseline')
ax.legend(fontsize=8, loc='upper right')

naive_theory = "Naive theory: weaker lira -> cheaper Turkey -> MORE tourists (positive slope)"
observed     = (
    "Observed: NEGATIVE slope — crisis/instability effect dominates price effect"
    if r4 < 0 else
    "Observed: POSITIVE slope — price effect apparent in this sample"
)
stat4_txt = f"Pearson r = {r4:.3f},  p = {p4:.3f}\n{naive_theory}\n{observed}"
ax.text(
    0.02, 0.04, stat4_txt, transform=ax.transAxes,
    fontsize=8, va='bottom', ha='left',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
)

# Finding-driven title
if r4 < 0 and p4 < 0.05:
    fig4_title = (
        f"Weaker Lira Accompanies Larger Visitor Drops — Crisis Effect Dominates Price Effect "
        f"(r={r4:.2f}, p={p4:.3f})"
    )
elif r4 < 0:
    fig4_title = (
        f"Negative Lira-Visitor Relationship — Not Statistically Significant "
        f"(r={r4:.2f}, p={p4:.3f})"
    )
else:
    fig4_title = (
        f"No Evidence Crisis Effect Dominates — Observed Positive Slope "
        f"(r={r4:.2f}, p={p4:.3f})"
    )

ax.set_title(fig4_title, fontweight='bold', pad=10, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig4_h4_lira_weakness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")


# ============================================================
# H5: Source country GDP per capita positively predicts visitor volume
# Test: Pearson + Spearman correlation on all country-year observations
# Figure 5: Scatter — log(GDP per capita) vs log(visitors)
# ============================================================

print_h(5, "Source country GDP per capita positively predicts visitor volume")

df_h5 = df.dropna(subset=['gdp_per_capita', 'log_visitors'])
r5,  p5  = stats.pearsonr(df_h5['gdp_per_capita'], df_h5['log_visitors'])
r5s, p5s = stats.spearmanr(df_h5['gdp_per_capita'], df_h5['log_visitors'])

print(f"\n  n = {len(df_h5)} country-year observations")
print(f"  Pearson r  = {r5:.4f},  p = {p5:.6f}")
print(f"  Spearman r = {r5s:.4f},  p = {p5s:.6f}")
print(f"  {'REJECT H0' if p5s < 0.05 else 'FAIL TO REJECT H0'} at alpha=0.05 (Spearman)")

# --- Fig 5 ---
fig, ax = plt.subplots(figsize=(10, 7))

for group in GROUP_COLORS:
    sub = df_h5[df_h5['market_group'] == group]
    ax.scatter(
        np.log(sub['gdp_per_capita']), sub['log_visitors'],
        color=GROUP_COLORS[group], alpha=0.40, s=40,
        edgecolors='none', label=group
    )

x5 = np.log(df_h5['gdp_per_capita'])
y5 = df_h5['log_visitors']
m5, b5 = np.polyfit(x5, y5, 1)
x5_range = np.linspace(x5.min(), x5.max(), 100)
ax.plot(x5_range, m5 * x5_range + b5, 'r--', lw=2, label='OLS trend')

ax.set_xlabel('log(Source Country GDP per Capita, USD)')
ax.set_ylabel('log(Annual Visitors to Turkey)')

stat5_txt = (
    f"Pearson r = {r5:.3f},  p = {p5:.4f}\n"
    f"Spearman r = {r5s:.3f},  p = {p5s:.4f}\n"
    f"n = {len(df_h5)} country-year obs."
)
ax.text(
    0.02, 0.97, stat5_txt, transform=ax.transAxes,
    fontsize=9, va='top', ha='left',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
)
ax.legend(fontsize=8)

# Finding-driven title (use Spearman as primary; flag strength)
strength = "Weakly" if abs(r5s) < 0.3 else "Moderately"
if p5s < 0.05 and r5s > 0:
    fig5_title = (
        f"Wealthier Source Countries Send More Tourists to Turkey — "
        f"{strength} Positive and Significant (Spearman r={r5s:.2f}, p={p5s:.3f})"
    )
elif p5s < 0.05:
    fig5_title = (
        f"GDP per Capita and Visitor Volume Are Significantly Related "
        f"(Spearman r={r5s:.2f}, p={p5s:.3f})"
    )
else:
    fig5_title = (
        f"GDP per Capita Does Not Significantly Predict Visitor Volume "
        f"(Spearman r={r5s:.2f}, p={p5s:.3f})"
    )

ax.set_title(fig5_title, fontweight='bold', pad=10, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig5_h5_gdp_visitors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved")


# ============================================================
# H6: Post-2023 MENA tensions reduced MENA-origin tourism more
#     than non-MENA markets (2023-25 mean vs 2017-19 baseline)
# Test: Welch t-test (MENA vs non-MENA % vs baseline)
# Figure 6: Horizontal bar chart — % vs 2017-19 baseline, MENA highlighted
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

print("\n% vs 2017-19 baseline, 2023-25 mean:")
print(h6_df.sort_values('market_group').to_string(index=False))

t6_stat, p6_val = stats.ttest_ind(mena_h6, nonmena_h6, equal_var=False)
print(f"\nMENA mean:     {mena_h6.mean():+.1f}%")
print(f"Non-MENA mean: {nonmena_h6.mean():+.1f}%")
print(f"Welch t-test: t={t6_stat:.3f},  p={p6_val:.4f}")
print(f"{'REJECT H0' if p6_val < 0.05 else 'FAIL TO REJECT H0'} at alpha=0.05")
print(
    "Interpretation: Post-2023 MENA tensions"
    + (" DID suppress MENA-origin tourism significantly more than non-MENA."
       if p6_val < 0.05 and mena_h6.mean() < nonmena_h6.mean()
       else " did NOT significantly suppress MENA markets more than non-MENA.")
)

# --- Fig 6 ---
fig, ax = plt.subplots(figsize=(12, 7))

h6_sorted = h6_df.sort_values('pct_vs_baseline')
bar_colors = [GROUP_COLORS[g] for g in h6_sorted['market_group']]
alphas     = [1.0 if g == 'MENA' else 0.50 for g in h6_sorted['market_group']]

bars = ax.barh(
    h6_sorted['country'], h6_sorted['pct_vs_baseline'],
    color=bar_colors, edgecolor='white', lw=0.8
)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)

ax.axvline(0, color='black', lw=1.5)
ax.set_xlabel('% Change in Mean Visitors vs 2017-19 Baseline  (2023-25 average)')

for bar, val in zip(bars, h6_sorted['pct_vs_baseline']):
    ax.text(
        val + (2 if val >= 0 else -2),
        bar.get_y() + bar.get_height() / 2,
        f'{val:+.0f}%', va='center',
        ha='left' if val >= 0 else 'right', fontsize=8.5
    )

stat6_txt = (
    f"Welch t = {t6_stat:.2f},  p = {p6_val:.3f}\n"
    f"MENA mean: {mena_h6.mean():+.0f}%    "
    f"Non-MENA mean: {nonmena_h6.mean():+.0f}%"
)
ax.text(
    0.02, 0.97, stat6_txt, transform=ax.transAxes,
    fontsize=9, va='top', ha='left',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85)
)

mena_patch  = mpatches.Patch(color=GROUP_COLORS['MENA'],   label='MENA markets')
other_patch = mpatches.Patch(color='#aaaaaa', alpha=0.5,   label='Non-MENA (faded)')
ax.legend(handles=[mena_patch, other_patch], fontsize=9, loc='lower right')

# Finding-driven title
if p6_val < 0.05 and mena_h6.mean() < nonmena_h6.mean():
    fig6_title = (
        f"Post-2023 MENA Tensions Hit MENA-Origin Tourism Harder Than Non-MENA "
        f"(Welch t={t6_stat:.2f}, p={p6_val:.3f})"
    )
elif p6_val < 0.05:
    fig6_title = (
        f"MENA and Non-MENA Show Significantly Different Post-2023 Recovery "
        f"(Welch t={t6_stat:.2f}, p={p6_val:.3f})"
    )
else:
    fig6_title = (
        f"No Significant Difference in Post-2023 Recovery: MENA vs Non-MENA "
        f"(Welch t={t6_stat:.2f}, p={p6_val:.3f})"
    )

ax.set_title(fig6_title, fontweight='bold', pad=12, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig6_h6_mena_tension_recent.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 6 saved")


# ============================================================
# SUMMARY TABLE
# ============================================================

print(f"\n{SEP}")
print("HYPOTHESIS TEST SUMMARY")
print(SEP)

h1_result = f"F={f1_stat:.3f}, p={p1_anova:.4f}  {'SIGNIFICANT' if p1_anova < 0.05 else 'not significant'}"
h2_result = f"t={t2_stat:.3f}, p={p2_val:.4f}   {'SIGNIFICANT' if p2_val < 0.05 else 'not significant'}  |  MENA: {mena_chg.mean():+.1f}%  Other: {other_chg.mean():+.1f}%"
h3_result = (
    f"F={f3_stat:.3f}, p={p3_anova:.4f}  {'SIGNIFICANT' if p3_anova < 0.05 else 'not significant'}"
    if not np.isnan(f3_stat) else "insufficient data"
)
h4_result = f"r={r4:.3f}, p={p4:.4f}   {'SIGNIFICANT' if p4 < 0.05 else 'not significant'}"
h5_result = f"Spearman r={r5s:.3f}, p={p5s:.4f}  {'SIGNIFICANT' if p5s < 0.05 else 'not significant'}"
h6_result = f"t={t6_stat:.3f}, p={p6_val:.4f}   {'SIGNIFICANT' if p6_val < 0.05 else 'not significant'}  |  MENA: {mena_h6.mean():+.1f}%  Non-MENA: {nonmena_h6.mean():+.1f}%"

print(f"""
  H1  2016 coup sensitivity differs by group
      ANOVA {h1_result}
      Tukey HSD sig. pairs: {sig_pairs_h1 if sig_pairs_h1 else 'none'}

  H2  MENA fell harder than non-MENA during Syria war (2011-15)
      Welch t  {h2_result}

  H3  COVID recovery speed differs by group
      ANOVA {h3_result}
      Tukey HSD sig. pairs: {sig_pairs_h3 if 'sig_pairs_h3' in dir() else 'see above'}
      Not recovered by 2025: {not_rec_h3}

  H4  Turkey lira weakness -> larger visitor drops during shocks
      Pearson  {h4_result}

  H5  Source GDP per capita predicts visitor volume
      {h5_result}

  H6  Post-2023 MENA tensions reduced MENA tourism more than non-MENA
      Welch t  {h6_result}
""")
