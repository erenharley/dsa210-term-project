"""
DSA 210 - Tourism Demand Under Shock
Diagnostic: Per-Country EDA (visitors + macro context)
Author: Eren Sean Harley | 36054

Outputs (all under images/per_country/):
  _grid_visitors.png      — 4×4 grid, all 15 visitor trajectories
  _turkey_macro.png       — Turkey-side macro indicators
  <country>/overview.png  — per-country 5-panel figure

Run AFTER 01_data_pipeline.py has been run (reads panel_dataset.csv as-is).
Does NOT modify any other script, README, or CLAUDE.md.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR.parent / 'data'
OUT_DIR      = BASE_DIR.parent / 'images' / 'per_country'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'panel_dataset.csv')
df = df.sort_values(['country', 'year']).reset_index(drop=True)

all_countries = sorted(df['country'].unique())

# ── Market groups and colors (mirroring 02_eda_and_hypothesis.py) ──────────
GROUP_COLORS = {
    'Western Europe': '#2196F3',
    'Eastern Europe': '#4CAF50',
    'Former Soviet':  '#FF5722',
    'MENA':           '#9C27B0',
    'Other':          '#607D8B',
}

COUNTRY_ORDER = [
    # Western Europe
    'France', 'Germany', 'Netherlands', 'United Kingdom',
    # Eastern Europe
    'Bulgaria', 'Greece', 'Ukraine',
    # Former Soviet
    'Azerbaijan', 'Georgia', 'Russia',
    # MENA
    'Iran', 'Iraq', 'Israel', 'Syria',
    # Other
    'USA',
]

# ── Crisis windows ─────────────────────────────────────────────────────────
CRISIS_WINDOWS = [
    # start/end   = axvspan drawing coordinates (can be fractional)
    # marker_s/e  = integer years of first/last affected data point (for hollow circles)
    {'label': 'Syria conflict',          'start': 2011,   'end': 2015,   'marker_s': 2011, 'marker_e': 2015, 'color': '#E53935'},
    {'label': '2016 coup',               'start': 2015.7, 'end': 2016.3, 'marker_s': 2016, 'marker_e': 2016, 'color': '#FB8C00'},
    {'label': 'COVID-19',                'start': 2019,   'end': 2021,   'marker_s': 2019, 'marker_e': 2021, 'color': '#1E88E5'},
    {'label': 'Russia–Ukraine war',      'start': 2022,   'end': 2022.5, 'marker_s': 2022, 'marker_e': 2022, 'color': '#FDD835'},
    {'label': 'Post-2023 MENA tensions', 'start': 2023,   'end': 2025,   'marker_s': 2023, 'marker_e': 2024, 'color': '#8E24AA'},
]

# Crisis edge-marker windows (visitors panel only, skip 2016 coup narrow band)
CRISIS_EDGE_WINDOWS = [c for c in CRISIS_WINDOWS if c['label'] != '2016 coup']


# ── Helpers ────────────────────────────────────────────────────────────────

def get_group_color(country):
    grp = df[df['country'] == country]['market_group'].iloc[0]
    return GROUP_COLORS.get(grp, '#607D8B')


def add_crisis_bands(ax):
    """Draw all crisis bands on a given axes."""
    for w in CRISIS_WINDOWS:
        ax.axvspan(w['start'], w['end'], color=w['color'], alpha=0.28,
                   linewidth=0, zorder=0)


def format_mean_visitors(val):
    """Format mean visitors as e.g. '4.2M', '850K', '12K'."""
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    elif val >= 1_000:
        return f"{val / 1_000:.0f}K"
    else:
        return f"{val:.0f}"


def thousands_formatter(x, pos):
    return f'{x:,.0f}'


def make_crisis_legend_handles():
    return [
        mpatches.Patch(facecolor=w['color'], alpha=0.55, label=w['label'])
        for w in CRISIS_WINDOWS
    ]


# ── Global rcParams ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   10,
    'axes.labelsize':   9,
    'legend.fontsize':  9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
})

PANEL_FACECOLOR = '#fafafa'
GRID_COLOR      = '#cccccc'
GRID_ALPHA      = 0.4


# ══════════════════════════════════════════════════════════════════════════
#  1. PER-COUNTRY OVERVIEW FIGURES
# ══════════════════════════════════════════════════════════════════════════

PANEL_SPECS = [
    # (column,             ylabel,                         axhline_zero)
    ('visitors',           'Visitors (annual)',             False),
    ('gdp_growth',         'GDP growth (% YoY)',            True),
    ('gdp_per_capita',     'GDP per capita (USD)',          False),
    ('inflation',          'Inflation (% CPI)',             False),
    ('political_stability','WGI Political Stability',       True),
]

PANEL_TITLES = ['Visitors', 'GDP growth', 'GDP per capita', 'Inflation', 'Political stability']

for country in all_countries:
    cdf = df[df['country'] == country].sort_values('year')
    years = cdf['year'].values
    color = get_group_color(country)
    slug = country.lower().replace(' ', '-')

    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True,
                              gridspec_kw={'hspace': 0.35},
                              facecolor='white')

    for i, (col, ylabel, zero_line) in enumerate(PANEL_SPECS):
        ax = axes[i]
        ax.set_facecolor(PANEL_FACECOLOR)
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.8, zorder=-1)

        # Crisis bands first (zorder=0)
        add_crisis_bands(ax)

        # Zero reference line for signed panels
        if zero_line:
            ax.axhline(0, color='#888888', lw=0.8, ls='--', zorder=1)

        # Data
        y = cdf[col].values.astype(float)
        mask = ~np.isnan(y)
        ax.plot(years[mask], y[mask],
                lw=2.0, color=color, marker='o', ms=5,
                mec='white', mew=0.8, zorder=3)

        # Crisis edge markers on the visitors panel only
        if i == 0:
            for w in CRISIS_EDGE_WINDOWS:
                for edge_year in [w['marker_s'], w['marker_e']]:
                    idx = np.where(years == edge_year)[0]
                    if len(idx) and not np.isnan(y[idx[0]]):
                        ax.plot(edge_year, y[idx[0]],
                                marker='o', ms=10, mfc='none',
                                mew=2, color=w['color'], zorder=4)

        # Y-axis formatting
        if col in ('visitors', 'gdp_per_capita'):
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

        # Explicit y-axis limits based on this country's actual data range
        valid_y = y[mask]
        if len(valid_y) > 0:
            lo, hi = valid_y.min(), valid_y.max()
            pad = (hi - lo) * 0.10 if hi != lo else abs(hi) * 0.10 or 1
            # Visitors and counts never go below 0
            if col in ('visitors', 'gdp_per_capita'):
                ax.set_ylim(max(0, lo - pad), hi + pad)
            else:
                ax.set_ylim(lo - pad, hi + pad)

        # Panel title (bold, left-aligned, small)
        ax.set_title(PANEL_TITLES[i], fontsize=10, fontweight='bold',
                     loc='left', pad=3)
        ax.set_ylabel(ylabel, fontsize=9)

        ax.set_xlim(2002.5, 2025.5)

    # X-axis ticks (bottom panel only, shared)
    axes[-1].set_xticks(range(2003, 2026, 2))
    axes[-1].set_xlabel('Year', fontsize=9)

    # Suptitle and subtitle
    fig.suptitle(f'{country} — Visitors and Macro Context (2003–2025)',
                 fontsize=15, fontweight='bold', y=0.995)
    fig.text(0.5, 0.978,
             'Source: TÜİK (visitors), IMF DataMapper (GDP, inflation), '
             'World Bank Data360 (political stability) | Crisis windows shaded',
             fontsize=9, color='#555555', ha='center')

    # Shared legend at bottom
    handles = make_crisis_legend_handles()
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.005), ncol=5, fontsize=9,
               framealpha=0.9, edgecolor='#cccccc')

    # Save
    out_path = OUT_DIR / f'{slug}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Console summary
    vis = cdf['visitors'].values.astype(float)
    vis_valid = vis[~np.isnan(vis)]
    vis_years = years[~np.isnan(vis)]

    mean_v = vis_valid.mean() if len(vis_valid) else float('nan')
    min_idx = int(np.argmin(vis_valid)) if len(vis_valid) else None
    max_idx = int(np.argmax(vis_valid)) if len(vis_valid) else None

    def _get_year_val(yr):
        row = cdf[cdf['year'] == yr]['visitors']
        return f"{int(row.values[0]):,}" if len(row) and not np.isnan(row.values[0]) else 'N/A'

    macro_nan_cols = []
    for col in ('gdp_growth', 'gdp_per_capita', 'inflation', 'political_stability'):
        if cdf[col].isna().any():
            macro_nan_cols.append(col)

    print(
        f"{country:20s} | visitors mean={mean_v:>12,.0f} "
        f"| min={vis_years[min_idx]}:{vis_valid[min_idx]:>12,.0f} "
        f"| max={vis_years[max_idx]}:{vis_valid[max_idx]:>12,.0f} "
        f"| 2019={_get_year_val(2019):>12s} "
        f"| 2020={_get_year_val(2020):>12s} "
        f"| 2024={_get_year_val(2024):>12s} "
        f"| macro NaN cols: {macro_nan_cols}"
    )


# ══════════════════════════════════════════════════════════════════════════
#  2. _grid_visitors.png — 4×4 grid of all 15 visitor trajectories
# ══════════════════════════════════════════════════════════════════════════

fig_grid, axes_grid = plt.subplots(4, 4, figsize=(20, 16), facecolor='white')
axes_flat = axes_grid.flatten()

for i, country in enumerate(COUNTRY_ORDER):
    ax = axes_flat[i]
    cdf = df[df['country'] == country].sort_values('year')
    years = cdf['year'].values
    vis = cdf['visitors'].values.astype(float)
    color = get_group_color(country)

    ax.set_facecolor(PANEL_FACECOLOR)
    ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.8, zorder=-1)
    add_crisis_bands(ax)

    mask = ~np.isnan(vis)
    ax.plot(years[mask], vis[mask],
            lw=1.8, color=color, marker='o', ms=4,
            mec='white', mew=0.6, zorder=3)

    ax.set_title(country, fontsize=11, fontweight='bold')
    ax.set_xticks(range(2003, 2026, 4))
    ax.tick_params(axis='x', labelsize=7, rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xlim(2002.5, 2025.5)

    # Explicit y-limits per country so Iran and UK don't look the same scale
    valid_vis = vis[mask]
    if len(valid_vis) > 0:
        lo, hi = valid_vis.min(), valid_vis.max()
        pad = (hi - lo) * 0.10 if hi != lo else hi * 0.10 or 1
        ax.set_ylim(max(0, lo - pad), hi + pad)

    # Mean annotation top-right
    mean_v = vis[mask].mean() if mask.sum() else float('nan')
    ax.text(0.97, 0.95, f'μ: {format_mean_visitors(mean_v)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.75, edgecolor='none'))

# Hide unused 16th cell
axes_flat[15].set_visible(False)

# Suptitle
fig_grid.suptitle('Per-Country Visitor Trajectories with Crisis Windows (2003–2025)',
                  fontsize=15, fontweight='bold', y=1.002)

# Shared legend at bottom
handles = make_crisis_legend_handles()
fig_grid.legend(handles=handles, loc='lower center',
                bbox_to_anchor=(0.5, -0.012), ncol=5, fontsize=9,
                framealpha=0.9, edgecolor='#cccccc')

fig_grid.tight_layout()
grid_path = OUT_DIR / '_grid_visitors.png'
fig_grid.savefig(grid_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_grid)


# ══════════════════════════════════════════════════════════════════════════
#  3. _turkey_macro.png — Turkey-side macro indicators
# ══════════════════════════════════════════════════════════════════════════

tur_cols = ['year', 'tur_gdp_growth', 'tur_gdp_per_capita', 'tur_inflation', 'tur_ppp_rate']
tur_df = (
    df[tur_cols]
    .drop_duplicates(subset='year')
    .sort_values('year')
    .reset_index(drop=True)
)

TUR_PANEL_SPECS = [
    ('tur_gdp_growth',     'Turkey GDP growth (% YoY)',         True),
    ('tur_gdp_per_capita', 'Turkey GDP per capita (USD)',        False),
    ('tur_inflation',      'Turkey inflation (% CPI)',           False),
    ('tur_ppp_rate',       'Turkey PPP rate (higher = weaker ₺)',False),
]
TUR_PANEL_TITLES = [
    'Turkey GDP growth',
    'Turkey GDP per capita',
    'Turkey inflation',
    'Turkey PPP rate (lira weakness proxy)',
]
TUR_COLOR = '#C62828'  # Turkish red

fig_tur, axes_tur = plt.subplots(4, 1, figsize=(11, 13), sharex=True,
                                   gridspec_kw={'hspace': 0.35},
                                   facecolor='white')

years_tur = tur_df['year'].values

for i, (col, ylabel, zero_line) in enumerate(TUR_PANEL_SPECS):
    ax = axes_tur[i]
    ax.set_facecolor(PANEL_FACECOLOR)
    ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=0.8, zorder=-1)
    add_crisis_bands(ax)

    if zero_line:
        ax.axhline(0, color='#888888', lw=0.8, ls='--', zorder=1)

    y = tur_df[col].values.astype(float)
    mask = ~np.isnan(y)
    ax.plot(years_tur[mask], y[mask],
            lw=2.0, color=TUR_COLOR, marker='o', ms=5,
            mec='white', mew=0.8, zorder=3)

    if col == 'tur_gdp_per_capita':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))

    ax.set_title(TUR_PANEL_TITLES[i], fontsize=10, fontweight='bold',
                 loc='left', pad=3)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(2002.5, 2025.5)

axes_tur[-1].set_xticks(range(2003, 2026, 2))
axes_tur[-1].set_xlabel('Year', fontsize=9)

fig_tur.suptitle('Turkey — Macroeconomic Indicators (2003–2025)',
                 fontsize=15, fontweight='bold', y=0.995)
fig_tur.text(0.5, 0.978,
             'Source: IMF DataMapper | Crisis windows shaded',
             fontsize=9, color='#555555', ha='center')

handles = make_crisis_legend_handles()
fig_tur.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.005), ncol=5, fontsize=9,
               framealpha=0.9, edgecolor='#cccccc')

tur_path = OUT_DIR / '_turkey_macro.png'
fig_tur.savefig(tur_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_tur)

# ── Turkey NaN report ─────────────────────────────────────────────────────
tur_nan_parts = []
for col in ['tur_gdp_growth', 'tur_gdp_per_capita', 'tur_inflation', 'tur_ppp_rate']:
    nan_years = tur_df[tur_df[col].isna()]['year'].tolist()
    if nan_years:
        tur_nan_parts.append(f"{col} NaN years: {nan_years}")
if tur_nan_parts:
    print("Turkey macro | " + " | ".join(tur_nan_parts))
else:
    print("Turkey macro | no NaN years in any column")

print(f"\nWrote 15 country overviews + _grid_visitors.png + _turkey_macro.png "
      f"to images/per_country/")
