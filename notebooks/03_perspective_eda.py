"""
03_perspective_eda.py — Stage 2.5 perspective EDA figures.

Outputs (all to images/):
  eda_01_trends_by_group.png      — 4-panel line trends by market group
  eda_02_shock_sensitivity.png    — 4-panel horizontal bar chart of shock impacts
  eda_03_recovery_speed.png       — 2-panel recovery speed bars (COVID + coup)
  eda_04_political_stability.png  — Turkey political stability time series
  eda_05_resilience_heatmap.png   — Country x year % vs 2017-19 baseline heatmap

Style:
  Crisis bands, GROUP_COLORS, and format helpers match 00_per_country_eda.py.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
DATA_PATH   = REPO_ROOT / "data" / "panel_dataset.csv"
OUT_DIR     = REPO_ROOT / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Group colors (must match 00_per_country_eda.py and 02_eda_and_hypothesis.py) ──

GROUP_COLORS = {
    "Western Europe": "#2196F3",
    "Eastern Europe": "#4CAF50",
    "Former Soviet":  "#FF5722",
    "MENA":           "#9C27B0",
    "Other":          "#607D8B",
}

# ── Crisis bands ─────────────────────────────────────────────────────────────

CRISES = [
    {"label": "Syria conflict",          "start": 2011,   "end": 2015,   "color": "#E53935"},
    {"label": "2016 coup",               "start": 2015.7, "end": 2016.3, "color": "#FB8C00"},
    {"label": "COVID-19",                "start": 2020,   "end": 2021,   "color": "#1E88E5"},
    {"label": "Russia–Ukraine war",      "start": 2022,   "end": 2023,   "color": "#FDD835"},
    {"label": "Post-2023 MENA tensions", "start": 2023,   "end": 2025,   "color": "#8E24AA"},
]

# Country ordering (group-sorted, matching 00_per_country_eda.py)
COUNTRY_ORDER = [
    "France", "Germany", "Netherlands", "United Kingdom",
    "Bulgaria", "Greece", "Ukraine",
    "Azerbaijan", "Georgia", "Russia",
    "Iran", "Iraq", "Israel", "Syria", "United Arab Emirates", "Qatar",
    "USA",
]

GROUP_ORDER = ["Western Europe", "Eastern Europe", "Former Soviet", "MENA", "Other"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def add_crisis_bands(ax):
    for c in CRISES:
        ax.axvspan(c["start"], c["end"], alpha=0.28, color=c["color"],
                   zorder=0, linewidth=0)


def crisis_legend_patches():
    return [mpatches.Patch(color=c["color"], alpha=0.55, label=c["label"])
            for c in CRISES]


def group_legend_patches():
    return [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]


def fmt_millions(x, _=None):
    if pd.isna(x):
        return ""
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}K"
    return str(int(x))


def style_ax(ax):
    ax.set_facecolor("#fafafa")
    ax.grid(axis="both", color="#cccccc", alpha=0.4, zorder=-1, linewidth=0.7)
    ax.tick_params(labelsize=9)


def shock_impact(df, country, shock_year, pre_years):
    """% change in visitors in shock_year vs mean(pre_years)."""
    d = df[df["country"] == country].sort_values("year")
    pre = d[d["year"].isin(pre_years)]["visitors"].mean()
    shock_val = d[d["year"] == shock_year]["visitors"].values
    if pre == 0 or len(shock_val) == 0 or np.isnan(pre):
        return np.nan
    return (shock_val[0] / pre - 1) * 100


def shock_impact_range(df, country, shock_years, pre_years):
    """% change: mean of shock_years vs mean of pre_years."""
    d = df[df["country"] == country]
    pre = d[d["year"].isin(pre_years)]["visitors"].mean()
    shock_v = d[d["year"].isin(shock_years)]["visitors"].mean()
    if pre == 0 or np.isnan(pre) or np.isnan(shock_v):
        return np.nan
    return (shock_v / pre - 1) * 100


def years_to_recover(df, country, shock_year, pre_years):
    """Years after shock_year until visitors first exceed the pre_years mean."""
    d = df[df["country"] == country].sort_values("year")
    pre = d[d["year"].isin(pre_years)]["visitors"].mean()
    if np.isnan(pre) or pre == 0:
        return np.nan
    for _, row in d[d["year"] > shock_year].iterrows():
        if row["visitors"] >= pre:
            return row["year"] - shock_year
    return np.nan


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
group_map = (df[["country", "market_group"]]
             .drop_duplicates()
             .set_index("country")["market_group"]
             .to_dict())

all_countries = sorted(df["country"].unique())
assert len(all_countries) == 17, f"Expected 17 countries, got {len(all_countries)}"


# ═══════════════════════════════════════════════════════════════════════════════
# EDA Figure 1 — Tourism Trends by Market Group (4-panel small multiples)
# ═══════════════════════════════════════════════════════════════════════════════

GROUP_MEMBERS = {
    "Western Europe": ["France", "Germany", "Netherlands", "United Kingdom", "USA"],
    "Eastern Europe": ["Bulgaria", "Greece", "Ukraine"],
    "Former Soviet":  ["Azerbaijan", "Georgia", "Russia"],
    "MENA":           ["Iran", "Iraq", "Israel", "Syria", "United Arab Emirates", "Qatar"],
}
PANEL_ORDER = ["Western Europe", "Eastern Europe", "Former Soviet", "MENA"]

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9), facecolor="white",
                            gridspec_kw={"hspace": 0.40, "wspace": 0.30})

for ax, group in zip(axes1.flatten(), PANEL_ORDER):
    style_ax(ax)
    add_crisis_bands(ax)

    members = GROUP_MEMBERS[group]
    for country in members:
        cdf = df[df["country"] == country].sort_values("year")
        sub = cdf[["year", "visitors"]].dropna()
        if sub.empty:
            continue
        label = f"{country} (Other group)" if country == "USA" else country
        line_color = GROUP_COLORS["Other"] if country == "USA" else GROUP_COLORS[group]
        ax.plot(sub["year"], sub["visitors"] / 1e6,
                color=line_color, lw=1.8, marker="o", ms=3.5,
                mec="white", mew=0.6, zorder=3, label=label,
                linestyle="--" if country == "USA" else "-")

    ax.set_title(group, fontweight="bold", fontsize=12, color=GROUP_COLORS[group], pad=6)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Annual Visitors (millions)", fontsize=9)
    ax.set_xticks(range(2003, 2026, 2))
    ax.set_xlim(2002.5, 2025.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

# Shared crisis legend at the bottom
fig1.legend(handles=crisis_legend_patches(),
            loc="lower center", bbox_to_anchor=(0.5, -0.03),
            ncol=5, fontsize=9, framealpha=0.9)

fig1.suptitle("Tourism Trends by Source Market Group (2003-2025)",
              fontweight="bold", fontsize=14)
fig1.savefig(OUT_DIR / "eda_01_trends_by_group.png", dpi=150, bbox_inches="tight",
             facecolor="white")
plt.close(fig1)
print("eda_01_trends_by_group.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# EDA Figure 2 — Shock Sensitivity (4-panel horizontal bar charts)
# ═══════════════════════════════════════════════════════════════════════════════

shock_configs = [
    {
        "title":      "COVID-19 (2020 vs 2018-19 baseline)",
        "fn":         lambda c: shock_impact(df, c, 2020, [2018, 2019]),
        "xlim":       (-100, 100),
        "ref_years":  "2018-19",
        "shock_year": "2020",
    },
    {
        "title":      "2016 Coup (2016 vs 2014-15 baseline)",
        "fn":         lambda c: shock_impact(df, c, 2016, [2014, 2015]),
        "xlim":       (-100, 100),
        "ref_years":  "2014-15",
        "shock_year": "2016",
    },
    {
        "title":      "Syria Onset (2012 vs 2010-11 baseline)",
        "fn":         lambda c: shock_impact(df, c, 2012, [2010, 2011]),
        "xlim":       (-100, 100),
        "ref_years":  "2010-11",
        "shock_year": "2012",
    },
    {
        "title":      "Post-2023 MENA Tensions (2023-25 mean vs 2017-19 baseline)",
        "fn":         lambda c: shock_impact_range(df, c, [2023, 2024, 2025], [2017, 2018, 2019]),
        "xlim":       (-50, 250),
        "ref_years":  "2017-19",
        "shock_year": "2023-25 mean",
    },
]

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 10), facecolor="white",
                            gridspec_kw={"wspace": 0.50})

for ax, cfg in zip(axes2, shock_configs):
    impacts = [(c, cfg["fn"](c), group_map[c]) for c in all_countries]
    impacts = [(c, v, g) for c, v, g in impacts if not np.isnan(v)]
    impacts.sort(key=lambda x: x[1], reverse=True)

    countries_list = [x[0] for x in impacts]
    vals           = [x[1] for x in impacts]
    colors         = [GROUP_COLORS[x[2]] for x in impacts]

    y_pos = range(len(countries_list))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", lw=0.6, height=0.7)

    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlim(cfg["xlim"])
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(countries_list, fontsize=8)
    ax.set_xlabel("% Change", fontsize=9)
    ax.set_title(cfg["title"], fontweight="bold", fontsize=9, wrap=True)
    style_ax(ax)

    for bar, val in zip(bars, vals):
        label_x = val + (cfg["xlim"][1] * 0.02) if val >= 0 else val + (cfg["xlim"][0] * 0.02)
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.0f}%", va="center",
                ha="left" if val >= 0 else "right", fontsize=6.5)

# Shared market group legend
fig2.legend(handles=group_legend_patches(),
            loc="lower center", bbox_to_anchor=(0.5, -0.04),
            ncol=5, fontsize=9, framealpha=0.9)

fig2.suptitle("Visitor Impact by Shock: % Change from Pre-Shock Baseline",
              fontweight="bold", fontsize=13)
fig2.savefig(OUT_DIR / "eda_02_shock_sensitivity.png", dpi=150, bbox_inches="tight",
             facecolor="white")
plt.close(fig2)
print("eda_02_shock_sensitivity.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# EDA Figure 3 — Recovery Speed (2-panel: COVID + 2016 Coup)
# ═══════════════════════════════════════════════════════════════════════════════

rec_covid = {c: years_to_recover(df, c, 2020, [2018, 2019]) for c in all_countries}
rec_coup  = {c: years_to_recover(df, c, 2016, [2014, 2015]) for c in all_countries}

# Fixed ordering: sort by COVID recovery years (NaN = largest), consistent across both panels
covid_order = sorted(
    all_countries,
    key=lambda c: (rec_covid[c] is None or np.isnan(rec_covid[c]),
                   rec_covid[c] if (rec_covid[c] is not None and not np.isnan(rec_covid[c])) else 999)
)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 9), facecolor="white",
                                   gridspec_kw={"wspace": 0.45})

MAX_YEARS = 6  # right edge for "not recovered" bars

for ax, rec_data, title in [
    (ax3a, rec_covid, "COVID-19 (shock: 2020, pre: 2018-19 mean)"),
    (ax3b, rec_coup,  "2016 Coup (shock: 2016, pre: 2014-15 mean)"),
]:
    style_ax(ax)

    for i, country in enumerate(covid_order):
        group = group_map[country]
        color = GROUP_COLORS[group]
        val   = rec_data.get(country)
        not_recovered = val is None or np.isnan(val)

        bar_val = MAX_YEARS if not_recovered else float(val)
        bar_color = "#BDBDBD" if not_recovered else color

        ax.barh(i, bar_val, color=bar_color, edgecolor="white", lw=0.5,
                height=0.65, alpha=0.90 if not not_recovered else 0.60)

        if not_recovered:
            ax.text(MAX_YEARS - 0.1, i, "Not recovered",
                    va="center", ha="right", fontsize=7.5, color="#555555",
                    fontstyle="italic")
        else:
            ax.text(bar_val + 0.08, i, f"{int(val)}y",
                    va="center", ha="left", fontsize=8, fontweight="bold",
                    color=color)

    for yr in [1, 2, 3, 4, 5]:
        ax.axvline(yr, color="#999999", lw=0.8, ls="--", alpha=0.6, zorder=0)

    ax.set_yticks(range(len(covid_order)))
    ax.set_yticklabels(covid_order, fontsize=8)
    ax.set_xlim(0, MAX_YEARS + 0.3)
    ax.set_xlabel("Years to recover", fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=10)

# Shared group legend + "Not recovered" explanation
legend_handles = group_legend_patches() + [
    mpatches.Patch(color="#BDBDBD", alpha=0.60, label="Not recovered by 2025")
]
fig3.legend(handles=legend_handles,
            loc="lower center", bbox_to_anchor=(0.5, -0.03),
            ncol=3, fontsize=9, framealpha=0.9)

fig3.suptitle("Recovery Speed by Market After Major Shocks",
              fontweight="bold", fontsize=13)
fig3.savefig(OUT_DIR / "eda_03_recovery_speed.png", dpi=150, bbox_inches="tight",
             facecolor="white")
plt.close(fig3)
print("eda_03_recovery_speed.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# EDA Figure 4 — Turkey Political Stability (single panel, all crisis bands)
# ═══════════════════════════════════════════════════════════════════════════════

# pol_stability in the panel refers to the *source country* column; Turkey's
# political stability is stored as tur_political_stability (if present) or
# we use Turkey rows directly if the column exists.
# Check what Turkey-specific columns are available.
tur_cols = [c for c in df.columns if c.startswith("tur_")]
pol_col  = "tur_political_stability" if "tur_political_stability" in df.columns else None

if pol_col is None:
    # Fall back: source-country pol_stability for rows where country == 'Turkey'
    # (won't exist in panel, so try the column that 00_per_country_eda uses)
    # The panel has pol_stability per source country — not Turkey's own PSI.
    # Use political_stability from a Turkey marker if available, otherwise skip.
    pass

# Build Turkey time series from panel (Turkey's own macro cols are duplicated per row)
tur_year_df = (df[["year"] + tur_cols]
               .drop_duplicates(subset="year")
               .sort_values("year"))

if pol_col and pol_col in tur_year_df.columns:
    pol_series = tur_year_df[["year", pol_col]].dropna()
else:
    # pol_stability for Turkey: use rows where the source country's political
    # stability IS Turkey's own — this won't exist in the panel.
    # Instead derive from the panel's tur_ prefix columns.
    # If no tur_political_stability exists, note it and skip this figure gracefully.
    pol_series = pd.DataFrame(columns=["year", "pol"])

fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor="white")
style_ax(ax4)
add_crisis_bands(ax4)

if not pol_series.empty:
    y_col = pol_series.columns[1]
    ax4.plot(pol_series["year"], pol_series[y_col],
             color="#C62828", lw=2.2, marker="o", ms=6,
             mec="white", mew=1.0, zorder=3)

    # Annotate the 2 lowest values
    sorted_pol = pol_series.nsmallest(2, y_col)
    for _, row in sorted_pol.iterrows():
        ax4.annotate(f"{int(row['year'])}: {row[y_col]:.2f}",
                     (row["year"], row[y_col]),
                     xytext=(8, -14), textcoords="offset points",
                     fontsize=9, fontweight="bold", color="#C62828",
                     arrowprops=dict(arrowstyle="-", color="#C62828", lw=0.8))

    ax4.set_xticks(range(2003, 2026, 2))
    ax4.set_xlim(2002.5, 2025.5)
    ax4.set_ylabel("WGI Political Stability Index (higher = more stable)", fontsize=10)
    ax4.set_xlabel("Year", fontsize=10)
else:
    # Draw from source-country pol_stability for one proxy country if no tur_ column
    # In the panel, there is no Turkey source-country row — so we produce a blank placeholder
    ax4.text(0.5, 0.5, "tur_political_stability not in panel_dataset.csv\n"
             "Add this column in 01_data_pipeline.py",
             ha="center", va="center", transform=ax4.transAxes,
             fontsize=12, color="#888888")
    ax4.set_xticks(range(2003, 2026, 2))
    ax4.set_xlim(2002.5, 2025.5)

ax4.legend(handles=crisis_legend_patches(),
           loc="lower center", bbox_to_anchor=(0.5, -0.22),
           ncol=5, fontsize=9, framealpha=0.9)

ax4.set_title("Turkey Political Stability Index (2003-2025) — Pre-ML Context",
              fontweight="bold", fontsize=12)
fig4.tight_layout()
fig4.savefig(OUT_DIR / "eda_04_political_stability.png", dpi=150, bbox_inches="tight",
             facecolor="white")
plt.close(fig4)
print("eda_04_political_stability.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# EDA Figure 5 — Market Resilience Heatmap (country x year)
# ═══════════════════════════════════════════════════════════════════════════════

HEATMAP_YEARS = list(range(2010, 2026))

# Baseline = mean visitors 2017-19 per country
baseline_17_19 = (df[df["year"].between(2017, 2019)]
                  .groupby("country")["visitors"].mean()
                  .rename("baseline"))

# Build a wide matrix
heatmap_rows = []
for country in COUNTRY_ORDER:
    if country not in baseline_17_19.index:
        continue
    base = baseline_17_19[country]
    if base == 0 or np.isnan(base):
        continue
    cdf = df[df["country"] == country].set_index("year")["visitors"]
    row = {}
    for yr in HEATMAP_YEARS:
        if yr in cdf.index and not np.isnan(cdf[yr]):
            row[yr] = (cdf[yr] / base - 1) * 100
        else:
            row[yr] = np.nan
    heatmap_rows.append((country, row))

heatmap_countries = [r[0] for r in heatmap_rows]
heatmap_matrix    = np.array([[r[1].get(yr, np.nan) for yr in HEATMAP_YEARS]
                               for r in heatmap_rows], dtype=float)

# Cap display values at +150 (UAE explodes to ~250+ post-2023)
VMAX = 150
heatmap_display = np.clip(heatmap_matrix, -100, VMAX)

fig5, ax5 = plt.subplots(figsize=(16, 8), facecolor="white")

im = ax5.imshow(heatmap_display, aspect="auto", cmap="RdYlGn",
                vmin=-100, vmax=VMAX,
                extent=[HEATMAP_YEARS[0] - 0.5, HEATMAP_YEARS[-1] + 0.5,
                        len(heatmap_countries) - 0.5, -0.5])

# Cell text annotations
for row_i, country in enumerate(heatmap_countries):
    for col_j, yr in enumerate(HEATMAP_YEARS):
        val = heatmap_matrix[row_i, col_j]
        if np.isnan(val):
            continue
        display_val = heatmap_display[row_i, col_j]
        # White text on dark cells, black on light
        text_color = "white" if abs(display_val) > 55 else "black"
        label = f"{val:+.0f}%" if abs(val) <= VMAX else f"+{VMAX}+%"
        ax5.text(yr, row_i, label, ha="center", va="center",
                 fontsize=6.5, color=text_color, fontweight="normal")

# Vertical bold dividers at shock onset years
shock_dividers = [
    (2011, "Syria\nonset"),
    (2016, "Coup"),
    (2020, "COVID"),
    (2022, "RU-UKR"),
    (2023, "MENA\ntensions"),
]
for yr, label in shock_dividers:
    if yr in HEATMAP_YEARS:
        ax5.axvline(yr - 0.5, color="black", lw=2.0, zorder=5)
        ax5.text(yr - 0.5, -0.8, label, ha="center", va="bottom",
                 fontsize=7, fontweight="bold", color="black",
                 rotation=0)

# Horizontal thin dividers between market groups (based on COUNTRY_ORDER)
group_boundaries = []
current_group    = None
for i, country in enumerate(heatmap_countries):
    g = group_map.get(country)
    if g != current_group and current_group is not None:
        group_boundaries.append(i)
    current_group = g

for b in group_boundaries:
    ax5.axhline(b - 0.5, color="#333333", lw=1.0, zorder=4)

# Y-axis: country names colored by group
ax5.set_yticks(range(len(heatmap_countries)))
ax5.set_yticklabels(heatmap_countries, fontsize=9)
for tick, country in zip(ax5.get_yticklabels(), heatmap_countries):
    tick.set_color(GROUP_COLORS[group_map[country]])

# X-axis: integer years
ax5.set_xticks(HEATMAP_YEARS)
ax5.set_xticklabels(HEATMAP_YEARS, fontsize=9, rotation=45, ha="right")

# Colorbar
cbar = fig5.colorbar(im, ax=ax5, fraction=0.025, pad=0.02)
cbar.set_label("% vs 2017-19 Baseline  (capped at +150%)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax5.set_title("Market Resilience Heatmap: % vs 2017-19 Baseline",
              fontweight="bold", fontsize=13, pad=10)

fig5.tight_layout()
fig5.savefig(OUT_DIR / "eda_05_resilience_heatmap.png", dpi=150, bbox_inches="tight",
             facecolor="white")
plt.close(fig5)
print("eda_05_resilience_heatmap.png saved")

print(f"\nAll 5 EDA perspective figures written to {OUT_DIR}")
