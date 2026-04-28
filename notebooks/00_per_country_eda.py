"""
00_per_country_eda.py — Stage 1 diagnostic figures (do NOT modify other scripts).

Outputs:
  images/per_country/<country_slug>/overview.png  (one per country, 17 total)
  images/per_country/_grid_visitors.png
  images/per_country/_turkey_macro.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
DATA_PATH   = REPO_ROOT / "data" / "panel_dataset.csv"
OUT_DIR     = REPO_ROOT / "images" / "per_country"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Group colors (must match 02_eda_and_hypothesis.py) ──────────────────────

GROUP_COLORS = {
    "Western Europe": "#2196F3",
    "Eastern Europe": "#4CAF50",
    "Former Soviet":  "#FF5722",
    "MENA":           "#9C27B0",
    "Other":          "#607D8B",
}

# ── Crisis bands ─────────────────────────────────────────────────────────────

CRISES = [
    {"label": "Syria conflict",          "start": 2011,    "end": 2015,    "color": "#E53935"},
    {"label": "2016 coup",               "start": 2015.7,  "end": 2016.3,  "color": "#FB8C00"},
    {"label": "COVID-19",                "start": 2020,    "end": 2021,    "color": "#1E88E5"},
    {"label": "Russia–Ukraine war",      "start": 2022,    "end": 2023,    "color": "#FDD835"},
    {"label": "Post-2023 MENA tensions", "start": 2023,    "end": 2025,    "color": "#8E24AA"},
]

# Crisis-edge marker years per crisis (visitors panel only); skip 2016 coup (narrow)
CRISIS_EDGE_YEARS = {
    "Syria conflict":          [2011, 2015],
    "COVID-19":                [2020, 2021],
    "Russia–Ukraine war":      [2022, 2023],
    "Post-2023 MENA tensions": [2023, 2025],
}

# ── Country ordering for grid ────────────────────────────────────────────────

COUNTRY_ORDER = [
    # Western Europe
    "France", "Germany", "Netherlands", "United Kingdom",
    # Eastern Europe
    "Bulgaria", "Greece", "Ukraine",
    # Former Soviet
    "Azerbaijan", "Georgia", "Russia",
    # MENA
    "Iran", "Iraq", "Israel", "Syria", "United Arab Emirates", "Qatar",
    # Other
    "USA",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def add_crisis_bands(axes):
    """Draw all crisis axvspan bands on a list (or single) Axes."""
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        for c in CRISES:
            ax.axvspan(c["start"], c["end"], alpha=0.28, color=c["color"],
                       zorder=0, linewidth=0)


def format_millions(x, _=None):
    if pd.isna(x):
        return "N/A"
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return str(int(x))


def crisis_legend_patches():
    return [mpatches.Patch(color=c["color"], alpha=0.55, label=c["label"])
            for c in CRISES]


def style_ax(ax, zero_line=False):
    """Apply shared panel styling."""
    ax.set_facecolor("#fafafa")
    ax.grid(axis="both", color="#cccccc", alpha=0.4, zorder=-1, linewidth=0.7)
    ax.tick_params(labelsize=8)
    if zero_line:
        ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=1)


def slug(country_name):
    return country_name.lower().replace(" ", "-")


def format_stat(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return f"{val:,.0f}"


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)

# Build country → group mapping
group_map = (df[["country", "market_group"]]
             .drop_duplicates()
             .set_index("country")["market_group"]
             .to_dict())

countries = sorted(df["country"].unique())
assert len(countries) == 17, f"Expected 17 countries, got {len(countries)}: {countries}"


# ── Per-country overview pages ────────────────────────────────────────────────

for country in countries:
    cdf = df[df["country"] == country].sort_values("year").copy()
    group = group_map[country]
    color = GROUP_COLORS[group]

    fig, axes = plt.subplots(5, 1, sharex=True,
                             figsize=(11, 16),
                             gridspec_kw={"hspace": 0.35},
                             facecolor="white")

    panels = [
        ("Visitors",            "visitors",            "Visitors (annual)",        False),
        ("GDP growth",          "gdp_growth",          "GDP growth (% YoY)",       True),
        ("GDP per capita",      "gdp_per_capita",      "GDP per capita (USD)",     False),
        ("Inflation",           "inflation",           "Inflation (% CPI)",        False),
        ("Political stability", "political_stability", "WGI Political Stability",  True),
    ]

    add_crisis_bands(axes)

    for ax, (panel_name, col, ylabel, zero_line) in zip(axes, panels):
        style_ax(ax, zero_line=zero_line)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(panel_name, fontsize=10, fontweight="bold", loc="left")

        sub = cdf[["year", col]].dropna()
        if not sub.empty:
            ax.plot(sub["year"], sub[col],
                    color=color, lw=2.0, marker="o", ms=5,
                    mec="white", mew=0.8, zorder=3)

        # Crisis-edge markers on visitors panel only
        if panel_name == "Visitors":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: format_millions(x)))
            for crisis_name, edge_years in CRISIS_EDGE_YEARS.items():
                c_color = next(c["color"] for c in CRISES if c["label"] == crisis_name)
                for ey in edge_years:
                    row = sub[sub["year"] == ey]
                    if not row.empty:
                        ax.plot(row["year"].values, row[col].values,
                                marker="o", ms=10, mfc="none",
                                mec=c_color, mew=2, zorder=4, linestyle="none")
        elif panel_name == "GDP per capita":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # X-axis ticks on bottom panel only (sharex=True)
    axes[-1].set_xlabel("Year", fontsize=9)
    axes[-1].set_xticks(range(2003, 2026, 2))
    axes[-1].set_xlim(2002.5, 2025.5)

    # Suptitle + subtitle
    fig.suptitle(f"{country} — Visitors and Macro Context (2003–2025)",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.978,
             "Source: TÜİK (visitors), IMF DataMapper (GDP, inflation), "
             "World Bank Data360 (political stability) | Crisis windows shaded",
             fontsize=9, color="#555555", ha="center")

    # Shared legend
    fig.legend(handles=crisis_legend_patches(),
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               ncol=5, fontsize=9, framealpha=0.9)

    # Save
    out_path = OUT_DIR / slug(country)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "overview.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # Console summary
    vis = cdf["visitors"].dropna()
    min_idx = vis.idxmin() if not vis.empty else None
    max_idx = vis.idxmax() if not vis.empty else None
    min_year = int(cdf.loc[min_idx, "year"]) if min_idx is not None else "N/A"
    max_year = int(cdf.loc[max_idx, "year"]) if max_idx is not None else "N/A"
    min_val  = cdf.loc[min_idx, "visitors"] if min_idx is not None else None
    max_val  = cdf.loc[max_idx, "visitors"] if max_idx is not None else None

    v2019 = cdf.loc[cdf["year"] == 2019, "visitors"].values
    v2020 = cdf.loc[cdf["year"] == 2020, "visitors"].values
    v2024 = cdf.loc[cdf["year"] == 2024, "visitors"].values

    nan_cols = [c for c in ["gdp_growth", "gdp_per_capita", "inflation", "political_stability"]
                if cdf[c].isna().any()]

    print(
        f"{country} | "
        f"visitors mean={vis.mean():,.0f} | "
        f"min={min_year}:{format_stat(min_val)} | "
        f"max={max_year}:{format_stat(max_val)} | "
        f"2019={format_stat(v2019[0] if len(v2019) else None)} | "
        f"2020={format_stat(v2020[0] if len(v2020) else None)} | "
        f"2024={format_stat(v2024[0] if len(v2024) else None)} | "
        f"macro NaN cols: {nan_cols}"
    )


# ── Turkey macro page ─────────────────────────────────────────────────────────

turkey_cols = ["year", "tur_gdp_growth", "tur_gdp_per_capita", "tur_inflation", "tur_ppp_rate"]
tdf = (df[turkey_cols]
       .drop_duplicates(subset="year")
       .sort_values("year"))

# Turkey NaN report
for col in ["tur_gdp_growth", "tur_gdp_per_capita", "tur_inflation", "tur_ppp_rate"]:
    nan_years = tdf.loc[tdf[col].isna(), "year"].tolist()
    print(f"Turkey macro | {col} NaN years: {nan_years}")

fig, axes = plt.subplots(4, 1, sharex=True,
                         figsize=(11, 13),
                         gridspec_kw={"hspace": 0.35},
                         facecolor="white")

tur_panels = [
    ("Turkey GDP growth",     "tur_gdp_growth",     "Turkey GDP growth (% YoY)",              True),
    ("Turkey GDP per capita", "tur_gdp_per_capita", "Turkey GDP per capita (USD)",            False),
    ("Turkey inflation",      "tur_inflation",      "Turkey inflation (% CPI)",               False),
    ("Turkey PPP rate",       "tur_ppp_rate",       "Turkey PPP rate (higher = weaker lira)", False),
]

TUR_COLOR = "#C62828"
add_crisis_bands(axes)

for ax, (panel_name, col, ylabel, zero_line) in zip(axes, tur_panels):
    style_ax(ax, zero_line=zero_line)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(panel_name, fontsize=10, fontweight="bold", loc="left")
    sub = tdf[["year", col]].dropna()
    if not sub.empty:
        ax.plot(sub["year"], sub[col],
                color=TUR_COLOR, lw=2.0, marker="o", ms=5,
                mec="white", mew=0.8, zorder=3)

axes[-1].set_xlabel("Year", fontsize=9)
axes[-1].set_xticks(range(2003, 2026, 2))
axes[-1].set_xlim(2002.5, 2025.5)

fig.suptitle("Turkey — Macroeconomic Indicators (2003–2025)",
             fontsize=15, fontweight="bold", y=0.995)
fig.text(0.5, 0.978,
         "Source: IMF DataMapper | Crisis windows shaded",
         fontsize=9, color="#555555", ha="center")
fig.legend(handles=crisis_legend_patches(),
           loc="lower center", bbox_to_anchor=(0.5, -0.005),
           ncol=5, fontsize=9, framealpha=0.9)

fig.savefig(OUT_DIR / "_turkey_macro.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)


# ── Grid visitors page ────────────────────────────────────────────────────────

fig, axes = plt.subplots(5, 4, figsize=(20, 20),
                         facecolor="white",
                         gridspec_kw={"hspace": 0.45, "wspace": 0.3})
axes_flat = axes.flatten()

for i, country in enumerate(COUNTRY_ORDER):
    ax = axes_flat[i]
    cdf = df[df["country"] == country].sort_values("year")
    group = group_map[country]
    color = GROUP_COLORS[group]

    add_crisis_bands(ax)
    style_ax(ax)

    sub = cdf[["year", "visitors"]].dropna()
    if not sub.empty:
        ax.plot(sub["year"], sub["visitors"],
                color=color, lw=1.8, marker="o", ms=4,
                mec="white", mew=0.6, zorder=3)

    ax.set_title(country, fontsize=11, fontweight="bold")
    ax.set_xticks(range(2003, 2026, 4))
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: format_millions(x)))
    ax.set_xlim(2002.5, 2025.5)

    # Mean annotation
    mean_val = sub["visitors"].mean() if not sub.empty else np.nan
    ax.text(0.97, 0.95, f"μ: {format_millions(mean_val)}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))

# Hide the 3 unused cells (17 countries in a 5×4 = 20-cell grid)
for j in range(len(COUNTRY_ORDER), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("Per-Country Visitor Trajectories with Crisis Windows (2003–2025)",
             fontsize=15, fontweight="bold", y=1.002)
fig.legend(handles=crisis_legend_patches(),
           loc="lower center", bbox_to_anchor=(0.5, -0.012),
           ncol=5, fontsize=10, framealpha=0.9)

fig.savefig(OUT_DIR / "_grid_visitors.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

print(f"\nWrote 17 country overviews + _grid_visitors.png + _turkey_macro.png to {OUT_DIR}")
