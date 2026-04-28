# Paste this to Claude Code (Opus 4.7) in VS Code — Stage 1 of 3

---

## Context

This is **Stage 1 of a three-stage Phase 3 redesign** (see CLAUDE.md → "Phase 3 redesign — work order"). Read CLAUDE.md first; it contains the test plan, the lecture-scope constraints, and the sequencing.

In this stage, you build only the per-country diagnostic figures. **No other files are modified.** I'll review the figures before stage 2 (the aggregate redesign + pipeline edit + new tests).

## What to build

Create the folder structure under `images/per_country/`:

```
images/per_country/
├── _grid_visitors.png          ← 4×4 grid of all 15 countries' visitor trajectories
├── _turkey_macro.png            ← single page of Turkey-side macro indicators
├── azerbaijan/
│   └── overview.png
├── bulgaria/
│   └── overview.png
├── france/
│   └── overview.png
├── georgia/
│   └── overview.png
├── germany/
│   └── overview.png
├── greece/
│   └── overview.png
├── iran/
│   └── overview.png
├── iraq/
│   └── overview.png
├── israel/
│   └── overview.png
├── netherlands/
│   └── overview.png
├── russia/
│   └── overview.png
├── syria/
│   └── overview.png
├── ukraine/
│   └── overview.png
├── united-kingdom/
│   └── overview.png
└── usa/
    └── overview.png
```

Each `overview.png` is a single tall figure with **5 vertically stacked panels** sharing the x-axis. Crisis bands span all 5 panels with `axvspan`, so the reader can see "did visitors drop during this crisis? did GDP also drop? did inflation spike?" all at one glance — vertical alignment IS the analytical insight.

Create a new script `notebooks/00_per_country_eda.py`. Do **NOT** modify `01_data_pipeline.py`, `02_eda_and_hypothesis.py`, `README.md`, or `CLAUDE.md` in this stage.

## Crisis windows (apply to all panels in every figure)

Each window is an `axvspan` (semi-transparent fill, **no outline**) drawn behind the line. Lines and markers must remain fully visible — use `alpha=0.28`, spans at `zorder=0`, lines at `zorder=3`. Use **vivid, distinct colors**.

| Window | Span (year start, year end) | Color (hex) |
|---|---|---|
| Syria conflict | (2011, 2015) | `#E53935` (vivid red) |
| 2016 coup | (2015.7, 2016.3) — narrow band centered on 2016 | `#FB8C00` (vivid orange) |
| COVID-19 | (2020, 2021) | `#1E88E5` (vivid blue) |
| Russia–Ukraine war | (2022, 2023) | `#FDD835` (vivid yellow) |
| Post-2023 MENA tensions | (2023, 2025) | `#8E24AA` (vivid purple) |

**Intentional overlap:** Russia–Ukraine (yellow) and Post-2023 MENA (purple) overlap on year 2023. With both at alpha 0.28, the overlap reads as a brownish/muted tone — that visually flags the compounded shock period. Do not try to remove the overlap.

## Per-country `overview.png` spec

**Figure:** `figsize=(11, 16)`, white facecolor, `dpi=150`, `bbox_inches='tight'`.

**Layout:** 5 stacked subplots via `plt.subplots(5, 1, sharex=True, gridspec_kw={'hspace': 0.35})`. All subplots span 2003–2025 on the x-axis.

**Panel order, top to bottom:**

| # | Panel | Y data | Y label |
|---|---|---|---|
| 1 | Visitors | `visitors` (raw counts, thousands separator on ticks) | `Visitors (annual)` |
| 2 | GDP growth | `gdp_growth` | `GDP growth (% YoY)` |
| 3 | GDP per capita | `gdp_per_capita` (USD, thousands separator) | `GDP per capita (USD)` |
| 4 | Inflation | `inflation` | `Inflation (% CPI)` |
| 5 | Political stability | `political_stability` | `WGI Political Stability` |

**Per-panel styling:**
- Line: `lw=2.0`, marker `'o'`, `ms=5`, white edge `mec='white'`, `mew=0.8`, color = a single accent color shared across all 5 panels for that country (use the country's market group color from the existing `GROUP_COLORS` dict in `02_eda_and_hypothesis.py` — read it in, don't hardcode).
- For panels 2 and 5 (GDP growth, political stability), add `axhline(0, color='#888', lw=0.8, ls='--', zorder=1)` since these are signed quantities where zero is meaningful.
- Light gray major-gridlines, alpha 0.4, `zorder=-1`.
- Pale gray axes facecolor `#fafafa`.
- Crisis bands as specified above, applied to **every panel**.
- Panel title (small, left-aligned, bold, `fontsize=10`, `loc='left'`): just the indicator name (e.g. "Visitors", "GDP growth").

**Crisis-edge markers (visitors panel only):** at the start and end years of each shaded window, draw a slightly larger hollow circle (`ms=10`, `mfc='none'`, `mew=2`, edge color matching the band color) on top of the regular line marker. Skip for the 2016 coup (narrow band — regular marker is fine). This makes pre/during/post comparison visually obvious. Don't do this for the macro panels — they would get cluttered.

**Figure title** (`fig.suptitle`): `<Country> — Visitors and Macro Context (2003–2025)`, fontsize 15, bold, `y=0.995`.
**Subtitle** below it via `fig.text(0.5, 0.978, ...)`: `Source: TÜİK (visitors), IMF DataMapper (GDP, inflation), World Bank Data360 (political stability) | Crisis windows shaded`, fontsize 9, gray, ha='center'.

**Shared legend** at the bottom of the figure (`fig.legend` with `mpatches.Patch` for each crisis window), `loc='lower center'`, `bbox_to_anchor=(0.5, -0.005)`, `ncol=5`, fontsize 9.

**X-axis (only on bottom panel since `sharex=True`):** integer year ticks every 2 years, label "Year".

**File save path:** `images/per_country/<country_slug>/overview.png` where `country_slug = country.lower().replace(' ', '-')`. Create the country folder with `Path(...).mkdir(parents=True, exist_ok=True)`.

**Missing data:** if a country has NaN in any macro column for some years, plot only the non-NaN points (line will have gaps, that's fine and accurate — don't interpolate).

## `_grid_visitors.png` spec (visitors-only overview)

A 4×4 grid showing only the visitors panel for all 15 countries side by side. Same crisis bands applied to each subplot. This complements the per-country overviews — gives one page where you see all 15 visitor trajectories at once.

- `figsize=(20, 16)`, `dpi=150`.
- 16 cells, 15 used, bottom-right empty.
- Each subplot: visitors line plot with crisis bands. No macro panels here. Compact styling: marker size 4, no crisis-edge markers (would crowd the small subplots), x-ticks every 4 years.
- In each subplot's top-right corner, annotate mean annual visitors as `μ: <formatted>` (e.g. `4.2M`, `850K`, `12K`) using `transform=ax.transAxes` at `(0.97, 0.95)`, `ha='right'`, `va='top'`, `fontsize=8`, with a white semi-transparent bbox.
- Subplot title: country name only, fontsize 11.
- **Do not share y-axis** — visitor counts span 3+ orders of magnitude; a shared scale would flatten everything.
- Country ordering by market group, then alphabetical within group:
  1. Western Europe: France, Germany, Netherlands, United Kingdom
  2. Eastern Europe: Bulgaria, Greece, Ukraine
  3. Former Soviet: Azerbaijan, Georgia, Russia
  4. MENA: Iran, Iraq, Israel, Syria
  5. Other: USA
- Figure title: `Per-Country Visitor Trajectories with Crisis Windows (2003–2025)`, fontsize 15, bold.
- One shared legend at the bottom for all 5 crisis windows.

## `_turkey_macro.png` spec (Turkey-side indicators)

A single page showing Turkey's own macroeconomic indicators with the same crisis bands. These are constant across all source countries in the panel, so plotting them once instead of repeating on every country page saves clutter.

**Figure:** `figsize=(11, 13)`, 4 stacked subplots via `plt.subplots(4, 1, sharex=True, gridspec_kw={'hspace': 0.35})`.

**Panel order:**

| # | Panel | Y data | Y label |
|---|---|---|---|
| 1 | Turkey GDP growth | `tur_gdp_growth` | `Turkey GDP growth (% YoY)` |
| 2 | Turkey GDP per capita | `tur_gdp_per_capita` | `Turkey GDP per capita (USD)` |
| 3 | Turkey inflation | `tur_inflation` | `Turkey inflation (% CPI)` |
| 4 | Turkey PPP rate (lira weakness proxy) | `tur_ppp_rate` | `Turkey PPP rate (higher = weaker lira)` |

Use a single dark accent color (e.g. `#C62828` — Turkish red) for all four lines. Same per-panel styling rules as per-country pages: bold left-aligned panel title, light grid, pale axes facecolor, axhline at zero for the GDP growth panel.

**Important — deduplicate before plotting.** Turkey-side variables are repeated across all 15 countries in the panel CSV. Pull them from one country only (or use `df[['year', 'tur_gdp_growth', 'tur_gdp_per_capita', 'tur_inflation', 'tur_ppp_rate']].drop_duplicates(subset='year').sort_values('year')`) before plotting. Otherwise you'll plot 15 overlapping identical lines.

Figure title: `Turkey — Macroeconomic Indicators (2003–2025)`, fontsize 15, bold.
Subtitle: `Source: IMF DataMapper | Crisis windows shaded`, fontsize 9, gray.
Shared legend at bottom for the 5 crisis windows.

Save to `images/per_country/_turkey_macro.png`.

## Console output

For each country, print one line:
```
<Country> | visitors mean=<X> | min=<year>:<value> | max=<year>:<value> | 2019=<value> | 2020=<value> | 2024=<value> | macro NaN cols: [<list>]
```
Format numbers with thousands separators. The `macro NaN cols` field lists any of `gdp_growth`, `gdp_per_capita`, `inflation`, `political_stability` that have one or more missing years for that country, so I can see at a glance which countries have spotty macro coverage.

Then print:
- A separate line for Turkey: `Turkey macro | tur_gdp_growth NaN years: [...] | tur_inflation NaN years: [...]` etc., for each Turkey-side column that has NaNs.
- Final summary: `Wrote 15 country overviews + _grid_visitors.png + _turkey_macro.png to images/per_country/`.

## Hard rules

- Do not regenerate `data/panel_dataset.csv`. Use whatever's currently on disk.
- Do not modify any other script, the README, or CLAUDE.md.
- Do not smooth, log-transform, or normalize any series. Plot raw values as-is.
- Do not interpolate NaN values. Let lines have gaps where data is missing.
- Crisis bands are drawn for **every** country regardless of whether the crisis is hypothesized to affect that country — this is diagnostic, the point is to *see* whether each country reacts.
- Read `GROUP_COLORS` (or define an equivalent dict at the top of the new script) so country line colors are consistent with the existing aggregate figures. Don't pick random new colors.
- If `scipy`, `matplotlib`, `pandas`, or `numpy` isn't installed, just stop and tell me — don't auto-install. (These are the only dependencies needed for this stage.)

When you're done, list the contents of `images/per_country/` recursively, paste the console summaries, and stop. I'll review the visuals before we proceed to stage 2 (aggregate redesign + new tests).
