# DSA 210 — Tourism Demand Under Shock

**Author:** Eren Sean Harley | 36054
**Course:** DSA 210 Introduction to Data Science, Sabancı University, Spring 2026

---

## Project Goal

Analyze how different types of shocks (political, regional conflict, global pandemic) affect tourism demand in Turkey across 17 source markets (2003–2025). Core questions:

1. **Market sensitivity** — which markets over/underreact to each shock type?
2. **Recovery rate** — how fast does each market bounce back, and does it differ by shock type?
3. **Actionable output** — which markets should tourism firms target given rising MENA tensions?

---

## Repository Structure

```
dsa210-term-project/
├── data/
│   ├── Çıkış Yapan Yabancı ve Vatandaşlar.xls   # TÜİK visitor counts by nationality (2003-2025)
│   ├── imf_macro_data.csv                         # IMF: GDP growth, inflation, GDP/capita (17 markets)
│   ├── imf_turkey_data.csv                        # IMF: Turkey macro + PPP exchange rate
│   ├── wb_political_stability.csv                 # World Bank WGI: Political Stability Index
│   ├── panel_dataset.csv                          # Final merged panel (17 markets × 23 years)
│   └── fetch_macro_data.py                        # One-time script that fetched IMF + WB data
├── notebooks/
│   ├── 00_per_country_eda.py                      # Per-country diagnostic figures
│   ├── 01_data_pipeline.py                        # Merges all sources → data/panel_dataset.csv
│   ├── 02_eda_and_hypothesis.py                   # 6 hypothesis tests (H1-H6) with finding-driven figures
│   ├── 03_perspective_eda.py                      # 5 perspective EDA figures (trends, shocks, heatmap)
│   └── project_analysis.ipynb                     # Interactive notebook
├── images/
│   ├── h1_coup_sensitivity.png                    # H1: 2016 coup sensitivity by market group
│   ├── h2_syria_proximity.png                     # H2: Syria-bordering vs non-bordering
│   ├── h3_covid_recovery.png                      # H3: COVID recovery speed by group
│   ├── h4_lira_weakness.png                       # H4: Lira weakness and visitor drops
│   ├── h5_gdp_visitors.png                        # H5: GDP per capita and visitor volume
│   ├── h6_mena_tension_recent.png                 # H6: Post-2023 MENA tensions
│   ├── eda_01_trends_by_group.png                 # Visitor trends 2003-2025 by group
│   ├── eda_02_shock_sensitivity.png               # % impact by shock and country
│   ├── eda_03_recovery_speed.png                  # Recovery speed: COVID + coup
│   ├── eda_04_political_stability.png             # Turkey political stability index
│   ├── eda_05_resilience_heatmap.png              # Country × year resilience heatmap
│   └── per_country/
│       ├── _grid_visitors.png                     # 17-country visitor grid
│       ├── _turkey_macro.png                      # Turkey macro context
│       └── <country>/overview.png                 # Per-country diagnostic (17 subfolders)
├── DSA210_Project_Proposal.pdf
├── requirements.txt
└── README.md
```

---

## Data Sources

| Dataset | Source | API / URL |
|---|---|---|
| Visitor counts by nationality | TÜİK (Turkish Statistical Institute) | Manual download |
| GDP growth, inflation, GDP per capita | IMF World Economic Outlook DataMapper | `https://www.imf.org/external/datamapper/api/v1/` |
| Turkey macro + PPP exchange rate | IMF DataMapper | `https://www.imf.org/external/datamapper/api/v1/` |
| Political Stability Index (WGI) | World Bank Data360 | `https://data360api.worldbank.org/data360/data` |

---

## How to Reproduce

### 1. Clone the repo and set up the environment

```bash
git clone https://github.com/erenharley/dsa210-term-project.git
cd dsa210-term-project
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the panel dataset

```bash
python notebooks/01_data_pipeline.py
```

This reads from `data/` and writes the merged panel to `data/panel_dataset.csv`.

### 3. Generate per-country diagnostic figures

```bash
python notebooks/00_per_country_eda.py
```

Writes `images/per_country/<country>/overview.png` for all 17 markets, plus `_grid_visitors.png` and `_turkey_macro.png`.

### 4. Run hypothesis tests

```bash
python notebooks/02_eda_and_hypothesis.py
```

Reads `data/panel_dataset.csv`, runs H1–H6, writes `images/h1_*.png` … `images/h6_*.png`, and prints all test results to stdout.

### 5. Generate perspective EDA figures

```bash
python notebooks/03_perspective_eda.py
```

Writes `images/eda_01_*.png` … `images/eda_05_*.png` (trends by group, shock sensitivity, recovery speed, political stability, resilience heatmap).

### 6. (Optional) Open the interactive notebook

```bash
jupyter lab notebooks/project_analysis.ipynb
```

---

## Key Findings (Phase 3, post-redesign)

Results are reported from actual tests run on `data/panel_dataset.csv`. Only
statistically significant findings are stated as findings; non-significant
results are reported as descriptive observations.

**Statistically significant (α = 0.05):**
- Turkey lira weakness (YoY % change in PPP rate) correlates strongly with *larger*
  visitor drops during shock years — Pearson r = −0.80, p < 0.001 (H4). The crisis
  and instability effect dominates the naive "cheap Turkey = more tourists" price
  signal. Note: correlation is driven by only 3 distinct shock years (2009, 2016,
  2020) — reflects year-level differences, not a continuous lira-visitor relationship.
- Source country GDP per capita shows a significant aggregate Spearman correlation
  with visitor volume (H5), but within-group directions diverge (MENA: r = −0.62;
  Former Soviet: r = +0.72), suggesting a potential Simpson's paradox.

**Descriptive (not statistically significant at α = 0.05):**
- Market groups did not differ significantly in 2016 coup sensitivity
  (ANOVA F = 0.75, p = 0.54; Kruskal-Wallis H = 3.24, p = 0.36, H1).
  Western Europe showed the largest mean drop (−32%), but high within-group
  variance prevents a formal directional claim.
- Former Soviet markets recovered most slowly post-COVID (mean 4.0 years vs
  2.0 for Western Europe), but recovery speed differences across groups were not
  significant (ANOVA F = 2.08, p = 0.18, H3). Georgia, Greece, Iraq, and Qatar
  had not recovered by 2025.
- MENA markets averaged +21% above their 2017–19 baseline in 2023–25, while
  non-MENA averaged +38%, but the difference was not significant
  (independent t = −0.48, p = 0.64, H6; n = 6 MENA vs n = 11 non-MENA).
  Within MENA, conflict-affected markets (Iran, Iraq, Israel, Syria) averaged
  +1.4% while stable Gulf markets (UAE, Qatar) averaged +60%, suggesting the
  aggregate MENA result is driven by home-country conditions rather than
  regional spillover.
- Iraq visitor counts grew +160% during the Syria war period (2011–15 vs 2008–10
  baseline), which masked any aggregate proximity suppression effect in H2
  (independent t = −0.21, p = 0.84; Syria-bordering mean +37% vs non-bordering +48%).

---

## Shock Events Modelled

| Event | Years | Dummy variable |
|---|---|---|
| Syrian Civil War onset | 2011–2015 | `syria_conflict` |
| Russia-Turkey jet crisis & tourism ban | 2015 | `russia_turkey_crisis` |
| Turkey coup attempt | 2016 | `coup_2016` |
| COVID-19 pandemic | 2020–2021 | `covid` |
| Russia-Ukraine war | 2022+ | `russia_ukraine_war` |
| Rising MENA tensions | 2023+ | `mena_tension_recent` |

---

## AI Assistance Disclosure

This project used Claude (Anthropic) for code review, data pipeline structuring, and repository cleanup, in accordance with the Sabancı University DSA 210 academic integrity policy requiring explicit disclosure of AI assistance.
