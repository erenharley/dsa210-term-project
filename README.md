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

### 5. Run ML models (Phase 4)

```bash
python notebooks/04_ml_models.py
```

Reads `data/panel_dataset.csv`, applies imputation, and produces `images/ml_01_*.png` … `images/ml_06_*.png`. Prints all metrics to stdout.

### 6. Generate perspective EDA figures

```bash
python notebooks/03_perspective_eda.py
```

Writes `images/eda_01_*.png` … `images/eda_05_*.png` (trends by group, shock sensitivity, recovery speed, political stability, resilience heatmap).

### 7. (Optional) Open the interactive notebook

```bash
jupyter lab notebooks/project_analysis.ipynb
```

---

## Key Findings (Phase 3, post-redesign)

Results are reported from actual tests run on `data/panel_dataset.csv`. Only
statistically significant findings are stated as findings; non-significant
results are reported as descriptive observations.

**Statistically significant (α = 0.05):**
- Turkey lira weakness (YoY % change in PPP rate) correlates with larger visitor
  drops during shock years — Pearson r = −0.80, p < 0.001 (H4). Note: the
  correlation is driven by only 3 distinct shock years (2009, 2016, 2020) — it
  reflects year-level differences, not a continuous lira-visitor relationship.

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
- Aggregate Spearman correlation between source country GDP per capita and visitor
  volume is not significant (Spearman r = −0.07, p = 0.17; H5), though within-group
  patterns diverge strongly: Former Soviet r = +0.72 (p < 0.001), MENA r = −0.62
  (p < 0.001), suggesting a Simpson's paradox — wealthier MENA markets send fewer
  tourists (conflict suppression), while wealthier Former Soviet markets send more.

---

## Phase 4 — ML Methods

Script: `python notebooks/04_ml_models.py`

Outputs: `images/ml_01_*.png` … `images/ml_06_*.png`

**Section 1 — Linear Regression (log_visitors; temporal split train <= 2019, test 2020-2025)**
Train R² = 0.705, Test R² = -3.24. The model fits pre-COVID visitor patterns reasonably well
(~70% variance explained in training) but fails to generalize to 2020-2025: COVID, the
Russia-Ukraine war, and rising MENA tensions are structurally unlike anything in the training
window. This is the expected and honest outcome of the temporal split — do not tune to improve
test R². Residual plot (`ml_01`) and predicted-vs-actual (`ml_02`) show systematic
under-prediction during COVID and over-prediction in the recovery. Key finding: shock dummies
whose events fall after 2019 (`covid`, `russia_ukraine_war`, `mena_tension_recent`) receive
near-zero coefficients because they have zero variance in training — a model trained before
a shock cannot learn that shock's coefficient.

**Section 2 — Classification: COVID-resilient vs non-resilient (LOOCV, n=17)**
Label: 1 if 2022 visitors >= 2019 visitors (9 resilient / 8 non-resilient). Features use
pre-2020 data only (2015-2019 macro averages, log baseline visitors, market group, MENA flags).
Logistic Regression LOOCV accuracy = 0.65; Random Forest = 0.71. Feature importance
(`ml_03`) shows log baseline market size and political stability as most informative.
ROC AUC is approximate given n=17 LOOCV points. All results are descriptive/exploratory
— n=17 is too small to claim statistical superiority of one model over another.

**Section 3 — Clustering: k-means (k=3) + hierarchical dendrogram**
Per-country feature vector: 2016 coup drop, 2020 COVID drop, 2023-25 % vs 2017-19 baseline,
post-COVID recovery years (capped at 5), log mean baseline visitors. Elbow plot (`ml_04`)
supports k=3. Cluster 0 (UAE, USA): fast recovery, strong 2023-25 gains. Cluster 1 (Western
Europe, Iran, Israel, Ukraine, Bulgaria): moderate COVID drop, recovered by 2022, moderate
post-2023 gains. Cluster 2 (Former Soviet, Greece, Iraq, Qatar, Syria): high coup sensitivity,
slow COVID recovery (mean 4.7 years), subdued 2023-25 performance. Critically, UAE separates
from conflict-affected MENA markets (Iraq, Syria, Qatar) — the hierarchical dendrogram
(`ml_05`) confirms this split. This aligns with the H6 within-MENA descriptive evidence:
Gulf-stable markets have a categorically different shock-response profile.

**Section 4 — Scenario 2026 (extending Section 1)**
Holds each country's macro features at their 2025 values with `mena_tension_recent=1`
and all other shocks off. Because `mena_tension_recent` has a near-zero coefficient (zero
variance in training), this scenario is equivalent to asking: "what does the pre-2020 model
expect from 2025 macro conditions?" The predictions are driven by the GDP per capita and
market group coefficients, not by the MENA tension variable — an honest limitation of
the temporal split. Bar chart (`ml_06`) shows the structural gap between the model's
pre-COVID expectations and 2025 actual visitor levels. The scenario confirms the linear
regression cannot serve as a MENA-tension forecaster without post-2023 training data.

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
