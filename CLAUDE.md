# DSA 210 — Tourism Demand Under Shock
**Author:** Eren Sean Harley | 36054  
**Course:** DSA 210 Introduction to Data Science, Sabancı University, Spring 2026  
**Assistant:** Read this file at the start of every session before doing anything.

---

## Project Goal
Analyze how different types of shocks (political, regional conflict, global pandemic) affect tourism demand in Turkey across 15 source markets (2003–2025). The core analytical questions are:
1. **Market sensitivity** — which markets over/underreact to each shock type?
2. **Recovery rate** — how fast does each market bounce back, and does it differ by shock type?
3. **Actionable output** — which markets should tourism firms target given rising MENA tensions?

This feeds into Phase 4 ML: predicting recovery trajectories and classifying market resilience.

---

## Repository Structure

```
dsa210-term-project/
├── data/
│   ├── Çıkış_Yapan_Yabancı_ve_Vatandaşlar.xls   # TÜİK visitor counts by nationality (2003-2025)
│   ├── imf_macro_data.csv                         # IMF DataMapper API: GDP growth, inflation, GDP/capita (15 source markets)
│   ├── imf_turkey_data.csv                        # IMF DataMapper API: Turkey macro + PPP exchange rate
│   ├── wb_political_stability.csv                 # World Bank WGI API: Political Stability Index (all 16 countries incl. Turkey)
│   └── fetch_macro_data.py                        # Script that fetched the IMF + WB data (run once, do not re-run)
├── notebooks/
│   ├── data_pipeline.py                           # Merges all sources → panel_dataset.csv
│   ├── eda_and_hypothesis.py                      # EDA figures + 5 hypothesis tests
│   └── panel_dataset.csv                          # Final merged panel (345 obs × 24 cols)
├── figures/                                       # All output figures
├── CLAUDE.md                                      # This file
└── README.md
```

---

## Data Sources (always cite these)
| Dataset | Source | API / URL |
|---|---|---|
| Visitor counts | TÜİK (Turkish Statistical Institute) | Manual download |
| GDP growth, inflation, GDP per capita | IMF World Economic Outlook | `https://www.imf.org/external/datamapper/api/v1/` |
| Turkey macro + PPP exchange rate | IMF DataMapper | Same API above |
| Political Stability Index | World Bank WGI via Data360 | `https://data360api.worldbank.org/data360/data` |

---

## Panel Dataset Columns (panel_dataset.csv)
| Column | Description |
|---|---|
| `country` | Source market (15 countries) |
| `year` | 2003–2025 |
| `visitors` | Annual departing foreign visitors from TÜİK |
| `gdp_growth` | Source country real GDP growth % (IMF) |
| `gdp_per_capita` | Source country GDP per capita USD (IMF) |
| `inflation` | Source country CPI inflation % (IMF) |
| `political_stability` | Source country WGI Political Stability score (WB) |
| `tur_gdp_growth` | Turkey GDP growth % (IMF) |
| `tur_gdp_per_capita` | Turkey GDP per capita USD (IMF) |
| `tur_inflation` | Turkey CPI inflation % (IMF) |
| `tur_ppp_rate` | Turkey PPP conversion rate — higher = weaker lira (IMF) |
| `tur_political_stability` | Turkey WGI Political Stability (WB) |
| `covid` | 1 if year ∈ {2020, 2021} |
| `coup_2016` | 1 if year = 2016 |
| `syria_conflict` | 1 if year ∈ {2011–2015} |
| `russia_turkey_crisis` | 1 if year = 2015 |
| `russia_ukraine_war` | 1 if year ≥ 2022 |
| `mena_tension` | 1 if year ∈ {2011–2015} or year ≥ 2023 |
| `visitors_yoy` | YoY % change in visitors |
| `baseline_visitors` | Country mean visitors 2017–2019 |
| `visitors_vs_baseline` | % vs 2017-19 baseline |
| `log_visitors` | ln(visitors) |
| `tur_currency_weakness` | YoY % change in Turkey PPP rate |
| `market_group` | Western Europe / Eastern Europe / Former Soviet / MENA |

---

## Market Groups
| Group | Countries |
|---|---|
| Western Europe | Germany, United Kingdom, France, Netherlands |
| Eastern Europe | Bulgaria, Greece, Ukraine |
| Former Soviet | Russia, Azerbaijan, Georgia |
| MENA | Iran, Iraq, Syria, Israel |
| Other | USA |

---

## Shock Events (as dummy variables)
| Event | Years | Variable |
|---|---|---|
| Syrian Civil War onset | 2011–2015 | `syria_conflict` |
| Russia-Turkey jet crisis + tourism ban | 2015 | `russia_turkey_crisis` |
| Turkey coup attempt | 2016 | `coup_2016` |
| COVID-19 pandemic | 2020–2021 | `covid` |
| Russia-Ukraine war | 2022+ | `russia_ukraine_war` |
| Rising MENA tensions | 2023+ | `mena_tension` |

---

## Phase Progress
- [x] **Phase 1** — GitHub repo created (17 March)
- [x] **Phase 2** — Proposal submitted (31 March)
- [x] **Phase 3** — Data collection, EDA, hypothesis tests (14 April)
  - Real API data: IMF DataMapper + World Bank Data360
  - 5 hypothesis tests focused on market sensitivity & recovery rates
  - Key findings:
    - Western Europe most sensitive to Turkey domestic shocks (-32% in 2016)
    - Former Soviet markets slowest to recover post-COVID (avg 4 years)
    - Turkey currency weakness correlates with larger drops (r=-0.68, p<0.001) — crisis effect dominates price effect
    - Source country GDP per capita significantly predicts visitor volume (Spearman r=0.22, p<0.001)
    - Georgia, Greece, Iraq still not recovered to 2017-19 baseline by 2025
- [ ] **Phase 4** — ML methods (due 5 May)
- [ ] **Phase 5** — Final report (due 18 May)

---

## Phase 4 Plan (next)
Apply ML methods to:
1. **Regression** — predict log(visitors) from macro + shock variables (panel OLS with country fixed effects)
2. **Classification** — classify markets as "resilient" vs "vulnerable" per shock type
3. **Clustering** — group markets by shock-response profile
4. **Forecasting** — project visitor recovery under current MENA tension scenario

The panel_dataset.csv is ready for ML. Use scikit-learn. All code in Python.

---

## Code Style Requirements (from course guidelines)
- All code in Python
- Well-documented with comments
- `requirements.txt` must be maintained
- README.md must have instructions to reproduce analysis
- Cite AI assistance explicitly (as per academic integrity policy)
