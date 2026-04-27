# DSA 210 — Tourism Demand Under Shock

**Author:** Eren Sean Harley | 36054
**Course:** DSA 210 Introduction to Data Science, Sabancı University, Spring 2026

---

## Project Goal

Analyze how different types of shocks (political, regional conflict, global pandemic) affect tourism demand in Turkey across 15 source markets (2003–2025). Core questions:

1. **Market sensitivity** — which markets over/underreact to each shock type?
2. **Recovery rate** — how fast does each market bounce back, and does it differ by shock type?
3. **Actionable output** — which markets should tourism firms target given rising MENA tensions?

---

## Repository Structure

```
dsa210-term-project/
├── data/
│   ├── Çıkış Yapan Yabancı ve Vatandaşlar.xls   # TÜİK visitor counts by nationality (2003-2025)
│   ├── imf_macro_data.csv                         # IMF: GDP growth, inflation, GDP/capita (15 markets)
│   ├── imf_turkey_data.csv                        # IMF: Turkey macro + PPP exchange rate
│   ├── wb_political_stability.csv                 # World Bank WGI: Political Stability Index
│   ├── panel_dataset.csv                          # Final merged panel (345 obs × 24 cols)
│   └── fetch_macro_data.py                        # One-time script that fetched IMF + WB data
├── notebooks/
│   ├── 01_data_pipeline.py                        # Merges all sources → data/panel_dataset.csv
│   ├── 02_eda_and_hypothesis.py                   # EDA figures + 6 hypothesis tests (H1-H6)
│   └── project_analysis.ipynb                     # Interactive notebook
├── images/                                        # Output figures (fig1–fig6)
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

### 3. Run EDA and hypothesis tests

```bash
python notebooks/02_eda_and_hypothesis.py
```

This reads `data/panel_dataset.csv`, produces figures in `images/`, and prints hypothesis test results to stdout.

### 4. (Optional) Open the interactive notebook

```bash
jupyter lab notebooks/project_analysis.ipynb
```

---

## Key Findings (Phase 3, post-redesign)

Results are reported from actual tests run on `data/panel_dataset.csv`. Only
statistically significant findings are stated as findings; non-significant
results are reported as descriptive observations.

**Statistically significant (α = 0.05):**
- Turkey currency weakness (PPP rate) correlates strongly with *larger* visitor
  drops during shock years — Pearson r = −0.79, p < 0.001 (H4). The crisis and
  instability effect dominates the naive "cheap Turkey = more tourists" price effect.
- Source country GDP per capita positively predicts visitor volume —
  Spearman r = 0.22, p < 0.001 (H5). Wealthier source markets send more tourists,
  though the effect is weak.

**Descriptive (not statistically significant at α = 0.05):**
- Western Europe showed the largest mean drop in 2016 (−32%), followed by
  Former Soviet (−21%) and MENA (−16%), but group differences did not reach
  significance (ANOVA F = 0.64, p = 0.61, H1). High within-group variance prevents
  a formal claim.
- Former Soviet markets recovered most slowly post-COVID (mean 4.0 years vs
  2.0 for Western Europe), but recovery speed differences across groups were not
  significant (ANOVA F = 2.95, p = 0.11, H3).
- MENA markets averaged only +1.4% above their 2017–19 baseline in 2023–25,
  well below the non-MENA average of +37.8%, but the small MENA sample (n = 4)
  prevented significance (Welch t = −1.45, p = 0.17, H6).
- Georgia, Greece, and Iraq had not recovered to their 2017–19 baseline by 2025.
- Iraq visitor counts grew +160% during the Syria war period (2011–15 vs 2008–10
  baseline), which masked any aggregate MENA suppression effect in H2
  (Welch t = 0.41, p = 0.71).

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
