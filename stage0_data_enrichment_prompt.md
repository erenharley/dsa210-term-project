# Paste this to Claude Code (Opus 4.7) in VS Code — Stage 0 of 4

---

## Context

This is **Stage 0 of a four-stage Phase 3 redesign** (see CLAUDE.md → "Phase 3 redesign — work order"). Read CLAUDE.md first; pay particular attention to the "Markets and groups (17 countries)" section and the "MENA enrichment rationale" subsection — they explain *why* we're adding UAE and Qatar and why Saudi Arabia was excluded.

In this stage, you extend the existing data pipeline to include **United Arab Emirates (ISO3: ARE)** and **Qatar (ISO3: QAT)** as two additional MENA-region source markets. The goal: regenerate `data/panel_dataset.csv` so it has 17 countries instead of 15, with all macro variables fetched from real APIs (no hardcoding).

**No analysis or figure code is touched in this stage.** Just the data fetching and pipeline. Stage 1 (per-country diagnostic figures) comes after I've reviewed the regenerated panel.

## What to do

### Step 1 — Inventory the current data fetch logic

Read `data/fetch_macro_data.py` and `notebooks/01_data_pipeline.py`. Identify:

- Where the current list of source countries is defined (likely a `COUNTRIES` dict or similar mapping country name → ISO3)
- Where IMF API calls are made (which endpoints, which indicators)
- Where World Bank Data360 calls are made (the `GOV_WGI_PV` indicator with `COMP_BREAKDOWN_1` filter)
- Where TÜİK visitor data is read in (the XLS file, what country names it uses)

Show me what you found before making changes — I want to confirm we're modifying the right places.

### Step 2 — Verify TÜİK has UAE and Qatar visitor data

The TÜİK XLS (`data/Çıkış Yapan Yabancı ve Vatandaşlar.xls`) lists inbound tourism by source nationality. UAE and Qatar are major Gulf source markets for Turkey, so they should be present, but the **column names need verification** because TÜİK uses Turkish-language country labels in some sheets.

Possible label variants to look for (case-insensitive):
- UAE: `Birleşik Arap Emirlikleri`, `B. Arap Emir.`, `BAE`, `United Arab Emirates`, `UAE`
- Qatar: `Katar`, `Qatar`

Read the XLS and print the full list of country labels present. If UAE and/or Qatar are not in the XLS at all, **stop and tell me** — we'll need a different visitor data source.

### Step 3 — Add UAE and Qatar to the country mapping

Wherever the existing pipeline holds the canonical country list (most likely a dict mapping display name → ISO3), add:

```python
"United Arab Emirates": "ARE",
"Qatar":                "QAT",
```

Both go into the `MENA` market group, joining Iran, Iraq, Israel, Syria.

If the script also has a separate group-assignment dict, add UAE and Qatar to MENA there too. The MENA group should now have 6 countries.

### Step 4 — Re-fetch macro data for UAE and Qatar

Run the existing fetch logic for these two new ISO3 codes. Specifically:

- **IMF DataMapper:** GDP growth (`NGDP_RPCH`), inflation (`PCPIPCH`), GDP per capita USD (`NGDPDPC`) — same indicators the existing pipeline uses for the other 15 countries. Fetch the full 2003–2025 range.
- **World Bank Data360:** `GOV_WGI_PV` political stability with `COMP_BREAKDOWN_1` filter for ARE and QAT.

If the pipeline currently fetches all countries in a single call (e.g., one IMF query with all ISO3 codes batched), just add `ARE` and `QAT` to the list and re-run. If it's per-country in a loop, the new countries will get picked up automatically once added to the mapping.

**Print the fetched macro data shape** for ARE and QAT (years × indicators) so I can sanity-check that the API returned full coverage. If either has missing years or NaN-heavy columns, flag it explicitly — UAE and Qatar are wealthy stable countries with reliable IMF reporting, so >90% coverage is expected.

### Step 5 — Regenerate the panel

Run `python notebooks/01_data_pipeline.py`. Confirm:

- `data/panel_dataset.csv` now has **17 unique countries**.
- Each country has the same set of years (typically 2003–2025).
- Total rows ≈ 17 × 23 = 391 (or whatever the year range yields).
- New rows for UAE and Qatar have non-null values for `gdp_growth`, `gdp_per_capita`, `inflation`, `political_stability` (allow some NaNs at the boundary years).

### Step 6 — Sanity-check the new data

Print, for UAE and Qatar specifically:

```
UAE | visitors mean=<X> | min=<year>:<value> | max=<year>:<value> | 2019=<value> | 2020=<value> | 2024=<value>
UAE macro | gdp_growth NaN years: [...] | gdp_per_capita NaN years: [...] | inflation NaN years: [...] | political_stability NaN years: [...]

Qatar | visitors mean=<X> | ... (same fields)
Qatar macro | ... (same fields)
```

Also print the **2017–2022 visitor trajectory** for UAE specifically (one row per year showing year and visitors). The CLAUDE.md flags a UAE-Turkey diplomatic chill 2017–2021, and I want to see whether that's visible in the raw data before we proceed.

### Step 7 — Don't touch anything else

- Do **not** modify `02_eda_and_hypothesis.py`, `00_per_country_eda.py` (if it exists from a previous attempt), the README, or `CLAUDE.md`.
- Do **not** edit `mena_tension` / `syria_conflict` / `mena_tension_recent` dummy logic in this stage. That happens in Stage 2.
- Do **not** add hardcoded data for UAE/Qatar. If an API returns nothing for a year, leave it as NaN.

### Step 8 — Commit and stop

When everything runs cleanly:

1. Show me the final list of countries in the panel (`sorted(df['country'].unique())`).
2. Show me the printed UAE 2017–2022 trajectory.
3. Show me the macro NaN summaries for UAE and Qatar.
4. Stop. Do not proceed to Stage 1.

## Hard rules

- **Real API data only.** No hardcoded fallback values. If an API fails, surface the error.
- **ISO3 codes**, not ISO2. UAE = `ARE`, Qatar = `QAT`. Verify before fetching.
- **Don't break the existing 15 countries.** After running, all original countries should still be present with their original data unchanged. Spot-check Germany or France to confirm.
- **Don't auto-install packages.** If something's missing, stop and tell me.
- **No silent fallbacks.** If TÜİK doesn't have UAE/Qatar columns under any of the listed label variants, stop and report; don't paper over it.

When done, paste the country list, the UAE trajectory, and the NaN summaries. I'll review before approving Stage 1.
