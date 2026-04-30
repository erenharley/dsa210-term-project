"""
DSA 210 — Tourism Demand Under Shock
Phase 4: ML Methods (LOCO-CV Rewrite)
Author: Eren Sean Harley | 36054

Design change from initial commit:
  Temporal split was rejected because every shock dummy is identically zero
  in the pre-2020 training window, giving shock coefficients of ~1e-17
  (machine epsilon). A model trained that way cannot answer any shock-
  sensitivity question.

  Fix: Leave-One-Country-Out cross-validation (LOCO) on the full
  17-country × 23-year panel. For each fold, train on 16 countries and
  predict on the held-out country's 23 rows. Shock columns now vary across
  folds — the model genuinely learns shock coefficients.

Lecture scope (Weeks 8a-11a):
  Week 8b  — missing-data imputation (MNAR / MCAR strategies)
  Week 8c  — cross-validation (LOCO)
  Weeks 9a/9b — logistic regression, performance metrics (ROC/AUC)
  Week 9c  — regression (baseline reference)
  Week 10  — ensemble learning (random forest, XGBoost)
  Week 11a — clustering (k-means, hierarchical/Ward)
"""

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    classification_report, roc_curve, auc,
)
from sklearn.cluster import KMeans

from xgboost import XGBRegressor
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent / 'data'
FIGURES_DIR = BASE_DIR.parent / 'images'
FIGURES_DIR.mkdir(exist_ok=True)

# ── Style (consistent with Phase 3 notebooks) ──────────────────────────────────
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

SEP = '=' * 65

# ── Load panel ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'panel_dataset.csv')
all_countries = sorted(df['country'].unique())

print(f'\n{SEP}')
print(f'Panel loaded: {len(df)} rows, {df["country"].nunique()} countries, '
      f'years {df["year"].min()}-{df["year"].max()}')
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# MISSING DATA IMPUTATION  (Week 8b)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('MISSING DATA IMPUTATION  (Week 8b)')
print(SEP)
print("""
Sources of missingness identified from panel audit:

  gdp_growth, gdp_per_capita  — Syria 2011-2025 (15 nulls each)
    Cause: IMF stopped reporting when civil war began (MNAR).
    Strategy: forward-fill Syria's last pre-war value (2010).
    Rationale (Week 8b): forward-fill is simple and honest — carries the
    last observed state rather than guessing a collapsed economy. The
    alternative (group-median imputation) would impose MENA averages on
    a country whose economy is categorically different post-2010.

  inflation  — Iraq 2003-2004 + Syria 2011-2025 (17 nulls)
    Iraq 2003-04: series starts in 2005; backward-fill from 2005 value.
    Syria: same forward-fill as above.
    Rationale: symmetric to the forward-fill approach; bfill for the
    two Iraq rows uses the first observed value as a conservative proxy
    for the immediately preceding unobserved period.

  political_stability, tur_political_stability  — all 17 countries in 2025 (17 nulls each)
    Cause: WGI series not yet published for 2025 (MCAR).
    Strategy: forward-fill from 2024 per country.
    Rationale: political stability changes slowly year-to-year;
    carrying 2024 forward is standard practice for lagged indicators.

  tur_currency_weakness, visitors_yoy  — all countries in 2003 (17 nulls each)
    Cause: structural — no prior year to compute a YoY ratio (MCAR).
    tur_currency_weakness: impute 0 (no change relative to unobserved 2002).
    visitors_yoy: not used as a ML feature; left unchanged.

Panel on disk (panel_dataset.csv) is NOT modified.
All imputation applied downstream in the working copy df_ml.
""")

df_ml = df.copy()

# gdp_growth, gdp_per_capita: Syria forward-fill from 2010
for col in ['gdp_growth', 'gdp_per_capita']:
    df_ml[col] = df_ml.groupby('country')[col].transform(lambda s: s.ffill())

# inflation: Syria forward-fill; Iraq 2003-04 backward-fill
df_ml['inflation'] = df_ml.groupby('country')['inflation'].transform(
    lambda s: s.ffill().bfill()
)

# political_stability, tur_political_stability: 2025 forward-fill from 2024
for col in ['political_stability', 'tur_political_stability']:
    df_ml[col] = df_ml.groupby('country')[col].transform(lambda s: s.ffill())

# tur_currency_weakness 2003: no prior year → impute 0
df_ml['tur_currency_weakness'] = df_ml['tur_currency_weakness'].fillna(0.0)

print('Remaining nulls after imputation:')
check_cols = [
    'gdp_growth', 'gdp_per_capita', 'inflation',
    'political_stability', 'tur_political_stability', 'tur_currency_weakness',
]
remaining = df_ml[check_cols].isnull().sum()
print(remaining[remaining > 0].to_string() if remaining.any() else '  none')

# ── log_baseline_visitors: market-size identifier required for LOCO ────────────
# Without a country dummy (which would be always-zero for held-out countries),
# the model needs a continuous scale proxy. log(mean visitors 2017-2019) does
# this while being computed from pre-shock years and varying across countries.
baseline_map = (
    df_ml[df_ml['year'].between(2017, 2019)]
    .groupby('country')['visitors']
    .mean()
    .apply(lambda x: np.log(x) if x > 0 else np.nan)
    .rename('log_baseline_visitors')
)
df_ml = df_ml.join(baseline_map, on='country')

# ── Group dummies (drop first = Western Europe baseline) ──────────────────────
grp_dummies = pd.get_dummies(df_ml['market_group'], prefix='grp', drop_first=True)
grp_dummies.columns = [c.replace(' ', '_') for c in grp_dummies.columns]
df_ml = pd.concat([df_ml, grp_dummies], axis=1)
GRP_COLS = grp_dummies.columns.tolist()

# ── Feature lists ──────────────────────────────────────────────────────────────
FEAT_CONTINUOUS = [
    'gdp_growth', 'gdp_per_capita', 'inflation', 'political_stability',
    'log_baseline_visitors', 'tur_currency_weakness', 'tur_political_stability',
]
FEAT_SHOCKS = [
    'covid', 'coup_2016', 'syria_conflict',
    'russia_turkey_crisis', 'russia_ukraine_war', 'mena_tension_recent',
]
FEAT_MENA = ['is_mena_stable', 'is_mena_conflict']
ALL_FEATURES = FEAT_CONTINUOUS + FEAT_SHOCKS + GRP_COLS + FEAT_MENA
TARGET = 'log_visitors'

print(f'\nFeature set ({len(ALL_FEATURES)} features):')
print(f'  {ALL_FEATURES}')

# ── XGBoost factory ────────────────────────────────────────────────────────────
def make_xgb_reg(max_depth=4, n_estimators=300, learning_rate=0.05,
                 early_stopping_rounds=20):
    return XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        verbosity=0,
        eval_metric='rmse',
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — XGBOOST LOCO REGRESSION  (Weeks 9c, 10)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 1 — XGBOOST LOCO REGRESSION  (Goals 1 & 4)')
print(SEP)
print("""
Validation: Leave-One-Country-Out (LOCO) across all 17 countries.
For each fold: train on 16 countries × 23 years = 368 rows,
predict on held-out country × 23 rows.
Overall R² and RMSE aggregated over all 391 predictions.

Early stopping uses a 10% internal holdout drawn from the training fold.

Temporal split was rejected: shock dummies (covid, russia_ukraine_war,
mena_tension_recent) are zero for every pre-2020 row, so a model trained
on 2003-2019 data assigns them ~1e-17 coefficients. LOCO lets all six
shock columns vary in training so their coefficients are estimable.

Syria is included. Poor LOCO predictions on Syria are expected and honest
— the model is extrapolating a conflict-collapsed economy from patterns
learned on functioning markets.
""")

df_s1 = df_ml.dropna(subset=ALL_FEATURES + [TARGET]).copy()
print(f'Section 1 dataset: {len(df_s1)} rows, {df_s1["country"].nunique()} countries')

loco_results = []

for held_out in all_countries:
    train_df = df_s1[df_s1['country'] != held_out]
    test_df  = df_s1[df_s1['country'] == held_out]
    if len(test_df) == 0:
        continue

    X_tr = train_df[ALL_FEATURES].values.astype(float)
    y_tr = train_df[TARGET].values.astype(float)
    X_te = test_df[ALL_FEATURES].values.astype(float)
    y_te = test_df[TARGET].values.astype(float)

    # 10% internal validation for early stopping
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.10, random_state=42
    )

    mdl = make_xgb_reg()
    mdl.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = mdl.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    r2     = r2_score(y_te, y_pred)

    loco_results.append({
        'country': held_out,
        'y_true':  y_te,
        'y_pred':  y_pred,
        'rmse':    rmse,
        'r2':      r2,
        'group':   test_df['market_group'].values[0],
    })

# Aggregate
all_y_true = np.concatenate([r['y_true'] for r in loco_results])
all_y_pred = np.concatenate([r['y_pred'] for r in loco_results])
loco_r2   = r2_score(all_y_true, all_y_pred)
loco_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))

print(f'\nOverall LOCO  R² = {loco_r2:.3f},  RMSE = {loco_rmse:.3f}')
if loco_r2 < 0.5:
    print('  NOTE: R² < 0.5 — verify imputation pipeline; feature coverage may be incomplete.')

loco_tbl = pd.DataFrame([
    {'country': r['country'], 'group': r['group'], 'rmse': r['rmse'], 'r2': r['r2']}
    for r in loco_results
]).sort_values('rmse')

print('\nPer-country LOCO RMSE (ascending):')
print(loco_tbl.to_string(index=False))

# Highlight Syria explicitly as required by honesty rules
if 'Syria' in loco_tbl['country'].values:
    syria_rmse = loco_tbl.loc[loco_tbl['country'] == 'Syria', 'rmse'].values[0]
    print(f'\n  Syria LOCO RMSE = {syria_rmse:.3f}  (expected worst: post-2011 visitor data gap;'
          ' the model extrapolates a conflict economy from peacetime training data)')

# ── Full model trained on all 391 rows for SHAP ────────────────────────────────
X_full = df_s1[ALL_FEATURES].values.astype(float)
y_full = df_s1[TARGET].values.astype(float)

X_full_fit, X_full_val, y_full_fit, y_full_val = train_test_split(
    X_full, y_full, test_size=0.10, random_state=42
)
full_model = make_xgb_reg()
full_model.fit(X_full_fit, y_full_fit, eval_set=[(X_full_val, y_full_val)], verbose=False)

# ── SHAP values ────────────────────────────────────────────────────────────────
print('\nComputing SHAP values on full model...')
explainer   = shap.TreeExplainer(full_model)
shap_array  = explainer.shap_values(X_full)          # (N, n_features)

shap_df = pd.DataFrame(shap_array, columns=ALL_FEATURES)
shap_df['country'] = df_s1['country'].values

# 17 × 6 country-shock sensitivity matrix.
# Average SHAP over shock-ACTIVE rows only (where the dummy == 1).
# Averaging over all 23 rows per country would dilute ~3 active years into
# 20 non-shock zero rows, producing near-zero values for every country.
shock_shap_dict = {}
for shock in FEAT_SHOCKS:
    active_mask = df_s1[shock].values == 1
    if active_mask.sum() > 0:
        shock_shap_dict[shock] = (
            shap_df.loc[active_mask]
            .groupby('country')[shock]
            .mean()
        )
    else:
        shock_shap_dict[shock] = pd.Series(dtype=float)

shock_shap = (
    pd.DataFrame(shock_shap_dict)
    .reindex(sorted(shap_df['country'].unique()))
    .fillna(0.0)
)

# 17 × 1 currency sensitivity
currency_shap = (
    shap_df.groupby('country')['tur_currency_weakness']
    .mean()
    .rename('mean_shap_currency')
    .to_frame()
    .join(
        df_s1[['country', 'market_group']]
        .drop_duplicates('country')
        .set_index('country')
    )
    .sort_values('mean_shap_currency')
)

# ── Fig ml_01: Shock sensitivity heatmap (17 × 6) ─────────────────────────────
country_order = shock_shap.index.tolist()
shock_labels  = [s.replace('_', '\n') for s in FEAT_SHOCKS]

fig, ax = plt.subplots(figsize=(12, 9))
vmax = max(np.abs(shock_shap.values).max(), 0.01)
im   = ax.imshow(shock_shap.values, cmap='RdBu_r', aspect='auto',
                 vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(FEAT_SHOCKS)))
ax.set_xticklabels(shock_labels, fontsize=9)
ax.set_yticks(range(len(country_order)))
ax.set_yticklabels(country_order, fontsize=9)

# Color country labels by market group
country_grp_map = df.drop_duplicates('country').set_index('country')['market_group'].to_dict()
for i, c in enumerate(country_order):
    grp = country_grp_map.get(c, 'Other')
    ax.get_yticklabels()[i].set_color(GROUP_COLORS.get(grp, 'black'))

plt.colorbar(im, ax=ax, label='Mean SHAP value (contribution to log visitors)')
fig.text(
    0.5, -0.02,
    f'Overall LOCO R²={loco_r2:.3f}, RMSE={loco_rmse:.3f}  |  '
    'SHAP averaged over shock-active rows only (where dummy==1)  |  '
    'Red = shock associated with more visitors, Blue = fewer  |  '
    'Country label color = market group',
    ha='center', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
)
ax.set_title(
    f'Shock Contributions Vary by Market Group During Active Shock Periods '
    f'(XGBoost LOCO R²={loco_r2:.2f})',
    fontweight='bold', pad=10,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_01_shock_sensitivity_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_01_shock_sensitivity_heatmap.png saved')

# ── Fig ml_02: LOCO predicted vs actual ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
plotted_groups = set()
for r in loco_results:
    grp = r['group']
    ax.scatter(r['y_true'], r['y_pred'],
               color=GROUP_COLORS[grp], alpha=0.55, s=40,
               edgecolors='white', lw=0.4, zorder=3,
               label=grp if grp not in plotted_groups else '_nolegend_')
    plotted_groups.add(grp)

lo = min(all_y_true.min(), all_y_pred.min()) - 0.2
hi = max(all_y_true.max(), all_y_pred.max()) + 0.2
ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, alpha=0.6)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel('Actual log(Visitors)')
ax.set_ylabel('Predicted log(Visitors)')
ax.legend(fontsize=8)
fig.text(
    0.5, -0.02,
    f'LOCO CV: overall R²={loco_r2:.3f}, RMSE={loco_rmse:.3f}  |  '
    'Each point = one country-year held out from training',
    ha='center', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
)
ax.set_title(
    f'XGBoost LOCO: Explanatory Model Generalizes Across Markets (R²={loco_r2:.2f})',
    fontweight='bold', pad=10,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_02_loco_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_02_loco_predicted_vs_actual.png saved')

# ── Fig ml_05: Currency sensitivity ranking ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
colors_curr = [GROUP_COLORS[g] for g in currency_shap['market_group']]
ax.barh(currency_shap.index, currency_shap['mean_shap_currency'],
        color=colors_curr, edgecolor='white', lw=0.8, alpha=0.85)
ax.axvline(0, color='black', lw=1.2, ls='--', alpha=0.6)
ax.set_xlabel('Mean SHAP value for tur_currency_weakness (log visitors)')
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if g in currency_shap['market_group'].values]
ax.legend(handles=legend_patches, fontsize=8)
fig.text(
    0.5, -0.02,
    'Negative SHAP = lira weakness associated with fewer visitors from that market  |  '
    'Full model SHAP (n=391)',
    ha='center', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
)
ax.set_title(
    'Markets Vary in Sensitivity to Turkish Lira Weakness '
    '(mean SHAP for tur_currency_weakness)',
    fontweight='bold', pad=10,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_05_currency_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_05_currency_sensitivity.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLASSIFICATION: large visitor-drop years  (Weeks 9a, 9b, 10)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 2 — CLASSIFICATION: did visitors_yoy drop > 30%?')
print(SEP)
print("""
Target: binary — did visitors_yoy < -30 in year t?
Shock dummies are lagged by 1 year within each country, so the model
predicts next year's large drop from this year's shock configuration.
First year per country is dropped after lagging (NaN lag values).

LOCO validation: train on all rows of 16 countries, predict on 17th.
Models: LogisticRegression(C=1) and RandomForestClassifier(n=200).
""")

df_cls = df_ml.copy().sort_values(['country', 'year'])

# Lag shock dummies by 1 year within each country
for s in FEAT_SHOCKS:
    df_cls[f'{s}_lag'] = df_cls.groupby('country')[s].shift(1)

SHOCK_LAG_COLS = [f'{s}_lag' for s in FEAT_SHOCKS]
FEAT_CLS = FEAT_CONTINUOUS + SHOCK_LAG_COLS + GRP_COLS + FEAT_MENA
TARGET_CLS = 'big_drop'

df_cls[TARGET_CLS] = (df_cls['visitors_yoy'] < -30).astype(int)
df_cls = df_cls.dropna(subset=FEAT_CLS + [TARGET_CLS, 'visitors_yoy'])

print(f'Classification dataset: {len(df_cls)} rows (after dropping lag-NaN first years)')
n_pos = int(df_cls[TARGET_CLS].sum())
n_neg = int(len(df_cls) - n_pos)
print(f'Class balance: {n_pos} drops >30%  /  {n_neg} no big drop')


def loco_classify(clf_factory, df_c, feat_cols, target_col, countries):
    rows = []
    for held in countries:
        tr = df_c[df_c['country'] != held]
        te = df_c[df_c['country'] == held]
        if len(te) == 0:
            continue
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(tr[feat_cols].values.astype(float))
        X_te_s = sc.transform(te[feat_cols].values.astype(float))
        clf = clf_factory()
        clf.fit(X_tr_s, tr[target_col].values)
        y_pred = clf.predict(X_te_s)
        y_prob = (clf.predict_proba(X_te_s)[:, 1]
                  if hasattr(clf, 'predict_proba')
                  else np.full(len(te), np.nan))
        for i, (_, row_te) in enumerate(te.iterrows()):
            rows.append({
                'country': held,
                'y_true':  int(row_te[target_col]),
                'y_pred':  int(y_pred[i]),
                'y_prob':  float(y_prob[i]),
                'year':    int(row_te['year']),
            })
    return pd.DataFrame(rows)


all_cls_countries = sorted(df_cls['country'].unique())

res_lr = loco_classify(
    lambda: LogisticRegression(C=1, max_iter=2000, random_state=42),
    df_cls, FEAT_CLS, TARGET_CLS, all_cls_countries,
)
res_rf = loco_classify(
    lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    df_cls, FEAT_CLS, TARGET_CLS, all_cls_countries,
)

print('\n--- Logistic Regression (LOCO) ---')
print(classification_report(res_lr['y_true'], res_lr['y_pred'],
                             target_names=['No big drop', 'Drop >30%'],
                             zero_division=0))
print('\n--- Random Forest (LOCO) ---')
print(classification_report(res_rf['y_true'], res_rf['y_pred'],
                             target_names=['No big drop', 'Drop >30%'],
                             zero_division=0))

print('\nPer-country prediction accuracy (Random Forest, held-out rows):')
for c in all_cls_countries:
    sub = res_rf[res_rf['country'] == c]
    n_correct = int((sub['y_true'] == sub['y_pred']).sum())
    n_wrong   = int((sub['y_true'] != sub['y_pred']).sum())
    print(f'  {c:25s}: {n_correct} correct, {n_wrong} incorrect')

print('\nNote: with country-year granularity, these metrics are descriptive/exploratory.')

# ── Fig ml_03: ROC curves ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for label, res, color in [
    ('Logistic Regression', res_lr, '#2196F3'),
    ('Random Forest',       res_rf, '#FF5722'),
]:
    valid = ~pd.isna(res['y_prob'])
    if valid.any() and res.loc[valid, 'y_true'].nunique() == 2:
        fpr, tpr, _ = roc_curve(res.loc[valid, 'y_true'], res.loc[valid, 'y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{label} (AUC={roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.50)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9)
fig.text(
    0.5, -0.02,
    f'LOCO predictions, {len(res_lr)} country-years  |  '
    f'Target: visitors_yoy < -30%  |  Shock features lagged 1 year',
    ha='center', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
)
ax.set_title(
    'ROC — Classifying Large Visitor Drops (>30% YoY) by Market (LOCO)',
    fontweight='bold',
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_03_classification_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_03_classification_roc.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RECOVERY SPEED REGRESSION  (Week 10)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 3 — RECOVERY SPEED REGRESSION  (Goal 2)')
print(SEP)
print('WARNING: n=34 is small (17 countries × 2 shocks); results are exploratory.')

SHOCK_EVENTS = {'coup_2016': 2016, 'covid_2020': 2020}


def compute_recovery(country, shock_year, pre_window=2):
    d = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    if np.isnan(pre) or pre == 0:
        return np.nan, np.nan
    for _, row in d[d['year'] > shock_year].iterrows():
        if row['visitors'] >= pre:
            return pre, float(row['year'] - shock_year)
    return pre, np.nan  # never recovered within data window


rec_rows = []
for c in all_countries:
    for shock_name, shock_yr in SHOCK_EVENTS.items():
        pre, ytr = compute_recovery(c, shock_yr)
        ytr_capped = min(ytr, 5.0) if (ytr is not None and not np.isnan(ytr)) else 5.0

        pre_df   = df_ml[(df_ml['country'] == c) & df_ml['year'].between(shock_yr - 2, shock_yr - 1)]
        pol_stab = pre_df['political_stability'].mean()
        gdp_pc   = pre_df['gdp_per_capita'].mean()
        log_base = np.log(pre) if (pre is not None and pre > 0 and not np.isnan(pre)) else np.nan
        grp      = df[df['country'] == c]['market_group'].values[0]
        is_ms    = int(df[df['country'] == c]['is_mena_stable'].values[0])
        is_mc    = int(df[df['country'] == c]['is_mena_conflict'].values[0])

        rec_rows.append({
            'country':            c,
            'shock':              shock_name,
            'years_to_recover':   ytr_capped,
            'log_baseline':       log_base,
            'political_stability': pol_stab,
            'gdp_per_capita':     gdp_pc,
            'market_group':       grp,
            'is_mena_stable':     is_ms,
            'is_mena_conflict':   is_mc,
        })

rec_df = pd.DataFrame(rec_rows)

# One-hot shock type (drop first = coup_2016 baseline)
shock_dum = pd.get_dummies(rec_df['shock'], prefix='shock', drop_first=True)
grp_dum_r = pd.get_dummies(rec_df['market_group'], prefix='grp', drop_first=True)
grp_dum_r.columns = [c.replace(' ', '_') for c in grp_dum_r.columns]

rec_feat = pd.concat([
    rec_df[['log_baseline', 'political_stability', 'gdp_per_capita',
            'is_mena_stable', 'is_mena_conflict']].reset_index(drop=True),
    shock_dum.reset_index(drop=True),
    grp_dum_r.reset_index(drop=True),
], axis=1).dropna()

valid_idx   = rec_feat.index
rec_y_full  = rec_df.loc[valid_idx, 'years_to_recover'].values.astype(float)
rec_X_full  = rec_feat.values.astype(float)
rec_ctries  = rec_df.loc[valid_idx, 'country'].values
rec_shocks  = rec_df.loc[valid_idx, 'shock'].values

print(f'Recovery dataset: {len(rec_X_full)} rows')

# 5-fold CV with XGBoost (n=34 too small for LOCO)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rec_cv_preds = np.zeros(len(rec_X_full))

for train_idx, val_idx in kf.split(rec_X_full):
    X_t, X_v = rec_X_full[train_idx], rec_X_full[val_idx]
    y_t, y_v = rec_y_full[train_idx], rec_y_full[val_idx]
    # Internal split for early stopping within the fold
    n_int = max(1, int(len(X_t) * 0.2))
    X_ti, X_vi = X_t[:-n_int], X_t[-n_int:]
    y_ti, y_vi = y_t[:-n_int], y_t[-n_int:]
    mdl_r = make_xgb_reg(max_depth=3, n_estimators=200, learning_rate=0.05,
                         early_stopping_rounds=15)
    mdl_r.fit(X_ti, y_ti, eval_set=[(X_vi, y_vi)], verbose=False)
    rec_cv_preds[val_idx] = mdl_r.predict(X_v)

rec_rmse = np.sqrt(mean_squared_error(rec_y_full, rec_cv_preds))
print(f'5-fold CV RMSE (recovery speed): {rec_rmse:.3f} years')
print('WARNING: n=34 is small; results exploratory.')

# Full model on all 34 rows for the plot
n_int_full = max(1, int(len(rec_X_full) * 0.2))
mdl_rec_full = make_xgb_reg(max_depth=3, n_estimators=200, learning_rate=0.05,
                             early_stopping_rounds=15)
mdl_rec_full.fit(rec_X_full[:-n_int_full], rec_y_full[:-n_int_full],
                 eval_set=[(rec_X_full[-n_int_full:], rec_y_full[-n_int_full:])],
                 verbose=False)

rec_df_plot = rec_df.loc[valid_idx].copy().reset_index(drop=True)
rec_df_plot['predicted_ytr'] = mdl_rec_full.predict(rec_X_full)

# ── Fig ml_04: Recovery speed bar chart (faceted by shock) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

for ax_i, (shock_name, shock_yr) in enumerate(SHOCK_EVENTS.items()):
    sub = rec_df_plot[rec_df_plot['shock'] == shock_name].sort_values('predicted_ytr')
    colors_r = [GROUP_COLORS.get(g, '#607D8B') for g in sub['market_group']]
    axes[ax_i].barh(sub['country'], sub['predicted_ytr'],
                    color=colors_r, edgecolor='white', lw=0.8, alpha=0.85)
    axes[ax_i].axvline(5, color='red', lw=1.5, ls='--', alpha=0.7, label='Cap = 5 years')
    axes[ax_i].set_xlabel('Predicted Years to Recover')
    axes[ax_i].set_xlim(0, 5.5)
    shock_label = '2016 Coup Attempt' if 'coup' in shock_name else 'COVID-19 (2020)'
    axes[ax_i].set_title(shock_label, fontweight='bold')
    axes[ax_i].legend(fontsize=8)

legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_patches, fontsize=8, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.05))
fig.text(
    0.5, -0.09,
    f'XGBoost 5-fold CV RMSE = {rec_rmse:.2f} years  |  '
    'WARNING: n=34 exploratory  |  Cap at 5 years = not recovered within data window',
    ha='center', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
)
fig.suptitle(
    'Explanatory Recovery Speed by Market and Shock Type (n=34, exploratory)',
    fontweight='bold', y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_04_recovery_speed.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_04_recovery_speed.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESILIENCE RANKING  (Goal 3, descriptive)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 4 — RESILIENCE RANKING  (Descriptive Index, not ML)')
print(SEP)
print("""
Composite resilience index = weighted mean of visitors_vs_baseline across:
  2016        (weight 0.2) — coup shock
  2020-2021   (weight 0.4) — COVID shock
  2023-2025   (weight 0.4) — post-MENA-tension period

visitors_vs_baseline is drawn directly from panel_dataset.csv
(= (visitors / mean_2017-19_visitors - 1) × 100, percentage points).
Index > 0 means above pre-shock baseline; index < 0 means below.
This is a purely descriptive ranking — no ML involved.
""")

rank_rows = []
for c in all_countries:
    sub = df[df['country'] == c]
    w1 = sub[sub['year'] == 2016]['visitors_vs_baseline'].mean()
    w2 = sub[sub['year'].isin([2020, 2021])]['visitors_vs_baseline'].mean()
    w3 = sub[sub['year'].isin([2023, 2024, 2025])]['visitors_vs_baseline'].mean()
    parts = [(0.2, w1), (0.4, w2), (0.4, w3)]
    valid = [(wt, v) for wt, v in parts if not np.isnan(v)]
    if valid:
        total_w = sum(wt for wt, _ in valid)
        idx = sum(wt * v for wt, v in valid) / total_w
    else:
        idx = np.nan
    grp = df[df['country'] == c]['market_group'].values[0]
    rank_rows.append({'country': c, 'resilience_index': idx, 'market_group': grp})

rank_df = pd.DataFrame(rank_rows).sort_values('resilience_index', ascending=True)

print('Resilience ranking — all 17 countries:')
print(rank_df[['country', 'market_group', 'resilience_index']]
      .to_string(index=False, float_format='%.2f'))

# ── Fig ml_06: Resilience ranking bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
colors_rank = [GROUP_COLORS.get(g, '#607D8B') for g in rank_df['market_group']]
ax.barh(rank_df['country'], rank_df['resilience_index'],
        color=colors_rank, edgecolor='white', lw=0.8, alpha=0.85)
ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.6,
           label='0 = at 2017-19 baseline')
ax.set_xlabel('Resilience Index (weighted % vs 2017-19 baseline; weights: coup=0.2, COVID=0.4, 2023-25=0.4)')
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if g in rank_df['market_group'].values]
legend_patches.append(mpatches.Patch(color='none', label='0 = at baseline'))
ax.legend(handles=legend_patches, fontsize=8)
fig.text(
    0.5, -0.02,
    'Descriptive Index, not ML  |  Higher = more resilient across all three shock windows',
    ha='center', fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
)
ax.set_title(
    'Descriptive Index: Market Resilience Across Three Shock Windows '
    '(Descriptive Index, not ML)',
    fontweight='bold', pad=10,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_06_resilience_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_06_resilience_ranking.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CLUSTERING  (Week 11a, descriptive)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 5 — CLUSTERING (descriptive sanity check on hand-coded market_group)')
print(SEP)
print("""
Clustering is used here as a descriptive sanity check on the hand-coded
market_group variable. If k-means and hierarchical Ward clustering
reproduce the hand-coded groups, it validates the group structure.
Sub-groupings that emerge within MENA (Gulf stable vs conflict-affected)
serve as descriptive evidence for the H6 narrative.
Do NOT interpret cluster membership as predictive or causal.
""")


def shock_impact_pct(country, shock_year, pre_window=2):
    d   = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    sv  = d[d['year'] == shock_year]['visitors'].values
    if pre == 0 or len(sv) == 0 or np.isnan(pre):
        return np.nan
    return (sv[0] / pre - 1) * 100


clust_rows = []
for c in all_countries:
    drop_2016 = shock_impact_pct(c, 2016)
    drop_2020 = shock_impact_pct(c, 2020)

    base_1719 = df[(df['country'] == c) & df['year'].between(2017, 2019)]['visitors'].mean()
    mean_2325 = df[(df['country'] == c) & df['year'].between(2023, 2025)]['visitors'].mean()
    pct_2325  = (mean_2325 / base_1719 - 1) * 100 if base_1719 > 0 else np.nan

    # COVID recovery (years until back to 2018-19 average)
    d_c     = df[df['country'] == c].sort_values('year')
    pre_cov = d_c[d_c['year'].between(2018, 2019)]['visitors'].mean()
    rec_yr  = np.nan
    for _, row in d_c[d_c['year'] > 2020].iterrows():
        if row['visitors'] >= pre_cov:
            rec_yr = float(row['year'] - 2020)
            break
    rec_capped = min(rec_yr, 5.0) if not np.isnan(rec_yr) else 5.0

    log_base = np.log(base_1719) if base_1719 > 0 else np.nan
    grp = df[df['country'] == c]['market_group'].values[0]
    clust_rows.append({
        'country':           c,
        'market_group':      grp,
        'drop_2016':         drop_2016,
        'drop_2020':         drop_2020,
        'pct_vs_base_2325':  pct_2325,
        'years_to_recover':  rec_capped,
        'log_base_visitors': log_base,
    })

clust_df = pd.DataFrame(clust_rows).set_index('country')

# Syria 2016 drop is NaN (visitor series starts post-war); fill with MENA mean
mena_mean_2016 = clust_df[clust_df['market_group'] == 'MENA']['drop_2016'].mean()
clust_df['drop_2016'] = clust_df['drop_2016'].fillna(mena_mean_2016)

feat_cols_clust = ['drop_2016', 'drop_2020', 'pct_vs_base_2325',
                   'years_to_recover', 'log_base_visitors']
X_clust   = clust_df[feat_cols_clust].values.astype(float)
sc_clust  = StandardScaler()
X_clust_s = sc_clust.fit_transform(X_clust)

# ── Elbow plot ─────────────────────────────────────────────────────────────────
inertias = []
K_range  = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_clust_s)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(K_range), inertias, marker='o', color='#2196F3', lw=2.5, ms=8)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Within-Cluster Sum of Squares (Inertia)')
ax.set_xticks(list(K_range))

delta = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
try:
    elbow_idx = int(np.argmax(np.diff(delta)) + 2)
except Exception:
    elbow_idx = 3
ax.axvline(elbow_idx, color='red', lw=1.5, ls='--', alpha=0.7,
           label=f'Suggested elbow k={elbow_idx}')
ax.legend(fontsize=9)
ax.set_title(
    'Elbow Plot — Clustering on Shock-Response Features '
    '(descriptive sanity check on hand-coded market groups)',
    fontweight='bold',
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_07_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_07_elbow.png saved')

# k-means with k=3
CHOSEN_K = 3
km_final = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42)
km_final.fit(X_clust_s)
clust_df['cluster_km'] = km_final.labels_

print(f'\nk-means k={CHOSEN_K}:')
for cl in sorted(clust_df['cluster_km'].unique()):
    members = clust_df[clust_df['cluster_km'] == cl]
    print(f'  Cluster {cl}: {", ".join(members.index.tolist())}')

# Within-MENA split
uae_qatar_clust = clust_df.loc[
    [c for c in ['United Arab Emirates', 'Qatar'] if c in clust_df.index], 'cluster_km'
].values
conflict_clust = clust_df.loc[
    [c for c in ['Iran', 'Iraq', 'Israel', 'Syria'] if c in clust_df.index], 'cluster_km'
].values
print(f'\nWithin-MENA: UAE/Qatar clusters = {uae_qatar_clust}, '
      f'conflict-affected clusters = {conflict_clust}')
same_cluster = all(c in uae_qatar_clust for c in conflict_clust)
if same_cluster:
    print('  -> Gulf-stable and conflict-affected MENA land in the SAME cluster;'
          ' 5-feature k-means cannot separate them.')
else:
    print('  -> UAE/Qatar separate from conflict-affected MENA -- aligns with H6 descriptive evidence.')

# ── Dendrogram ─────────────────────────────────────────────────────────────────
Z = linkage(X_clust_s, method='ward')

fig, ax = plt.subplots(figsize=(14, 6))
dend = dendrogram(
    Z,
    labels=clust_df.index.tolist(),
    ax=ax,
    color_threshold=0.7 * max(Z[:, 2]),
    leaf_rotation=45,
    leaf_font_size=10,
)
for lbl in ax.get_xticklabels():
    grp = country_grp_map.get(lbl.get_text(), 'Other')
    lbl.set_color(GROUP_COLORS.get(grp, 'black'))

ax.set_ylabel('Ward Linkage Distance')
ax.set_title(
    'Hierarchical Clustering — Descriptive Sanity Check on market_group Structure '
    '(label color = market group)',
    fontweight='bold',
)
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
ax.legend(handles=legend_patches, fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_08_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_08_dendrogram.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PHASE 4 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 6 — PHASE 4 SUMMARY')
print(SEP)

# Compute AUC for summary
def safe_auc(res):
    valid = ~pd.isna(res['y_prob'])
    if valid.sum() > 0 and res.loc[valid, 'y_true'].nunique() == 2:
        fpr, tpr, _ = roc_curve(res.loc[valid, 'y_true'], res.loc[valid, 'y_prob'])
        return auc(fpr, tpr)
    return float('nan')

lr_auc = safe_auc(res_lr)
rf_auc = safe_auc(res_rf)
lr_acc = (res_lr['y_true'] == res_lr['y_pred']).mean()
rf_acc = (res_rf['y_true'] == res_rf['y_pred']).mean()

top3_countries    = rank_df.dropna(subset=['resilience_index']).tail(3)['country'].tolist()
bottom3_countries = rank_df.dropna(subset=['resilience_index']).head(3)['country'].tolist()

print(f"""
Section 1 — XGBoost LOCO Regression  (Goals 1 & 4)
  Overall LOCO R²   = {loco_r2:.3f}
  Overall LOCO RMSE = {loco_rmse:.3f}
  Figures: ml_01_shock_sensitivity_heatmap.png
           ml_02_loco_predicted_vs_actual.png
           ml_05_currency_sensitivity.png

Section 2 — Classification: large drop years  (lecture coverage Weeks 9a/9b/10)
  Logistic Regression  accuracy={lr_acc:.2f},  AUC={lr_auc:.2f}
  Random Forest        accuracy={rf_acc:.2f},  AUC={rf_auc:.2f}
  Figure: ml_03_classification_roc.png

Section 3 — Recovery Speed Regression  (Goal 2)
  5-fold CV RMSE = {rec_rmse:.2f} years  (n=34, exploratory)
  Figure: ml_04_recovery_speed.png

Section 4 — Resilience Ranking  (Goal 3)
  Most resilient  (top 3):    {', '.join(top3_countries)}
  Least resilient (bottom 3): {', '.join(bottom3_countries)}
  Figure: ml_06_resilience_ranking.png  (Descriptive Index, not ML)

Section 5 — Clustering  (lecture coverage Week 11a)
  Figures: ml_07_elbow.png, ml_08_dendrogram.png

All generated figures:
  ml_01_shock_sensitivity_heatmap.png
  ml_02_loco_predicted_vs_actual.png
  ml_03_classification_roc.png
  ml_04_recovery_speed.png
  ml_05_currency_sensitivity.png
  ml_06_resilience_ranking.png
  ml_07_elbow.png
  ml_08_dendrogram.png
""")
