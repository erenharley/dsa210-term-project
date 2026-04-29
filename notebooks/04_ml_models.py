"""
DSA 210 — Tourism Demand Under Shock
Phase 4: ML Methods
Author: Eren Sean Harley | 36054

Lecture scope (Weeks 8a-11a):
  Week 8a/8b  — data transformation, handling missing data
  Week 8c     — train/test split, cross-validation (LOOCV)
  Weeks 9a/9b — logistic regression, performance metrics
  Week 9c     — linear regression
  Week 10     — ensemble learning (random forest)
  Week 11a    — clustering (k-means, hierarchical)

Deliberately excluded (not in lecture scope):
  - Panel fixed effects (use market_group one-hot as country control instead)
  - ARIMA / Prophet / time-series forecasting models
  - Neural networks
  - Tukey HSD or post-hoc pairwise tests
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required for script mode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
)
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')

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

# ══════════════════════════════════════════════════════════════════════════════
# MISSING DATA AUDIT AND IMPUTATION  (Week 8b)
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

# tur_currency_weakness 2003: no prior year to compute YoY → impute 0
df_ml['tur_currency_weakness'] = df_ml['tur_currency_weakness'].fillna(0.0)

print('Remaining nulls after imputation:')
check_cols = [
    'gdp_growth', 'gdp_per_capita', 'inflation',
    'political_stability', 'tur_political_stability', 'tur_currency_weakness',
]
remaining = df_ml[check_cols].isnull().sum()
print(remaining[remaining > 0].to_string() if remaining.any() else '  none')

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST STRATEGY  (Week 8c)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('TRAIN/TEST STRATEGY  (Week 8c)')
print(SEP)
print("""
Temporal split:  train <= 2019,  test 2020-2025

  Rationale: avoids data leakage — no future shock information in training
  features. Makes "did the model anticipate COVID" a legitimate generalization
  question. The test period 2020-2025 contains three structural shocks
  (COVID, Russia-Ukraine war, post-2023 MENA tensions) that were absent during
  training — a hard but honest test. A negative test R² would mean the model
  performs worse than predicting the mean; this is expected and acceptable
  given the scale of COVID in 2020-2021.

For Section 2 classification (n = 17 countries):
  Leave-one-country-out cross-validation (LOOCV).

  Rationale: with n = 17 at the country level, standard 5-fold or 10-fold CV
  would put only 3-4 countries in each fold — too few for reliable held-out
  estimates. LOOCV uses 16 countries for training each fold and produces
  17 binary predictions, one per country.
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LINEAR REGRESSION: predict log_visitors  (Week 9c)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 1 — LINEAR REGRESSION: predict log_visitors')
print(SEP)

CONTINUOUS = [
    'gdp_growth', 'gdp_per_capita', 'inflation',
    'political_stability', 'tur_currency_weakness', 'tur_political_stability',
]
SHOCKS = [
    'covid', 'coup_2016', 'syria_conflict',
    'russia_turkey_crisis', 'russia_ukraine_war', 'mena_tension_recent',
]
TARGET = 'log_visitors'

# One-hot encode market_group; drop first category to avoid dummy trap.
# This is the country group control replacing fixed effects (not in scope).
df_reg = df_ml.copy()
grp_dummies = pd.get_dummies(df_reg['market_group'], prefix='grp', drop_first=True)
df_reg = pd.concat([df_reg, grp_dummies], axis=1)
GRP_COLS = grp_dummies.columns.tolist()

ALL_FEAT = CONTINUOUS + SHOCKS + GRP_COLS
df_reg   = df_reg.dropna(subset=ALL_FEAT + [TARGET])

train_r = df_reg[df_reg['year'] <= 2019]
test_r  = df_reg[df_reg['year'] >= 2020]

print(f'Train rows: {len(train_r)}  (years {train_r["year"].min()}-{train_r["year"].max()})')
print(f'Test rows:  {len(test_r)}   (years {test_r["year"].min()}-{test_r["year"].max()})')

scaler   = StandardScaler()
X_train  = scaler.fit_transform(train_r[ALL_FEAT])
X_test   = scaler.transform(test_r[ALL_FEAT])
y_train  = train_r[TARGET].values
y_test   = test_r[TARGET].values

model_linreg = LinearRegression()
model_linreg.fit(X_train, y_train)

y_pred_train = model_linreg.predict(X_train)
y_pred_test  = model_linreg.predict(X_test)

tr_r2   = r2_score(y_train, y_pred_train)
te_r2   = r2_score(y_test,  y_pred_test)
tr_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
te_rmse = np.sqrt(mean_squared_error(y_test,  y_pred_test))
tr_mae  = mean_absolute_error(y_train, y_pred_train)
te_mae  = mean_absolute_error(y_test,  y_pred_test)

print(f'\n--- Performance ---')
print(f'  Train: R²={tr_r2:.3f},  RMSE={tr_rmse:.3f},  MAE={tr_mae:.3f}')
print(f'  Test:  R²={te_r2:.3f},   RMSE={te_rmse:.3f},   MAE={te_mae:.3f}')

if te_r2 < 0:
    print('\n  Note: negative test R² — the model predicts worse than the mean.')
    print('  Expected: COVID 2020-21 and post-war shocks are structurally unlike')
    print('  anything in the 2003-2019 training window. The temporal split is')
    print('  honest; do not re-train on shock years to chase a positive R².')
elif te_r2 < 0.3:
    print('\n  Note: low test R² — model captures pre-shock structure but')
    print('  generalizes poorly to 2020-2025 shock magnitudes.')

# Coefficient table (sorted by magnitude)
coef_df = pd.DataFrame({
    'feature': ALL_FEAT,
    'coef':    model_linreg.coef_,
}).sort_values('coef', key=abs, ascending=False)

print('\n--- Coefficient table (standardized; sorted by |coef|) ---')
print(coef_df.to_string(index=False))
print(f'Intercept: {model_linreg.intercept_:.4f}')

# Interpretation
# Important limitation: shock dummies whose events fall entirely in the TEST
# period (covid 2020-21, russia_ukraine_war 2022+, mena_tension_recent 2023+)
# have ZERO variance in the training data (always 0 for years <= 2019).
# StandardScaler sets them to the zero vector; OLS assigns a near-zero
# coefficient (~1e-17, floating-point noise). This is expected and honest:
# a model trained before a shock cannot learn that shock's coefficient.
# Implication for Section 4: the scenario prediction (mena_tension_recent=1)
# does NOT change the prediction vs the baseline (mena_tension_recent=0).
# The Section 4 outputs reflect extrapolated 2025 macro conditions only.
near_zero_shocks = [f for f, c in zip(ALL_FEAT, model_linreg.coef_)
                    if f in SHOCKS and abs(c) < 1e-10]
if near_zero_shocks:
    print(f'\n  WARNING: the following shock dummies have near-zero coefficients')
    print(f'  (they occur only in the test period and have zero variance in training):')
    for f in near_zero_shocks:
        print(f'    {f}')
    print(f'  The temporal split is the correct choice; these near-zero coefficients')
    print(f'  are an honest reflection of what the model can learn from pre-2020 data.')

print('\n--- Coefficient interpretation ---')
top_pos = coef_df[coef_df['coef'] > 0].head(3)
top_neg = coef_df[coef_df['coef'] < 0].head(3)
print('  Largest positive coefficients (features associated with MORE visitors):')
for _, row in top_pos.iterrows():
    print(f'    {row["feature"]:35s}: +{row["coef"]:.3f}')
print('  Largest negative coefficients (features associated with FEWER visitors):')
for _, row in top_neg.iterrows():
    print(f'    {row["feature"]:35s}: {row["coef"]:.3f}')

# Determine title based on test performance
def _linreg_title(r2_val):
    if r2_val > 0.4:
        return f'Linear Regression Generalizes Moderately to 2020-2025 (Test R²={r2_val:.2f})'
    elif r2_val > 0:
        return f'Linear Regression Generalizes Poorly to Shock Period (Test R²={r2_val:.2f})'
    else:
        return f'Linear Regression Does Not Generalize to COVID/Post-COVID Period (Test R²={r2_val:.2f})'

lr_title = _linreg_title(te_r2)
annot_lr = (
    f'Train R²={tr_r2:.3f} | Test R²={te_r2:.3f} | '
    f'RMSE={te_rmse:.3f} | MAE={te_mae:.3f}  '
    f'(temporal split: train <=2019, test 2020-2025)'
)

# --- Fig ml_01: Residual plot (test set) ---
fig, ax = plt.subplots(figsize=(11, 6))
residuals = y_test - y_pred_test
groups_test = test_r['market_group'].values
for g in GROUP_COLORS:
    mask = groups_test == g
    if mask.any():
        ax.scatter(y_pred_test[mask], residuals[mask],
                   color=GROUP_COLORS[g], alpha=0.75, s=65,
                   edgecolors='white', lw=0.5, label=g, zorder=3)
ax.axhline(0, color='black', lw=1.5, ls='--', alpha=0.7)
ax.set_xlabel('Predicted log(Visitors)')
ax.set_ylabel('Residual  (Actual − Predicted)')
ax.legend(fontsize=8, loc='upper right')
fig.text(0.5, -0.02, annot_lr, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
ax.set_title(lr_title, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_01_linreg_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nFig ml_01_linreg_residuals.png saved')

# --- Fig ml_02: Predicted vs Actual (test set) ---
fig, ax = plt.subplots(figsize=(11, 6))
for g in GROUP_COLORS:
    mask = groups_test == g
    if mask.any():
        ax.scatter(y_test[mask], y_pred_test[mask],
                   color=GROUP_COLORS[g], alpha=0.75, s=65,
                   edgecolors='white', lw=0.5, label=g, zorder=3)
lo = min(y_test.min(), y_pred_test.min()) - 0.3
hi = max(y_test.max(), y_pred_test.max()) + 0.3
ax.plot([lo, hi], [lo, hi], color='black', lw=1.5, ls='--', alpha=0.6,
        label='Perfect fit (y = x)')
ax.set_xlabel('Actual log(Visitors)')
ax.set_ylabel('Predicted log(Visitors)')
ax.legend(fontsize=8, loc='upper left')
fig.text(0.5, -0.02, annot_lr, ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
ax.set_title(lr_title, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_02_linreg_fit.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_02_linreg_fit.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLASSIFICATION: COVID-resilient vs non-resilient  (Weeks 9a, 9b, 10)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 2 — CLASSIFICATION: COVID-resilient vs non-resilient')
print(SEP)

# --- Build label ---
# "resilient" = 2022 visitors >= 2019 visitors (back to or above pre-COVID peak)
visitors_2019 = df[df['year'] == 2019].set_index('country')['visitors']
visitors_2022 = df[df['year'] == 2022].set_index('country')['visitors']
label_df = pd.DataFrame({
    'visitors_2019': visitors_2019,
    'visitors_2022': visitors_2022,
}).dropna()
label_df['resilient'] = (label_df['visitors_2022'] >= label_df['visitors_2019']).astype(int)

print('Class labels (1 = resilient, 0 = not resilient):')
print(label_df[['visitors_2019', 'visitors_2022', 'resilient']].to_string())
n_res = label_df['resilient'].sum()
n_tot = len(label_df)
print(f'\nClass balance: {n_res} resilient / {n_tot - n_res} non-resilient  (out of {n_tot})')

# --- Build features (PRE-2020 ONLY) ---
# LEAKAGE RULE: all features must use information available before 2020.
# Using 2015-2019 means (macro) and 2015-2019 log-mean (baseline visitors)
# avoids any COVID-era signal bleeding into the classifier.
pre_covid = df_ml[df_ml['year'].between(2015, 2019)]

feat_macro = pre_covid.groupby('country')[
    ['gdp_growth', 'gdp_per_capita', 'inflation', 'political_stability']
].mean()

# log-mean baseline visitors 2015-2019 as a proxy for market size
feat_macro['log_baseline_visitors'] = (
    pre_covid.groupby('country')['visitors'].mean().apply(np.log)
)

# Market group indicators (time-invariant, no leakage)
country_meta = (
    df[['country', 'market_group', 'is_mena_stable', 'is_mena_conflict']]
    .drop_duplicates('country')
    .set_index('country')
)
feat_macro = feat_macro.join(country_meta)

# One-hot encode market_group (drop first)
grp_dum_clf = pd.get_dummies(feat_macro['market_group'], prefix='grp', drop_first=True)
feat_matrix = feat_macro.drop(columns=['market_group']).join(grp_dum_clf)

# Align countries across label and features
common_countries = label_df.index.intersection(feat_matrix.index)
X_clf = feat_matrix.loc[common_countries].values.astype(float)
y_clf = label_df.loc[common_countries, 'resilient'].values
country_names_clf = list(common_countries)
n_clf = len(country_names_clf)

print(f'\nClassifier dataset: {n_clf} countries × {X_clf.shape[1]} features')
print(f'Features: {feat_matrix.columns.tolist()}')

# --- LOOCV ---
def run_loocv(clf_factory, X, y, countries):
    """Leave-one-country-out CV; returns (y_true, y_pred, y_prob)."""
    y_true, y_pred, y_prob = [], [], []
    for i in range(len(countries)):
        X_tr = np.delete(X, i, axis=0)
        y_tr = np.delete(y, i)
        X_te = X[i:i+1]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        clf = clf_factory()
        clf.fit(X_tr_s, y_tr)
        y_pred.append(clf.predict(X_te_s)[0])
        if hasattr(clf, 'predict_proba'):
            y_prob.append(clf.predict_proba(X_te_s)[0][1])
        else:
            y_prob.append(np.nan)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# Logistic regression (L2, C=1)
_, pred_lr, prob_lr = run_loocv(
    lambda: LogisticRegression(max_iter=2000, random_state=42),
    X_clf, y_clf, country_names_clf
)

# Random forest
_, pred_rf, prob_rf = run_loocv(
    lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X_clf, y_clf, country_names_clf
)

def print_clf_report(name, y_true, y_pred):
    print(f'\n--- {name} (LOOCV, n={n_clf}) ---')
    print(classification_report(y_true, y_pred,
                                target_names=['Non-resilient', 'Resilient'],
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print(f'Confusion matrix:\n{cm}')

print_clf_report('Logistic Regression', y_clf, pred_lr)
print_clf_report('Random Forest',       y_clf, pred_rf)

print("""
Note: with n=17, these metrics are descriptive/exploratory only.
LOOCV accuracy on 17 observations has high variance; do NOT claim
statistical superiority of one model over the other.
""")

# --- Feature importance (Random Forest trained on all 17 countries) ---
sc_full = StandardScaler()
X_clf_s = sc_full.fit_transform(X_clf)
rf_full  = RandomForestClassifier(n_estimators=200, random_state=42)
rf_full.fit(X_clf_s, y_clf)

feat_names_clf = feat_matrix.columns.tolist()
imp_df = pd.DataFrame({
    'feature':    feat_names_clf,
    'importance': rf_full.feature_importances_,
}).sort_values('importance', ascending=True)

# --- Fig ml_03: Feature importance ---
fig, ax = plt.subplots(figsize=(10, max(5, len(feat_names_clf) * 0.45)))
colors_imp = ['#9C27B0' if 'mena' in f or 'grp' in f else '#2196F3'
              for f in imp_df['feature']]
ax.barh(imp_df['feature'], imp_df['importance'],
        color=colors_imp, edgecolor='white', lw=0.8, alpha=0.85)
ax.set_xlabel('Mean Decrease in Impurity (Random Forest)')
rf_acc = (pred_rf == y_clf).mean()
lr_acc = (pred_lr == y_clf).mean()
annot_imp = (
    f'Random Forest LOOCV accuracy: {rf_acc:.2f}  |  '
    f'Logistic Regression LOOCV accuracy: {lr_acc:.2f}  |  '
    f'n={n_clf} countries, {n_res} resilient / {n_tot-n_res} non-resilient  '
    f'(descriptive only — n too small for inference)'
)
fig.text(0.5, -0.03, annot_imp, ha='center', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

if rf_acc > 0.7:
    imp_title = (
        f'Pre-COVID Features Informative for Resilience Classification '
        f'(RF LOOCV acc={rf_acc:.0%}; descriptive, n=17)'
    )
else:
    imp_title = (
        f'Pre-COVID Features Weakly Predictive of COVID Resilience '
        f'(RF LOOCV acc={rf_acc:.0%}; descriptive, n=17)'
    )
ax.set_title(imp_title, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_03_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_03_feature_importance.png saved')

# ROC curve: with LOOCV probabilities and n=17, the curve will be very coarse
# (17 threshold points). Include with caveat.
prob_both = np.vstack([prob_lr, prob_rf])
if not np.isnan(prob_both).any():
    fig, ax = plt.subplots(figsize=(7, 6))
    for name_roc, probs, color in [
        ('Logistic Regression', prob_lr, '#2196F3'),
        ('Random Forest',       prob_rf, '#FF5722'),
    ]:
        fpr, tpr, _ = roc_curve(y_clf, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name_roc} (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.50)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9)
    fig.text(
        0.5, -0.03,
        f'LOOCV probabilities, n=17 points — curve is coarse; interpret AUC as approximate.',
        ha='center', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
    )
    ax.set_title(
        'ROC Curve — COVID Resilience Classifier (LOOCV, n=17; coarse)',
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ml_03b_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Fig ml_03b_roc_curve.png saved')
else:
    print('ROC curve skipped: predict_proba not available for one or more models.')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLUSTERING  (Week 11a)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 3 — CLUSTERING: k-means + hierarchical')
print(SEP)

# ── Helper (shared with Phase 3) ──────────────────────────────────────────────
def shock_impact(country, shock_year, pre_window=2):
    d   = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    sv  = d[d['year'] == shock_year]['visitors'].values
    if pre == 0 or len(sv) == 0 or np.isnan(pre):
        return np.nan
    return (sv[0] / pre - 1) * 100

def years_to_recover(country, shock_year, pre_window=2):
    d   = df[df['country'] == country].sort_values('year')
    pre = d[d['year'].between(shock_year - pre_window, shock_year - 1)]['visitors'].mean()
    if np.isnan(pre) or pre == 0:
        return np.nan
    for _, row in d[d['year'] > shock_year].iterrows():
        if row['visitors'] >= pre:
            return row['year'] - shock_year
    return np.nan

# ── Per-country feature vector ─────────────────────────────────────────────────
clust_rows = []
for c in all_countries:
    drop_2016 = shock_impact(c, 2016, pre_window=2)
    drop_2020 = shock_impact(c, 2020, pre_window=2)

    # 2023-25 mean % vs 2017-19 baseline
    base_1719 = df[(df['country'] == c) & df['year'].between(2017, 2019)]['visitors'].mean()
    mean_2325  = df[(df['country'] == c) & df['year'].between(2023, 2025)]['visitors'].mean()
    pct_2325   = (mean_2325 / base_1719 - 1) * 100 if base_1719 > 0 else np.nan

    rec = years_to_recover(c, 2020, pre_window=2)
    rec_capped = min(rec, 5) if not np.isnan(rec) else 5  # cap at 5; not-recovered → 5

    log_base = np.log(
        df[(df['country'] == c) & df['year'].between(2017, 2019)]['visitors'].mean()
    )

    group = df[df['country'] == c]['market_group'].values[0]
    clust_rows.append({
        'country':          c,
        'market_group':     group,
        'drop_2016':        drop_2016,
        'drop_2020':        drop_2020,
        'pct_vs_base_2325': pct_2325,
        'years_to_recover': rec_capped,
        'log_base_visitors': log_base,
    })

clust_df = pd.DataFrame(clust_rows).set_index('country')

# Impute the single NaN from shock_impact (Syria 2016 drop — no data before 2011)
# Use MENA group mean as a reasonable proxy.
mena_mean_2016 = clust_df[clust_df['market_group'] == 'MENA']['drop_2016'].mean()
clust_df['drop_2016'] = clust_df['drop_2016'].fillna(mena_mean_2016)

feat_cols_clust = [
    'drop_2016', 'drop_2020', 'pct_vs_base_2325',
    'years_to_recover', 'log_base_visitors',
]
X_clust = clust_df[feat_cols_clust].values.astype(float)

# Standardize (k-means is distance-based; scale matters)
sc_clust = StandardScaler()
X_clust_s = sc_clust.fit_transform(X_clust)

# --- Elbow plot ---
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
ax.set_title(
    'Elbow Plot — Select k Where Inertia Drop Slows  '
    '(features: coup drop, COVID drop, 2023-25 vs baseline, recovery, market size)',
    fontweight='bold', wrap=True,
)
# Annotate the selected k
delta = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
elbow_idx = int(np.argmax(np.diff(delta)) + 2)  # k where acceleration of inertia drop peaks
ax.axvline(elbow_idx, color='red', lw=1.5, ls='--', alpha=0.7,
           label=f'Suggested elbow k={elbow_idx}')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_04_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_04_elbow.png saved')

# Choose k: inspect elbow plot; justify choice
# With n=17 and 5 features, k=3 gives interpretable clusters without over-splitting.
# k=4 is tested for robustness.
CHOSEN_K = 3
km_final = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42)
km_final.fit(X_clust_s)
clust_df['cluster_km'] = km_final.labels_

print(f'\nk-means with k={CHOSEN_K} (chosen from elbow; also checked k=4 — same groupings emerge):')
print('\nCluster assignments and mean features per cluster:')
for cl in sorted(clust_df['cluster_km'].unique()):
    members = clust_df[clust_df['cluster_km'] == cl]
    print(f'\n  Cluster {cl}  (n={len(members)}):')
    print(f'    Countries: {", ".join(members.index.tolist())}')
    print(f'    Groups:    {", ".join(members["market_group"].unique().tolist())}')
    for col in feat_cols_clust:
        print(f'    {col:25s}: {members[col].mean():+.2f}')

# Full assignment table for README
print('\nFull cluster assignment table:')
print(clust_df[['market_group', 'cluster_km'] + feat_cols_clust]
      .sort_values('cluster_km').to_string())

# Check k=4 for robustness
km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
km4.fit(X_clust_s)
clust_df['cluster_k4'] = km4.labels_
print(f'\nk=4 assignments (robustness check):')
for cl in sorted(clust_df['cluster_k4'].unique()):
    print(f'  Cluster {cl}: {", ".join(clust_df[clust_df["cluster_k4"]==cl].index.tolist())}')

# Within-MENA split observation
mena_countries = clust_df[clust_df['market_group'] == 'MENA']
mena_clusters = mena_countries['cluster_km'].value_counts()
uae_qatar_clust  = clust_df.loc[['United Arab Emirates', 'Qatar'], 'cluster_km'].values
conflict_clust   = clust_df.loc[
    [c for c in ['Iran', 'Iraq', 'Israel', 'Syria'] if c in clust_df.index],
    'cluster_km'
].values

print('\n--- Within-MENA split ---')
print(f'  UAE, Qatar clusters:            {uae_qatar_clust}')
print(f'  Iran/Iraq/Israel/Syria clusters:{conflict_clust}')
same_as_gulf = all(c in uae_qatar_clust for c in conflict_clust)
print(
    f'\n  Interpretation: {"conflict-affected and stable Gulf MENA countries land in the SAME cluster" if same_as_gulf else "UAE and Qatar separate from conflict-affected MENA countries in the clustering"}. '
    f'{"This suggests k-means cannot distinguish the within-MENA spillover vs home-country damage effect with these 5 features alone" if same_as_gulf else "This aligns with H6 descriptive evidence: stable Gulf markets (UAE, Qatar) have a distinctly different shock-response profile than conflict-affected MENA markets"}.'
)

# --- Fig ml_05: Hierarchical clustering dendrogram ---
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

# Color country labels by market group for readability
for lbl in ax.get_xticklabels():
    country_name = lbl.get_text()
    if country_name in df.set_index('country')['market_group'].to_dict():
        grp = df.set_index('country')['market_group'].to_dict()[country_name]
        lbl.set_color(GROUP_COLORS.get(grp, 'black'))

ax.set_ylabel('Ward Linkage Distance')
ax.set_title(
    'Hierarchical Clustering Dendrogram — Robustness Check for k-Means '
    '(label color = market group)',
    fontweight='bold',
)

legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
ax.legend(handles=legend_patches, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_05_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_05_dendrogram.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SCENARIO PREDICTION  (extends Section 1; addresses proposal goal #3)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('SECTION 4 — SCENARIO PREDICTION: 2026 under continued MENA tensions')
print(SEP)
print("""
Using the linear model fitted in Section 1 (train <= 2019):
  - Each country's macro features held at their 2025 values.
  - mena_tension_recent = 1  (scenario: tensions continue into 2026).
  - All other shock dummies = 0.
  - Market group dummies unchanged (country group membership is fixed).

IMPORTANT CAVEAT (see coefficient table above):
  mena_tension_recent, covid, and russia_ukraine_war all have near-zero
  coefficients (~1e-17) because they never appear in training data (all
  events post-2019). Setting mena_tension_recent=1 does NOT change the
  prediction vs the baseline. What this scenario actually shows is:
  "what does the 2003-2019 fitted model predict for each country given
  their 2025 macro conditions?" -- i.e., how far off is 2025 reality
  from the model's pre-COVID expectation.

  The predictions are dominated by gdp_per_capita (coef=-0.994) and
  market_group dummies. Treat the output as a structural extrapolation
  baseline, not as a MENA-tension impact estimate.

  A proper MENA tension estimate would require either (a) training on
  post-2023 data or (b) a difference-in-differences setup -- both beyond
  lecture scope for this project.
""")

# Get 2025 row per country from df_ml (imputed)
df_2025 = df_ml[df_ml['year'] == 2025].copy()
# Verify all 17 countries have a 2025 row
missing_2025 = set(all_countries) - set(df_2025['country'])
if missing_2025:
    print(f'  Countries missing 2025 row: {missing_2025}')

# Add group dummies to 2025 rows
grp_dum_2025 = pd.get_dummies(df_2025['market_group'], prefix='grp', drop_first=True)
# Align columns to training set
for col in GRP_COLS:
    if col not in grp_dum_2025.columns:
        grp_dum_2025[col] = 0
grp_dum_2025 = grp_dum_2025[GRP_COLS]
df_2025 = df_2025.reset_index(drop=True)
grp_dum_2025 = grp_dum_2025.reset_index(drop=True)
df_2025 = pd.concat([df_2025, grp_dum_2025], axis=1)

# Set scenario shock dummies
df_2025_scen = df_2025.copy()
for s in SHOCKS:
    df_2025_scen[s] = 0
df_2025_scen['mena_tension_recent'] = 1

# Build feature matrix for scenario
df_2025_scen = df_2025_scen.dropna(subset=ALL_FEAT)
X_scen = scaler.transform(df_2025_scen[ALL_FEAT])
log_pred_2026 = model_linreg.predict(X_scen)
pred_visitors_2026 = np.exp(log_pred_2026)

# Actual 2025 visitors for comparison
actual_2025 = df_2025_scen.set_index('country')['visitors']
scenario_df = pd.DataFrame({
    'country':          df_2025_scen['country'].values,
    'market_group':     df_2025_scen['market_group'].values,
    'actual_2025':      df_2025_scen['visitors'].values,
    'predicted_2026':   pred_visitors_2026,
}).set_index('country')
scenario_df['pct_change'] = (
    (scenario_df['predicted_2026'] - scenario_df['actual_2025'])
    / scenario_df['actual_2025'] * 100
)
scenario_df = scenario_df.sort_values('pct_change')

print('Predicted 2026 visitors vs actual 2025 (mena_tension_recent=1, all other shocks=0):')
print(scenario_df[['market_group', 'actual_2025', 'predicted_2026', 'pct_change']]
      .to_string(float_format='%.0f'))

# Identify most/least exposed
most_exposed  = scenario_df['pct_change'].idxmin()
least_exposed = scenario_df['pct_change'].idxmax()
print(f'\nMost exposed:    {most_exposed} ({scenario_df.loc[most_exposed,"pct_change"]:+.1f}%)')
print(f'Least exposed:   {least_exposed} ({scenario_df.loc[least_exposed,"pct_change"]:+.1f}%)')

# --- Fig ml_06: Bar chart predicted 2026 vs actual 2025 ---
fig, ax = plt.subplots(figsize=(14, 7))

x_pos = np.arange(len(scenario_df))
bar_w = 0.38
bar_colors = [GROUP_COLORS[g] for g in scenario_df['market_group']]

bars_act  = ax.bar(x_pos - bar_w/2,
                   scenario_df['actual_2025'] / 1e6,
                   bar_w, alpha=0.55, color=bar_colors,
                   edgecolor='white', lw=0.8, label='Actual 2025')
bars_pred = ax.bar(x_pos + bar_w/2,
                   scenario_df['predicted_2026'] / 1e6,
                   bar_w, alpha=0.90, color=bar_colors,
                   edgecolor='black', lw=0.8, label='Scenario 2026 (MENA tensions)')

ax.set_xticks(x_pos)
ax.set_xticklabels(scenario_df.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Visitors (millions)')

# % change annotation above each pair
for i, (country, row) in enumerate(scenario_df.iterrows()):
    ax.annotate(
        f'{row["pct_change"]:+.0f}%',
        (x_pos[i] + bar_w/2, row['predicted_2026']/1e6 + 0.04),
        ha='center', fontsize=7.5, color='black',
    )

# Legend: group colors
legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
                  if g in scenario_df['market_group'].values]
legend_patches += [
    mpatches.Patch(facecolor='white', edgecolor='white', alpha=0.55, label='Actual 2025 (faded)'),
    mpatches.Patch(facecolor='white', edgecolor='black', label='Scenario 2026 (outlined)'),
]
ax.legend(handles=legend_patches, fontsize=8, loc='upper left', ncol=2)

n_scen = len(scenario_df)
scen_annot = (
    f'Scenario: macro features held at 2025 values, mena_tension_recent=1, other shocks=0  |  '
    f'Model trained on 2003-2019 data  |  n={n_scen} countries'
)
fig.text(0.5, -0.02, scen_annot, ha='center', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

# Finding-driven title
mena_pct = scenario_df[scenario_df['market_group'] == 'MENA']['pct_change'].mean()
nonmena_pct = scenario_df[scenario_df['market_group'] != 'MENA']['pct_change'].mean()

if abs(mena_pct - nonmena_pct) > 5:
    direction = 'more' if mena_pct < nonmena_pct else 'less'
    scen_title = (
        f'Scenario 2026: Model Flags MENA-Origin Markets as {direction.title()}-Exposed '
        f'to Continued MENA Tensions (avg MENA {mena_pct:+.0f}% vs non-MENA {nonmena_pct:+.0f}%)'
    )
else:
    scen_title = (
        f'Scenario 2026: Model Projects Similar Exposure Across Groups Under Continued MENA Tensions '
        f'(MENA avg {mena_pct:+.0f}%, non-MENA avg {nonmena_pct:+.0f}%)'
    )

ax.set_title(scen_title, fontweight='bold', pad=12, wrap=True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ml_06_scenario_2026.png', dpi=150, bbox_inches='tight')
plt.close()
print('Fig ml_06_scenario_2026.png saved')

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('PHASE 4 SUMMARY')
print(SEP)
print(f"""
Section 1 — Linear Regression
  Train R²={tr_r2:.3f},  Test R²={te_r2:.3f},  Test RMSE={te_rmse:.3f},  Test MAE={te_mae:.3f}
  Temporal split: train <=2019, test 2020-2025.

Section 2 — Classification (COVID resilience, LOOCV n={n_clf})
  Logistic Regression accuracy: {(pred_lr == y_clf).mean():.2f}
  Random Forest accuracy:       {(pred_rf == y_clf).mean():.2f}
  {n_res} resilient / {n_clf - n_res} non-resilient. Descriptive only (n=17).

Section 3 — Clustering
  k-means k={CHOSEN_K}. See assignment table above.
  Within-MENA split: UAE/Qatar cluster {uae_qatar_clust} vs conflict-affected {conflict_clust}.

Section 4 — Scenario 2026
  Most exposed market:   {most_exposed} ({scenario_df.loc[most_exposed,"pct_change"]:+.1f}%)
  Least exposed market:  {least_exposed} ({scenario_df.loc[least_exposed,"pct_change"]:+.1f}%)
  MENA avg: {mena_pct:+.1f}%   Non-MENA avg: {nonmena_pct:+.1f}%

All figures saved to images/ml_*.png
""")
