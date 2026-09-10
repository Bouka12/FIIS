"""
vertebral_pipeline.py
---------------------
Case study: Vertebral Column Diagnosis (UCI, n=310, IR=2.1)

Applies the FIIS diagnostic framework to a single medical dataset:
  - Normal (minority, n=100) vs Abnormal (majority, n=210)
  - 6 biomechanical features, all continuous, no missing values

Protocol: 5x6 repeated stratified CV = 30 runs
Classifiers: LR, RF, XGBoost, SVM (same as main pipeline)
Remedies: No Correction, Threshold Correction, ROS, SMOTE,
          BorderlineSMOTE, Class Weighting
FIIS: computed once per fold on original training data via LR

Output:
  results/vertebral/vertebral_raw.csv     — per-run results
  results/vertebral/vertebral_fiis.csv    — FIIS per fold
  results/vertebral/vertebral_agg.csv     — aggregated per remedy×classifier
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "vertebral")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS  = 5
N_REPEATS = 6     # 30 runs total
SEED      = 42
EPSILON   = 0.05  # compensation threshold


# ─────────────────────────────────────────────
# Classifiers
# ─────────────────────────────────────────────

def get_classifiers(seed=SEED):
    clfs = {
        'logistic_regression': LogisticRegression(
            max_iter=1000, solver='lbfgs', random_state=seed),
        'random_forest': RandomForestClassifier(
            n_estimators=100, random_state=seed),
    }
    if XGBOOST_AVAILABLE:
        clfs['xgboost'] = XGBClassifier(
            n_estimators=100, random_state=seed,
            eval_metric='logloss', verbosity=0,
            use_label_encoder=False)
    return clfs


def clone_classifier(clf, seed):
    params = clf.get_params()
    if 'random_state' in params:
        params['random_state'] = seed
    return clf.__class__(**params)


# ─────────────────────────────────────────────
# FIIS computation
# ─────────────────────────────────────────────

def compute_fiis(X, y, seed=SEED):
    clf = LogisticRegression(max_iter=1000, solver='lbfgs',
                             random_state=seed)
    clf.fit(X, y)
    p  = clf.predict_proba(X)[:, 1]
    w  = p * (1.0 - p)
    ns = np.sum(X ** 2, axis=1)

    m0 = (y == 0); m1 = (y == 1)
    n0 = m0.sum(); n1 = m1.sum()
    if n0 == 0 or n1 == 0:
        return {}

    mw0  = w[m0].mean();  mw1  = w[m1].mean()
    mns0 = ns[m0].mean(); mns1 = ns[m1].mean()
    mp0  = (w[m0] * ns[m0]).mean()
    mp1  = (w[m1] * ns[m1]).mean()

    tr0 = n0 * mp0; tr1 = n1 * mp1
    R   = tr1 / tr0 if tr0 > 0 else np.nan
    R_n = n1 / n0
    R_p = mw1 / mw0   if mw0  > 0 else np.nan
    R_g = mns1 / mns0 if mns0 > 0 else np.nan

    d0 = mw0 * mns0; d1 = mw1 * mns1
    e0 = (mp0 - d0) / d0 if d0 > 0 else np.nan
    e1 = (mp1 - d1) / d1 if d1 > 0 else np.nan
    C  = (1 + e1) / (1 + e0) \
         if (not np.isnan(e0) and not np.isnan(e1)
             and abs(1 + e0) > 1e-10) else np.nan

    # Archetype assignment (no log, natural scale)
    if not np.isnan(R_p) and not np.isnan(R_g):
        if R_p >= 1 and R_g >= 1:
            archetype = 'A'
        elif R_g < 1 and (np.isnan(R_p) or R_g <= R_p):
            archetype = 'C'
        elif not np.isnan(R_p) and R_p < 1:
            archetype = 'B'
        else:
            archetype = 'A'
    else:
        archetype = 'A'

    # Compensation regime
    if not np.isnan(R):
        if R > 1 + EPSILON:
            comp = 'O'
        elif R >= 1 - EPSILON:
            comp = 'B'
        else:
            comp = 'D'
    else:
        comp = 'D'

    return {
        'R_n': R_n, 'R_p': R_p, 'R_g': R_g, 'C': C, 'R': R,
        'archetype': archetype, 'compensation': comp,
        'trace_0': tr0, 'trace_1': tr1,
    }


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_prob, tau):
    y_pred = (y_prob >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gm   = np.sqrt(sens * spec)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    ba   = (sens + spec) / 2
    try:    auc = roc_auc_score(y_true, y_prob)
    except: auc = np.nan
    return {'AUC': auc, 'G_mean': gm, 'F1': f1,
            'sensitivity': sens, 'specificity': spec,
            'balanced_accuracy': ba}


# ─────────────────────────────────────────────
# Remedies
# ─────────────────────────────────────────────

def apply_remedies(X_tr, y_tr, X_te, y_te,
                   clf_template, fiis, seed=SEED):
    n1  = (y_tr == 1).sum()
    pi  = n1 / len(y_tr)
    res = {}

    def fit_predict(X, y):
        clf = clone_classifier(clf_template, seed)
        clf.fit(X, y)
        return clf.predict_proba(X_te)[:, 1]

    # No correction + threshold correction (same model)
    try:
        prob = fit_predict(X_tr, y_tr)
        res['no_correction']        = compute_metrics(y_te, prob, 0.5)
        res['threshold_correction'] = compute_metrics(y_te, prob, pi)
    except Exception as e:
        res['no_correction'] = {}
        res['threshold_correction'] = {}

    # ROS
    try:
        Xr, yr = RandomOverSampler(random_state=seed).fit_resample(X_tr, y_tr)
        res['ROS'] = compute_metrics(y_te, fit_predict(Xr, yr), 0.5)
    except: res['ROS'] = {}

    # SMOTE
    try:
        k = min(5, n1 - 1)
        if k >= 1:
            Xs, ys = SMOTE(random_state=seed,
                           k_neighbors=k).fit_resample(X_tr, y_tr)
            res['SMOTE'] = compute_metrics(y_te, fit_predict(Xs, ys), 0.5)
    except: res['SMOTE'] = {}

    # BorderlineSMOTE
    try:
        k = min(5, n1 - 1); m = max(min(10, n1-1), k)
        if k >= 1:
            Xb, yb = BorderlineSMOTE(
                random_state=seed, k_neighbors=k,
                m_neighbors=m).fit_resample(X_tr, y_tr)
            res['BorderlineSMOTE'] = compute_metrics(
                y_te, fit_predict(Xb, yb), 0.5)
    except: res['BorderlineSMOTE'] = {}

    # Class weighting
    try:
        clf_cw = clone_classifier(clf_template, seed)
        params = clf_cw.get_params()
        if 'class_weight' in params:
            clf_cw.set_params(class_weight='balanced')
        elif hasattr(clf_cw, 'scale_pos_weight'):
            clf_cw.set_params(
                scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1))
        clf_cw.fit(X_tr, y_tr)
        res['class_weighting'] = compute_metrics(
            y_te, clf_cw.predict_proba(X_te)[:,1], 0.5)
    except: res['class_weighting'] = {}

    return res


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # Load data
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__),
                     'data', 'vertebral', 'Vertebral.csv'))
    X = df.drop(columns=['Outcome']).values.astype(float)
    y = df['Outcome'].values.astype(int)



    n0 = (y == 0).sum(); n1 = (y == 1).sum()
    IR = n0 / n1
    print(f"Vertebral Column: n={len(y)}, n0={n0}, n1={n1}, IR={IR:.2f}")

    clf_dict = get_classifiers(SEED)
    rskf     = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    all_rows  = []
    fiis_rows = []

    for rep_fold, (tr_idx, te_idx) in enumerate(rskf.split(X, y)):
        repeat = rep_fold // N_SPLITS
        fold   = rep_fold %  N_SPLITS
        seed   = SEED + rep_fold

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Standardise
        scaler = StandardScaler()
        X_tr      = scaler.fit_transform(X_tr)
        X_te      = scaler.transform(X_te)
        
        # FIIS — once per fold
        fiis = compute_fiis(X_tr, y_tr, seed=seed)

        fiis_rows.append({
            'repeat': repeat, 'fold': fold, 'run': rep_fold,
            **{f'fiis_{k}': v for k, v in fiis.items()}
        })

        # Per classifier
        for clf_name, clf_template in clf_dict.items():
            remedy_res = apply_remedies(
                X_tr, y_tr, X_te, y_te,
                clf_template=clf_template,
                fiis=fiis, seed=seed)

            for remedy, metrics in remedy_res.items():
                if not metrics:
                    continue
                row = {
                    'dataset': 'vertebral_column',
                    'classifier': clf_name,
                    'remedy': remedy,
                    'repeat': repeat, 'fold': fold, 'run': rep_fold,
                    'IR': IR, 'n': len(y), 'n0': int(n0), 'n1': int(n1),
                }
                row.update(metrics)
                row.update({f'fiis_{k}': v for k, v in fiis.items()})
                all_rows.append(row)

    # Save raw
    df_raw  = pd.DataFrame(all_rows)
    df_fiis = pd.DataFrame(fiis_rows)

    raw_path  = os.path.join(RESULTS_DIR, 'vertebral_raw.csv')
    fiis_path = os.path.join(RESULTS_DIR, 'vertebral_fiis.csv')
    df_raw.to_csv(raw_path,   index=False)
    df_fiis.to_csv(fiis_path, index=False)
    print(f"Saved: {raw_path}  ({len(df_raw)} rows)")
    print(f"Saved: {fiis_path} ({len(df_fiis)} rows)")

    # Aggregate per remedy × classifier
    metric_cols = ['G_mean','AUC','F1','sensitivity','specificity',
                   'balanced_accuracy']
    agg = (df_raw.groupby(['classifier','remedy'])[metric_cols]
                 .agg(['mean','std']).round(4))
    agg_path = os.path.join(RESULTS_DIR, 'vertebral_agg.csv')
    agg.to_csv(agg_path)
    print(f"Saved: {agg_path}")

    return df_raw, df_fiis


if __name__ == "__main__":
    os.makedirs(os.path.join(
        os.path.dirname(__file__), 'data', 'vertebral'), exist_ok=True)
    import shutil
    shutil.copy(r'real_data_validation\results\vertebral\Vertebral.csv',
                os.path.join(os.path.dirname(__file__),
                             'data', 'vertebral', 'Vertebral.csv'))
    main()