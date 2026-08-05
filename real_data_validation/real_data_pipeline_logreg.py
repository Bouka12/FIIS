"""
real_data_pipeline.py
---------------------
Downloads and processes all real-world imbalanced datasets:
  - 24 KEEL datasets
  - 26 imblearn datasets

Evaluation: 5x6 repeated stratified CV = 30 runs per configuration.

Key design principle:
  FIIS is computed ONCE on the original training data (X_tr, y_tr)
  and assigned identically to ALL remedies. This is correct because:
  - FIIS diagnoses the information imbalance structure of the original data
  - Resampled data FIIS reflects the artificial distribution, not the
    actual imbalance — meaningless for diagnosis
  - All remedies receive the same FIIS so the diagnostic is consistent

Fixes:
  - Sex {M,F,I} one-hot encoded (3 binary columns)
  - BorderlineSMOTE m_neighbors adaptive
  - fiis_dominant (string) excluded from numeric aggregation
  - winequality/poker: IR-range URL tried first
  - FIIS computed on original training data for ALL remedies
"""

import os
import io
import zipfile
import urllib.request
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    average_precision_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE
from imblearn.datasets import fetch_datasets

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "keel")
N_SPLITS    = 5
N_REPEATS   = 6      # 5x6 = 30 runs
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)


# ─────────────────────────────────────────────
# KEEL dataset catalogue
# ─────────────────────────────────────────────
KEEL_DATASETS = {
    'new-thyroid1':             'imb_IRlowerThan9', # no missing values
    'new-thyroid2':             'imb_IRlowerThan9', # no missing values
    'glass-0-1-2-3_vs_4-5-6':  'imb_IRlowerThan9', # no missing values
    'ecoli-0_vs_1':             'imb_IRlowerThan9', # no missing values
    'vehicle0':                 'imb_IRlowerThan9', # no missing values
    'vehicle1':                 'imb_IRlowerThan9', # no missing values
    'vehicle2':                 'imb_IRlowerThan9', # no missing values
    'vehicle3':                 'imb_IRlowerThan9', # no missing values
    'segment0':                 'imb_IRlowerThan9', # no missing values
    'wisconsin':                'imb_IRlowerThan9', # no missing values
    'glass1':                   'imb_IRlowerThan9', # no missing values
    'pima':                     'imb_IRlowerThan9', # no missing values
    'iris0':                     'imb_IRlowerThan9', # no missing values
    'glass0':                    'imb_IRlowerThan9', # no missing values
    'yeast1':                    'imb_IRlowerThan9', # no missing values
    'haberman':                  'imb_IRlowerThan9', # no missing values
    'ecoli1':                    'imb_IRlowerThan9', # no missing values
    'ecoli2':                    'imb_IRlowerThan9', # no missing values
    'yeast3':                    'imb_IRlowerThan9', # no missing values
    'glass6':                    'imb_IRlowerThan9', # no missing values
    'page-blocks0':                'imb_IRlowerThan9', # no missing values
    #----------------------higher than 9 Part I----------------------#
    'yeast-2_vs_4':                 'imb_IR_9To15',
    'yeast-0-5-6-7-9_vs_4':          'imb_IR_9To15',
    'vowel0':                        'imb_IR_9To15',
    'glass-0-1-6_vs_2':             'imb_IR_9To15',
    'glass2':                        'imb_IR_9To15',
    'shuttle-c0-vs-c4':        'imb_IR_9To15',
    'yeast-1_vs_7':             'imb_IR_9To15',
    'glass4':                        'imb_IR_9To15',
    'ecoli4':                   'imb_IR_15To35',
    'page-blocks-1-3_vs_4':     'imb_IR_9To15',
    'abalone9-18':              'imb_IR_9To15',
    'glass-0-1-6_vs_5':        'imb_IR_9To15',
    'shuttle-c2-vs-c4':             'imb_IR_9To15',
    'yeast-1-4-5-8_vs_7':             'imb_IR_9To15',
    'glass5':                        'imb_IR_9To15',
    'yeast-2_vs_8':                 'imb_IR_9To15',
    'yeast4':                      'imb_IR_9To15',
    'yeast-1-2-8-9_vs_7':           'imb_IR_9To15',
    'yeast5':                   'imb_IR_9To15',
    'ecoli-0-1-3-7_vs_2-6':     'imb_IR_9To15',
    'yeast6':                   'imb_IR_9To15',
    #----------------------higher than 9 Part II-----------------------#
    'ecoli-0-3-4_vs_5':         'imb_IR_9To15',
    'ecoli-0-6-7_vs_3-5':       'imb_IR_9To15',
    
    # --------------------------------------------#
    'yeast4':                   'imb_IR_15To35',
    
    'glass-0-1-4-6_vs_2':      'imb_IR_15To35',
    'abalone-17_vs_7-8-9-10':   'imb_IR_15To35',
    'yeast-1-2-8-9_vs_7':             'imb_IR_15To35',
    'yeast-2_vs_8':                        'imb_IR_15To35',
    'shuttle-c2-vs-c4':                     'imb_IR_15To35',
    'page-blocks-1-3_vs_4':                          'imb_IR_15To35',
    'yeast5':                   'imb_IR_35To100',
    'yeast6':                   'imb_IR_35To100',
    'ecoli-0-1-3-7_vs_2-6':    'imb_IR_35To100'

}

KEEL_URL_PATTERNS = [
    "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/{ir_range}/{name}.zip",
    "https://sci2s.ugr.es/keel/dataset/data/imbalanced/{name}.zip",
]


# ─────────────────────────────────────────────
# KEEL .dat parser — one-hot encoding for
# categorical attributes (Sex {M,F,I})
# ─────────────────────────────────────────────

def parse_keel_dat(content: str):
    """
    Parse KEEL .dat file into X (float array), y (binary array).

    Categorical attributes (e.g. Sex {M,F,I}) are one-hot encoded,
    producing one binary column per category value, consistent with
    Alcala-Fdez et al. (2011) KEEL preprocessing.
    """
    lines      = content.strip().split('\n')
    attributes = []
    attr_types = []
    attr_cats  = []
    data_start = False
    data_lines = []
    class_idx  = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        low = line.lower()

        if low.startswith('@relation'):
            continue

        elif low.startswith('@attribute'):
            rest = line[len('@attribute'):].strip()
            if rest.startswith("'") or rest.startswith('"'):
                quote = rest[0]
                end   = rest.index(quote, 1)
                attr_name = rest[1:end]
                attr_type = rest[end+1:].strip().lower()
            else:
                parts     = rest.split(None, 1)
                attr_name = parts[0]
                attr_type = parts[1].strip().lower() if len(parts) > 1 else ''

            attributes.append(attr_name)
            if attr_type.startswith('{') or attr_type in ('string', 'nominal'):
                attr_types.append('categorical')
                cats_str = attr_type.strip('{}')
                cats = [c.strip().lower() for c in cats_str.split(',')]
                attr_cats.append(cats)
            else:
                attr_types.append('numeric')
                attr_cats.append([])

        elif (low.startswith('@input') or low.startswith('@output') or
              low.startswith('@datos')):
            continue

        elif low.startswith('@data'):
            data_start = True
            for i, a in enumerate(attributes):
                if a.lower() in ('class', 'clase', 'target', 'output'):
                    class_idx = i
                    break
            if class_idx is None:
                class_idx = len(attributes) - 1
            continue

        elif data_start:
            line = line.split('%')[0].strip()
            if line:
                data_lines.append(line)

    if not attributes or not data_lines:
        return None, None, []

    if class_idx is None:
        class_idx = len(attributes) - 1

    rows = []
    for line in data_lines:
        vals = [v.strip() for v in line.split(',')]
        if len(vals) == len(attributes):
            rows.append(vals)

    if not rows:
        return None, None, []

    df = pd.DataFrame(rows, columns=attributes)

    class_col = attributes[class_idx]
    y_raw     = df[class_col].str.strip().str.lower()
    valid_cls = ~y_raw.isin(['?', 'nan', '', 'none'])
    df        = df[valid_cls].reset_index(drop=True)
    y_raw     = y_raw[valid_cls].reset_index(drop=True)

    if len(df) == 0:
        return None, None, []

    counts    = y_raw.value_counts()
    min_class = counts.idxmin()
    y         = (y_raw == min_class).astype(int).values

    feat_cols  = [a for i, a in enumerate(attributes) if i != class_idx]
    feat_types = [t for i, t in enumerate(attr_types) if i != class_idx]
    feat_cats  = [c for i, c in enumerate(attr_cats)  if i != class_idx]

    X_parts   = []
    col_names = []

    for col, ftype, cats in zip(feat_cols, feat_types, feat_cats):
        col_vals = df[col].str.strip()
        if ftype == 'categorical':
            col_lower = col_vals.str.lower()
            for cat in cats:
                X_parts.append((col_lower == cat).astype(float).values)
                col_names.append(f"{col}_{cat}")
        else:
            num_vals = col_vals.replace('?', np.nan)
            X_parts.append(pd.to_numeric(num_vals, errors='coerce').values)
            col_names.append(col)

    if not X_parts:
        return None, None, []

    X_arr = np.column_stack(X_parts).astype(float)
    valid_rows = ~np.isnan(X_arr).any(axis=1)
    X_arr = X_arr[valid_rows]
    y     = y[valid_rows]

    if len(X_arr) == 0:
        return None, None, col_names

    return X_arr, y, col_names


def download_keel(name, ir_range):
    """Download and parse one KEEL dataset with local npz cache."""
    cache_path = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        if len(d['X']) > 0:
            return d['X'], d['y']
        else:
            os.remove(cache_path)

    for pattern in KEEL_URL_PATTERNS:
        url = pattern.format(name=name, ir_range=ir_range)
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as r:
                if r.status != 200:
                    continue
                content = r.read()
                if len(content) == 0:
                    continue

            with zipfile.ZipFile(io.BytesIO(content)) as z:
                dat_files = [
                    f for f in z.namelist()
                    if f.endswith('.dat') and
                    not f.startswith('__MACOSX') and
                    not any(x in f.lower() for x in
                            ['fold', '-tra', '-tst', '5-', '10-'])
                ]
                if not dat_files:
                    dat_files = [f for f in z.namelist()
                                 if f.endswith('.dat') and
                                 not f.startswith('__MACOSX')]
                if not dat_files:
                    continue

                best_X, best_y = None, None
                for fname in dat_files:
                    dat_content = z.read(fname).decode('utf-8', errors='ignore')
                    X, y, _ = parse_keel_dat(dat_content)
                    if X is not None and len(X) > 0:
                        if best_X is None or len(X) > len(best_X):
                            best_X, best_y = X, y

                if best_X is not None and len(best_X) > 0:
                    np.savez(cache_path, X=best_X, y=best_y)
                    return best_X, best_y

        except Exception as e:
            print(f"    URL failed: {str(e)[:80]}")
            continue

    return None, None


# ─────────────────────────────────────────────
# FIIS computation — always on original data
# ─────────────────────────────────────────────

def compute_fiis(clf, X_orig, y_orig):
    """
    Compute FIIS from the model fitted on original training data.

    Parameters
    ----------
    clf    : fitted LogisticRegression
    X_orig : original training features (before any resampling)
    y_orig : original training labels (before any resampling)

    This must always be called with the original (X_tr, y_tr),
    never with resampled data, to ensure the FIIS reflects the
    true information imbalance structure of the dataset.
    """
    X  = np.array(X_orig, dtype=float)
    p  = clf.predict_proba(X)[:, 1]
    w  = p * (1.0 - p)
    ns = np.sum(X ** 2, axis=1)

    m0 = (y_orig == 0); m1 = (y_orig == 1)
    n0 = m0.sum();       n1 = m1.sum()
    if n0 == 0 or n1 == 0:
        return {}

    mw0  = w[m0].mean();  mw1  = w[m1].mean()
    mns0 = ns[m0].mean(); mns1 = ns[m1].mean()
    mp0  = (w[m0] * ns[m0]).mean()
    mp1  = (w[m1] * ns[m1]).mean()

    tr0 = n0 * mp0; tr1 = n1 * mp1
    R   = tr1 / tr0 if tr0 > 0 else np.nan
    R_n = n1 / n0
    R_p = mw1  / mw0  if mw0  > 0 else np.nan
    R_g = mns1 / mns0 if mns0 > 0 else np.nan

    d0  = mw0 * mns0; d1 = mw1 * mns1
    e0  = (mp0 - d0) / d0 if d0 > 0 else np.nan
    e1  = (mp1 - d1) / d1 if d1 > 0 else np.nan
    C   = (1 + e1) / (1 + e0) \
          if (not np.isnan(e0) and not np.isnan(e1)
              and abs(1 + e0) > 1e-10) else np.nan

    def slog(x):
        return np.log(x) if (x is not None and
                              not np.isnan(x) and x > 0) else np.nan

    lRn = slog(R_n); lRp = slog(R_p)
    lRg = slog(R_g); lC  = slog(C)

    abs_logs = {
        'n': abs(lRn) if lRn is not None and not np.isnan(lRn) else 0,
        'p': abs(lRp) if lRp is not None and not np.isnan(lRp) else 0,
        'g': abs(lRg) if lRg is not None and not np.isnan(lRg) else 0,
        'C': abs(lC)  if lC  is not None and not np.isnan(lC)  else 0,
    }
    total = sum(abs_logs.values())
    phi   = {k: v/total for k, v in abs_logs.items()} \
            if total > 0 else {k: 0.0 for k in abs_logs}

    return {
        'R_n': R_n, 'R_p': R_p, 'R_g': R_g, 'C': C, 'R': R,
        'log_Rn': lRn, 'log_Rp': lRp, 'log_Rg': lRg, 'log_C': lC,
        'phi_n': phi['n'], 'phi_p': phi['p'],
        'phi_g': phi['g'], 'phi_C': phi['C'],
        'dominant': max(phi, key=phi.get),
        'trace_0': tr0, 'trace_1': tr1,
    }


# ─────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_prob, tau):
    y_pred = (y_prob >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    gm   = np.sqrt(sens * spec)
    ba   = (sens + spec) / 2
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    try:    auc = roc_auc_score(y_true, y_prob)
    except: auc = np.nan
    try:    pr_auc = average_precision_score(y_true, y_prob)
    except: pr_auc = np.nan
    return {
        'AUC': auc, 'PR_AUC': pr_auc, 'G_mean': gm, 'F1': f1,
        'sensitivity': sens, 'specificity': spec,
        'precision': prec, 'balanced_accuracy': ba,
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
    }


# ─────────────────────────────────────────────
# Remedy strategies
# FIIS computed ONCE on original data, shared by all remedies
# ─────────────────────────────────────────────

def fit_logreg(X, y, class_weight=None, seed=SEED):
    clf = LogisticRegression(
        max_iter=1000, solver='lbfgs',
        random_state=seed, class_weight=class_weight)
    clf.fit(X, y)
    return clf


def apply_remedies(X_tr, y_tr, X_te, y_te, seed=SEED):
    """
    Apply all remedy strategies and evaluate on test set.

    FIIS is computed once on the original (X_tr, y_tr) using the
    baseline logistic regression model, then assigned identically
    to all remedies. This ensures FIIS always reflects the original
    information imbalance structure, not the resampled distribution.
    """
    n1  = (y_tr == 1).sum()
    pi  = n1 / len(y_tr)
    res = {}

    # ── Compute FIIS ONCE on original training data ──
    # Use baseline model (no correction) fitted on original data
    clf_base  = fit_logreg(X_tr, y_tr, seed=seed)
    prob_base = clf_base.predict_proba(X_te)[:, 1]
    fiis_orig = compute_fiis(clf_base, X_tr, y_tr)
    # This FIIS will be assigned to ALL remedies below

    # 1. No correction
    res['no_correction'] = {
        'metrics': compute_metrics(y_te, prob_base, tau=0.5),
        'fiis':    fiis_orig,
    }

    # 2. Threshold correction
    res['threshold_correction'] = {
        'metrics': compute_metrics(y_te, prob_base, tau=pi),
        'fiis':    fiis_orig,
    }

    # 3. ROS
    try:
        Xr, yr = RandomOverSampler(
            random_state=seed).fit_resample(X_tr, y_tr)
        clf_r = fit_logreg(Xr, yr, seed=seed)
        res['ROS'] = {
            'metrics': compute_metrics(
                y_te, clf_r.predict_proba(X_te)[:, 1], tau=0.5),
            'fiis': fiis_orig,   # ← original data FIIS
        }
    except Exception as e:
        res['ROS'] = {'metrics': {}, 'fiis': fiis_orig,
                      'error': str(e)}

    # 4. SMOTE
    try:
        k = min(5, n1 - 1)
        if k >= 1:
            Xs, ys = SMOTE(
                random_state=seed, k_neighbors=k).fit_resample(X_tr, y_tr)
            clf_s = fit_logreg(Xs, ys, seed=seed)
            res['SMOTE'] = {
                'metrics': compute_metrics(
                    y_te, clf_s.predict_proba(X_te)[:, 1], tau=0.5),
                'fiis': fiis_orig,   # ← original data FIIS
            }
        else:
            res['SMOTE'] = {'metrics': {}, 'fiis': fiis_orig,
                            'error': 'too few minority'}
    except Exception as e:
        res['SMOTE'] = {'metrics': {}, 'fiis': fiis_orig,
                        'error': str(e)}

    # 5. BorderlineSMOTE
    # k_neighbors: synthesis neighbours (adaptive)
    # m_neighbors: borderline detection neighbours (adaptive, >= k)
    try:
        k = min(5,  n1 - 1)
        m = min(10, n1 - 1)
        m = max(m, k)
        if k >= 1:
            Xb, yb = BorderlineSMOTE(
                random_state=seed,
                k_neighbors=k,
                m_neighbors=m
            ).fit_resample(X_tr, y_tr)
            clf_b = fit_logreg(Xb, yb, seed=seed)
            res['BorderlineSMOTE'] = {
                'metrics': compute_metrics(
                    y_te, clf_b.predict_proba(X_te)[:, 1], tau=0.5),
                'fiis': fiis_orig,   # ← original data FIIS
            }
        else:
            res['BorderlineSMOTE'] = {'metrics': {}, 'fiis': fiis_orig,
                                       'error': 'too few minority'}
    except Exception as e:
        res['BorderlineSMOTE'] = {'metrics': {}, 'fiis': fiis_orig,
                                   'error': str(e)}

    # 6. Class weighting
    try:
        clf_cw = fit_logreg(X_tr, y_tr,
                             class_weight='balanced', seed=seed)
        res['class_weighting'] = {
            'metrics': compute_metrics(
                y_te, clf_cw.predict_proba(X_te)[:, 1], tau=0.5),
            'fiis': fiis_orig,   # ← original data FIIS
        }
    except Exception as e:
        res['class_weighting'] = {'metrics': {}, 'fiis': fiis_orig,
                                   'error': str(e)}

    return res


# ─────────────────────────────────────────────
# Cross-validated experiment
# ─────────────────────────────────────────────

def run_dataset(name, X, y, source):
    """5x6 repeated stratified CV = 30 runs."""
    if X is None or len(X) == 0:
        print(f"  ✗ {name}: empty dataset, skipping")
        return []

    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    if n1 < N_SPLITS:
        print(f"  ✗ {name}: too few minority samples ({n1}), skipping")
        return []

    ir = n0 / n1

    try:
        scaler = StandardScaler()
        X      = scaler.fit_transform(X)
    except Exception as e:
        print(f"  ✗ {name}: scaling failed: {e}")
        return []

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    rows = []
    for rep_fold, (tr_idx, te_idx) in enumerate(rskf.split(X, y)):
        repeat = rep_fold // N_SPLITS
        fold   = rep_fold %  N_SPLITS
        seed   = SEED + rep_fold

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if (y_tr == 1).sum() < 2 or (y_te == 1).sum() < 1:
            continue

        try:
            remedy_results = apply_remedies(
                X_tr, y_tr, X_te, y_te, seed=seed)
        except Exception as e:
            print(f"  ✗ {name} run {rep_fold}: {e}")
            continue

        for remedy, res in remedy_results.items():
            if not res.get('metrics'):
                continue
            row = {
                'dataset': name, 'source': source,
                'n': int(len(y)), 'n0': int(n0),
                'n1': int(n1), 'IR': round(ir, 3),
                'd': int(X.shape[1]),
                'repeat': repeat, 'fold': fold,
                'run': rep_fold, 'remedy': remedy,
            }
            row.update(res['metrics'])
            for k, v in res.get('fiis', {}).items():
                row[f'fiis_{k}'] = v
            rows.append(row)

    return rows


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    all_rows = []
    failed   = []

    # ── KEEL ──
    print("\n" + "="*60)
    print("KEEL Datasets")
    print("="*60)

    for name, ir_range in KEEL_DATASETS.items():
        print(f"\n  [{name}]...")
        X, y = download_keel(name, ir_range)

        if X is None or len(X) == 0:
            print(f"  ✗ Download/parse failed — skipping")
            failed.append(('KEEL', name))
            continue

        ir = (y == 0).sum() / max((y == 1).sum(), 1)
        print(f"  Downloaded: X={X.shape}, IR={ir:.2f}")
        rows = run_dataset(name, X, y, source='KEEL')
        all_rows.extend(rows)
        print(f"  ✓ {len(rows)} result rows")

    # ── imblearn ──
    print("\n" + "="*60)
    print("imblearn Datasets")
    print("="*60)

    try:
        imblearn_data = fetch_datasets()
        for name, data in imblearn_data.items():
            print(f"\n  [{name}]...")
            X = np.array(data.data,   dtype=float)
            y = np.array(data.target, dtype=int)
            y = np.where(y == 1, 1, 0)
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            rows = run_dataset(name, X, y, source='imblearn')
            all_rows.extend(rows)
            print(f"  ✓ {len(rows)} result rows")
    except Exception as e:
        print(f"  ✗ imblearn fetch failed: {e}")
        failed.append(('imblearn', 'all'))

    # ── Save ──
    if not all_rows:
        print("\n✗ No results.")
        return

    df = pd.DataFrame(all_rows)

    # Raw results
    raw_path = os.path.join(RESULTS_DIR, "real_data_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\n✓ Raw:  {raw_path}  ({len(df)} rows, "
          f"{df['dataset'].nunique()} datasets)")

    # Aggregated results — numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    exclude  = ['repeat', 'fold', 'run', 'n', 'n0', 'n1', 'd',
                'TP', 'FP', 'FN', 'TN']
    agg_cols = [c for c in numeric_cols if c not in exclude]
    agg = (df.groupby(['dataset', 'source', 'IR', 'n', 'd', 'remedy'])
             [agg_cols].agg(['mean', 'std']).round(4))
    agg_path = os.path.join(RESULTS_DIR, "real_data_agg.csv")
    agg.to_csv(agg_path)
    print(f"✓ Agg:  {agg_path}")

    # FIIS summary — from no_correction, numeric cols + dominant mode
    fiis_num_cols = [c for c in df.columns
                     if c.startswith('fiis_') and
                     c != 'fiis_dominant' and
                     pd.api.types.is_numeric_dtype(df[c])]

    nc = df[df['remedy'] == 'no_correction']
    grp_cols = ['dataset', 'source', 'IR', 'n', 'd']

    fiis_mean = (nc.groupby(grp_cols)[fiis_num_cols]
                   .mean().round(4).reset_index())

    # Dominant factor: mode + stability (fraction of runs agreeing)
    def dominant_mode(x):
        return x.mode()[0] if len(x) > 0 else 'n'

    def dominant_stability(x):
        return round(x.value_counts().iloc[0] / len(x), 3) \
               if len(x) > 0 else 0.0

    dom_agg = (nc.groupby(grp_cols)['fiis_dominant']
                 .agg(fiis_dominant=dominant_mode,
                      fiis_dominant_stability=dominant_stability)
                 .reset_index())

    fiis_sum = fiis_mean.merge(dom_agg, on=grp_cols)
    fiis_path = os.path.join(RESULTS_DIR, "real_data_fiis.csv")
    fiis_sum.to_csv(fiis_path, index=False)
    print(f"✓ FIIS: {fiis_path}")

    if failed:
        print(f"\n⚠  Failed ({len(failed)}): {failed}")
    print("\nDone.")


if __name__ == "__main__":
    main()