"""
fiis_core.py
------------
Core utilities for the Fisher Information Imbalance Signature (FIIS).

Provides:
  - FIIS computation from a fitted logistic regression model
  - Classification metrics
  - Remedy application (No correction, Threshold correction,
    ROS, SMOTE, Class weighting)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler


# ─────────────────────────────────────────────
# 1.  FIIS COMPUTATION
# ─────────────────────────────────────────────

def compute_fiis(clf, X, y):
    """
    Compute the Fisher Information Imbalance Signature (FIIS)
    from a fitted logistic regression model.

    Parameters
    ----------
    clf : fitted LogisticRegression
    X   : array (n, d)  — training features (already augmented or not,
                          we work with raw X since sklearn handles intercept)
    y   : array (n,)    — binary labels {0, 1}, 1 = minority

    Returns
    -------
    dict with keys:
        R_n, R_p, R_g, C,
        eps_0, eps_1,
        trace_0, trace_1, R,
        log_Rn, log_Rp, log_Rg, log_C
    """
    X = np.array(X)
    y = np.array(y)

    # Predicted probabilities
    p = clf.predict_proba(X)[:, 1]          # shape (n,)
    w = p * (1.0 - p)                        # logistic weights p(1-p)

    # Squared norms
    x_norm_sq = np.sum(X ** 2, axis=1)      # shape (n,)

    # Class masks
    mask0 = (y == 0)
    mask1 = (y == 1)
    n0 = mask0.sum()
    n1 = mask1.sum()

    # ── Per-class means ──
    mean_w0   = w[mask0].mean()
    mean_w1   = w[mask1].mean()
    mean_ns0  = x_norm_sq[mask0].mean()
    mean_ns1  = x_norm_sq[mask1].mean()

    # ── Per-class mean of product p(1-p)*||x||^2 ──
    mean_prod0 = (w[mask0] * x_norm_sq[mask0]).mean()
    mean_prod1 = (w[mask1] * x_norm_sq[mask1]).mean()

    # ── Traces ──
    trace0 = n0 * mean_prod0
    trace1 = n1 * mean_prod1
    R      = trace1 / trace0 if trace0 > 0 else np.nan

    # ── Decomposition factors ──
    R_n = n1 / n0
    R_p = mean_w1  / mean_w0  if mean_w0  > 0 else np.nan
    R_g = mean_ns1 / mean_ns0 if mean_ns0 > 0 else np.nan

    # ── Coupling correction ──
    # eps_k = Cov_k / (mean_w_k * mean_ns_k)
    #       = (mean_prod_k - mean_w_k * mean_ns_k) / (mean_w_k * mean_ns_k)
    denom0 = mean_w0 * mean_ns0
    denom1 = mean_w1 * mean_ns1

    eps0 = (mean_prod0 - denom0) / denom0 if denom0 > 0 else np.nan
    eps1 = (mean_prod1 - denom1) / denom1 if denom1 > 0 else np.nan

    C = (1 + eps1) / (1 + eps0) if (not np.isnan(eps0)
                                     and not np.isnan(eps1)
                                     and (1 + eps0) != 0) else np.nan

    # ── Log-space FIIS vector ──
    log_Rn = np.log(R_n) if R_n > 0 else np.nan
    log_Rp = np.log(R_p) if R_p and R_p > 0 else np.nan
    log_Rg = np.log(R_g) if R_g and R_g > 0 else np.nan
    log_C  = np.log(C)   if C   and C  > 0 else np.nan

    return {
        "R_n": R_n, "R_p": R_p, "R_g": R_g, "C": C,
        "eps_0": eps0, "eps_1": eps1,
        "trace_0": trace0, "trace_1": trace1, "R": R,
        "log_Rn": log_Rn, "log_Rp": log_Rp,
        "log_Rg": log_Rg, "log_C": log_C,
    }


def dominance_profile(fiis: dict):
    """
    Compute the dominance weights phi_j for each FIIS component.

    phi_j = |log R_j| / sum_j |log R_j|

    Returns dict with keys phi_n, phi_p, phi_g, phi_C
    and 'dominant' (the name of the dominant factor).
    """
    logs = {
        "n": abs(fiis["log_Rn"]) if not np.isnan(fiis["log_Rn"]) else 0,
        "p": abs(fiis["log_Rp"]) if not np.isnan(fiis["log_Rp"]) else 0,
        "g": abs(fiis["log_Rg"]) if not np.isnan(fiis["log_Rg"]) else 0,
        "C": abs(fiis["log_C"])  if not np.isnan(fiis["log_C"])  else 0,
    }
    total = sum(logs.values())
    if total == 0:
        return {"phi_n": 0, "phi_p": 0, "phi_g": 0, "phi_C": 0,
                "dominant": "none"}
    phis = {k: v / total for k, v in logs.items()}
    dominant = max(phis, key=phis.get)
    return {
        "phi_n": phis["n"], "phi_p": phis["p"],
        "phi_g": phis["g"], "phi_C": phis["C"],
        "dominant": dominant,
    }


# ─────────────────────────────────────────────
# 2.  CLASSIFICATION METRICS
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_prob, tau=0.5):
    """
    Compute classification metrics at a given threshold tau.

    Returns dict with:
        AUC, PR_AUC, F1, G_mean,
        sensitivity (recall), specificity,
        precision, balanced_accuracy,
        TP, FP, FN, TN
    """
    y_pred = (y_prob >= tau).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                       labels=[0, 1]).ravel()

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision    = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1           = f1_score(y_true, y_pred, zero_division=0)
    g_mean       = np.sqrt(sensitivity * specificity)
    bal_acc      = (sensitivity + specificity) / 2.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan

    return {
        "AUC": auc, "PR_AUC": pr_auc,
        "F1": f1, "G_mean": g_mean,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "balanced_accuracy": bal_acc,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ─────────────────────────────────────────────
# 3.  REMEDY STRATEGIES
# ─────────────────────────────────────────────

def fit_logistic(X_train, y_train, class_weight=None, seed=42):
    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs",
        random_state=seed,
        class_weight=class_weight
    )
    clf.fit(X_train, y_train)
    return clf


def apply_remedies(X_train, y_train, X_test, y_test, seed=42):
    """
    Apply all five remedy strategies and return metrics + FIIS per remedy.

    Remedies
    --------
    1. No correction          : fit on original data, tau = 0.5
    2. Threshold correction   : fit on original data,
                                tau = n1/n (Van den Goorbergh)
    3. ROS                    : random oversampling, refit, tau = 0.5
    4. SMOTE                  : SMOTE oversampling, refit, tau = 0.5
    4.2 BorderlineSMOTE       : BorderlineSMOTE oversampling, refit, tau = 0.5
    5. Class weighting        : fit with inverse-frequency weights, tau = 0.5


    Returns
    -------
    dict keyed by remedy name, each value is a dict with
    'metrics' and 'fiis' sub-dicts.
    """
    n1 = (y_train == 1).sum()
    n  = len(y_train)
    pi_train = n1 / n          # training prevalence for threshold correction

    results = {}

    # ── 1. No correction ──
    clf_base = fit_logistic(X_train, y_train, seed=seed)
    prob_test = clf_base.predict_proba(X_test)[:, 1]
    results["no_correction"] = {
        "metrics": compute_metrics(y_test, prob_test, tau=0.5),
        "fiis":    compute_fiis(clf_base, X_train, y_train),
    }

    # ── 2. Threshold correction ──
    # Same fitted model as no_correction, only tau changes
    results["threshold_correction"] = {
        "metrics": compute_metrics(y_test, prob_test, tau=pi_train),
        "fiis":    results["no_correction"]["fiis"],  # identical model
    }

    # ── 3. ROS ──
    ros = RandomOverSampler(random_state=seed)
    X_ros, y_ros = ros.fit_resample(X_train, y_train)
    clf_ros = fit_logistic(X_ros, y_ros, seed=seed)
    prob_ros = clf_ros.predict_proba(X_test)[:, 1]
    results["ROS"] = {
        "metrics": compute_metrics(y_test, prob_ros, tau=0.5),
        "fiis":    compute_fiis(clf_ros, X_ros, y_ros),
    }

    # ── 4. SMOTE ──
    # SMOTE requires at least k_neighbors + 1 minority samples
    n1_train = (y_train == 1).sum()
    k_neighbors = min(5, n1_train - 1) if n1_train > 1 else 1
    try:
        smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
        X_smt, y_smt = smote.fit_resample(X_train, y_train)
        clf_smt = fit_logistic(X_smt, y_smt, seed=seed)
        prob_smt = clf_smt.predict_proba(X_test)[:, 1]
        results["SMOTE"] = {
            "metrics": compute_metrics(y_test, prob_smt, tau=0.5),
            "fiis":    compute_fiis(clf_smt, X_smt, y_smt),
        }
    except Exception as e:
        results["SMOTE"] = {"metrics": {}, "fiis": {}, "error": str(e)}

    # ── 4.2 BorderlineSMOTE ──
    try:
        from imblearn.over_sampling import BorderlineSMOTE
        bsmote = BorderlineSMOTE(random_state=seed, k_neighbors=k_neighbors)
        X_bsm, y_bsm = bsmote.fit_resample(X_train, y_train)
        clf_bsm = fit_logistic(X_bsm, y_bsm, seed=seed)
        prob_bsm = clf_bsm.predict_proba(X_test)[:, 1]
        results["BorderlineSMOTE"] = {
            "metrics": compute_metrics(y_test, prob_bsm, tau=0.5),
            "fiis":    compute_fiis(clf_bsm, X_bsm, y_bsm),
        }
    except Exception as e:
        results["BorderlineSMOTE"] = {"metrics": {}, "fiis": {}, "error": str(e)}

    # ── 5. Class weighting ──
    clf_cw = fit_logistic(X_train, y_train,
                          class_weight="balanced", seed=seed)
    prob_cw = clf_cw.predict_proba(X_test)[:, 1]
    results["class_weighting"] = {
        "metrics": compute_metrics(y_test, prob_cw, tau=0.5),
        "fiis":    compute_fiis(clf_cw, X_train, y_train),
    }

    return results