"""
dgp_utils.py
------------
Shared data generation utilities for all DGPs.

Provides:
  - generate_gaussian()  : generate samples from two Gaussian classes
  - make_test_set()      : generate one fixed test set per scenario
                           matching training prevalence (pi_test = pi_train)
  - get_class_counts()   : compute n0, n1 from N and ratio
  - baseline_params()    : shared baseline mu, Sigma configuration
  - aggregate_results()  : aggregate Monte Carlo results
  - stability_check()    : marginal stability for DGP-G and DGP-C

Baseline configuration (shared across all DGPs):
    mu_0 = 0_d
    mu_1 = delta * e_1
    Sigma_0 = Sigma_1 = I_d

Test set design:
    One fixed test set per scenario, generated with the same
    class ratio as training (pi_test = pi_train).
    This is consistent with the Van den Goorbergh threshold
    correction tau = pi_train, which assumes deployment
    conditions match training prevalence.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
D         = 3        # feature dimensionality
N_TRAIN   = 5_000    # training set size
N_TEST    = 20_000   # test set size (imbalanced, matching training ratio)
M         = 100      # Monte Carlo replications
SEED_BASE = 42       # base random seed
DELTA     = 1.0      # default class separation


# ─────────────────────────────────────────────
# Class count helper
# ─────────────────────────────────────────────

def get_class_counts(N, ratio):
    """
    Compute n0, n1 from total N and minority/majority ratio n1/n0.

    Parameters
    ----------
    N     : int   total samples
    ratio : float n1/n0

    Returns
    -------
    n0, n1 : int
    """
    n0 = int(N / (1 + ratio))
    n1 = N - n0
    return n0, n1


# ─────────────────────────────────────────────
# Gaussian data generation
# ─────────────────────────────────────────────

def generate_gaussian(n0, n1, d, mu0, mu1, Sigma0, Sigma1, seed):
    """
    Generate binary classification data from two Gaussian classes.

    Parameters
    ----------
    n0, n1          : int    class sample sizes
    d               : int    feature dimensionality
    mu0, mu1        : array  class means (d,)
    Sigma0, Sigma1  : array  class covariances (d, d)
    seed            : int    random seed

    Returns
    -------
    X : array (n0+n1, d)
    y : array (n0+n1,)  — 0=majority, 1=minority
    """
    rng = np.random.default_rng(seed)
    X0  = rng.multivariate_normal(mu0, Sigma0, size=n0)
    X1  = rng.multivariate_normal(mu1, Sigma1, size=n1)
    X   = np.vstack([X0, X1])
    y   = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(int)
    return X, y


# ─────────────────────────────────────────────
# Baseline parameter constructor
# ─────────────────────────────────────────────

def baseline_params(d=D, delta=DELTA):
    """
    Shared baseline for all DGPs:
        mu_0 = 0_d
        mu_1 = delta * e_1
        Sigma_0 = Sigma_1 = I_d
    """
    mu0   = np.zeros(d)
    mu1   = np.zeros(d)
    mu1[0] = delta
    Sigma = np.eye(d)
    return mu0, mu1, Sigma, Sigma


# ─────────────────────────────────────────────
# Test set generation
# ─────────────────────────────────────────────

def make_test_set(mu0, mu1, Sigma0, Sigma1,
                  ratio, d=D, n_test=N_TEST, seed=0):
    """
    Generate one fixed test set per scenario matching training prevalence.
    Called once per scenario before the replication loop.

    The test set mirrors the training class ratio so that:
      - Threshold correction tau = pi_train is correctly evaluated
      - Metrics reflect deployment conditions matching training

    Parameters
    ----------
    mu0, mu1        : array  class means
    Sigma0, Sigma1  : array  class covariances
    ratio           : float  n1/n0 — same as training scenario
    d               : int    dimensionality
    n_test          : int    total test size
    seed            : int    fixed seed — same for all replications

    Returns
    -------
    X_test : array (n_test, d)
    y_test : array (n_test,)
    """
    n0_test, n1_test = get_class_counts(n_test, ratio)
    return generate_gaussian(
        n0_test, n1_test, d,
        mu0, mu1, Sigma0, Sigma1,
        seed=seed
    )


# ─────────────────────────────────────────────
# Results aggregation
# ─────────────────────────────────────────────

def aggregate_results(df, group_cols):
    """
    Aggregate Monte Carlo results across replications.
    Returns mean +/- std per group.

    Parameters
    ----------
    df         : pd.DataFrame  raw replication-level results
    group_cols : list          columns to group by

    Returns
    -------
    pd.DataFrame with mean and std for all numeric columns
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    exclude      = ["rep", "n0", "n1", "seed"]
    agg_cols     = [c for c in numeric_cols if c not in exclude]

    agg = (df.groupby(group_cols)[agg_cols]
             .agg(["mean", "std"])
             .round(4))
    return agg


# ─────────────────────────────────────────────
# Marginal stability check
# ─────────────────────────────────────────────

def stability_check(df, target_component, baseline_param_val,
                    threshold=0.1):
    """
    For DGP-G and DGP-C: check whether non-target FIIS components
    drift beyond threshold across scenario parameter values.

    Parameters
    ----------
    df                  : pd.DataFrame  raw results (no_correction only)
    target_component    : str           e.g. 'fiis_log_Rg'
    baseline_param_val  : float         parameter value where all
                                        components are at baseline
    threshold           : float         maximum allowed drift in
                                        |log R_j| for non-target components

    Returns
    -------
    pd.DataFrame: fraction of flagged replications per scenario
    """
    non_targets = [c for c in ["fiis_log_Rn", "fiis_log_Rp",
                                "fiis_log_Rg", "fiis_log_C"]
                   if c != target_component]

    base  = df[df["scenario_param"] == baseline_param_val]
    flags = []

    for param_val, grp in df.groupby("scenario_param"):
        for nt in non_targets:
            base_mean = base[nt].mean()
            drift     = (grp[nt] - base_mean).abs()
            flagged   = (drift > threshold).mean()
            flags.append({
                "scenario_param": param_val,
                "non_target":     nt,
                "mean_drift":     round(float(drift.mean()), 4),
                "frac_flagged":   round(float(flagged), 4),
            })

    return pd.DataFrame(flags)