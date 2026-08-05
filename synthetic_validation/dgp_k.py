"""
dgp_k.py
--------
DGP-K: Compensation Trajectory

Target     : R (overall information imbalance ratio)
Archetype  : A -> F trajectory (deficit through balance to overbalance)
Scenarios  : gamma in {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}

Generation
----------
Fixed strong sampling deficit: n_1/n_0 = 0.1
Joint compensation via single parameter gamma:
    mu_0   = 0_d                (majority fixed at origin)
    mu_1   = gamma * e_1        (minority mean scales with gamma)
    Sigma_0 = I_d               (majority fixed isotropic)
    Sigma_1 = gamma * I_d       (minority covariance scales with gamma)

As gamma increases:
    - R_p increases (minority moves relative to boundary)
    - R_g increases (minority spreads geometrically)
    - R traces from deficit through balance to overbalance
    - R_n remains fixed at 0.1 throughout

Pilot results (M=20):
    gamma=0.5  -> R=0.076  (severe deficit)
    gamma=1.0  -> R=0.279  (moderate deficit)
    gamma=2.0  -> R=0.771  (near balance)
    gamma=3.0  -> R=1.049  (balance/overbalance threshold)
    gamma=5.0  -> R=1.317  (overbalance)
    gamma=8.0  -> R=1.360  (overbalance)

Test set
--------
One fixed test set per scenario, generated with ratio=0.1
matching training prevalence. Size: N_TEST = 20,000.

Expected behavior
-----------------
- R increases monotonically with gamma
- Remedy effectiveness decreases as R increases
- At R~1: no correction approximately as good as any remedy
- At R>1: resampling harmful (reduces effective sample size
  without improving information structure)
- AUC remains approximately stable (prevalence-invariant)
- Core validation: same R_n=0.1 produces radically different
  information structures depending on minority geometry
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fiis_core import compute_fiis, dominance_profile, apply_remedies
from dgp_utils import (
    D, N_TRAIN, N_TEST, M, SEED_BASE,
    get_class_counts, generate_gaussian,
    make_test_set, aggregate_results
)

# ─────────────────────────────────────────────
# DGP-K Configuration
# ─────────────────────────────────────────────
GAMMAS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]   # compensation parameter
RATIO  = 0.1                                  # fixed strong sampling deficit


def make_dgpk_params(gamma, d=D):
    """
    Construct DGP-K class parameters for a given gamma.

    Joint compensation:
        mu_1   = gamma * e_1   (minority mean)
        Sigma_1 = gamma * I_d  (minority covariance)
    Majority fixed at baseline throughout.
    """
    mu0    = np.zeros(d)
    mu1    = np.zeros(d);  mu1[0] = gamma
    Sigma0 = np.eye(d)
    Sigma1 = gamma * np.eye(d)
    return mu0, mu1, Sigma0, Sigma1


# ─────────────────────────────────────────────
# Run DGP-K
# ─────────────────────────────────────────────

def run_dgp_k(
    gammas=GAMMAS,
    ratio=RATIO,
    d=D, N=N_TRAIN, n_test=N_TEST,
    M=M, seed_base=SEED_BASE,
    save_dir="results"
):
    """
    Run DGP-K Monte Carlo experiment.

    For each scenario (gamma):
      1. Generate one fixed test set matching ratio=0.1
      2. Run M replications: generate training data,
         fit model, compute FIIS, apply all remedies
      3. Collect replication-level results

    Returns
    -------
    df_raw : pd.DataFrame  replication-level results
    """
    n0, n1   = get_class_counts(N, ratio)
    pi_train = n1 / N
    all_rows = []

    for gamma in gammas:
        mu0, mu1, Sigma0, Sigma1 = make_dgpk_params(gamma, d=d)

        E_x1 = gamma**2 + gamma * d
        E_x0 = d
        print(f"DGP-K | gamma={gamma} | "
              f"n0={n0}, n1={n1}, pi={pi_train:.4f} | "
              f"E[||x1||^2]={E_x1:.2f}, E[||x0||^2]={E_x0:.2f}")

        # One fixed test set per scenario — ratio fixed at 0.1
        X_test, y_test = make_test_set(
            mu0, mu1, Sigma0, Sigma1,
            ratio=ratio,
            d=d, n_test=n_test,
            seed=seed_base
        )

        for m in range(M):
            seed = seed_base + m + 1
            X_train, y_train = generate_gaussian(
                n0, n1, d,
                mu0, mu1, Sigma0, Sigma1,
                seed=seed
            )

            remedy_results = apply_remedies(
                X_train, y_train,
                X_test,  y_test,
                seed=seed
            )

            for remedy_name, res in remedy_results.items():
                if not res.get("metrics"):
                    continue

                row = {
                    "dgp":            "DGP-K",
                    "scenario_param": gamma,
                    "gamma":          gamma,
                    "ratio":          ratio,
                    "n0":             n0,
                    "n1":             n1,
                    "pi":             pi_train,
                    "rep":            m,
                    "seed":           seed,
                    "remedy":         remedy_name,
                }
                row.update(res["metrics"])

                fiis = res.get("fiis", {})
                for k, v in fiis.items():
                    row[f"fiis_{k}"] = v

                if remedy_name == "no_correction" and fiis:
                    dp = dominance_profile(fiis)
                    for k, v in dp.items():
                        row[f"dom_{k}"] = v

                all_rows.append(row)

        print(f"  Replications: {M} done")

    df_raw = pd.DataFrame(all_rows)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "dgp_k_raw.csv")
    df_raw.to_csv(out_path, index=False)
    print(f"\nRaw results saved to {out_path}")

    return df_raw


# ─────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────

def analyse_dgp_k(df_raw, save_dir):
    """
    Aggregate and print key summaries for DGP-K.
    Returns aggregated DataFrame.
    """
    base = df_raw[df_raw["remedy"] == "no_correction"].copy()

    print("\n=== FIIS Components (no correction) ===")
    fiis_cols = ["fiis_R_n", "fiis_R_p", "fiis_R_g",
                 "fiis_C",   "fiis_R",
                 "fiis_log_Rn", "fiis_log_Rp",
                 "fiis_log_Rg", "fiis_log_C"]
    fiis_cols = [c for c in fiis_cols if c in base.columns]
    summary   = base.groupby("gamma")[fiis_cols].agg(["mean","std"]).round(4)
    print(summary.to_string())
    # save FIIS summary
    os.makedirs(save_dir, exist_ok=True)
    fiis_path = os.path.join(save_dir, "dgp_k_fiis_summary.csv")
    summary.to_csv(fiis_path)
    print(f"\nFIIS summary saved to {fiis_path}")

    print("\n=== Dominance Profile (mean phi per gamma) ===")
    dom_cols = [c for c in base.columns if c.startswith("dom_phi")]
    if dom_cols:
        dom = base.groupby("gamma")[dom_cols].mean().round(4)
        print(dom.to_string())
    # save dominance profile summary
    os.makedirs(save_dir, exist_ok=True)
    dom_path = os.path.join(save_dir, "dgp_k_dominance_profile.csv")
    dom.to_csv(dom_path)
    print(f"\nDominance profile summary saved to {dom_path}")

    print("\n=== Classification Metrics by Remedy ===")
    metric_cols = ["AUC", "G_mean", "F1",
                   "sensitivity", "specificity"]
    metric_cols = [c for c in metric_cols if c in df_raw.columns]
    perf = (df_raw.groupby(["gamma", "remedy"])[metric_cols]
                  .mean().round(4))
    print(perf.to_string())
    # save classification metrics summary
    os.makedirs(save_dir, exist_ok=True)
    perf_path = os.path.join(save_dir, "dgp_k_classification_metrics.csv")
    perf.to_csv(perf_path)
    print(f"\nClassification metrics summary saved to {perf_path}")

    print("\n=== R trajectory (mean R per gamma, no correction) ===")
    print(base.groupby("gamma")["fiis_R"].agg(["mean","std"]).round(4).to_string())

    print("\n=== Remedy effectiveness vs R trajectory ===")
    # G-mean improvement over no_correction per gamma
    pivot = df_raw.groupby(["gamma","remedy"])["G_mean"].mean().unstack()
    if "no_correction" in pivot.columns:
        for col in pivot.columns:
            if col != "no_correction":
                pivot[f"delta_{col}"] = pivot[col] - pivot["no_correction"]
    print(pivot.round(4).to_string())

    agg = aggregate_results(df_raw, group_cols=["gamma", "remedy"])
    agg_path = os.path.join(save_dir, "dgp_k_agg.csv")
    agg.to_csv(agg_path)
    print(f"\nAggregated results saved to {agg_path}")

    return agg


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), "results", "synthetic_validation", "dgp_k")
    df_raw   = run_dgp_k(save_dir=save_dir)
    agg      = analyse_dgp_k(df_raw, save_dir=save_dir)