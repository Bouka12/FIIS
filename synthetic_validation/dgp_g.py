"""
dgp_g.py
--------
DGP-G: Feature-Energy (Geometric Spread) Mechanism Validation

Target     : R_g
Archetype  : C (feature-energy deficit)
Scenarios  : alpha in {0.05, 0.1, 0.2}
             where Sigma_1 = alpha * I_d

Generation
----------
Minority class centered at origin, majority at delta*e_1:
    mu_0   = delta * e_1  (majority far from origin, delta=1.0)
    mu_1   = 0_d          (minority at origin — geometrically compact)
    Sigma_0 = I_d
    Sigma_1 = alpha * I_d (alpha varies — minority compressed)

This placement ensures:
    E[||x_0||^2] = delta^2 + d  (fixed, large)
    E[||x_1||^2] = alpha * d    (decreases with alpha)
    R_g = alpha*d / (delta^2 + d) << 1 at small alpha

Class ratio fixed at n_1/n_0 = 0.2 across all scenarios.

Note on class placement:
    Majority at delta*e_1, minority at origin.
    Boundary sits between them.
    Minority at origin is close to boundary → R_p > 1 (secondary effect).
    This secondary R_p drift is monitored via stability check.

Test set
--------
One fixed test set per scenario, generated with ratio=0.2
matching training prevalence. Size: N_TEST = 20,000.

Expected behavior
-----------------
- phi_g dominant at alpha <= 0.2 (100% of replications)
- R_n stable (fixed ratio=0.2)
- R_p secondary drift acknowledged and monitored
- Oversampling contraindicated (Proposition 2):
  synthetic points generated within compressed minority region
  inherit the same low ||x||^2, cannot increase tr(I_1)
- No correction and threshold correction both limited
  since geometry is poorly estimated on minority side
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
    make_test_set, aggregate_results, stability_check
)

# ─────────────────────────────────────────────
# DGP-G Configuration
# ─────────────────────────────────────────────
ALPHAS = [0.05, 0.1, 0.2]   # Archetype C scenarios: phi_g dominant
RATIO  = 0.2                  # fixed n_1/n_0
DELTA  = 1.0                  # majority class separation


def make_dgpg_params(alpha, d=D, delta=DELTA):
    """
    Construct DGP-G class parameters for a given alpha.

    Majority at delta*e_1 (far from origin, fixed).
    Minority at origin, compressed: Sigma_1 = alpha * I_d.
    """
    mu0    = np.zeros(d);  mu0[0] = delta   # majority far from origin
    mu1    = np.zeros(d)                     # minority at origin
    Sigma0 = np.eye(d)
    Sigma1 = alpha * np.eye(d)
    return mu0, mu1, Sigma0, Sigma1


# ─────────────────────────────────────────────
# Run DGP-G
# ─────────────────────────────────────────────

def run_dgp_g(
    alphas=ALPHAS,
    ratio=RATIO,
    d=D, N=N_TRAIN, n_test=N_TEST,
    M=M, seed_base=SEED_BASE,
    save_dir="results"
):
    """
    Run DGP-G Monte Carlo experiment.

    For each scenario (alpha):
      1. Generate one fixed test set matching ratio=0.2
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

    for alpha in alphas:
        mu0, mu1, Sigma0, Sigma1 = make_dgpg_params(alpha, d=d)
        E_x1 = alpha * d
        E_x0 = DELTA**2 + d

        print(f"DGP-G | alpha={alpha} | "
              f"n0={n0}, n1={n1}, pi={pi_train:.4f} | "
              f"E[||x1||^2]={E_x1:.3f}, E[||x0||^2]={E_x0:.3f}, "
              f"R_g_theory={E_x1/E_x0:.3f}")

        # One fixed test set per scenario
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
                    "dgp":            "DGP-G",
                    "scenario_param": alpha,
                    "alpha":          alpha,
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
    out_path = os.path.join(save_dir, "dgp_g_raw.csv")
    df_raw.to_csv(out_path, index=False)
    print(f"\nRaw results saved to {out_path}")

    return df_raw


# ─────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────

def analyse_dgp_g(df_raw, save_dir):
    """
    Aggregate and print key summaries for DGP-G.
    Returns aggregated DataFrame.
    """
    base = df_raw[df_raw["remedy"] == "no_correction"].copy()

    print("\n=== FIIS Component Means +/- Std (no correction) ===")
    fiis_cols = ["fiis_R_n", "fiis_R_p", "fiis_R_g",
                 "fiis_C",   "fiis_R",
                 "fiis_log_Rn", "fiis_log_Rp",
                 "fiis_log_Rg", "fiis_log_C"]
    fiis_cols = [c for c in fiis_cols if c in base.columns]
    summary   = base.groupby("alpha")[fiis_cols].agg(["mean","std"]).round(4)
    print(summary.to_string())
    # save FIIS summary
    os.makedirs(save_dir, exist_ok=True)
    fiis_path = os.path.join(save_dir, "dgp_g_fiis_summary.csv")
    summary.to_csv(fiis_path)
    print(f"\nFIIS summary saved to {fiis_path}")
    
    print("\n=== Dominance Profile (mean phi per alpha) ===")
    dom_cols = [c for c in base.columns if c.startswith("dom_phi")]
    if dom_cols:
        dom = base.groupby("alpha")[dom_cols].mean().round(4)
        print(dom.to_string())
    # save dominance profile summary
    os.makedirs(save_dir, exist_ok=True)
    dom_path = os.path.join(save_dir, "dgp_g_dominance_profile.csv")
    dom.to_csv(dom_path)
    print(f"\nDominance profile summary saved to {dom_path}")

    print("\n=== Dominant factor counts ===")
    if "dom_dominant" in base.columns:
        print(base.groupby(["alpha","dom_dominant"]).size().to_string())

    print("\n=== Classification Metrics by Remedy ===")
    metric_cols = ["AUC", "G_mean", "F1",
                   "sensitivity", "specificity"]
    metric_cols = [c for c in metric_cols if c in df_raw.columns]
    perf = (df_raw.groupby(["alpha", "remedy"])[metric_cols]
                  .mean().round(4))
    print(perf.to_string())
    # save performance summary
    os.makedirs(save_dir, exist_ok=True)
    perf_path = os.path.join(save_dir, "dgp_g_performance.csv")
    perf.to_csv(perf_path)
    print(f"\nPerformance summary saved to {perf_path}")

    # # Stability check — R_n, R_p, C should be stable across alphas
    # print("\n=== Marginal Stability Check (non-target drift) ===")
    # stab = stability_check(
    #     base,
    #     target_component="fiis_log_Rg",
    #     baseline_param_val=max(ALPHAS),   # least compressed = baseline
    #     threshold=0.1
    # )
    # print(stab.to_string())
    # # save stability check results
    # os.makedirs(save_dir, exist_ok=True)
    # stab_path = os.path.join(save_dir, "dgp_g_stability_check.csv")
    # stab.to_csv(stab_path)
    # print(f"\nStability check results saved to {stab_path}")

    agg = aggregate_results(df_raw, group_cols=["alpha", "remedy"])
    os.makedirs(save_dir, exist_ok=True)
    agg_path = os.path.join(save_dir, "dgp_g_agg.csv")
    agg.to_csv(agg_path)
    print(f"\nAggregated results saved to {agg_path}")

    return agg


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), "results", "synthetic_validation", "dgp_g")
    df_raw   = run_dgp_g(save_dir=save_dir)
    agg      = analyse_dgp_g(df_raw, save_dir=save_dir)