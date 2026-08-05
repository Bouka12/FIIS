"""
dgp_s.py
--------
DGP-S: Sampling Mechanism Validation

Target     : R_n
Archetype  : A (sampling-dominated deficit)
Scenarios  : n_1/n_0 in {0.3, 0.1, 0.05}

Generation
----------
Shared baseline configuration:
    mu_0 = 0_d
    mu_1 = delta * e_1   (delta = 1.0)
    Sigma_0 = Sigma_1 = I_d

Only n_1/n_0 varies across scenarios.
All geometric parameters fixed.

Test set
--------
One fixed test set per scenario, generated with the same
class ratio as training (pi_test = pi_train = n1/N).
Size: N_TEST = 20,000.

Expected behavior
-----------------
- R_n decreases monotonically with ratio
- R_p, R_g, C approximately stable across scenarios
- phi_n dominant in all scenarios
- No correction: sensitivity collapses at extreme imbalance
- Threshold correction (tau = pi_train): restores G-mean
- Oversampling: effective (Hypothesis A)
- AUC: invariant across remedies and ratios
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")) 
sys.path.insert(0, os.path.dirname(__file__)) 

from fiis_core import compute_fiis, dominance_profile, apply_remedies
from dgp_utils import (
    D, N_TRAIN, N_TEST, M, SEED_BASE, DELTA,
    get_class_counts, generate_gaussian,
    baseline_params, make_test_set,
    aggregate_results, stability_check
)

# ─────────────────────────────────────────────
# DGP-S Scenarios
# ─────────────────────────────────────────────
RATIOS = [0.3, 0.1, 0.05]   # n_1/n_0


# ─────────────────────────────────────────────
# Run DGP-S
# ─────────────────────────────────────────────

def run_dgp_s(
    ratios=RATIOS,
    d=D, N=N_TRAIN, n_test=N_TEST,
    M=M, seed_base=SEED_BASE, delta=DELTA,
    save_dir="results"
):
    """
    Run DGP-S Monte Carlo experiment.

    For each scenario (ratio):
      1. Generate one fixed test set matching training ratio
      2. Run M replications: generate training data,
         fit model, compute FIIS, apply all remedies
      3. Collect replication-level results

    Returns
    -------
    df_raw : pd.DataFrame  replication-level results
    """
    mu0, mu1, Sigma0, Sigma1 = baseline_params(d=d, delta=delta)
    all_rows = []

    for ratio in ratios:
        n0, n1   = get_class_counts(N, ratio)
        pi_train = n1 / N

        print(f"DGP-S | ratio={ratio} | "
              f"n0={n0}, n1={n1}, pi={pi_train:.4f}")

        # One fixed test set per scenario — same ratio as training
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
                    "dgp":            "DGP-S",
                    "scenario_param": ratio,
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
    out_path = os.path.join(save_dir, "dgp_s_raw.csv")
    df_raw.to_csv(out_path, index=False)
    print(f"\nRaw results saved to {out_path}")

    return df_raw


# ─────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────

def analyse_dgp_s(df_raw, save_dir):
    """
    Aggregate and print key summaries for DGP-S.
    Returns aggregated DataFrame.
    """
    base = df_raw[df_raw["remedy"] == "no_correction"].copy()

    print("\n=== FIIS Component Means +/- Std (no correction) ===")
    fiis_cols = ["fiis_R_n", "fiis_R_p", "fiis_R_g",
                 "fiis_C",   "fiis_R",
                 "fiis_log_Rn", "fiis_log_Rp",
                 "fiis_log_Rg", "fiis_log_C"]
    fiis_cols = [c for c in fiis_cols if c in base.columns]
    summary   = base.groupby("ratio")[fiis_cols].agg(["mean","std"]).round(4)
    print(summary.to_string())
    # save FIIS summary
    os.makedirs(save_dir, exist_ok=True)
    fiis_path = os.path.join(save_dir, "dgp_s_fiis_summary.csv")
    summary.to_csv(fiis_path)

    print("\n=== Dominance Profile (mean phi per ratio) ===")
    dom_cols = [c for c in base.columns if c.startswith("dom_phi")]
    if dom_cols:
        dom = base.groupby("ratio")[dom_cols].mean().round(4)
        print(dom.to_string())
    # save dominance profile summary
    os.makedirs(save_dir, exist_ok=True)
    dom_path = os.path.join(save_dir, "dgp_s_dominance_profile.csv")
    dom.to_csv(dom_path)

    print("\n=== Dominant factor counts ===")
    if "dom_dominant" in base.columns:
        print(base.groupby(["ratio","dom_dominant"]).size().to_string())

    print("\n=== Classification Metrics by Remedy ===")
    metric_cols = ["AUC", "G_mean", "F1",
                   "sensitivity", "specificity"]
    metric_cols = [c for c in metric_cols if c in df_raw.columns]
    perf = (df_raw.groupby(["ratio", "remedy"])[metric_cols]
                  .mean().round(4))
    print(perf.to_string())

    # save performance summary
    os.makedirs(save_dir, exist_ok=True)
    perf_path = os.path.join(save_dir, "dgp_s_performance.csv")
    perf.to_csv(perf_path)
    print(f"\nPerformance summary saved to {perf_path}")

    # print("\n=== Marginal Stability Check ===")
    # stab = stability_check(
    #     base,
    #     target_component="fiis_log_Rn",
    #     baseline_param_val=max(RATIOS),
    #     threshold=0.1
    # )
    # print(stab.to_string())
    # # save stability check results
    # os.makedirs(save_dir, exist_ok=True)
    # stab_path = os.path.join(save_dir, "dgp_s_stability_check.csv")
    # stab.to_csv(stab_path)
    # print(f"\nStability check results saved to {stab_path}")

    agg = aggregate_results(df_raw, group_cols=["ratio", "remedy"])
    
    os.makedirs(save_dir, exist_ok=True)
    agg_path = os.path.join(save_dir, "dgp_s_agg.csv")
    
    agg.to_csv(agg_path)
    print(f"\nAggregated results saved to {agg_path}")

    return agg


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), "results", "synthetic_validation", "dgp_s")
    df_raw   = run_dgp_s(save_dir=save_dir)
    agg      = analyse_dgp_s(df_raw, save_dir=save_dir)