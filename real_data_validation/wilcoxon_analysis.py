"""
wilcoxon_analysis.py
---------------------
Pairwise Wilcoxon signed-rank tests for each hypothesis group.
Compares all pairs of remedies on per-dataset mean G-mean.

For each hypothesis group (H1, H2, H3):
  - All 15 pairwise comparisons (6 choose 2)
  - Wilcoxon signed-rank test on paired G-mean values
  - Bonferroni correction for multiple comparisons
  - Effect size: rank-biserial correlation r = 1 - 2W/(n*(n+1)/2)... 
    using common approximation r = Z/sqrt(n)

Outputs:
  results/wilcoxon/wilcoxon_H1.csv/.tex
  results/wilcoxon/wilcoxon_H2.csv/.tex
  results/wilcoxon/wilcoxon_H3.csv/.tex
  results/wilcoxon/wilcoxon_summary.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
import scipy 
from itertools import combinations

# warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "wilcoxon")
os.makedirs(OUT_DIR, exist_ok=True)

REMEDY_ORDER = [
    'no_correction', 'threshold_correction', 'ROS',
    'SMOTE', 'BorderlineSMOTE', 'class_weighting',
]
REMEDY_LABELS = {
    'no_correction':        'No Correction',
    'threshold_correction': 'Threshold Corr.',
    'ROS':                  'ROS',
    'SMOTE':                'SMOTE',
    'BorderlineSMOTE':      'Borderline SMOTE',
    'class_weighting':      'Class Weighting',
}

ALPHA       = 0.05
N_PAIRS     = 15   # C(6,2)
ALPHA_BONF  = ALPHA / N_PAIRS


def run_wilcoxon_group(pivot, group_label):
    """
    Pairwise two-sided Wilcoxon signed-rank tests on dataset-level
    mean G-mean values.

    effect_r: signed matched-pairs rank-biserial correlation.
    Positive values favour Remedy 1; negative values favour Remedy 2.

    Bonferroni correction is applied to 15 comparisons within each
    hypothesis group.
    """
    pivot = pivot.loc[:, REMEDY_ORDER]

    if pivot.empty:
        raise ValueError(f"{group_label}: no datasets.")

    if not pivot.index.is_unique:
        raise ValueError(f"{group_label}: duplicate dataset identifiers.")

    values = pivot.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{group_label}: missing or non-finite G-mean values."
        )

    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{group_label}: G-mean values outside [0, 1].")

    n = len(pivot)
    rows = []

    for r1, r2 in combinations(REMEDY_ORDER, 2):
        x = pivot[r1].to_numpy(dtype=float)
        y = pivot[r2].to_numpy(dtype=float)

        # The supplied aggregate CSV stores G-mean to four decimals.
        # Remove floating-point subtraction artefacts at that precision.
        d = np.round(x - y, decimals=4)

        nonzero = d[d != 0]
        n_nonzero = len(nonzero)

        if n_nonzero == 0:
            # Explicit convention when every paired value is identical.
            stat, p_val = 0.0, 1.0
        else:
            result = wilcoxon(
                d,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )

            stat = float(result.statistic)
            p_val = float(result.pvalue)

            if not np.isfinite(stat) or not np.isfinite(p_val):
                raise ValueError(
                    f"{group_label}: invalid Wilcoxon result "
                    f"for {r1} versus {r2}."
                )


        mean_diff = float(np.mean(x - y))
        p_bonf = min(p_val * N_PAIRS, 1.0)

        # Descriptive direction of the arithmetic mean difference.
        # This is not a declaration of statistical superiority.
        if np.all(d == 0):
            better = "Tie"
        elif mean_diff > 0:
            better = REMEDY_LABELS[r1]
        elif mean_diff < 0:
            better = REMEDY_LABELS[r2]
        else:
            better = "Tie"

        rows.append({
            "Remedy_1": REMEDY_LABELS[r1],
            "Remedy_2": REMEDY_LABELS[r2],
            "W_stat": stat,
            "p_value": p_val,
            "p_bonf": p_bonf,
            "sig_nominal": p_val < ALPHA,
            "sig_bonferroni": p_bonf < ALPHA,
            "mean_diff": mean_diff,
            "better": better,
            "n": n,
            "n_nonzero": n_nonzero,
        })

    df = pd.DataFrame(rows)

    print(
        f"\n{group_label}: n={n}, "
        f"Bonferroni threshold={ALPHA_BONF:.6f}, "
        f"SciPy={scipy.__version__}"
    )
    print(
        f"{'Remedy 1':<22} {'Remedy 2':<22} "
        f"{'p':>10} {'p_B':>10} {'Sig.':>5} "
        f"{'diff':>9}"
    )

    for _, row in df.iterrows():
        sig = (
            "**" if row["sig_bonferroni"]
            else "*" if row["sig_nominal"]
            else ""
        )
        print(
            f"{row['Remedy_1']:<22} {row['Remedy_2']:<22} "
            f"{row['p_value']:>10.4g} {row['p_bonf']:>10.4g} "
            f"{sig:>5} {row['mean_diff']:>+9.4f}"
        )

    return df


def make_latex_table(df, group_label, hyp, tex_path):
    """LaTeX table of pairwise Wilcoxon results."""
    n = int(df["n"].iloc[0])

    def format_p(value):
        if value < 0.0001:
            return r"$<0.0001$"
        return f"{value:.4f}"

    caption = (
        "Pairwise two-sided Wilcoxon signed-rank tests --- "
        + group_label
        + f" ($n={n}$ datasets). "
        + r"$p$: nominal; $p_B$: Bonferroni-adjusted across "
        + r"15 comparisons within this group. "
        + r"$\Delta$: arithmetic mean G-mean difference "
        + r"(Remedy 1 $-$ Remedy 2). "
        + r"Zero differences are excluded from signed ranks. "
        + r"$\ast$: $p<0.05$; $\ast\ast$: $p_B<0.05$."
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{" + caption + "}",
        r"\label{tab:wilcoxon_" + hyp.lower() + "}",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"\textbf{Remedy 1} & \textbf{Remedy 2} & "
        r"\textbf{$p$} & \textbf{$p_B$} & "
        r"\textbf{$\Delta$} & \textbf{Sig.} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        significant = bool(row["sig_bonferroni"])

        if significant:
            sig = r"$\ast\ast$"
        elif row["sig_nominal"]:
            sig = r"$\ast$"
        else:
            sig = "---"

        def emphasize(text):
            return r"\textbf{" + text + "}" if significant else text

        cells = [
            emphasize(row["Remedy_1"]),
            emphasize(row["Remedy_2"]),
            format_p(row["p_value"]),
            format_p(row["p_bonf"]),
            emphasize(f"{row['mean_diff']:+.4f}"),
            sig,
        ]

        lines.append("    " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {tex_path}")



def main():
    agg_path = os.path.join(RESULTS_DIR,
                            'remedy_effectiveness',
                            'remedy_effectiveness_agg.csv')
    agg = pd.read_csv(agg_path)

    all_results = []

    for hyp, label in [
        ('H1', 'H1: Archetype A + Deficiency'),
        ('H2', 'H2: Archetype C + Deficiency'),
        ('H3', 'H3: Balance/Overbalance'),
    ]:
        sub   = agg[agg['hyp_group'] == hyp]
        pivot = sub.pivot(index='dataset', columns='remedy',
                          values='G_mean')[REMEDY_ORDER]

        df = run_wilcoxon_group(pivot, label)
        df['group'] = hyp
        all_results.append(df)

        csv_path = os.path.join(OUT_DIR, f'wilcoxon_{hyp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        tex_path = os.path.join(OUT_DIR, f'wilcoxon_{hyp}.tex')
        make_latex_table(df, label, hyp, tex_path)

    # Summary across groups
    summary = pd.concat(all_results, ignore_index=True)
    summary_path = os.path.join(OUT_DIR, 'wilcoxon_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    for hyp in ['H1','H2','H3']:
        sub = summary[summary['group']==hyp]
        n_sig_nom  = sub['sig_nominal'].sum()
        n_sig_bonf = sub['sig_bonferroni'].sum()
        print(f"\n{hyp}: {n_sig_nom}/15 nominally significant, "
              f"{n_sig_bonf}/15 Bonferroni significant")
        bonf_sig = sub[sub['sig_bonferroni']]
        if len(bonf_sig) > 0:
            print("  Bonferroni significant pairs:")
            for _, row in bonf_sig.iterrows():
                print(
                    f"    {row['Remedy_1']} vs {row['Remedy_2']}: "
                    f"p_B={row['p_bonf']:.4g}, "
                    f"delta={row['mean_diff']:+.4f}"
                )


if __name__ == "__main__":
    main()