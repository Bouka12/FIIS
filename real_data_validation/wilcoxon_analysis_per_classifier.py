"""
wilcoxon_analysis_per_classifier.py
-----------------------------------
Pairwise Wilcoxon signed-rank tests for each hypothesis group and EACH classifier.
Compares all pairs of remedies on per-dataset mean G-mean.

For each classifier and each hypothesis group (H1, H2, H3):
  - All 15 pairwise comparisons (6 choose 2)
  - Wilcoxon signed-rank test on paired G-mean values
  - Bonferroni correction for multiple comparisons
  - Effect size: rank-biserial correlation

Inputs:
  results/remedy_effectiveness_per_classifier/remedy_effectiveness_agg_per_classifier.csv

Outputs:
  results/wilcoxon_per_classifier/{classifier}/
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import scipy
from itertools import combinations

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
# Using the output from the previous script as input
INPUT_FILE  = os.path.join(RESULTS_DIR, "remedy_effectiveness_per_classifier", "remedy_effectiveness_agg_per_classifier.csv")
OUT_DIR     = os.path.join(RESULTS_DIR, "wilcoxon_per_classifier")
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

GROUP_LABELS = {
    'H1': 'H1: Archetype A + Deficiency',
    'H2': 'H2: Archetype C + Deficiency',
    'H3': 'H3: Balance/Overbalance',
}

ALPHA       = 0.05
N_PAIRS     = 15   # C(6,2)
ALPHA_BONF  = ALPHA / N_PAIRS


def run_wilcoxon_group(pivot, group_label, classifier_name):
    """
    Run all pairwise Wilcoxon signed-rank tests for one group and one classifier.
    pivot: DataFrame (datasets x remedies) of mean G-mean values.
    Returns DataFrame of results.
    """
    n = len(pivot)
    rows = []

    if n < 5: # Wilcoxon is generally not recommended for n < 5
        print(f"  Skipping Wilcoxon for {classifier_name} - {group_label}: n={n} too small.")
        return None

    for r1, r2 in combinations(REMEDY_ORDER, 2):
        x = pivot[r1].values
        y = pivot[r2].values
        d = np.round(x - y, 4)
        n_nonzero = int(np.count_nonzero(d))

        # Skip if all differences are zero
        if n_nonzero == 0:
            stat, p_val = np.nan, 1.0
        else:
            
            # Using alternative='two-sided' and zero_method='wilcox' as in original
            result = wilcoxon(d, alternative='two-sided',
                                       zero_method='wilcox', correction=False, method='auto')
            stat, p_val = float(result.statistic), float(result.pvalue)

            if not np.isfinite(stat) or not np.isfinite(p_val):
                raise ValueError(
                    f"{classifier_name} - {group_label}: "
                    f"invalid test result for {r1} versus {r2}."
                )


        sig_bonf = p_val < ALPHA_BONF
        sig_nom  = p_val < ALPHA

        # Which remedy is better?
        mean_diff = np.mean(x - y)
        if mean_diff > 0:
            better = REMEDY_LABELS[r1]
        elif mean_diff < 0:
            better = REMEDY_LABELS[r2]
        else:
            better = 'Tie'
        # better = REMEDY_LABELS[r1] if mean_diff > 0 else REMEDY_LABELS[r2]

        rows.append({
            'Classifier':   classifier_name,
            'Group':        group_label,
            'Remedy_1':     REMEDY_LABELS[r1],
            'Remedy_2':     REMEDY_LABELS[r2],
            'W_stat':       round(stat, 2) if not np.isnan(stat) else np.nan,
            'p_value':      round(p_val, 6),
            'p_bonf':       round(min(p_val * N_PAIRS, 1.0), 6),
            'sig_nominal':  sig_nom,
            'sig_bonferroni': sig_bonf,
            'mean_diff':    round(mean_diff, 4),
            'better':       better,
            'n':            n,
            'n_nonzero':    n_nonzero
        })

    df = pd.DataFrame(rows)
    return df


def make_latex_table(df, classifier_name, group_label, hyp, tex_path):
    """LaTeX table of pairwise Wilcoxon results for a specific classifier."""
    if df is None or df.empty: return
    
    n = df['n'].iloc[0]
    ab = round(ALPHA_BONF, 4)
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\scriptsize')
    cap = (
        f'Pairwise Wilcoxon signed-rank tests for {classifier_name} --- ' +
        group_label + ' ($n=' + str(n) + '$ datasets). '
        '$p$: nominal; $p_B$: Bonferroni-corrected '
        '($\\alpha_B=' + str(ab) + '$). '
        '$\\Delta$: mean G-mean difference (Remedy 1 $-$ Remedy 2). '
        '$\\ast$: $p<0.05$; $\\ast\\ast$: $p_B<0.05$.'
    )
    lines.append('\\caption{' + cap + '}')
    lines.append('\\label{tab:wilcoxon_' + classifier_name.lower().replace(" ", "_") + '_' + hyp.lower() + '}')
    lines.append(r'\begin{tabular}{llrrrl}')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{Remedy 1} & \textbf{Remedy 2} & '
        r'\textbf{$p$} & \textbf{$p_B$} & '
        r'\textbf{$\Delta$} &  \textbf{Sig.} \\\\'
    )
    lines.append(r'\midrule')

    for _, row in df.iterrows():
        if row['sig_bonferroni']:
            sig_str = r'$\ast\ast$'
            bf = lambda s: '\\textbf{' + str(s) + '}'
        elif row['sig_nominal']:
            sig_str = r'$\ast$'
            bf = lambda s: str(s)
        else:
            sig_str = '---'
            bf = lambda s: str(s)

        pv  = f"{row['p_value']:.4f}"
        pb  = f"{row['p_bonf']:.4f}"
        md  = f"{row['mean_diff']:+.4f}"
        r1  = row['Remedy_1']
        r2  = row['Remedy_2']
        row_str = (bf(r1) + " & " + bf(r2) + " & " +
                   bf(pv) + " & " + bf(pb) + " & " +
                   bf(md) + " & " +
                   sig_str + " \\\\")
        lines.append("    " + row_str)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {tex_path}")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Please run the effectiveness script first.")
        return

    agg = pd.read_csv(INPUT_FILE)
    classifiers = agg['classifier'].unique()
    
    all_results = []

    for classifier in classifiers:
        print(f"\n--- Wilcoxon Analysis for Classifier: {classifier} ---")
        classifier_out_dir = os.path.join(OUT_DIR, classifier.replace(' ', '_'))
        os.makedirs(classifier_out_dir, exist_ok=True)

        for hyp, label in [
            ('H1', 'H1: Archetype A + Deficiency'),
            ('H2', 'H2: Archetype C + Deficiency'),
            ('H3', 'H3: Balance/Overbalance'),
        ]:
            sub = agg[(agg['hyp_group'] == hyp) & (agg['classifier'] == classifier)]
            if sub.empty:
                continue
                
            pivot = sub.pivot(index='dataset', columns='remedy',
                              values='G_mean')[REMEDY_ORDER]

            df = run_wilcoxon_group(pivot, hyp, classifier)
            
            if df is not None:
                all_results.append(df)
                
                csv_path = os.path.join(classifier_out_dir, f'wilcoxon_{hyp}.csv')
                df.to_csv(csv_path, index=False)
                print(f"Saved: {csv_path}")

                tex_path = os.path.join(classifier_out_dir, f'wilcoxon_{hyp}.tex')
                make_latex_table(df, classifier, label, hyp, tex_path)

    # Summary across all classifiers and groups
    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = os.path.join(OUT_DIR, 'wilcoxon_summary_per_classifier.csv')
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved Global Summary: {summary_path}")

        # Print some summary stats
        print(f"\n{'='*80}")
        print(f"{'Classifier':<15} {'Hyp':<5} {'Sig (Nom)':<10} {'Sig (Bonf)':<10}")
        print("-" * 80)
        for classifier in classifiers:
            for hyp in ['H1', 'H2', 'H3']:
                sub = summary[(summary['Classifier'] == classifier) & (summary['Group'] == hyp)]
                if not sub.empty:
                    n_sig_nom = sub['sig_nominal'].sum()
                    n_sig_bonf = sub['sig_bonferroni'].sum()
                    print(f"{classifier:<15} {hyp:<5} {n_sig_nom:>2}/15 {n_sig_bonf:>2}/15")
    else:
        print("No Wilcoxon results generated (possibly due to small sample sizes).")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
