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
from scipy.stats import wilcoxon
from itertools import combinations

warnings.filterwarnings('ignore')

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
    Run all pairwise Wilcoxon signed-rank tests for one group.
    pivot: DataFrame (datasets x remedies) of mean G-mean values.
    Returns DataFrame of results.
    """
    n = len(pivot)
    rows = []

    for r1, r2 in combinations(REMEDY_ORDER, 2):
        x = pivot[r1].values
        y = pivot[r2].values
        d = x - y

        # Skip if all differences are zero
        if np.all(d == 0):
            stat, p_val = np.nan, 1.0
        else:
            try:
                stat, p_val = wilcoxon(x, y, alternative='two-sided',
                                       zero_method='wilcox')
            except Exception:
                stat, p_val = np.nan, 1.0

        # Effect size: rank-biserial correlation
        # r = Z / sqrt(n) where Z from normal approximation
        from scipy.stats import norm
        if not np.isnan(stat) and p_val < 1.0:
            # Mean and std of W under H0
            n_pairs = n * (n - 1) / 2
            mu_w    = n_pairs / 2
            sig_w   = np.sqrt(n * (n+1) * (2*n+1) / 24)
            z       = (stat - mu_w) / sig_w if sig_w > 0 else 0
            r_eff   = abs(z) / np.sqrt(n)
        else:
            r_eff = 0.0

        sig_bonf = p_val < ALPHA_BONF
        sig_nom  = p_val < ALPHA

        # Which remedy is better?
        mean_diff = np.mean(x - y)
        better = REMEDY_LABELS[r1] if mean_diff > 0 else REMEDY_LABELS[r2]

        rows.append({
            'Remedy_1':     REMEDY_LABELS[r1],
            'Remedy_2':     REMEDY_LABELS[r2],
            'W_stat':       round(stat, 2) if not np.isnan(stat) else np.nan,
            'p_value':      round(p_val, 6),
            'p_bonf':       round(min(p_val * N_PAIRS, 1.0), 6),
            'sig_nominal':  sig_nom,
            'sig_bonferroni': sig_bonf,
            'mean_diff':    round(mean_diff, 4),
            'effect_r':     round(r_eff, 3),
            'better':       better,
            'n':            n,
        })

    df = pd.DataFrame(rows)

    print(f"\n{'='*70}")
    print(f"{group_label} (n={n}, Bonferroni alpha={ALPHA_BONF:.4f})")
    print(f"{'='*70}")
    print(f"{'Remedy 1':<22} {'Remedy 2':<22} "
          f"{'p-val':>8} {'p-bonf':>8} {'sig*':>5} {'diff':>7} {'r':>5}")
    print("-"*80)
    for _, row in df.iterrows():
        sig_str = '**' if row['sig_bonferroni'] else \
                  ('*' if row['sig_nominal'] else '')
        print(f"{row['Remedy_1']:<22} {row['Remedy_2']:<22} "
              f"{row['p_value']:>8.4f} {row['p_bonf']:>8.4f} "
              f"{sig_str:>5} {row['mean_diff']:>7.4f} "
              f"{row['effect_r']:>5.3f}")

    return df


def make_latex_table(df, group_label, hyp, tex_path):
    """LaTeX table of pairwise Wilcoxon results."""
    n = df['n'].iloc[0]
    ab = round(ALPHA_BONF, 4)
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\scriptsize')
    cap = (
        'Pairwise Wilcoxon signed-rank tests --- ' +
        group_label + ' ($n=' + str(n) + '$ datasets). '
        '$p$: nominal; $p_B$: Bonferroni-corrected '
        '($\\alpha_B=' + str(ab) + '$). '
        '$r$: rank-biserial effect size. '
        '$\\Delta$: mean G-mean difference (Remedy 1 $-$ Remedy 2). '
        '$\\ast$: $p<0.05$; $\\ast\\ast$: $p_B<0.05$.'
    )
    lines.append('\\caption{' + cap + '}')
    lines.append('\\label{tab:wilcoxon_' + hyp.lower() + '}')
    lines.append(r'\begin{tabular}{llrrrrl}')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{Remedy 1} & \textbf{Remedy 2} & '
        r'\textbf{$p$} & \textbf{$p_B$} & '
        r'\textbf{$\Delta$} & \textbf{$r$} & \textbf{Sig.} \\\\'
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
        ef  = f"{row['effect_r']:.3f}"
        r1  = row['Remedy_1']
        r2  = row['Remedy_2']
        row_str = (bf(r1) + " & " + bf(r2) + " & " +
                   bf(pv) + " & " + bf(pb) + " & " +
                   bf(md) + " & " + bf(ef) + " & " +
                   sig_str + " \\\\")
        lines.append("    " + row_str)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
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
                print(f"    {row['Remedy_1']} vs {row['Remedy_2']}: "
                      f"p_B={row['p_bonf']:.4f}, "
                      f"delta={row['mean_diff']:+.4f}, "
                      f"r={row['effect_r']:.3f}")


if __name__ == "__main__":
    main()