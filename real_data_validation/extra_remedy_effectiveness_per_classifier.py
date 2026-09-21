"""
remedy_effectiveness_per_classifier.py
--------------------------------------
RQ3: Is remedy effectiveness associated with FIIS diagnosis?

This script adapts the original analysis to perform Friedman test, Nemenyi post-hoc analysis (with CD diagrams), and violin plots per classifier, in addition to the existing analysis per hypothesis group.

Three hypothesis groups based on revised archetype framework:
  H1 — Archetype A + Deficiency (R<1):   59 datasets
       oversampling effective, remedies differ
  H2 — Archetype C + Deficiency (R<1):   24 datasets
       remedies equivalent, limited benefit
  H3 — Balance/Overbalance (R>=1):        11 datasets
       no correction optimal, remedies equivalent

Analysis per group and per classifier:
  - Friedman test
  - Nemenyi post-hoc + CD diagram
  - Descriptive statistics
  - Violin plots
  - Delta G-mean strip plot

Inputs:
  results/real_data_raw.csv
  results/archetype_discovery/archetype_discovery.csv

Outputs:
  results/remedy_effectiveness_per_classifier/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import friedmanchisquare
from itertools import combinations
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'mathtext.fontset': 'cm',   # renders $...$ math in Computer Modern, matching LaTeX
    'legend.frameon': False,    # optional: cleaner legend box, common in papers
})
warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "remedy_effectiveness_per_classifier")
os.makedirs(OUT_DIR, exist_ok=True)

BG = 'white'

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
REMEDY_COLORS = {
    'no_correction':        '#888888',
    'threshold_correction': '#6B4226',
    'ROS':                  '#E84855',
    'SMOTE':                '#2E86AB',
    'BorderlineSMOTE':      '#9B59B6',
    'class_weighting':      '#F4A261',
}
GROUP_COLORS = {
    'H1': '#623991',   # purple — Archetype A + Deficiency
    'H2': '#C55D08',   # orange — Archetype C + Deficiency
    'H3': '#2E7D32',   # green  — Balance/Overbalance
}
GROUP_LABELS = {
    'H1': 'H1: Archetype A + Deficiency',
    'H2': 'H2: Archetype C + Deficiency',
    'H3': 'H3: Balance/Overbalance',
}

# Nemenyi q_alpha table (Demsar 2006, alpha=0.05)
Q_ALPHA = {2:1.960, 3:2.343, 4:2.569, 5:2.728,
           6:2.850, 7:2.949, 8:3.031, 9:3.102, 10:3.164}


# ─────────────────────────────────────────────
# Step 1 — Aggregate data
# ─────────────────────────────────────────────

def aggregate_data_per_classifier(raw_path, arch_path):
    """
    Average G-mean and AUC across 30 runs per (dataset, classifier, remedy).
    Merge archetype + compensation group.
    """
    raw  = pd.read_csv(raw_path)
    arch = pd.read_csv(arch_path)

    # Assign binary compensation group
    arch['comp_group'] = arch['R_mean'].apply(
        lambda r: 'D' if r < 1 else 'BO')

    # Assign hypothesis group
    def get_hyp_group(row):
        if row['archetype_main_label'] == 'A' and row['comp_group'] == 'D':
            return 'H1'
        elif row['archetype_main_label'] == 'C' and row['comp_group'] == 'D':
            return 'H2'
        else:
            return 'H3'
    arch['hyp_group'] = arch.apply(get_hyp_group, axis=1)

    # Average across runs, keeping classifier separate
    agg = (raw.groupby(['dataset', 'classifier', 'remedy'])
              [['G_mean', 'AUC', 'F1', 'sensitivity',
                'specificity', 'balanced_accuracy']]
              .mean().round(4).reset_index())

    # Merge
    agg = agg.merge(
        arch[['dataset', 'IR', 'R_mean', 'R_std',
              'archetype_main_label', 'comp_group', 'hyp_group',
              'compensation_label']],
        on='dataset', how='left'
    )

    out = os.path.join(OUT_DIR, 'remedy_effectiveness_agg_per_classifier.csv')
    agg.to_csv(out, index=False)
    print(f"Saved: {out}")
    return agg, arch


# ─────────────────────────────────────────────
# Friedman + Nemenyi
# ─────────────────────────────────────────────

def run_friedman_nemenyi_per_classifier(agg_data, classifier_name, group_label, hyp_group):
    """Run Friedman + Nemenyi for one hypothesis group and one classifier."""
    sub   = agg_data[(agg_data['hyp_group'] == hyp_group) & (agg_data['classifier'] == classifier_name)]
    if sub.empty:
        return None
        
    pivot = (sub.pivot(index='dataset', columns='remedy',
                       values='G_mean')[REMEDY_ORDER])
    n, k  = pivot.shape

    if n < 2: # Friedman test requires at least 2 observations per treatment
        print(f"  Skipping Friedman for {classifier_name} - {group_label}: Not enough datasets (n={n})")
        return None

    f_stat, f_p = friedmanchisquare(
        *[pivot[r].values for r in REMEDY_ORDER])

    # Average ranks
    ranks     = pivot.rank(axis=1, ascending=False, method='average')
    avg_ranks = ranks.mean(axis=0).values
    mean_gm   = pivot.mean().values
    std_gm    = pivot.std().values

    # Nemenyi CD
    cd = Q_ALPHA[k] * np.sqrt(k*(k+1) / (6*n))

    # Pairwise significance
    p_matrix = np.ones((k, k))
    for i, j in combinations(range(k), 2):
        diff = abs(avg_ranks[i] - avg_ranks[j])
        sig  = diff > cd
        p_matrix[i,j] = 0.01 if sig else 0.5
        p_matrix[j,i] = p_matrix[i,j]

    print(f"\n=== {classifier_name} - {group_label} (n={n}) ===")
    print(f"Friedman: chi2={f_stat:.3f}, p={f_p:.6f}")
    for i, r in enumerate(REMEDY_ORDER):
        print(f"  {REMEDY_LABELS[r]:<22} rank={avg_ranks[i]:.3f}  "
              f"G-mean={mean_gm[i]:.4f}±{std_gm[i]:.4f}")

    return {
        'pivot': pivot, 'n': n, 'k': k,
        'f_stat': f_stat, 'f_p': f_p,
        'avg_ranks': avg_ranks, 'cd': cd,
        'mean_gm': mean_gm, 'std_gm': std_gm,
        'p_matrix': p_matrix,
    }


# ─────────────────────────────────────────────
# Rank table (LaTeX)
# ─────────────────────────────────────────────

def make_rank_table_per_classifier(res, classifier_name, hyp_group, tex_path, csv_path):
    if res is None: return

    results = pd.DataFrame({
        'Remedy':     [REMEDY_LABELS[r] for r in REMEDY_ORDER],
        'Avg_Rank':   res['avg_ranks'].round(3),
        'G_mean':     res['mean_gm'].round(4),
        'G_mean_std': res['std_gm'].round(4),
    }).sort_values('Avg_Rank').reset_index(drop=True)

    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    p_str = ('$p < 0.001$' if res['f_p'] < 0.001
             else f'$p = {res["f_p"]:.4f}$' if res['f_p'] < 0.05 else f'$p = {res["f_p"]:.2f}$' )

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(
        f'\\caption{{Remedy effectiveness for {classifier_name} — {GROUP_LABELS[hyp_group]}. '
        f'Averaged across 30 CV runs. '
        f'Friedman: $\\chi^2_F = {res["f_stat"]:.3f}$, {p_str}. '
        f'CD$={res["cd"]:.3f}$ ($\\alpha=0.05$). '
        f'Average rank (lower = better).}}'
    )
    lines.append(f'\\label{{tab:remedy_{classifier_name.lower()}_{hyp_group.lower()}}}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lrr}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Remedy} & \textbf{Avg.\ Rank} '
                 r'& \textbf{G-mean} \\')
    lines.append(r'\midrule')
    for _, row in results.iterrows():
        lines.append(
            f"    {row['Remedy']} & {row['Avg_Rank']:.3f} & "
            f"${row['G_mean']:.4f} \\pm {row['G_mean_std']:.4f}$ \\\\"
        )
    lines.append(r'\midrule')
    
    # Fixed syntax error here
    summary_line = (f'\\multicolumn{{3}}{{l}}{{\\footnotesize Friedman: $\\chi^2_F={res["f_stat"]:.3f}$, '
                    f'{p_str}, CD$={res["cd"]:.3f}$}} \\\\')
    lines.append(summary_line)
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {tex_path}")


# ─────────────────────────────────────────────
# CD diagram
# ─────────────────────────────────────────────

def plot_cd_diagram_per_classifier(res, classifier_name, hyp_group, out_path):
    if res is None:
        return

    avg_ranks = np.asarray(res['avg_ranks'], dtype=float)
    cd = float(res['cd'])
    k = int(res['k'])

    order = np.argsort(avg_ranks, kind='stable')
    s_ranks = avg_ranks[order]
    s_labels = [REMEDY_LABELS[REMEDY_ORDER[i]] for i in order]

    half = k // 2
    left_idx = list(range(half))
    right_idx = list(range(half, k))

    # Every pair within a clique must satisfy the CD criterion.
    candidates = []
    for i in range(k - 1):
        j = i
        while j + 1 < k and s_ranks[j + 1] - s_ranks[i] <= cd:
            j += 1
        if j > i:
            candidates.append(set(range(i, j + 1)))

    # Keep maximal cliques; do not merge overlapping cliques.
    final_cliques = [
        cl for cl in candidates
        if not any(cl < other for other in candidates)
    ]

    ax_y = 0.0
    step = 0.70
    lbl_gap = 0.15

    clique_ys = [
        ax_y - 0.35 - 0.25 * i
        for i in range(len(final_cliques))
    ]

    # All vertical connectors extend below all clique bars.
    first_label_y = ax_y - step
    if clique_ys:
        first_label_y = min(first_label_y, min(clique_ys) - 0.35)

    lowest_label_y = first_label_y - step * (
        max(len(left_idx), len(right_idx)) - 1
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    rank_min = s_ranks[0] - 0.3
    rank_max = s_ranks[-1] + 0.3
    ax.set_xlim(rank_min - 2.5, rank_max + 2.5)
    ax.set_ylim(min(-4.5, lowest_label_y - 0.5), 2.8)

    ax.plot(
        [rank_min, rank_max], [ax_y, ax_y],
        color='black', linewidth=1.5, zorder=3
    )

    # Rank ticks and staggered labels.
    label_y_base = ax_y + 0.22
    label_y_high = ax_y + 0.48
    min_label_sep = 0.28
    last_x = -np.inf
    current_level = 0

    for r in s_ranks:
        ax.plot(
            [r, r], [ax_y - 0.1, ax_y + 0.1],
            color='black', linewidth=1.2, zorder=3
        )

        if r - last_x < min_label_sep:
            current_level = 1 - current_level
        else:
            current_level = 0

        y = label_y_high if current_level else label_y_base
        ax.text(
            r, y, f'{r:.2f}',
            ha='center', va='bottom', fontsize=7.5
        )
        last_x = r

    # CD bar.
    best = s_ranks[0]
    ax.annotate(
        '',
        xy=(best + cd, 2.2),
        xytext=(best, 2.2),
        arrowprops=dict(
            arrowstyle='<->',
            color='black',
            lw=1.5,
            mutation_scale=10
        )
    )
    ax.text(
        best + cd / 2, 2.42, 'CD',
        ha='center', va='bottom', fontsize=9
    )

    # Left labels.
    label_x_left = rank_min - 0.15
    for pos, i in enumerate(left_idx):
        rank = s_ranks[i]
        line_y = first_label_y - step * pos

        ax.plot(
            [rank, rank, label_x_left],
            [ax_y, line_y, line_y],
            color='black', lw=1.0, zorder=2
        )
        ax.text(
            label_x_left - lbl_gap, line_y, s_labels[i],
            ha='right', va='center', fontsize=8.5
        )

    # Right labels.
    label_x_right = rank_max + 0.15
    for pos, i in enumerate(reversed(right_idx)):
        rank = s_ranks[i]
        line_y = first_label_y - step * pos

        ax.plot(
            [rank, rank, label_x_right],
            [ax_y, line_y, line_y],
            color='black', lw=1.0, zorder=2
        )
        ax.text(
            label_x_right + lbl_gap, line_y, s_labels[i],
            ha='left', va='center', fontsize=8.5
        )

    # Draw overlapping cliques on separate rows.
    for cl, cy in zip(final_cliques, clique_ys):
        indices = sorted(cl)
        lo = s_ranks[indices[0]]
        hi = s_ranks[indices[-1]]

        ax.plot(
            [lo, hi], [cy, cy],
            color='black',
            linewidth=4,
            solid_capstyle='round',
            zorder=5
        )

        # Keep cliques with nearly identical ranks visible.
        if hi - lo < 0.03:
            ax.plot(
                [(lo + hi) / 2], [cy],
                marker='o',
                markersize=4,
                color='black',
                linestyle='none',
                zorder=6
            )

    fig.suptitle(
        f'CD Diagram — {classifier_name} - {GROUP_LABELS[hyp_group]}\n'
        f'Nemenyi post-hoc ($\\alpha=0.05$, CD$={cd:.3f}$); '
        r'bars: not significantly different',
        fontsize=9, color='black', y=1.01
    )

    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {out_path}')

# ─────────────────────────────────────────────
# Violin plot per classifier and hypothesis group
# ─────────────────────────────────────────────

def plot_violin_per_classifier_and_group(agg_data, classifier_name, out_path):
    """
    Three panels — one per hypothesis group for a given classifier.
    Each panel: violin per remedy.
    """
    groups = [('H1', 'H1: Archetype A + Deficiency'),
              ('H2', 'H2: Archetype C + Deficiency'),
              ('H3', 'H3: Balance / Overbalance')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.patch.set_facecolor(BG)

    for ax, (hyp, title) in zip(axes, groups):
        ax.set_facecolor(BG)
        sub    = agg_data[(agg_data['hyp_group'] == hyp) & (agg_data['classifier'] == classifier_name)]
        data   = [sub[sub['remedy']==r]['G_mean'].values
                  for r in REMEDY_ORDER if not sub[sub['remedy']==r]['G_mean'].empty]
        colors = [REMEDY_COLORS[r] for r in REMEDY_ORDER if not sub[sub['remedy']==r]['G_mean'].empty]
        labels = [REMEDY_LABELS[r] for r in REMEDY_ORDER if not sub[sub['remedy']==r]['G_mean'].empty]

        if not data: # Skip if no data for this group and classifier
            ax.set_title(f'{title} (No Data)', fontsize=9, fontweight='bold', color='#1A1A2E')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color); pc.set_alpha(0.70)
            pc.set_edgecolor('white')
        for pn in ['cmedians','cmins','cmaxes','cbars']:
            if pn in parts:
                parts[pn].set_color('#1A1A2E')
                parts[pn].set_linewidth(1.5)

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('G-mean', fontsize=9)
        ax.set_title(f'{title}', fontsize=9, fontweight='bold', color='#1A1A2E')
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle(
        f'Remedy Effectiveness by Hypothesis Group for {classifier_name}\n',
        fontsize=10, color='black', y=1.05
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# Delta G-mean strip plot per classifier
# ─────────────────────────────────────────────

def plot_delta_gmean_per_classifier(agg_data, classifier_name, out_path):
    """
    Delta G-mean strip plot per hypothesis group for a given classifier.
    """
    sub = agg_data[agg_data['classifier'] == classifier_name].copy()

    # Calculate delta G-mean relative to 'no_correction'
    no_correction_gmean = sub[sub['remedy'] == 'no_correction']\
        .set_index('dataset')['G_mean']

    sub['delta_gmean'] = sub.apply(
        lambda row: row['G_mean'] - no_correction_gmean.get(row['dataset'], np.nan),
        axis=1
    )
    sub.dropna(subset=['delta_gmean'], inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x_ticks = []; x_labels = []
    x_center = 0
    width = 0.8

    for remedy_idx, remedy in enumerate(REMEDY_ORDER):
        if remedy == 'no_correction': continue

        x_center += 1
        for group_idx, hyp_group in enumerate(['H1', 'H2', 'H3']):
            x_pos = x_center + (group_idx - 1) * (width / 3)
            vals = sub[(sub['remedy'] == remedy) & (sub['hyp_group'] == hyp_group)]['delta_gmean'].values
            color = GROUP_COLORS[hyp_group]

            if len(vals) >= 2:
                parts = ax.violinplot(vals, positions=[x_pos],
                                      widths=width/3, showmedians=True,
                                      showextrema=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.70)
                    pc.set_edgecolor('white')
                for pn in ['cmedians','cmins','cmaxes','cbars']:
                    if pn in parts:
                        parts[pn].set_color('#1A1A2E')
                        parts[pn].set_linewidth(1.5)
            elif len(vals) == 1:
                ax.scatter([x_pos], vals, color=color,
                           s=60, zorder=5, edgecolors='white',
                           linewidths=0.5, marker='D')

        x_ticks.append(x_center)
        x_labels.append(REMEDY_LABELS[remedy])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Delta G-mean (vs. No Correction)', fontsize=10)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    handles = [
        mpatches.Patch(color=GROUP_COLORS[h], alpha=0.8,
                       label=GROUP_LABELS[h])
        for h in ['H1', 'H2', 'H3']
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9,
              loc='lower right')

    fig.suptitle(
        f'Delta G-mean for {classifier_name} by Remedy and Hypothesis Group\n',
        fontsize=10, color='black', y=1.05
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    raw_path  = os.path.join(RESULTS_DIR, 'real_data_raw.csv')
    arch_path = os.path.join(RESULTS_DIR,
                             'archetype_discovery_noB', 'archetype_discovery.csv')

    if not os.path.exists(raw_path) or not os.path.exists(arch_path):
        print(f"Error: Could not find input files at {raw_path} or {arch_path}")
        return

    print("Step 1 — Aggregating data per classifier...")
    agg, arch = aggregate_data_per_classifier(raw_path, arch_path)

    classifiers = agg['classifier'].unique()
    print(f"Found classifiers: {', '.join(classifiers)}")

    for classifier in classifiers:
        print(f"\n--- Analyzing Classifier: {classifier} ---")
        classifier_out_dir = os.path.join(OUT_DIR, classifier.replace(' ', '_'))
        os.makedirs(classifier_out_dir, exist_ok=True)

        print("\nGroup sizes per classifier:")
        for h in ['H1','H2','H3']:
            n = agg[(agg['hyp_group']==h) & (agg['classifier']==classifier)]['dataset'].nunique()
            print(f"  {h}: {n} datasets for {classifier}")

        print("\nStep 2 — Friedman + Nemenyi per group and classifier...")
        results_per_classifier = {}
        for hyp, label in [('H1', 'H1: Archetype A + Deficiency'),
                            ('H2', 'H2: Archetype C + Deficiency'),
                            ('H3', 'H3: Balance/Overbalance')]:
            res = run_friedman_nemenyi_per_classifier(agg, classifier, label, hyp)
            results_per_classifier[hyp] = res

            make_rank_table_per_classifier(
                res, classifier, hyp,
                tex_path=os.path.join(classifier_out_dir, f'table_remedy_{hyp}.tex'),
                csv_path=os.path.join(classifier_out_dir, f'table_remedy_{hyp}.csv'),
            )

            # CD diagram for all groups
            if res is not None:
                plot_cd_diagram_per_classifier(
                    res, classifier, hyp,
                    os.path.join(classifier_out_dir, f'modified_plot_cd_{hyp}.png')
                )

        print("\nStep 3 — Violin plot per classifier and hypothesis group...")
        plot_violin_per_classifier_and_group(
            agg, classifier, os.path.join(classifier_out_dir, 'plot_violin_by_group.png'))

        print("\nStep 4 — Delta G-mean strip plot per classifier...")
        plot_delta_gmean_per_classifier(
            agg, classifier, os.path.join(classifier_out_dir, 'plot_delta_gmean.png'))

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
