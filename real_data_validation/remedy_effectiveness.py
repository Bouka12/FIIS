"""
remedy_effectiveness_revised.py
--------------------------------
RQ3: Is remedy effectiveness associated with FIIS diagnosis?

Three hypothesis groups based on revised archetype framework:
  H1 — Archetype A + Deficiency (R<1):   59 datasets
       oversampling effective, remedies differ
  H2 — Archetype C + Deficiency (R<1):   24 datasets
       remedies equivalent, limited benefit
  H3 — Balance/Overbalance (R>=1):        11 datasets
       no correction optimal, remedies equivalent

Analysis per group:
  - Friedman test (all groups, even H3)
  - Nemenyi post-hoc + CD diagram (H1, H2 if significant)
  - Descriptive for H3 (small n)
  - Violin plots per group
  - Delta G-mean strip plot

Inputs:
  results/real_data_raw.csv
  results/archetype_discovery/archetype_discovery.csv

Outputs:
  results/remedy_effectiveness/
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

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "remedy_effectiveness")
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
    'H1': 'H1: Archetype A + Deficiency (59 datasets)',
    'H2': 'H2: Archetype C + Deficiency (24 datasets)',
    'H3': 'H3: Balance/Overbalance (11 datasets)',
}

# Nemenyi q_alpha table (Demsar 2006, alpha=0.05)
Q_ALPHA = {2:1.960, 3:2.343, 4:2.569, 5:2.728,
           6:2.850, 7:2.949, 8:3.031, 9:3.102, 10:3.164}


# ─────────────────────────────────────────────
# Step 1 — Aggregate data
# ─────────────────────────────────────────────

def aggregate_data(raw_path, arch_path):
    """
    Average G-mean and AUC across 3 classifiers × 30 runs
    per (dataset, remedy). Merge archetype + compensation group.
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

    # Average across classifiers and runs
    agg = (raw.groupby(['dataset', 'remedy'])
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

    out = os.path.join(OUT_DIR, 'remedy_effectiveness_agg.csv')
    agg.to_csv(out, index=False)
    print(f"Saved: {out}")
    return agg, arch


# ─────────────────────────────────────────────
# Friedman + Nemenyi
# ─────────────────────────────────────────────

def run_friedman_nemenyi(agg, group_label, hyp_group):
    """Run Friedman + Nemenyi for one hypothesis group."""
    sub   = agg[agg['hyp_group'] == hyp_group]
    pivot = (sub.pivot(index='dataset', columns='remedy',
                       values='G_mean')[REMEDY_ORDER])
    n, k  = pivot.shape

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

    print(f"\n=== {group_label} (n={n}) ===")
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

def make_rank_table(res, hyp_group, tex_path, csv_path):
    results = pd.DataFrame({
        'Remedy':     [REMEDY_LABELS[r] for r in REMEDY_ORDER],
        'Avg_Rank':   res['avg_ranks'].round(3),
        'G_mean':     res['mean_gm'].round(4),
        'G_mean_std': res['std_gm'].round(4),
    }).sort_values('Avg_Rank').reset_index(drop=True)

    results.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    p_str = (f'$p < 0.001$' if res['f_p'] < 0.001
             else f'$p = {res["f_p"]:.4f}$')

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(
        f'\\caption{{Remedy effectiveness — {GROUP_LABELS[hyp_group]}. '
        f'Averaged across 3 classifiers and 30 CV runs. '
        f'Friedman: $\\chi^2_F = {res["f_stat"]:.3f}$, {p_str}. '
        f'CD$={res["cd"]:.3f}$ ($\\alpha=0.05$). '
        f'Average rank (lower = better).}}'
    )
    lines.append(f'\\label{{tab:remedy_{hyp_group.lower()}}}')
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
    lines.append(
        f'\\multicolumn{{3}}{{l}}{{'
        f'\\footnotesize Friedman: $\\chi^2_F={res["f_stat"]:.3f}$, '
        f'{p_str}, CD$={res["cd"]:.3f}$}} \\\\'
    )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {tex_path}")


# ─────────────────────────────────────────────
# CD diagram
# ─────────────────────────────────────────────

def plot_cd_diagram(res, hyp_group, out_path):
    avg_ranks = res['avg_ranks']
    cd        = res['cd']
    n         = res['n']
    k         = res['k']

    order    = np.argsort(avg_ranks)
    s_ranks  = avg_ranks[order]
    s_keys   = [REMEDY_ORDER[i] for i in order]
    s_labels = [REMEDY_LABELS[kk] for kk in s_keys]

    half      = k // 2
    left_idx  = list(range(half))
    right_idx = list(range(half, k))

    # Non-significant cliques
    cliques = []
    for i in range(k):
        for j in range(i+1, k):
            if abs(s_ranks[i] - s_ranks[j]) <= cd:
                merged = False
                for cl in cliques:
                    if i in cl or j in cl:
                        cl.update([i,j]); merged=True; break
                if not merged:
                    cliques.append({i,j})
    final_cliques = []
    for cl in cliques:
        found = False
        for mcl in final_cliques:
            if cl & mcl:
                mcl.update(cl); found=True; break
        if not found:
            final_cliques.append(set(cl))

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    rank_min = s_ranks[0] - 0.3
    rank_max = s_ranks[-1] + 0.3
    ax.set_xlim(rank_min - 2.5, rank_max + 2.5)
    ax.set_ylim(-4.5, 2.8)

    ax_y = 0.0; step = 0.70; lbl_gap = 0.15

    ax.plot([rank_min, rank_max], [ax_y, ax_y],
            color='black', linewidth=1.5, zorder=3)
    # Rank ticks and labels
    label_y_base = ax_y + 0.22
    label_y_high = ax_y + 0.48

    # Minimum horizontal separation required between labels
    min_label_sep = 0.28

    label_y = []
    last_x = -np.inf
    current_level = 0

    for r in s_ranks:
        ax.plot([r, r], [ax_y - 0.1, ax_y + 0.1],
                color='black', linewidth=1.2, zorder=3)

        # If the rank is too close to the previous one, move the label upward
        if r - last_x < min_label_sep:
            current_level = 1 - current_level
        else:
            current_level = 0

        y = label_y_high if current_level else label_y_base
        label_y.append(y)

        ax.text(r, y, f'{r:.2f}',
                ha='center', va='bottom', fontsize=7.5)

        last_x = r

    # CD bar
    best = s_ranks[0]
    ax.annotate('', xy=(best+cd, 2.2), xytext=(best, 2.2),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=1.5, mutation_scale=10))
    ax.text((best*2+cd)/2, 2.42, f'CD={cd:.2f}',
            ha='center', va='bottom', fontsize=9)

    # Left labels (best→worst, shallowest→deepest)
    label_x_left = rank_min - 0.15
    for pos, i in enumerate(left_idx):
        rank  = s_ranks[i]; label = s_labels[i]
        line_y = ax_y - step*(pos+1)
        ax.plot([rank,rank], [ax_y, line_y], color='black', lw=1.0, zorder=2)
        ax.plot([label_x_left,rank], [line_y,line_y], color='black', lw=1.0)
        ax.text(label_x_left-lbl_gap, line_y, label,
                ha='right', va='center', fontsize=8.5)

    # Right labels (worst→best, shallowest→deepest)
    label_x_right = rank_max + 0.15
    for pos, i in enumerate(reversed(right_idx)):
        rank  = s_ranks[i]; label = s_labels[i]
        line_y = ax_y - step*(pos+1)
        ax.plot([rank,rank], [ax_y, line_y], color='black', lw=1.0, zorder=2)
        ax.plot([rank,label_x_right], [line_y,line_y], color='black', lw=1.0)
        ax.text(label_x_right+lbl_gap, line_y, label,
                ha='left', va='center', fontsize=8.5)

    # Clique bars
    cy = ax_y - 0.35
    for cl in final_cliques:
        cl_ranks = [s_ranks[j] for j in sorted(cl)]
        ax.plot([min(cl_ranks), max(cl_ranks)], [cy,cy],
                color='black', linewidth=4, solid_capstyle='butt', zorder=5)
        cy -= 0.25

    # fig.suptitle(
    #     f'CD Diagram — {GROUP_LABELS[hyp_group]}\n'
    #     f'Nemenyi post-hoc ($\\alpha=0.05$, CD$={cd:.3f}$); '
    #     r'bars: not significantly different',
    #     fontsize=9, color='black', y=1.01
    # )
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# Violin plot per hypothesis group
# ─────────────────────────────────────────────

def plot_violin_by_group(agg, out_path):
    """
    Three panels — one per hypothesis group.
    Each panel: violin per remedy.
    """
    groups = [('H1', 'H1: Archetype A + Deficiency'),
              ('H2', 'H2: Archetype C + Deficiency'),
              ('H3', 'H3: Balance / Overbalance')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.patch.set_facecolor(BG)

    for ax, (hyp, title) in zip(axes, groups):
        ax.set_facecolor(BG)
        sub    = agg[agg['hyp_group'] == hyp]
        data   = [sub[sub['remedy']==r]['G_mean'].values
                  for r in REMEDY_ORDER]
        colors = [REMEDY_COLORS[r] for r in REMEDY_ORDER]
        labels = [REMEDY_LABELS[r] for r in REMEDY_ORDER]

        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color); pc.set_alpha(0.70)
            pc.set_edgecolor('white')
        for pn in ['cmedians','cmins','cmaxes','cbars']:
            if pn in parts:
                parts[pn].set_color('#1A1A2E')
                parts[pn].set_linewidth(1.5)

        ax.set_xticks(range(1, len(REMEDY_ORDER)+1))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('G-mean', fontsize=9)
        ax.set_title(title, fontsize=9, fontweight='bold', color='#1A1A2E')
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        ax.spines[['top','right']].set_visible(False)

    # fig.suptitle(
    #     'Remedy Effectiveness by Hypothesis Group\n'
    #     '(G-mean averaged across 3 classifiers × 30 runs)',
    #     fontsize=10, fontweight='bold', color='#1A1A2E'
    # )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# Delta G-mean strip plot
# ─────────────────────────────────────────────

def plot_delta_gmean(agg, out_path):
    """
    Delta G-mean vs no_correction per remedy,
    colored by hypothesis group.
    """
    nc    = agg[agg['remedy']=='no_correction'][['dataset','G_mean']]\
               .rename(columns={'G_mean':'G_nc'})
    delta = agg[agg['remedy']!='no_correction'].merge(nc, on='dataset')
    delta['delta_G'] = delta['G_mean'] - delta['G_nc']

    active = [r for r in REMEDY_ORDER if r != 'no_correction']

    fig, axes = plt.subplots(1, len(active), figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor(BG)

    for ax, remedy in zip(axes, active):
        ax.set_facecolor(BG)
        sub = delta[delta['remedy'] == remedy]

        for hyp, color in GROUP_COLORS.items():
            s = sub[sub['hyp_group'] == hyp]
            rng = np.random.default_rng(42)
            ax.scatter(
                rng.uniform(-0.15, 0.15, len(s)),
                s['delta_G'].values,
                color=color, alpha=0.65, s=25,
                edgecolors='white', linewidths=0.3, zorder=3
            )

        ax.plot([-0.3, 0.3],
                [sub['delta_G'].median()]*2,
                color='#1A1A2E', linewidth=2, zorder=4)
        ax.axhline(y=0, color='gray', linestyle='--',
                   linewidth=1, alpha=0.6)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(REMEDY_LABELS[remedy], fontsize=8,
                     fontweight='bold', color='#1A1A2E')
        ax.grid(True, axis='y', alpha=0.2, linestyle='--')
        ax.spines[['top','right','bottom']].set_visible(False)

    axes[0].set_ylabel(r'$\Delta$G-mean (vs no correction)', fontsize=9)

    handles = [
        mpatches.Patch(color=GROUP_COLORS[h], alpha=0.8,
                       label=GROUP_LABELS[h])
        for h in ['H1','H2','H3']
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    # fig.suptitle(
    #     r'$\Delta$G-mean per Remedy vs No Correction'
    #     '\n(dots = datasets, bar = median, colored by hypothesis group)',
    #     fontsize=10, fontweight='bold', color='#1A1A2E'
    # )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")



def plot_violin_three_groups(agg, out_path):
    """
    For each remedy: three violins side by side —
    H1 (purple), H2 (orange), H3 (green).
    Allows direct cross-group comparison per remedy.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
 
    width   = 0.25
    gap     = 0.05
    spacing = 1.3
    x_ticks  = []
    x_labels = []
 
    for i, remedy in enumerate(REMEDY_ORDER):
        sub      = agg[agg['remedy'] == remedy]
        x_center = i * spacing
 
        for j, (hyp, color) in enumerate(GROUP_COLORS.items()):
            vals  = sub[sub['hyp_group'] == hyp]['G_mean'].values
            x_pos = x_center + (j - 1) * (width + gap)
 
            if len(vals) >= 2:
                parts = ax.violinplot(vals, positions=[x_pos],
                                      widths=width, showmedians=True,
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
    ax.set_ylabel('G-mean', fontsize=10)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
 
    handles = [
        mpatches.Patch(color=GROUP_COLORS[h], alpha=0.8,
                       label=GROUP_LABELS[h])
        for h in ['H1', 'H2', 'H3']
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9,
              loc='lower right')
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")
 

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    raw_path  = os.path.join(RESULTS_DIR, 'real_data_raw.csv')
    arch_path = os.path.join(RESULTS_DIR,
                             'archetype_discovery', 'archetype_discovery.csv')

    print("Step 1 — Aggregating data...")
    agg, arch = aggregate_data(raw_path, arch_path)

    print("\nGroup sizes:")
    for h in ['H1','H2','H3']:
        n = arch[arch['hyp_group']==h]['dataset'].nunique()
        print(f"  {h}: {n} datasets")

    print("\nStep 2 — Friedman + Nemenyi per group...")
    results = {}
    for hyp, label in [('H1', 'H1: Archetype A + Deficiency'),
                        ('H2', 'H2: Archetype C + Deficiency'),
                        ('H3', 'H3: Balance/Overbalance')]:
        res = run_friedman_nemenyi(agg, label, hyp)
        results[hyp] = res

        make_rank_table(
            res, hyp,
            tex_path=os.path.join(OUT_DIR, f'table_remedy_{hyp}.tex'),
            csv_path=os.path.join(OUT_DIR, f'table_remedy_{hyp}.csv'),
        )

        # CD diagram for all groups
        plot_cd_diagram(
            res, hyp,
            os.path.join(OUT_DIR, f'plot_cd_{hyp}.pdf')
        )

    print("\nStep 3 — Violin plot by hypothesis group...")
    plot_violin_by_group(
        agg, os.path.join(OUT_DIR, 'plot_violin_by_group.pdf'))

    print("\nStep 4 — Delta G-mean strip plot...")
    plot_delta_gmean(
        agg, os.path.join(OUT_DIR, 'plot_delta_gmean.pdf'))

    plot_violin_three_groups(
        agg, os.path.join(OUT_DIR, 'plot_violin_three_groups.pdf'))

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()