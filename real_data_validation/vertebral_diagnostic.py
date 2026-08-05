"""
vertebral_diagnostic.py
------------------------
FIIS Diagnostic Case Study: Vertebral Column Dataset

Shows the full FIIS diagnostic pipeline applied to one medical dataset:
  1. FIIS components per fold (R_n, R_p, R_g, C, R)
  2. Archetype assignment stability across 30 folds
  3. Compensation regime stability
  4. Remedy effectiveness per classifier
  5. Summary diagnostic table + figures

Requires: vertebral_raw.csv and vertebral_fiis.csv
          (produced by vertebral_pipeline.py)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = os.path.join(os.path.dirname(__file__),
                           "results", "vertebral")
os.makedirs(RESULTS_DIR, exist_ok=True)

REMEDY_ORDER = ['no_correction','threshold_correction','ROS',
                'SMOTE','BorderlineSMOTE','class_weighting']
REMEDY_LABELS = {
    'no_correction':        'No Corr.',
    'threshold_correction': 'Threshold',
    'ROS':                  'ROS',
    'SMOTE':                'SMOTE',
    'BorderlineSMOTE':      'BL-SMOTE',
    'class_weighting':      'CW',
}
ARCH_COLORS = {'A':'#623991','B':'#2E86AB','C':'#C55D08'}
COMP_COLORS = {'O':'#2E7D32','B':'#1565C0','D':'#C62828'}
BG = 'white'
EPSILON = 0.05


# ─────────────────────────────────────────────
# 1. FIIS Diagnostic Summary Table
# ─────────────────────────────────────────────

def fiis_summary(df_fiis):
    """
    Compute mean ± std of FIIS components across 30 folds.
    Report archetype and compensation regime stability.
    """
    comp_cols = ['fiis_R_n','fiis_R_p','fiis_R_g','fiis_C','fiis_R']
    summary   = df_fiis[comp_cols].agg(['mean','std']).round(4)

    arch_counts = df_fiis['fiis_archetype'].value_counts()
    comp_counts = df_fiis['fiis_compensation'].value_counts()

    arch_label = arch_counts.index[0]
    arch_pct   = 100 * arch_counts.iloc[0] / len(df_fiis)
    comp_label = comp_counts.index[0]
    comp_pct   = 100 * comp_counts.iloc[0] / len(df_fiis)

    print("\n=== FIIS Diagnostic Summary (30 folds) ===")
    for col in comp_cols:
        m = df_fiis[col].mean()
        s = df_fiis[col].std()
        print(f"  {col:<12}: {m:.4f} ± {s:.4f}")
    print(f"\n  Archetype:    {arch_label} in {arch_pct:.1f}% of folds")
    print(f"  Compensation: {comp_label} in {comp_pct:.1f}% of folds")
    print(f"\n  Archetype distribution: {arch_counts.to_dict()}")
    print(f"  Compensation distribution: {comp_counts.to_dict()}")

    return summary, arch_label, arch_pct, comp_label, comp_pct


# ─────────────────────────────────────────────
# 2. FIIS Component Strip Plot (per fold)
# ─────────────────────────────────────────────

def plot_fiis_folds(df_fiis, out_path):
    """
    Strip plot of FIIS components across 30 folds.
    Shows stability and component values.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    components = [
        ('fiis_R_n', '$R_n$'),
        ('fiis_R_p', '$R_p$'),
        ('fiis_R_g', '$R_g$'),
        ('fiis_C',   '$C$'),
        ('fiis_R',   '$R$'),
    ]
    colors_by_arch = [ARCH_COLORS.get(a, '#888888')
                      for a in df_fiis['fiis_archetype']]
    rng = np.random.default_rng(42)

    for x_pos, (col, label) in enumerate(components):
        vals  = df_fiis[col].values
        x_jit = x_pos + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x_jit, vals, c=colors_by_arch,
                   alpha=0.65, s=40,
                   edgecolors='white', linewidths=0.4, zorder=3)
        ax.plot([x_pos-0.3, x_pos+0.3],
                [np.median(vals)]*2,
                color='#1A1A2E', linewidth=2.0, zorder=4)

    ax.axhline(y=1.0, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.6, label='$= 1$ (balance)')
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([l for _,l in components], fontsize=11)
    ax.set_ylabel('Component value', fontsize=10)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top','right']].set_visible(False)

    handles = [
        mpatches.Patch(color=ARCH_COLORS['A'], alpha=0.8,
                       label='Archetype A (Sampling Deficit)'),
        mpatches.Patch(color=ARCH_COLORS['C'], alpha=0.8,
                       label='Archetype C (Geometric Deficit)'),
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.9,
              loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# 3. Remedy Effectiveness Plot
# ─────────────────────────────────────────────

def plot_remedy_effectiveness(df_raw, out_path):
    """
    Boxplot of G-mean per remedy, one panel per classifier.
    """
    classifiers = df_raw['classifier'].unique()
    n_clf = len(classifiers)

    fig, axes = plt.subplots(1, n_clf,
                             figsize=(4*n_clf, 5), sharey=True)
    fig.patch.set_facecolor(BG)
    if n_clf == 1:
        axes = [axes]

    clf_labels = {
        'logistic_regression': 'LR',
        'random_forest':       'RF',
        'xgboost':             'XGB',
    }
    remedy_colors = {
        'no_correction':        '#888888',
        'threshold_correction': '#6B4226',
        'ROS':                  '#E84855',
        'SMOTE':                '#2E86AB',
        'BorderlineSMOTE':      '#9B59B6',
        'class_weighting':      '#F4A261',
    }

    for ax, clf in zip(axes, classifiers):
        ax.set_facecolor(BG)
        sub  = df_raw[df_raw['classifier']==clf]
        data = [sub[sub['remedy']==r]['G_mean'].values
                for r in REMEDY_ORDER]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          medianprops=dict(color='#1A1A2E',
                                           linewidth=2))
        for patch, r in zip(bp['boxes'], REMEDY_ORDER):
            patch.set_facecolor(remedy_colors[r])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(REMEDY_ORDER)+1))
        ax.set_xticklabels(
            [REMEDY_LABELS[r] for r in REMEDY_ORDER],
            rotation=30, ha='right', fontsize=7.5)
        ax.set_title(clf_labels.get(clf, clf),
                     fontsize=9, fontweight='bold')
        ax.set_ylabel('G-mean', fontsize=9)
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        ax.spines[['top','right']].set_visible(False)

    fig.suptitle('Vertebral Column: Remedy Effectiveness\n'
                 '(G-mean across 30 CV runs per classifier)',
                 fontsize=10, fontweight='bold', color='#1A1A2E')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# 4. LaTeX Diagnostic Summary Table
# ─────────────────────────────────────────────

def make_latex_table(df_fiis, df_raw, tex_path):
    """
    Compact LaTeX table:
      - FIIS components (mean ± std)
      - Archetype + compensation
      - Best remedy per classifier
    """
    comp_cols = ['fiis_R_n','fiis_R_p','fiis_R_g','fiis_C','fiis_R']
    stats     = df_fiis[comp_cols].agg(['mean','std'])

    arch_counts = df_fiis['fiis_archetype'].value_counts()
    comp_counts = df_fiis['fiis_compensation'].value_counts()
    arch_label  = arch_counts.index[0]
    arch_pct    = 100 * arch_counts.iloc[0] / len(df_fiis)
    comp_label  = comp_counts.index[0]
    comp_pct    = 100 * comp_counts.iloc[0] / len(df_fiis)

    # Best remedy per classifier
    clf_best = {}
    for clf, grp in df_raw.groupby('classifier'):
        pivot = grp.pivot_table(
            index='run', columns='remedy',
            values='G_mean', aggfunc='mean')[REMEDY_ORDER]
        ranks    = pivot.rank(axis=1, ascending=False, method='average')
        avg_r    = ranks.mean()
        best_r   = avg_r.idxmin()
        best_gm  = pivot[best_r].mean()
        nc_gm    = pivot['no_correction'].mean()
        clf_best[clf] = (REMEDY_LABELS[best_r], best_gm,
                         best_gm - nc_gm)

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\scriptsize')
    lines.append(
        r'\caption{FIIS diagnostic results for the Vertebral Column '
        r'dataset ($n=310$, IR$=2.1$). '
        r'Component values: mean\,$\pm$\,std across 30 CV folds. '
        r'Archetype and compensation regime are the most frequent '
        r'label across folds (frequency \%). '
        r'Best remedy: highest Nemenyi average rank per classifier '
        r'($\Delta$: G-mean improvement over no correction).}'
    )
    lines.append(r'\label{tab:vertebral_diagnostic}')

    # Part (a): FIIS components
    lines.append(r'\begin{tabular}{lrrrrr}')
    lines.append(r'\toprule')
    lines.append(
        r'\multicolumn{6}{c}{\textbf{(a) FIIS Component Summary}} \\')
    lines.append(r'\midrule')
    lines.append(
        r'& $R_n$ & $R_p$ & $R_g$ & $C$ & $R$ \\')
    lines.append(r'\midrule')

    means = [stats.loc['mean', c] for c in comp_cols]
    stds  = [stats.loc['std',  c] for c in comp_cols]
    row   = ' & '.join(
        f'${m:.3f}\\pm{s:.3f}$' for m, s in zip(means, stds))
    lines.append(f'    Mean\\,$\\pm$\\,std & {row} \\\\')

    lines.append(r'\midrule')
    lines.append(
        f'\\multicolumn{{3}}{{l}}{{Archetype: '
        f'\\textbf{{{arch_label}}} ({arch_pct:.0f}\\% of folds)}} &'
        f'\\multicolumn{{3}}{{r}}{{Compensation: '
        f'\\textbf{{{comp_label}}} ({comp_pct:.0f}\\% of folds)}} \\\\'
    )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    lines.append(r'\medskip')

    # Part (b): G-mean per remedy — mean row + std row per remedy
    clf_order = ['logistic_regression','random_forest','xgboost']

    lines.append(r'\begin{tabular}{lrrrr}')
    lines.append(r'\toprule')
    lines.append(
        r'\multicolumn{5}{c}{\textbf{(b) G-mean per Remedy '
        r'(mean / std across 30 runs)}} \\')
    lines.append(r'\midrule')
    lines.append(
        r'\textbf{Remedy} & \textbf{LR} & \textbf{RF} '
        r'& \textbf{XGB} & \textbf{Avg.} \\')
    lines.append(r'\midrule')

    for i, r in enumerate(REMEDY_ORDER):
        means = []
        stds  = []
        for clf in clf_order:
            sub = df_raw[(df_raw['remedy']==r) &
                         (df_raw['classifier']==clf)]['G_mean']
            means.append(f'{sub.mean():.3f}')
            stds.append(f'({sub.std():.3f})')
        avg_m = df_raw[df_raw['remedy']==r]['G_mean'].mean()
        avg_s = df_raw[df_raw['remedy']==r]['G_mean'].std()
        means.append(f'\\textbf{{{avg_m:.3f}}}')
        stds.append(f'\\textbf{{({avg_s:.3f})}}')

        lines.append(
            f"    {REMEDY_LABELS[r]} & " +
            ' & '.join(means) + r' \\\\')
        lines.append(
            f"    & " + ' & '.join(stds) + r' \\\\')
        if i < len(REMEDY_ORDER) - 1:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {tex_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    raw_path  = os.path.join(RESULTS_DIR, 'vertebral_raw.csv')
    fiis_path = os.path.join(RESULTS_DIR, 'vertebral_fiis.csv')

    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Run vertebral_pipeline.py first.")
        return

    df_raw  = pd.read_csv(raw_path)
    df_fiis = pd.read_csv(fiis_path)

    print(f"Loaded: {len(df_raw)} result rows, "
          f"{len(df_fiis)} FIIS fold records")

    # 1. Summary
    _, arch, arch_pct, comp, comp_pct = fiis_summary(df_fiis)

    # 2. FIIS strip plot
    plot_fiis_folds(
        df_fiis,
        os.path.join(RESULTS_DIR, 'plot_fiis_folds.png'))

    # 3. Remedy effectiveness
    plot_remedy_effectiveness(
        df_raw,
        os.path.join(RESULTS_DIR, 'plot_remedy_effectiveness.png'))

    # 4. LaTeX table
    make_latex_table(
        df_fiis, df_raw,
        os.path.join(RESULTS_DIR, 'table_vertebral_diagnostic.tex'))

    print(f"\nDiagnosis: Archetype {arch} ({arch_pct:.0f}% stable), "
          f"Compensation {comp} ({comp_pct:.0f}% stable)")


if __name__ == "__main__":
    main()