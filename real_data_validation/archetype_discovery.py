"""
archetype_discovery.py
---------------------------
Same as archetype_discovery.py but Archetype B is removed.
Only Archetype A (Sampling Deficit) and Archetype C (Geometric Deficit).

Rules:
- A: R_p >= 1 AND R_g >= 1  → pure sampling deficit
- C: R_g < 1                 → geometric deficit (regardless of R_p)

When both R_p < 1 and R_g < 1, C wins because we retain only
the geometric-energy archetype as the compounding mechanism.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'mathtext.fontset': 'cm',   # renders $...$ math in Computer Modern, matching LaTeX
    'legend.frameon': False,    # optional: cleaner legend box, common in papers
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SAVING_DIR  = os.path.join(RESULTS_DIR, "archetype_discovery_noB" \
"")
os.makedirs(SAVING_DIR, exist_ok=True)

EPSILON = 0.05

ARCH_COLORS = {
    'A': '#623991',
    'C': '#C55D08',
}
ARCH_LABELS = {
    'A': 'Archetype A (Sampling Deficit)',
    'C': 'Archetype C (Geometric Deficit)',
}
COMP_COLORS = {
    'O': '#2E7D32',
    'B': '#1565C0',
    'D': '#C62828',
}
BG = 'white'


def assign_archetype(row):
    """
    A: R_p >= 1 AND R_g >= 1 — no geometric compounding
    C: R_g < 1               — geometric deficit dominant
    Archetype B excluded per theoretical proposition and empirical absence.
    """
    Rg = row['fiis_R_g']
    Rp = row['fiis_R_p']
    if Rp >= 1 and Rg >= 1:
        return 'A'
    if Rg < 1:
        return 'C'
    # Rp < 1 only (no B) — assign A since no geometric energy deficit
    return 'A'


def assign_compensation(r):
    if r > 1 + EPSILON:   return 'O'
    elif r >= 1 - EPSILON: return 'B'
    else:                  return 'D'


def label_raw_data(raw_path):
    raw = pd.read_csv(raw_path)
    nc  = raw[
        (raw['remedy']     == 'no_correction') &
        (raw['classifier'] == 'logistic_regression')
    ].copy()
    nc['archetype_label']    = nc.apply(assign_archetype, axis=1)
    nc['compensation_label'] = nc['fiis_R'].apply(assign_compensation)
    out = os.path.join(SAVING_DIR, 'real_data_raw_labeled.csv')
    nc.to_csv(out, index=False)
    print(f"Saved: {out}")
    return nc


def compute_archetype_table(nc):
    grp_cols = ['dataset', 'source', 'n', 'd', 'IR']
    rows = []
    for keys, grp in nc.groupby(grp_cols):
        n_runs = len(grp)
        row    = dict(zip(grp_cols, keys))
        row['R_mean'] = grp['fiis_R'].mean()
        row['R_std']  = grp['fiis_R'].std()
        for col, label in [('fiis_R_n','R_n'),('fiis_R_p','R_p'),
                            ('fiis_R_g','R_g'),('fiis_C','C')]:
            row[f'{label}_mean'] = grp[col].mean()
            row[f'{label}_std']  = grp[col].std()

        arch_counts = grp['archetype_label'].value_counts()
        for label in ['A','C']:
            row[f'pct_arch_{label}'] = round(
                100 * arch_counts.get(label, 0) / n_runs, 1)

        arch_pcts = {l: row[f'pct_arch_{l}'] for l in ['A','C']}
        active    = [(l,p) for l,p in arch_pcts.items() if p > 0]
        active.sort(key=lambda x: -x[1])
        if len(active) == 1 and active[0][1] == 100.0:
            row['archetype_label']      = active[0][0]
            row['archetype_main_label'] = active[0][0]
        else:
            active = [
                (label, int(arch_counts.get(label, 0)))
                for label in ['A','C']
                if arch_counts.get(label, 0) > 0
            ]
            active.sort(key = lambda x: (-x[1], x[0]!= 'C'))
            
            row['archetype_label']      = ''.join(l for l,_ in active)
            row['archetype_main_label'] =  active[0][0] if active else 'A'

        comp_counts = grp['compensation_label'].value_counts()
        for label in ['O','B','D']:
            row[f'pct_comp_{label}'] = round(
                100 * comp_counts.get(label, 0) / n_runs, 1)
        comp_pcts = {l: row[f'pct_comp_{l}'] for l in ['O','B','D']}
        active_c  = [(l,p) for l,p in comp_pcts.items() if p > 0]
        active_c.sort(key=lambda x: -x[1])
        if len(active_c) == 1 and active_c[0][1] == 100.0:
            row['compensation_label']      = active_c[0][0]
            row['compensation_main_label'] = active_c[0][0]
        else:
            row['compensation_label']      = ''.join(l for l,_ in active_c)
            row['compensation_main_label'] = active_c[0][0] if active_c else 'D'
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('IR').reset_index(drop=True)
    out = os.path.join(SAVING_DIR, 'archetype_discovery.csv')
    df.round(4).to_csv(out, index=False)
    print(f"Saved: {out}")
    return df


def _strip_base(df, out_path, color_fn, legend_handles=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    components = [('R_p_mean','$R_p$'),('R_g_mean','$R_g$'),('C_mean','$C$')]
    rng = np.random.default_rng(42)
    for x_pos, (col, label) in enumerate(components):
        vals   = df[col].values
        colors = [color_fn(row) for _, row in df.iterrows()]
        x_jit  = x_pos + rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x_jit, vals, c=colors, alpha=0.65, s=35,
                   edgecolors='white', linewidths=0.3, zorder=3)
        ax.plot([x_pos-0.3, x_pos+0.3],
                [np.median(vals), np.median(vals)],
                color='#1A1A2E', linewidth=2.0, zorder=4)
    ax.axhline(y=1.0, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.6, label='Component $= 1$')
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([l for _,l in components], fontsize=11)
    ax.set_ylabel('Component value', fontsize=10)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9,
                  framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_strip_plain(df, out_path):
    _strip_base(df, out_path, color_fn=lambda row: '#2E86AB')


def plot_strip_archetype(df, out_path):
    handles = [mpatches.Patch(color=ARCH_COLORS[l], alpha=0.8,
                               label=ARCH_LABELS[l])
               for l in ['A','C']]
    _strip_base(df, out_path,
                color_fn=lambda row: ARCH_COLORS.get(
                    row['archetype_main_label'], '#888888'),
                legend_handles=handles)


def plot_strip_compensation(df, out_path):
    comp_names = {'O':'Overbalance','B':'Balance','D':'Deficiency'}
    handles = [mpatches.Patch(color=COMP_COLORS[l], alpha=0.8,
                               label=f'{l}: {comp_names[l]}')
               for l in ['O','B','D']]
    _strip_base(df, out_path,
                color_fn=lambda row: COMP_COLORS.get(
                    row['compensation_main_label'], '#888888'),
                legend_handles=handles)


def plot_stacked_bar(df, out_path):
    df_s = df.sort_values('IR').reset_index(drop=True)
    n = len(df_s)
    fig, ax = plt.subplots(figsize=(10, max(8, n*0.22)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    y = np.arange(n)
    pA = df_s['pct_arch_A'].values
    pC = df_s['pct_arch_C'].values
    ax.barh(y, pA, color=ARCH_COLORS['A'], alpha=0.85, label='A (Sampling Deficit)')
    ax.barh(y, pC, left=pA, color=ARCH_COLORS['C'], alpha=0.85, label='C (Geometric Deficit)')
    ax.set_yticks(y); ax.set_yticklabels(df_s['dataset'].tolist(), fontsize=5.5)
    ax.set_xlabel('Percentage of runs (%)', fontsize=9)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, framealpha=0.9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def compute_log_means(raw_path):
    raw = pd.read_csv(raw_path)
    nc  = raw[(raw['remedy']=='no_correction') &
              (raw['classifier']=='logistic_regression')].copy()
    nc['log_R_n'] = np.log(nc['fiis_R_n'].clip(lower=1e-10))
    nc['log_R']   = np.log(nc['fiis_R'].clip(lower=1e-10))
    return nc.groupby('dataset')[['log_R_n','log_R']].mean().round(4).reset_index()


def _scatter_RvsRn(df, log_df, out_path, color_fn, legend_handles=None):
    merged = df.merge(log_df, on='dataset')
    log_Rn = merged['log_R_n'].values
    log_R  = merged['log_R'].values
    colors = [color_fn(row) for _, row in merged.iterrows()]
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.scatter(log_Rn, log_R, c=colors, alpha=0.75, s=50,
               edgecolors='white', linewidths=0.5, zorder=3)
    lim = [min(log_Rn.min(), log_R.min())-0.3,
           max(log_Rn.max(), log_R.max())+0.3]
    ax.plot(lim, lim, color='#1A1A2E', linestyle='--',
            linewidth=1.5, alpha=0.6, label='$R=R_n$', zorder=2)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(0.02, 0.97, 'Compensation\n($R > R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#2E7D32', va='top', fontstyle='italic')
    ax.text(0.98, 0.03, 'Compounding\n($R < R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#C62828', va='bottom', ha='right', fontstyle='italic')
    n_above = (log_R > log_Rn).sum()
    n_below = (log_R < log_Rn).sum()
    ax.text(0.98, 0.97,
            f'Above: {n_above} ({100*n_above/len(merged):.0f}\\%)\n'
            f'Below: {n_below} ({100*n_below/len(merged):.0f}\\%)',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='#CCCCCC'))
    ax.set_xlabel(r'$\log R_n$ (class imbalance)', fontsize=11)
    ax.set_ylabel(r'$\log R$ (information imbalance)', fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9)
    else:
        ax.legend(fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def plot_RvsRn_plain(df, log_df, out_path):
    _scatter_RvsRn(df, log_df, out_path, color_fn=lambda row: '#2E86AB')


def plot_RvsRn_archetype(df, log_df, out_path):
    handles = [mpatches.Patch(color=ARCH_COLORS[l], alpha=0.8,
                               label=ARCH_LABELS[l])
               for l in ['A','C']]
    _scatter_RvsRn(df, log_df, out_path,
                   color_fn=lambda row: ARCH_COLORS.get(
                       row['archetype_main_label'], '#888888'),
                   legend_handles=handles)


def plot_RvsRn_compensation(df, log_df, out_path):
    comp_names = {'O':'Overbalance','B':'Balance','D':'Deficiency'}
    handles = [mpatches.Patch(color=COMP_COLORS[l], alpha=0.8,
                               label=f'{l}: {comp_names[l]}')
               for l in ['O','B','D']]
    _scatter_RvsRn(df, log_df, out_path,
                   color_fn=lambda row: COMP_COLORS.get(
                       row['compensation_main_label'], '#888888'),
                   legend_handles=handles)


def main():
    raw_path = os.path.join(RESULTS_DIR, 'real_data_raw.csv')

    print("Step 1 — Labeling (A/C only)...")
    nc = label_raw_data(raw_path)

    print("\nStep 2 — Archetype table...")
    df = compute_archetype_table(nc)
    print(f"Datasets: {len(df)}")
    print(df['archetype_label'].value_counts().to_string())
    print(df['compensation_label'].value_counts().to_string())

    print("\nStep 3 — Strip plots...")
    plot_strip_plain(df,        os.path.join(SAVING_DIR, 'plot_strip_plain.png'))
    plot_strip_archetype(df,    os.path.join(SAVING_DIR, 'plot_strip_archetype.png'))
    plot_strip_compensation(df, os.path.join(SAVING_DIR, 'plot_strip_compensation.png'))

    print("\nStep 4 — Stacked bar...")
    plot_stacked_bar(df, os.path.join(SAVING_DIR, 'plot_stacked_bar.png'))

    print("\nStep 5 — R vs R_n scatter...")
    log_df = compute_log_means(raw_path)
    plot_RvsRn_plain(df, log_df,        os.path.join(SAVING_DIR, 'plot_RvsRn_plain.png'))
    plot_RvsRn_archetype(df, log_df,    os.path.join(SAVING_DIR, 'plot_RvsRn_archetype.png'))
    plot_RvsRn_compensation(df, log_df, os.path.join(SAVING_DIR, 'plot_RvsRn_compensation.png'))

    print(f"\nAll outputs saved to: {SAVING_DIR}")


if __name__ == "__main__":
    main()