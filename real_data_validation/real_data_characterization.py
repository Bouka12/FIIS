""" 
this script is used to extract all evidence from real data to answer PART I of the research question (RQ2): "Does FIIS provide a meaningful characterisation
of real-world imbalanced datasets and reveal recurring empirical imbalance regimes?"

Part I: Characterization of real-world imbalanced datasets
    *  Get CSV/Latex Table (datasets summary) for appendix B-A with the following information:
        -   Dataset name 
        -   Number of instances
        -   Number of features
        -   Number of classes
        -   Imbalance ratio (IR)
        -   Source of the dataset (UCI, KEEL, OpenML, etc.)
    *   Get CSV/Latex Table (FIIS components) for appendix B-A with the following information:
        -   Dataset name
        -   Imbalance ratio (IR) is same as R_n so eliminate it from the table and keep only R_n, R_p, R_e, R_s, R_c
        -   R (mean+std)
        -   FIIS components (mean+std): R_n, R_p, R_e, R_s, R_c
    
    * Visualizations of the real-world imbalanced datasets characterization:
        - dot plot: on the x axis the variables R_j (j in {n, g, p, c}) and R, and the y axis the values of the variables and each dot is a dataset.
            the dots will be (1st version) without color coding    


real_data_characterization.py
------------------------------
Part I of RQ2: Characterisation of real-world imbalanced datasets.

Produces:
  1. Dataset summary table — LaTeX (longtable) + CSV
  2. FIIS components table — LaTeX (longtable) + CSV
  3. Strip/dot plot of FIIS components across datasets
  4. R vs R_n scatter plot (decoupling evidence)

Usage:
    python real_data_characterization.py
Input:
    results/real_data_fiis.csv
Outputs:
    results/table_dataset_summary.tex / .csv
    results/table_fiis_components.tex / .csv
    results/plot_fiis_strip.png
    results/plot_R_vs_Rn.png
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
# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BG = 'white'

DOMINANT_COLORS = {
    'n': "#681DCA",   # red    — Archetype A (sampling)
    'p': '#2E86AB',   # blue   — Archetype B (boundary)
    'g': "#0D7412",   # orange — Archetype C (feature energy)
    'C': '#6B4226',   # brown  — Coupling
}


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

def load_fiis(path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, 'real_data_fiis.csv')
    df = pd.read_csv(path)
    df = df.sort_values('IR').reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 1. Dataset summary table
# ─────────────────────────────────────────────

def make_dataset_summary_table(df, tex_path, csv_path):
    """LaTeX longtable + CSV: dataset, source, n, d, IR."""
    summary = df[['dataset', 'source', 'n', 'd', 'IR']].copy()
    summary = summary.sort_values(['source', 'IR']).reset_index(drop=True)

    # CSV
    csv_out = summary.copy()
    csv_out.columns = ['Dataset', 'Source', 'n', 'd', 'IR']
    csv_out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX
    lines = []
    lines.append(r'\begin{longtable}{llrrr}')
    lines.append(
        r'\caption{Real-world imbalanced benchmark datasets. '
        r'$n$: instances; $d$: features; IR: imbalance ratio ($n_0/n_1$). '
        r'Sorted by IR within each source group.}'
    )
    lines.append(r'\label{tab:datasets_summary} \\')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Dataset} & \textbf{Source} & '
                 r'\textbf{$n$} & \textbf{$d$} & \textbf{IR} \\')
    lines.append(r'\midrule')
    lines.append(r'\endfirsthead')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Dataset} & \textbf{Source} & '
                 r'\textbf{$n$} & \textbf{$d$} & \textbf{IR} \\')
    lines.append(r'\midrule')
    lines.append(r'\endhead')
    lines.append(r'\midrule \multicolumn{5}{r}{\textit{Continued}} \\')
    lines.append(r'\endfoot')
    lines.append(r'\bottomrule')
    lines.append(r'\endlastfoot')

    for source_label, source_key in [
        ('KEEL Imbalanced Repository', 'KEEL'),
        ('Imbalanced-learn Benchmark Suite', 'imblearn'),
    ]:
        lines.append(
            f'\\multicolumn{{5}}{{l}}'
            f'{{\\textit{{{source_label}}}}} \\\\'
        )
        lines.append(r'\midrule')
        sub = summary[summary['source'] == source_key]
        for _, row in sub.iterrows():
            name = row['dataset'].replace('_', r'\_')
            lines.append(
                f"    {name} & {row['source']} & "
                f"{int(row['n'])} & {int(row['d'])} & "
                f"{row['IR']:.2f} \\\\"
            )
        lines.append(r'\midrule')

    lines.append(r'\end{longtable}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {tex_path}")


# ─────────────────────────────────────────────
# 2. FIIS components table
# ─────────────────────────────────────────────

def make_fiis_table(df, tex_path, csv_path):
    """LaTeX longtable + CSV: FIIS components per dataset."""
    cols = ['dataset', 'IR',
            'fiis_R_n', 'fiis_R_p', 'fiis_R_g', 'fiis_C', 'fiis_R',
            'fiis_dominant', 'fiis_dominant_stability']
    out = df[cols].copy().sort_values('IR').reset_index(drop=True)
    out = out.round(4)

    # CSV
    csv_out = out.copy()
    csv_out.columns = ['Dataset', 'IR', 'R_n', 'R_p', 'R_g', 'C', 'R',
                       'Dominant', 'Stability']
    csv_out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX
    lines = []
    lines.append(r'\begin{longtable}{lrrrrrrrll}')
    lines.append(
        r'\caption{FIIS component summary for all benchmark datasets. '
        r'Values are means across $5\times6=30$ repeated stratified CV folds. '
        r'$R_n$: sampling; $R_p$: boundary-proximity; '
        r'$R_g$: feature-energy; $C$: coupling; $R$: overall ratio; '
        r'Dom.: dominant component; Stab.: stability (fraction of folds agreeing).}'
    )
    lines.append(r'\label{tab:fiis_components} \\')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{Dataset} & \textbf{IR} & \textbf{$R_n$} & \textbf{$R_p$} & '
        r'\textbf{$R_g$} & \textbf{$C$} & \textbf{$R$} & '
        r'\textbf{Dom.} & \textbf{Stab.} \\'
    )
    lines.append(r'\midrule')
    lines.append(r'\endfirsthead')
    lines.append(r'\toprule')
    lines.append(
        r'\textbf{Dataset} & \textbf{IR} & \textbf{$R_n$} & \textbf{$R_p$} & '
        r'\textbf{$R_g$} & \textbf{$C$} & \textbf{$R$} & '
        r'\textbf{Dom.} & \textbf{Stab.} \\'
    )
    lines.append(r'\midrule')
    lines.append(r'\endhead')
    lines.append(r'\midrule \multicolumn{9}{r}{\textit{Continued}} \\')
    lines.append(r'\endfoot')
    lines.append(r'\bottomrule')
    lines.append(r'\endlastfoot')

    for _, row in out.iterrows():
        name = row['dataset'].replace('_', r'\_')
        dom  = f"$\\phi_{{{row['fiis_dominant']}}}$"
        lines.append(
            f"    {name} & {row['IR']:.2f} & "
            f"{row['fiis_R_n']:.3f} & {row['fiis_R_p']:.3f} & "
            f"{row['fiis_R_g']:.3f} & {row['fiis_C']:.3f} & "
            f"{row['fiis_R']:.3f} & {dom} & "
            f"{row['fiis_dominant_stability']:.2f} \\\\"
        )

    lines.append(r'\end{longtable}')

    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {tex_path}")


# ─────────────────────────────────────────────
# 3. Strip/dot plot of FIIS components
# ─────────────────────────────────────────────

def plot_fiis_strip(df, out_path):
    """Strip plot: one dot per dataset per FIIS component."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    components = [
        ('fiis_R_n', '$R_n$'),
        ('fiis_R_p', '$R_p$'),
        ('fiis_R_g', '$R_g$'),
        ('fiis_C',   '$C$'),
        ('fiis_R',   '$R$'),
    ]

    rng    = np.random.default_rng(42)
    jitter = 0.15

    for x_pos, (col, label) in enumerate(components):
        vals   = df[col].values
        colors = [DOMINANT_COLORS.get(d, '#888888')
                  for d in df['fiis_dominant'].values]
        x_jit  = x_pos + rng.uniform(-jitter, jitter, size=len(vals))

        ax.scatter(x_jit, vals,
                   c=colors, alpha=0.65, s=30,
                   edgecolors='white', linewidths=0.3,
                   zorder=3)
        # Median line
        ax.plot([x_pos - 0.3, x_pos + 0.3],
                [np.median(vals), np.median(vals)],
                color='#1A1A2E', linewidth=2.0, zorder=4)

    ax.axhline(y=1.0, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.6, zorder=2)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([l for _, l in components], fontsize=11)
    ax.set_ylabel('Component value', fontsize=10)
    ax.set_xlabel('FIIS component', fontsize=10)
    ax.set_title(
        f'FIIS Component Distribution Across {len(df)} Benchmark Datasets\n'
        r'(dots = datasets, horizontal bar = median, colored by dominant archetype)',
        fontsize=10, fontweight='bold', color='#1A1A2E'
    )
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    handles = [
        mpatches.Patch(color=DOMINANT_COLORS['n'], alpha=0.8,
                       label='Archetype A ($\\phi_n$ dominant)'),
        mpatches.Patch(color=DOMINANT_COLORS['g'], alpha=0.8,
                       label='Archetype C ($\\phi_g$ dominant)'),
        Line2D([0],[0], color='gray', linestyle='--',
               linewidth=1.5, label='$R=1$ (balance threshold)'),
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.9,
              loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# 4. R vs R_n scatter — decoupling evidence
# ─────────────────────────────────────────────

def plot_R_vs_Rn(df, out_path):
    """
    R vs R_n scatter in natural scale (no log transformation).
    x-axis: R_n (class imbalance ratio, always in (0,1))
    y-axis: R (information imbalance ratio)
    Diagonal R = R_n marks no compensation.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    Rn        = df['fiis_R_n'].values
    R         = df['fiis_R'].values
    DOT_COLOR = '#2E86AB'

    ax.scatter(Rn, R,
               color=DOT_COLOR, alpha=0.70, s=50,
               edgecolors='white', linewidths=0.5,
               zorder=3)

    # Diagonal R = R_n
    lim = [0, max(Rn.max(), R.max()) + 0.1]
    ax.plot(lim, lim, color='#1A1A2E', linestyle='--',
            linewidth=1.5, alpha=0.6,
            label='$R = R_n$ (no compensation)', zorder=2)

    # Reference line at R = 1 (balance threshold)
    ax.axhline(y=1.0, color='gray', linestyle=':',
               linewidth=1, alpha=0.5,
               label='$R=1$ (information balance)')

    # Region labels
    ax.text(0.02, 0.97, 'Compensation\n($R > R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#2E7D32', va='top', fontstyle='italic')
    ax.text(0.98, 0.03, 'Compounding\n($R < R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#C62828', va='bottom', ha='right', fontstyle='italic')

    # Count above/below diagonal
    n_above = (R > Rn).sum()
    n_below = (R < Rn).sum()
    ax.text(0.98, 0.97,
            f'Above diagonal: {n_above} ({100*n_above/len(df):.0f}\\%)\n'
            f'Below diagonal: {n_below} ({100*n_below/len(df):.0f}\\%)',
            transform=ax.transAxes, fontsize=7.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='#CCCCCC'))

    ax.set_xlabel(r'$R_n$ (class imbalance)', fontsize=11)
    ax.set_ylabel(r'$R$ (information imbalance)', fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------
# Without color coding 
# ---------------------------------------------


# ─────────────────────────────────────────────
# 4. Strip/dot plot of FIIS components
# ─────────────────────────────────────────────
def plot_fiis_strip_wc(df, out_path):
    """Strip plot: one dot per dataset per FIIS component. No color coding."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
 
    components = [
        ('fiis_R_n', '$R_n$'),
        ('fiis_R_p', '$R_p$'),
        ('fiis_R_g', '$R_g$'),
        ('fiis_C',   '$C$'),
        ('fiis_R',   '$R$'),
    ]
 
    rng    = np.random.default_rng(42)
    jitter = 0.15
    DOT_COLOR = '#2E86AB'   # single neutral color for all dots
 
    for x_pos, (col, label) in enumerate(components):
        vals  = df[col].values
        x_jit = x_pos + rng.uniform(-jitter, jitter, size=len(vals))
 
        ax.scatter(x_jit, vals,
                   color=DOT_COLOR, alpha=0.60, s=30,
                   edgecolors='white', linewidths=0.3,
                   zorder=3)
        # Median line
        ax.plot([x_pos - 0.3, x_pos + 0.3],
                [np.median(vals), np.median(vals)],
                color='#1A1A2E', linewidth=2.0, zorder=4)
 
    ax.axhline(y=1.0, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.6, zorder=2,
               label='$R=1$ (balance threshold)')
 
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([l for _, l in components], fontsize=11)
    ax.set_ylabel('Component value', fontsize=10)
    ax.set_xlabel('FIIS component', fontsize=10)
    # ax.set_title(
    #     f'FIIS Component Distribution Across {len(df)} Benchmark Datasets\n'
    #     r'(dots = datasets, horizontal bar = median)',
    #     fontsize=10, fontweight='bold', color='#1A1A2E'
    # )
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")
 
 
# ─────────────────────────────────────────────
# 6. R vs R_n scatter — decoupling evidence
# ─────────────────────────────────────────────
 
def plot_R_vs_Rn_wc(df, out_path):
    """log R vs log R_n scatter — decoupling of class and info imbalance."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
 
    log_Rn    = np.log(df['fiis_R_n'].values)
    log_R     = np.log(df['fiis_R'].values)
    DOT_COLOR = '#2E86AB'   # single neutral color, no archetype coding
 
    ax.scatter(log_Rn, log_R,
               color=DOT_COLOR, alpha=0.70, s=50,
               edgecolors='white', linewidths=0.5,
               zorder=3)
 
    # Diagonal R = R_n
    lim = [min(log_Rn.min(), log_R.min()) - 0.3,
           max(log_Rn.max(), log_R.max()) + 0.3]
    ax.plot(lim, lim, color='#1A1A2E', linestyle='--',
            linewidth=1.5, alpha=0.6,
            label='$R = R_n$ (no compensation)', zorder=2)
 
    # Reference lines at log=0 (R=1 or R_n=1)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
 
    # Region labels
    ax.text(0.02, 0.97, 'Compensation\n($R > R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#2E7D32', va='top', fontstyle='italic')
    ax.text(0.98, 0.03, 'Compounding\n($R < R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#C62828', va='bottom', ha='right', fontstyle='italic')
 
    ax.set_xlabel(r'$\log R_n$ (class imbalance)', fontsize=11)
    ax.set_ylabel(r'$\log R$ (information imbalance)', fontsize=11)
    # ax.set_title(
    #     'Class Imbalance vs Information Imbalance\n'
    #     r'Decoupling of $R$ from $R_n$ across benchmark datasets',
    #     fontsize=11, fontweight='bold', color='#1A1A2E'
    # )
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
 
    # Count above/below diagonal
    n_above = (log_R > log_Rn).sum()
    n_below = (log_R < log_Rn).sum()
    ax.text(0.98, 0.97,
            f'Above diagonal: {n_above} ({100*n_above/len(df):.0f}\\%)\n'
            f'Below diagonal: {n_below} ({100*n_below/len(df):.0f}\\%)',
            transform=ax.transAxes, fontsize=7.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='#CCCCCC'))
 
    ax.legend(fontsize=8, framealpha=0.9)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")



# ---------------------------------------------
# FIIS Components in log scale
# ---------------------------------------------

 
def compute_log_fiis_from_raw(raw_path):
    """
    Compute mean of log(fiis_j) per dataset from raw data.
    log is applied per run THEN averaged — correct order.
    Uses no_correction rows only (original training data FIIS).
    Returns DataFrame with one row per dataset.
    """
    raw = pd.read_csv(raw_path)
    nc  = raw[raw['remedy'] == 'no_correction'].copy()
 
    # log of each component per run
    for col in ['fiis_R_n', 'fiis_R_p', 'fiis_R_g', 'fiis_C', 'fiis_R']:
        nc[f'log_{col}'] = np.log(nc[col].clip(lower=1e-10))
 
    log_cols = ['log_fiis_R_n', 'log_fiis_R_p',
                'log_fiis_R_g', 'log_fiis_C', 'log_fiis_R']
 
    grp = (nc.groupby(['dataset', 'source', 'IR', 'n', 'd'])
             [log_cols].mean().round(4).reset_index())
    return grp
 
 
def plot_fiis_strip_log(df_log, out_path):
    """
    Strip plot in log scale.
    x-axis: log FIIS components (log R_n, log R_p, log R_g, log C, log R)
    y-axis: mean log value per dataset (computed from per-run logs)
    Vertical dashed line at log=0 marks the balance threshold (value=1).
    No color coding.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    BG = 'white'
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
 
    components = [
        ('log_fiis_R_n', r'$\log R_n$'),
        ('log_fiis_R_p', r'$\log R_p$'),
        ('log_fiis_R_g', r'$\log R_g$'),
        ('log_fiis_C',   r'$\log C$'),
        ('log_fiis_R',   r'$\log R$'),
    ]
 
    rng       = np.random.default_rng(42)
    jitter    = 0.15
    DOT_COLOR = '#2E86AB'
 
    for x_pos, (col, label) in enumerate(components):
        vals  = df_log[col].values
        x_jit = x_pos + rng.uniform(-jitter, jitter, size=len(vals))
 
        ax.scatter(x_jit, vals,
                   color=DOT_COLOR, alpha=0.60, s=30,
                   edgecolors='white', linewidths=0.3,
                   zorder=3)
        # Median line
        ax.plot([x_pos - 0.3, x_pos + 0.3],
                [np.median(vals), np.median(vals)],
                color='#1A1A2E', linewidth=2.0, zorder=4)
 
    # Reference line at log = 0 (component value = 1)
    ax.axhline(y=0, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.6, zorder=2,
               label='$\\log = 0$ (component $= 1$, balance threshold)')
 
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([l for _, l in components], fontsize=11)
    ax.set_ylabel(r'Mean $\log$ value across folds', fontsize=10)
    ax.set_xlabel('FIIS component (log scale)', fontsize=10)
    # ax.set_title(
    #     f'FIIS Component Distribution Across {len(df_log)} Benchmark Datasets'
    #     ' (log scale)\n'
    #     r'(dots = datasets, horizontal bar = median)',
    #     fontsize=10, fontweight='bold', color='#1A1A2E'
    # )
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")



def plot_R_vs_Rn_log(df_log, out_path):
    """
    log R vs log R_n scatter using mean(log(R)) and mean(log(R_n))
    computed per run from raw data — correct order.
    Both axes use mean-of-log, not log-of-mean.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    BG = 'white'
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
 
    log_Rn    = df_log['log_fiis_R_n'].values
    log_R     = df_log['log_fiis_R'].values
    DOT_COLOR = '#2E86AB'
 
    ax.scatter(log_Rn, log_R,
               color=DOT_COLOR, alpha=0.70, s=50,
               edgecolors='white', linewidths=0.5,
               zorder=3)
 
    # Diagonal R = R_n
    lim = [min(log_Rn.min(), log_R.min()) - 0.3,
           max(log_Rn.max(), log_R.max()) + 0.3]
    ax.plot(lim, lim, color='#1A1A2E', linestyle='--',
            linewidth=1.5, alpha=0.6,
            label='$R = R_n$ (no compensation)', zorder=2)
 
    # Reference lines at log=0
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
 
    # Region labels
    ax.text(0.02, 0.97, 'Compensation\n($R > R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#2E7D32', va='top', fontstyle='italic')
    ax.text(0.98, 0.03, 'Compounding\n($R < R_n$)',
            transform=ax.transAxes, fontsize=8,
            color='#C62828', va='bottom', ha='right', fontstyle='italic')
 
    ax.set_xlabel(r'$\log R_n$ (class imbalance)', fontsize=11)
    ax.set_ylabel(r'$\log R$ (information imbalance)', fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
 
    # Count above/below diagonal
    n_above = (log_R > log_Rn).sum()
    n_below = (log_R < log_Rn).sum()
    ax.text(0.98, 0.97,
            f'Above diagonal: {n_above} ({100*n_above/len(df_log):.0f}\\%)\n'
            f'Below diagonal: {n_below} ({100*n_below/len(df_log):.0f}\\%)',
            transform=ax.transAxes, fontsize=7.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='#CCCCCC'))
 
    ax.legend(fontsize=8, framealpha=0.9)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    fiis_path = os.path.join(RESULTS_DIR, 'real_data_fiis.csv')
    df = load_fiis(fiis_path)

    print(f"Loaded {len(df)} datasets | "
          f"IR: {df['IR'].min():.2f}–{df['IR'].max():.2f} | "
          f"Sources: {df['source'].value_counts().to_dict()}")
    # print(f"Dominant: {df['fiis_dominant'].value_counts().to_dict()}")
    
    # 0. Real Data Characterization
    SAVING_DIR = os.path.join(RESULTS_DIR, "charcterization_results")
    os.makedirs(SAVING_DIR, exist_ok=True)

    # 1. Dataset summary
    make_dataset_summary_table(
        df,
        tex_path=os.path.join(SAVING_DIR, 'table_dataset_summary.tex'),
        csv_path=os.path.join(SAVING_DIR, 'table_dataset_summary.csv'),
    )

    # 2. FIIS components
    # make_fiis_table(
    #     df,
    #     tex_path=os.path.join(SAVING_DIR, 'table_fiis_components.tex'),
    #     csv_path=os.path.join(SAVING_DIR, 'table_fiis_components.csv'),
    # )


    # 3. Strip plot
    # plot_fiis_strip(
    #     df,
    #     os.path.join(SAVING_DIR, 'plot_fiis_strip.png'),
    # )

    # 4. R vs Rn scatter
    plot_R_vs_Rn(
        df,
        os.path.join(SAVING_DIR, 'plot_R_vs_Rn.png'),
    )

    # 5. Strip plot without color coding
    plot_fiis_strip_wc(
        df,
        os.path.join(SAVING_DIR, 'plot_fiis_strip_wcc_notitle.png')
    )

    # 6. R vs Rn scatter without color color coding
    plot_R_vs_Rn_wc(
        df,
        os.path.join(SAVING_DIR, 'plot_R_vs_Rn_wcc_notitle.png')
    )

    
    # # 5b. Log-scale strip plot (requires raw CSV)
    # # 5b. Log-scale strip plot (requires raw CSV)
    # raw_path = os.path.join(RESULTS_DIR, 'real_data_raw.csv')
    # if os.path.exists(raw_path):
    #     df_log = compute_log_fiis_from_raw(raw_path)

    #     # save log-scaled FIIS components
    #     log_saving_path = os.path.join(SAVING_DIR, 'table_log_fiis_components.csv')
    #     df_log.to_csv(log_saving_path, index=False)
    #     plot_fiis_strip_log(
    #         df_log,
    #         os.path.join(SAVING_DIR, 'plot_fiis_strip_log.png'),
    #     )
    #     plot_R_vs_Rn_log(
    #         df_log,
    #         os.path.join(SAVING_DIR, 'plot_R_vs_Rn_log.png'),
    #     )
    # else:
    #     print(f"Skipping log strip plot — raw CSV not found at {raw_path}")
 
    print(f"\nAll outputs saved to: {SAVING_DIR}")


if __name__ == "__main__":
    main()