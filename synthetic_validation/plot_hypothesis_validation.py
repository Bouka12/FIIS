"""
plot_hypothesis_validation.py
-----------------------------
Four-panel hypothesis validation figure using aggregated results
from user-provided CSV files (mean +/- std over M=100 replications).

Panels:
    H1: DGP-S — G-mean vs n1/n0 (Archetype A, Sampling)
    H2: DGP-G — G-mean vs alpha (Archetype C, Feature Energy)
    H3: DGP-K — Delta G-mean vs R (Balance & Overbalance)

Error bars show +/- 1 std across replications.
BorderlineSMOTE included.

Usage:
    python plot_hypothesis_validation.py
Output:
    results/hypothesis_validation_figure.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BG = 'white'

REMEDY_STYLES = {
    "no_correction":        ("#888888", "x", "--", 1.5, "No Correction"),
    "threshold_correction": ("#6B4226", "D", "-",  2.0, "Threshold Correction"),
    "ROS":                  ("#E84855", "o", "-",  2.0, "ROS"),
    "SMOTE":                ("#2E86AB", "s", "-",  2.0, "SMOTE"),
    "BorderlineSMOTE":      ("#9B59B6", "P", "-",  2.0, "Borderline SMOTE"),
    "class_weighting":      ("#F4A261", "^", "-",  2.0, "Class Weighting"),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ─────────────────────────────────────────────
# Data loading helper
# ─────────────────────────────────────────────

def load_agg(filename):
    """
    Load aggregated CSV with MultiIndex columns (metric, mean/std).
    Returns DataFrame with MultiIndex columns.
    """
    path = os.path.join(RESULTS_DIR, filename)
    df   = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    return df


def get_metric(df, metric):
    """Extract mean and std for a given metric across all index values."""
    mean = df[(metric, "mean")]
    std  = df[(metric, "std")]
    return mean, std


# ─────────────────────────────────────────────
# Panel plotting helper
# ─────────────────────────────────────────────

def plot_panel(ax, df, group_col, x_vals, metric,
               xlabel, title, remedies_order=None):
    """
    Plot G-mean (mean +/- std) per remedy across x_vals.

    Parameters
    ----------
    ax          : matplotlib axis
    df          : aggregated DataFrame (MultiIndex: group_col, remedy)
    group_col   : first index level name (e.g. 'ratio', 'delta')
    x_vals      : ordered list of scenario parameter values
    metric      : metric name to plot (e.g. 'G_mean')
    xlabel      : x-axis label
    title       : subplot title
    remedies_order : list of remedy keys in desired plot order
    """
    ax.set_facecolor(BG)

    if remedies_order is None:
        remedies_order = list(REMEDY_STYLES.keys())

    for rem in remedies_order:
        if rem not in REMEDY_STYLES:
            continue
        color, marker, ls, lw, label = REMEDY_STYLES[rem]

        means = []
        stds  = []
        xs    = []

        for xv in x_vals:
            try:
                row  = df.loc[(xv, rem)]
                m    = row[(metric, "mean")]
                s    = row[(metric, "std")]
                means.append(m)
                stds.append(s)
                xs.append(xv)
            except KeyError:
                continue

        if not xs:
            continue

        xs    = np.array(xs)
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(xs, means,
                color=color, marker=marker,
                linestyle=ls, linewidth=lw,
                markersize=6, label=label, zorder=3)
        ax.fill_between(xs,
                        means - stds,
                        means + stds,
                        color=color, alpha=0.12, zorder=2)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(metric.replace("_", "-"), fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold',
                 color='#1A1A2E', pad=5)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(-0.05, 1.05)


# ─────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────

def plot_hypothesis_validation(save_dir=None):

    if save_dir is None:
        save_dir = RESULTS_DIR

    # Load CSVs
    df_s = load_agg("synthetic_validation/dgp_s/dgp_s_agg.csv")
    # df_b = load_agg("synthetic_validation/dgp_b/dgp_b_agg.csv")
    df_g = load_agg("synthetic_validation/dgp_g/dgp_g_agg.csv")
    df_k = load_agg("synthetic_validation/dgp_k/dgp_k_agg.csv")

    # Scenario parameter values
    x_s = sorted(df_s.index.get_level_values("ratio").unique())
    # x_b = sorted(df_b.index.get_level_values("delta").unique())
    x_g = sorted(df_g.index.get_level_values("alpha").unique())
    x_k = sorted(df_k.index.get_level_values("gamma").unique())

    # R values for DGP-K x-axis (use fiis_R mean from no_correction)
    R_vals = []
    for g in x_k:
        try:
            R_vals.append(df_k.loc[(g, "no_correction"), ("fiis_R", "mean")])
        except KeyError:
            R_vals.append(np.nan)
    R_vals = np.array(R_vals)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.patch.set_facecolor(BG)

    remedies_order = list(REMEDY_STYLES.keys())

    # ── Panel H1: DGP-S ──
    plot_panel(axes[0], df_s, "ratio", x_s,
               metric="G_mean",
               xlabel=r"$n_1/n_0$",
               title="H1: DGP-S\nArchetype A (Sampling Deficit)",
               remedies_order=remedies_order)


    # ── Panel H2: DGP-G ──
    plot_panel(axes[1], df_g, "alpha", x_g,
               metric="G_mean",
               xlabel=r"$\alpha$",
               title="H2: DGP-G\nArchetype C (Feature Energy)",
               remedies_order=remedies_order)

    # ── Panel H3: DGP-K — Delta G-mean vs R ──
    ax4 = axes[2]
    ax4.set_facecolor(BG)

    # Get no_correction G_mean per gamma
    nc_means = np.array([
        df_k.loc[(g, "no_correction"), ("G_mean", "mean")]
        for g in x_k
    ])

    for rem in remedies_order:
        if rem == "no_correction":
            continue
        color, marker, ls, lw, label = REMEDY_STYLES[rem]

        delta_means = []
        delta_stds  = []
        xs_valid    = []
        R_valid     = []

        for i, g in enumerate(x_k):
            try:
                m_rem = df_k.loc[(g, rem), ("G_mean", "mean")]
                s_rem = df_k.loc[(g, rem), ("G_mean", "std")]
                s_nc  = df_k.loc[(g, "no_correction"), ("G_mean", "std")]
                delta = m_rem - nc_means[i]
                # propagate std (approximate)
                delta_std = np.sqrt(s_rem**2 + s_nc**2)
                delta_means.append(delta)
                delta_stds.append(delta_std)
                xs_valid.append(g)
                R_valid.append(R_vals[i])
            except KeyError:
                continue

        if not R_valid:
            continue

        R_arr = np.array(R_valid)
        dm    = np.array(delta_means)
        ds    = np.array(delta_stds)

        ax4.plot(R_arr, dm,
                 color=color, marker=marker,
                 linestyle=ls, linewidth=lw,
                 markersize=6, label=label, zorder=3)
        ax4.fill_between(R_arr, dm - ds, dm + ds,
                         color=color, alpha=0.12, zorder=2)

    ax4.axvline(x=1.0, color='gray', linestyle=':',
                linewidth=1.5, alpha=0.8, label='$R=1$ (balance)')
    ax4.axhline(y=0.0, color='black', linewidth=0.8, alpha=0.4)
    ax4.set_xlabel('Information ratio $R$', fontsize=9)
    ax4.set_ylabel(r'$\Delta$ G-mean (vs no correction)', fontsize=9)
    ax4.set_title('H3: DGP-K\nBalance & Overbalance',
                  fontsize=9, fontweight='bold',
                  color='#1A1A2E', pad=5)
    ax4.grid(True, alpha=0.25, linestyle='--')
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.set_xlim(0, max(R_vals) * 1.05)

    # ── Global legend ──
    handles, labels = axes[0].get_legend_handles_labels()
    # Add R=1 line from panel H3
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='gray',
                          linestyle=':', linewidth=1.5,
                          label='$R=1$ (balance)'))
    labels.append('$R=1$ (balance)')

    fig.legend(handles, labels,
               loc='lower center', ncol=7,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    # fig.suptitle(
    #     'Hypothesis Validation: Remedy Effectiveness per Archetype\n'
    #     r'Shaded bands: mean $\pm$ std over $M=100$ replications',
    #     fontsize=11, fontweight='bold',
    #     color='#1A1A2E', y=1.02
    # )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "hypothesis_validation_figure.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), "results", "hypothesis_validation")
    plot_hypothesis_validation(save_dir=save_dir)