"""
plot_dgpk_geometry.py
---------------------
3D scatter plots of training data geometry for DGP-K
across all compensation scenarios (gamma values).

Shows how the minority class geometry changes as gamma increases,
driving R from deficit through balance to overbalance while
R_n = 0.1 remains fixed throughout.

Layout: 2 rows x 3 cols (6 gamma scenarios)

Usage:
    python plot_dgpk_geometry.py
Output:
    results/dgpk_geometry_3d.png
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dgp_utils import (
    D, N_TRAIN, SEED_BASE,
    get_class_counts, generate_gaussian
)
from dgp_k import make_dgpk_params, GAMMAS, RATIO

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SEED    = SEED_BASE + 1
MAJ_CAP = 600     # majority cap for visual balance
d       = D       # 3 features

C0 = '#2E86AB'    # majority — steel blue
C1 = "#6242C2"     # minority — crimson
BG = 'white'    # background

# Pilot R values per gamma for annotation
R_VALUES = {
    0.5: 0.077,
    1.0: 0.283,
    2.0: 0.779,
    3.0: 1.064,
    5.0: 1.315,
    8.0: 1.417,
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def split_classes(X, y, maj_cap=MAJ_CAP, seed=SEED):
    """
    Sample both classes proportionally to preserve imbalance ratio.
    maj_cap controls majority sample size.
    Minority sampled at maj_cap * (n1/n0) to maintain ratio.
    """
    rng   = np.random.default_rng(seed)
    idx0  = np.where(y == 0)[0]
    idx1  = np.where(y == 1)[0]
    ratio = len(idx1) / len(idx0)

    n0_show = min(maj_cap, len(idx0))
    n1_show = min(int(n0_show * ratio), len(idx1))

    idx0 = rng.choice(idx0, n0_show, replace=False)
    idx1 = rng.choice(idx1, n1_show, replace=False)
    return X[idx0], X[idx1]


# ─────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────

def plot_dgpk_geometry(save_dir="results"):

    n0, n1 = get_class_counts(N_TRAIN, RATIO)

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor(BG)

    for idx, gamma in enumerate(GAMMAS):
        mu0, mu1, S0, S1 = make_dgpk_params(gamma, d=d)
        X, y = generate_gaussian(n0, n1, d, mu0, mu1, S0, S1, seed=SEED)
        X0, X1 = split_classes(X, y)

        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax.set_facecolor(BG)

        ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2],
                   c=C0, alpha=0.20, s=5, label='Majority')
        ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2],
                   c=C1, alpha=0.45, s=14, label='Minority')

        # R value annotation
        R_val = R_VALUES.get(gamma, "")
        regime = ("deficit" if R_val < 0.9
                  else "balance" if R_val < 1.1
                  else "overbalance")
        ax.set_title(
            f"$\\gamma={gamma}$   $R={R_val}$  ({regime})",
            fontsize=8, fontweight='bold',
            color='#1A1A2E', pad=3
        )

        ax.set_xlabel('$x_1$', fontsize=6, labelpad=0)
        ax.set_ylabel('$x_2$', fontsize=6, labelpad=0)
        ax.set_zlabel('$x_3$', fontsize=6, labelpad=0)
        ax.tick_params(labelsize=5)
        ax.view_init(elev=25, azim=135)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#DDDDDD')
        ax.yaxis.pane.set_edgecolor('#DDDDDD')
        ax.zaxis.pane.set_edgecolor('#DDDDDD')
        ax.grid(True, alpha=0.15)

    # ── Legend ──
    legend_els = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=C0, markersize=9,
               label=f'Majority (y=0, up to {MAJ_CAP} shown)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=C1, markersize=9,
               label='Minority (y=1, proportional to ratio)'),
    ]
    fig.legend(handles=legend_els, loc='lower center',
               ncol=2, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    # # ── Title ──
    # fig.suptitle(
    #     'DGP-K: Training Data Geometry Across Compensation Scenarios\n'
    #     r'($n_1/n_0=0.1$ fixed, $\mu_1=\gamma\cdot e_1$, '
    #     r'$\Sigma_1=\gamma\cdot I_d$)',
    #     fontsize=11, fontweight='bold',
    #     color='#1A1A2E', y=0.99
    # )

    plt.tight_layout(rect=[0.02, 0.05, 1.0, 0.95])

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "dgpk_geometry_3d.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), "results", "synthetic_validation", "dgpk_plots")
    plot_dgpk_geometry(save_dir=save_dir)