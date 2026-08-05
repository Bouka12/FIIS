"""
plot_dgp_geometry.py
--------------------
3D scatter plots of training data geometry for DGP-S, DGP-B, DGP-G.
One subplot per scenario, grouped by DGP in rows.

All minority points are shown.
Majority class is capped at maj_cap for visual balance only.

Layout:
    Row 0: DGP-S  (3 scenarios: n1/n0 in {0.30, 0.10, 0.05})
    Row 1: DGP-B  (2 scenarios: delta in {2.0, 3.0})
    Row 2: DGP-G  (3 scenarios: alpha in {0.05, 0.10, 0.20})

Usage:
    python plot_dgp_geometry.py
Output:
    results/dgp_geometry_3d.png
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
    D, N_TRAIN, SEED_BASE, DELTA,
    get_class_counts, generate_gaussian,
    baseline_params
)
# from dgp_b import make_dgpb_params
from dgp_g import make_dgpg_params

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SEED    = SEED_BASE + 1
MAJ_CAP = 800    # cap majority points for visual balance only
d       = D      # 3 features

C0 = '#2E86AB'   # majority — steel blue
C1 = "#6242C2"   # minority — crimson
BG = 'white'   # background


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def split_classes(X, y, maj_cap=800, seed=SEED):
    """
    Sample both classes proportionally to preserve the imbalance ratio.
    maj_cap controls the majority sample size.
    Minority is sampled at maj_cap * (n1/n0) to maintain the ratio.
    """
    rng    = np.random.default_rng(seed)
    idx0   = np.where(y == 0)[0]
    idx1   = np.where(y == 1)[0]
    ratio  = len(idx1) / len(idx0)
    
    n0_show = min(maj_cap, len(idx0))
    n1_show = min(int(n0_show * ratio), len(idx1))
    
    idx0 = rng.choice(idx0, n0_show, replace=False)
    idx1 = rng.choice(idx1, n1_show, replace=False)
    return X[idx0], X[idx1]


def plot_scenario(ax, X0, X1, title, elev=20, azim=45):
    """Plot one 3D scatter scenario on a given axis."""
    ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2],
               c=C0, alpha=0.25, s=5, label='Majority')
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2],
               c=C1, alpha=0.45, s=5, label='Minority')

    ax.set_title(title, fontsize=8, fontweight='bold',
                 pad=3, color='#1A1A2E')
    ax.set_xlabel('$x_1$', fontsize=6, labelpad=0)
    ax.set_ylabel('$x_2$', fontsize=6, labelpad=0)
    ax.set_zlabel('$x_3$', fontsize=6, labelpad=0)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#DDDDDD')
    ax.yaxis.pane.set_edgecolor('#DDDDDD')
    ax.zaxis.pane.set_edgecolor('#DDDDDD')
    ax.grid(True, alpha=0.15)


# ─────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────

DGPS_SCENARIOS = [
    (r"$n_1/n_0=0.30$", 0.30),
    (r"$n_1/n_0=0.10$", 0.10),
    (r"$n_1/n_0=0.05$", 0.05),
]

# DGPB_SCENARIOS = [
#     (r"$\delta=2.0$", 2.0),
#     (r"$\delta=3.0$", 3.0),
# ]

DGPG_SCENARIOS = [
    (r"$\alpha=0.05$", 0.05),
    (r"$\alpha=0.10$", 0.10),
    (r"$\alpha=0.20$", 0.20),
]


# ─────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────

def plot_dgp_geometry(save_dir="results"):

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor(BG)

    # ── Row 0: DGP-S ──
    for col, (label, ratio) in enumerate(DGPS_SCENARIOS):
        n0s, n1s = get_class_counts(N_TRAIN, ratio)
        mu0, mu1, S0, S1 = baseline_params(d, DELTA)
        X, y = generate_gaussian(n0s, n1s, d, mu0, mu1, S0, S1, seed=SEED)
        X0, X1 = split_classes(X, y)

        ax = fig.add_subplot(2, 3, col + 1, projection='3d',)
        ax.set_facecolor(BG)
        plot_scenario(ax, X0, X1,
                      f"DGP-S  {label}",
                      elev=25, azim=135)

    # # ── Row 1: DGP-B (2 scenarios, 1 empty slot) ──
    # for col, (label, delta) in enumerate(DGPB_SCENARIOS):
    #     n0b, n1b = get_class_counts(N_TRAIN, 0.2)
    #     mu0, mu1, S0, S1 = make_dgpb_params(delta, d)
    #     X, y = generate_gaussian(n0b, n1b, d, mu0, mu1, S0, S1, seed=SEED)
    #     X0, X1 = split_classes(X, y)

    #     ax = fig.add_subplot(3, 3, 4 + col, projection='3d')
    #     ax.set_facecolor(BG)
    #     plot_scenario(ax, X0, X1,
    #                   f"DGP-B  {label}",
    #                   elev=20, azim=60)

    # # Hide unused slot (position 6)
    # ax_empty = fig.add_subplot(3, 3, 6)
    # ax_empty.set_visible(False)

    # ── Row 2: DGP-G ──
    for col, (label, alpha) in enumerate(DGPG_SCENARIOS):
        n0g, n1g = get_class_counts(N_TRAIN, 0.2)
        mu0, mu1, S0, S1 = make_dgpg_params(alpha, d)
        X, y = generate_gaussian(n0g, n1g, d, mu0, mu1, S0, S1, seed=SEED)
        X0, X1 = split_classes(X, y)

        ax = fig.add_subplot(2, 3, 4 + col, projection='3d')
        ax.set_facecolor(BG)
        plot_scenario(ax, X0, X1,
                      f"DGP-G  {label}",
                      elev=35, azim=225)

    # ── Row labels ──
    row_labels = [
        'DGP-S',#'DGP-S\nArchetype A\n(Sampling Deficit)',
        # 'DGP-B',#'DGP-B\nArchetype B\n(Boundary Proximity)',
        'DGP-G',#'DGP-G\nArchetype C\n(Feature Energy)',
    ]
    for row_idx, label in enumerate(row_labels):
        fig.text(0.01, 0.83 - row_idx * 0.31,
                 label, fontsize=8, fontweight='bold',
                 color='#1A1A2E', va='center', ha='left',
                 rotation=90, linespacing=1.4)

    # ── Legend ──
    legend_els = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=C0, markersize=9,
               label=f'Majority (y=0, up to {MAJ_CAP} shown)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=C1, markersize=9,
               label='Minority (y=1, all shown)'),
    ]
    fig.legend(handles=legend_els, loc='lower center',
               ncol=2, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    # # ── Title ──
    # fig.suptitle(
    #     '31/07 Training Data Geometry per DGP and Scenario  ($d=3$)',
    #     fontsize=12, fontweight='bold', color='#1A1A2E', y=0.99
    # )

    plt.tight_layout(rect=[0.05, 0.05, 1.0, 0.97])

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "dgp_geometry_3d.pdf")
    plt.savefig(out, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    plot_dgp_geometry(save_dir=save_dir)