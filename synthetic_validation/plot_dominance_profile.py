"""
plot_dominance_profile.py
-------------------------
Grouped bar chart of FIIS dominance weights (phi_n, phi_p, phi_g, phi_C)
per scenario for DGP-S, DGP-B, DGP-G.

Each subplot shows one DGP with its subscenarios on the x-axis.
The target component bar is highlighted with a dark border.
Bar widths are equal and consistent across all subplots.

Usage:
    python plot_dominance_profile.py
Output:
    results/fiis_dominance_profile.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BG        = '#F8F9FA'
BAR_WIDTH = 0.18       # fixed bar width — same across all subplots
OFFSETS   = np.array([-1.5, -0.5, 0.5, 1.5]) * BAR_WIDTH

COLORS = {
    "phi_n": "#E84855",   # red    — sampling
    "phi_p": "#2E86AB",   # blue   — boundary proximity
    "phi_g": "#F4A261",   # orange — feature energy
    "phi_C": "#6B4226",   # brown  — coupling
}
LABELS = {
    "phi_n": r"$\phi_n$ (Sampling)",
    "phi_p": r"$\phi_p$ (Boundary Proximity)",
    "phi_g": r"$\phi_g$ (Feature Energy)",
    "phi_C": r"$\phi_C$ (Coupling)",
}
TARGET_COLORS = {
    "n": "#E84855",
    "p": "#2E86AB",
    "g": "#F4A261",
}
PHI_KEYS = ["phi_n", "phi_p", "phi_g", "phi_C"]

DGP_GROUPS = [
    ("DGP-S", "Archetype A\n(Sampling Deficit)",      "n"),
    ("DGP-B", "Archetype B\n(Boundary Proximity)",    "p"),
    ("DGP-G", "Archetype C\n(Feature Energy)",        "g"),
]


# ─────────────────────────────────────────────
# Load and prepare data
# ─────────────────────────────────────────────

def load_dominance_data(results_dir="results"):
    """Load phi values per scenario from raw result CSVs."""
    records = []

    # DGP-S
    df_s  = pd.read_csv(os.path.join(results_dir, "dgp_s_raw.csv"))
    base  = df_s[df_s["remedy"] == "no_correction"]
    for ratio, grp in base.groupby("ratio"):
        records.append({
            "dgp":      "DGP-S",
            "scenario": f"$n_1/n_0={ratio}$",
            "phi_n":    grp["dom_phi_n"].mean(),
            "phi_p":    grp["dom_phi_p"].mean(),
            "phi_g":    grp["dom_phi_g"].mean(),
            "phi_C":    grp["dom_phi_C"].mean(),
            "target":   "n",
        })

    # DGP-B
    df_b  = pd.read_csv(os.path.join(results_dir, "dgp_b_raw.csv"))
    base  = df_b[df_b["remedy"] == "no_correction"]
    for delta, grp in base.groupby("delta"):
        records.append({
            "dgp":      "DGP-B",
            "scenario": f"$\\delta={delta}$",
            "phi_n":    grp["dom_phi_n"].mean(),
            "phi_p":    grp["dom_phi_p"].mean(),
            "phi_g":    grp["dom_phi_g"].mean(),
            "phi_C":    grp["dom_phi_C"].mean(),
            "target":   "p",
        })

    # DGP-G
    df_g  = pd.read_csv(os.path.join(results_dir, "dgp_g_raw.csv"))
    base  = df_g[df_g["remedy"] == "no_correction"]
    for alpha, grp in base.groupby("alpha"):
        records.append({
            "dgp":      "DGP-G",
            "scenario": f"$\\alpha={alpha}$",
            "phi_n":    grp["dom_phi_n"].mean(),
            "phi_p":    grp["dom_phi_p"].mean(),
            "phi_g":    grp["dom_phi_g"].mean(),
            "phi_C":    grp["dom_phi_C"].mean(),
            "target":   "g",
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Plotting function
# ─────────────────────────────────────────────

def plot_dominance_profile(df, save_dir="results"):

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor(BG)

    for ax, (dgp, title, target) in zip(axes, DGP_GROUPS):
        ax.set_facecolor(BG)
        sub = df[df["dgp"] == dgp].reset_index(drop=True)
        n_scenarios = len(sub)

        # x positions — fixed spacing regardless of n_scenarios
        x = np.arange(n_scenarios)

        for j, key in enumerate(PHI_KEYS):
            vals = sub[key].values
            bars = ax.bar(
                x + OFFSETS[j], vals,
                width=BAR_WIDTH,
                color=COLORS[key],
                alpha=0.85,
                edgecolor='white',
                linewidth=0.5,
                label=LABELS[key] if ax == axes[0] else "_nolegend_",
            )
            # Highlight target component with dark border
            if key == f"phi_{target}":
                for bar in bars:
                    bar.set_edgecolor('#1A1A2E')
                    bar.set_linewidth(2.0)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["scenario"].values, fontsize=8)
        ax.set_xlim(-0.5, n_scenarios - 0.5)
        ax.set_ylim(0, 0.85)
        ax.set_title(f"{dgp}\n{title}",
                     fontsize=9, fontweight='bold',
                     color='#1A1A2E', pad=6)

        if ax == axes[0]:
            ax.set_ylabel(r"Dominance weight $\phi_j$", fontsize=9)

        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

        # Target annotation
        ax.text(
            0.98, 0.97,
            f"Target: $\\phi_{{{target}}}$",
            transform=ax.transAxes,
            fontsize=8, ha='right', va='top',
            color=TARGET_COLORS[target],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', alpha=0.7,
                      edgecolor=TARGET_COLORS[target])
        )

    # ── Global legend ──
    handles = [
        mpatches.Patch(color=COLORS[k], alpha=0.85, label=LABELS[k])
        for k in PHI_KEYS
    ]
    fig.legend(
        handles=handles,
        loc='lower center', ncol=4,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04)
    )

    fig.suptitle(
        r'FIIS Dominance Profile per DGP and Scenario'
        '\n'
        r'(target component highlighted with dark border)',
        fontsize=11, fontweight='bold',
        color='#1A1A2E', y=1.02
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "fiis_dominance_profile.png")
    plt.savefig(out, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    df          = load_dominance_data(results_dir)
    plot_dominance_profile(df, save_dir=results_dir)