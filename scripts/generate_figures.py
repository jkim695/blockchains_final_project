"""Generate PDF figures for the paper from benchmark results.

Usage:
    python scripts/generate_figures.py

Outputs PDF files to paper/figures/.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# -- Benchmark data (from pytest-benchmark run) --------------------------------

TREE_SIZES = [100, 1_000, 10_000]
TREE_SIZE_LABELS = ["100", "1K", "10K"]

# Proof generation latency (microseconds, mean)
MPT_PROOF_GEN = [6.23, 7.15, 7.21]
RBT_PROOF_GEN = [77.20, 729.66, 9487.03]

# Insert throughput (microseconds, all N keys)
MPT_INSERT = [932.46, 11723.08, 141132.20]
RBT_INSERT = [853.17, 14415.97, 190832.29]

# Delta cache: patch vs full regeneration (microseconds)
DELTA_PATCH = [7.52, 9.92, 10.38]
# Compare patch cost to RBT full proof regeneration (the real benefit)
RBT_FULL_REGEN = [77.20, 729.66, 9487.03]

# Memory (KB)
MPT_MEMORY = [101.6, 1671.9, 21668.5]
RBT_MEMORY = [34.4, 322.0, 5055.8]

# -- Style settings -----------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
ORANGE = "#ea580c"


def fig_proof_generation():
    """Figure 2: Proof generation latency (log-log scale)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(TREE_SIZES, MPT_PROOF_GEN, "o-", color=BLUE, linewidth=2,
            markersize=7, label="Merkle-Prefix Tree", zorder=5)
    ax.plot(TREE_SIZES, RBT_PROOF_GEN, "s-", color=RED, linewidth=2,
            markersize=7, label="Red-Black Tree", zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Orders (N)")
    ax.set_ylabel("Proof Generation Latency (\u00b5s)")
    ax.set_xticks(TREE_SIZES)
    ax.set_xticklabels(TREE_SIZE_LABELS)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(50, 20_000)

    # Add speedup annotations
    for i, n in enumerate(TREE_SIZES):
        speedup = RBT_PROOF_GEN[i] / MPT_PROOF_GEN[i]
        ax.annotate(f"{speedup:.0f}x", xy=(n, MPT_PROOF_GEN[i]),
                    xytext=(0, -18), textcoords="offset points",
                    ha="center", fontsize=9, color=BLUE, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "proof_generation.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> proof_generation.pdf")


def fig_delta_cache():
    """Figure 3: Delta cache patch vs RBT full regeneration."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(TREE_SIZES))
    width = 0.35

    bars1 = ax.bar(x - width / 2, DELTA_PATCH, width, label="Delta Cache Patch",
                   color=GREEN, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, RBT_FULL_REGEN, width,
                   label="RBT Full Regeneration", color=RED, edgecolor="white",
                   linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Orders (N)")
    ax.set_ylabel("Latency (\u00b5s)")
    ax.set_xticks(x)
    ax.set_xticklabels(TREE_SIZE_LABELS)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add speedup labels
    for i in range(len(TREE_SIZES)):
        speedup = RBT_FULL_REGEN[i] / DELTA_PATCH[i]
        ax.text(x[i], max(RBT_FULL_REGEN[i], DELTA_PATCH[i]) * 1.5,
                f"{speedup:.0f}x", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "delta_cache_speedup.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> delta_cache_speedup.pdf")


def fig_memory():
    """Figure 4: Memory comparison."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(TREE_SIZES))
    width = 0.35

    ax.bar(x - width / 2, [m / 1024 for m in MPT_MEMORY], width,
           label="Merkle-Prefix Tree", color=BLUE, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, [m / 1024 for m in RBT_MEMORY], width,
           label="Red-Black Tree", color=RED, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Number of Orders (N)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(TREE_SIZE_LABELS)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "memory_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> memory_comparison.pdf")


def fig_insert_throughput():
    """Figure 5: Insert throughput comparison."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(TREE_SIZES, [t / 1000 for t in MPT_INSERT], "o-", color=BLUE,
            linewidth=2, markersize=7, label="Merkle-Prefix Tree")
    ax.plot(TREE_SIZES, [t / 1000 for t in RBT_INSERT], "s-", color=RED,
            linewidth=2, markersize=7, label="Red-Black Tree")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Orders (N)")
    ax.set_ylabel("Total Insert Time (ms)")
    ax.set_xticks(TREE_SIZES)
    ax.set_xticklabels(TREE_SIZE_LABELS)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "insert_throughput.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> insert_throughput.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig_proof_generation()
    fig_delta_cache()
    fig_memory()
    fig_insert_throughput()
    print("Done.")
