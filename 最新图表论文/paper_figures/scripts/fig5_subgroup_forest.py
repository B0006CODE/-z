from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import (
    ABLATION_ORANGE,
    BASELINE_GRAY,
    KCH_BLUE,
    LIGHT_GRAY,
    despine,
    panel_label,
    save_figure,
    setup_matplotlib_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed" / "fig5_subgroup_forest.csv"


SUBSETS = [
    "Overall",
    "MeSH Overlap = 0",
    "MeSH Overlap >= 1",
    "Entity Overlap = 0",
    "Entity Overlap >= 1",
    "1 gold passage",
    ">= 2 gold passages",
]


def plot_metric(ax: plt.Axes, df: pd.DataFrame, metric: str, label: str) -> None:
    sub = df[df["metric"] == metric].set_index("subset").loc[SUBSETS].reset_index()
    y = np.arange(len(sub))
    ax.axvline(0, color=BASELINE_GRAY, lw=0.8, zorder=1)
    for idx, row in sub.iterrows():
        color = ABLATION_ORANGE if row["subset"] == "MeSH Overlap = 0" else KCH_BLUE
        ax.errorbar(
            row["delta"],
            idx,
            xerr=[[row["delta"] - row["ci_lower"]], [row["ci_upper"] - row["delta"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.25,
            capsize=3,
            markersize=4.5,
            zorder=3,
        )
        p = row["p-value"]
        text = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(row["ci_upper"] + 0.0012, idx, text, va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(SUBSETS)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(f"Delta {metric}")
    ax.set_title(metric)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.55)
    despine(ax)
    panel_label(ax, label)


def main() -> None:
    setup_matplotlib_style()
    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), constrained_layout=True)
    plot_metric(axes[0], df, "Recall@5", "A")
    plot_metric(axes[1], df, "Recall@10", "B")
    axes[1].set_yticklabels([])
    axes[0].set_xlim(-0.003, 0.029)
    axes[1].set_xlim(-0.009, 0.029)
    fig.suptitle("Hypergraph benefit is concentrated in indirect-evidence subgroups", y=1.04, fontsize=10)
    save_figure(fig, "fig5_subgroup_forest")
    plt.close(fig)


if __name__ == "__main__":
    main()
