from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
DATA = PROJECT_ROOT / "data" / "processed" / "fig4_structural_ablation.csv"


def color_for(method: str) -> str:
    if method == "Full KCH-MedRank":
        return KCH_BLUE
    if method == "Flat knowledge LTR":
        return ABLATION_ORANGE
    return BASELINE_GRAY


def main() -> None:
    setup_matplotlib_style()
    df = pd.read_csv(DATA)
    methods = (
        df[["method", "method_order"]]
        .drop_duplicates()
        .sort_values("method_order")["method"]
        .tolist()
    )
    metrics = ["Recall@10", "MRR@10", "nDCG@10"]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.55), sharey=True, constrained_layout=True)
    y_positions = list(range(len(methods)))
    for ax, metric, label in zip(axes, metrics, ["A", "B", "C"]):
        sub = df[df["metric"] == metric].set_index("method").loc[methods]
        values = sub["value"].values
        ax.plot(values, y_positions, color=LIGHT_GRAY, lw=1.2, zorder=1)
        for y, method, value in zip(y_positions, methods, values):
            ax.scatter(value, y, color=color_for(method), s=42 if method == "Full KCH-MedRank" else 30, zorder=3)
            ax.text(value + 0.0014, y, f"{value:.4f}", va="center", fontsize=7)
        ax.set_title(metric)
        ax.set_xlabel("Metric value")
        ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.55)
        despine(ax)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        panel_label(ax, label)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(methods)
    axes[0].invert_yaxis()
    fig.suptitle("Structural ablation: knowledge features carry most aggregate gain", y=1.04, fontsize=10)
    save_figure(fig, "fig4_structural_ablation")
    plt.close(fig)


if __name__ == "__main__":
    main()
