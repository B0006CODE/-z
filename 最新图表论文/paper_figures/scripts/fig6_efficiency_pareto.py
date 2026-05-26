from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from plot_style import (
    ABLATION_ORANGE,
    BASELINE_GRAY,
    GREEN,
    KCH_BLUE,
    LIGHT_GRAY,
    RED,
    despine,
    save_figure,
    setup_matplotlib_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed" / "fig6_efficiency_pareto.csv"


COLORS = {
    "MedCPT Cross-Encoder": RED,
    "KCH-MedRank": KCH_BLUE,
    "KCH-MedRank without semantic feature": GREEN,
    "Retrieval-feature-only LambdaMART": BASELINE_GRAY,
}


def main() -> None:
    setup_matplotlib_style()
    df = pd.read_csv(DATA)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for _, row in df.iterrows():
        method = row["method"]
        ax.scatter(
            row["ms/query"],
            row["Recall@10"],
            s=85 if method == "KCH-MedRank" else 58,
            color=COLORS.get(method, BASELINE_GRAY),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        offset_y = 0.006 if method != "Retrieval-feature-only LambdaMART" else -0.012
        ax.annotate(
            method.replace(" without semantic feature", "\nwithout semantic"),
            (row["ms/query"], row["Recall@10"]),
            xytext=(5, 8 if offset_y > 0 else -12),
            textcoords="offset points",
            fontsize=7.3,
            color=COLORS.get(method, BASELINE_GRAY),
        )

    ce = df[df["method"] == "MedCPT Cross-Encoder"].iloc[0]
    kch = df[df["method"] == "KCH-MedRank"].iloc[0]
    speedup = ce["ms/query"] / kch["ms/query"]
    ax.annotate(
        f"{speedup:.1f}x lower reranking latency",
        xy=(kch["ms/query"], kch["Recall@10"]),
        xytext=(155, 0.5368),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": BASELINE_GRAY},
        ha="center",
        va="bottom",
        fontsize=8,
        color=BASELINE_GRAY,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Reranking latency (ms/query, log scale)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Efficiency-performance tradeoff on BioASQ")
    ax.set_ylim(0.505, 0.538)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55, alpha=0.55)
    despine(ax)
    save_figure(fig, "fig6_efficiency_pareto")
    plt.close(fig)


if __name__ == "__main__":
    main()
