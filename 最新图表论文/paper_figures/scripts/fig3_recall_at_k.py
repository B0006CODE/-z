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
    despine,
    save_figure,
    setup_matplotlib_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed" / "fig3_recall_at_k.csv"


COLORS = {
    "BM25": "#9CA3AF",
    "Dense": "#4B5563",
    "Enhanced Hybrid RRF": ABLATION_ORANGE,
    "MedCPT Cross-Encoder": GREEN,
    "KCH-MedRank": KCH_BLUE,
}

LABEL_Y_OFFSETS = {
    "KCH-MedRank": 0.030,
    "MedCPT Cross-Encoder": 0.008,
    "Enhanced Hybrid RRF": -0.014,
    "BM25": 0.006,
    "Dense": -0.006,
}


def main() -> None:
    setup_matplotlib_style()
    df = pd.read_csv(DATA)
    methods = ["BM25", "Dense", "Enhanced Hybrid RRF", "MedCPT Cross-Encoder", "KCH-MedRank"]

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    for method in methods:
        sub = df[df["method"] == method].sort_values("k")
        if sub.empty:
            continue
        is_kch = method == "KCH-MedRank"
        ax.plot(
            sub["k"],
            sub["Recall@k"],
            marker="o",
            lw=2.4 if is_kch else 1.35,
            ms=5 if is_kch else 4,
            color=COLORS.get(method, BASELINE_GRAY),
            alpha=1.0 if is_kch else 0.82,
        )
        last = sub.iloc[-1]
        ax.text(
            last["k"] * 1.05,
            last["Recall@k"] + LABEL_Y_OFFSETS.get(method, 0.0),
            method,
            color=COLORS.get(method, BASELINE_GRAY),
            va="center",
            fontsize=7.5,
        )

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "3", "5", "10", "20", "50", "100"])
    ax.set_xlim(0.85, 185)
    ax.set_ylim(0.0, 0.80)
    ax.set_xlabel("Rank cutoff k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall improves across early and candidate-pool ranks")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55, alpha=0.55)
    despine(ax)
    save_figure(fig, "fig3_recall_at_k")
    plt.close(fig)


if __name__ == "__main__":
    main()
