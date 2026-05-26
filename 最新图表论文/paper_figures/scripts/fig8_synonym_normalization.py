from __future__ import annotations

from pathlib import Path

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
DATA = PROJECT_ROOT / "data" / "processed" / "fig8_synonym_normalization.csv"


CONFIGS = [
    "Hybrid RRF source (no rerank)",
    "Base KCH-MedRank (flat knowledge, no normalization)",
    "+ Synonym-aware features (16 dimensions)",
]
CONFIG_LABELS = {
    "Hybrid RRF source (no rerank)": "Hybrid RRF",
    "Base KCH-MedRank (flat knowledge, no normalization)": "Base KCH",
    "+ Synonym-aware features (16 dimensions)": "+ Synonym features",
}
CONFIG_COLORS = {
    "Hybrid RRF source (no rerank)": BASELINE_GRAY,
    "Base KCH-MedRank (flat knowledge, no normalization)": ABLATION_ORANGE,
    "+ Synonym-aware features (16 dimensions)": KCH_BLUE,
}


FEATURE_LABELS = {
    "norm_entity_overlap_count": "Normalized entity overlap",
    "norm_entity_jaccard": "Normalized entity Jaccard",
    "question_norm_entity_coverage": "Question normalized coverage",
    "mesh_normalized_overlap_count": "Normalized MeSH overlap",
    "mesh_major_overlap_count": "Major MeSH overlap",
    "mesh_weighted_overlap": "Weighted MeSH overlap",
    "q2p_mesh_alias_count": "Question-passage MeSH alias",
    "abbreviation_match_count": "Abbreviation match",
}


def plot_performance(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[df["panel"] == "performance"].copy()
    metrics = ["MRR@10", "Recall@10", "nDCG@10"]
    for config in CONFIGS:
        vals = sub[sub["configuration"] == config].set_index("metric").loc[metrics]["value"]
        ax.plot(
            metrics,
            vals,
            marker="o",
            color=CONFIG_COLORS[config],
            lw=1.8 if "Synonym" in config else 1.25,
            label=CONFIG_LABELS[config],
        )
    ax.set_ylabel("Metric value")
    ax.set_title("Synonym features add small consistent gains")
    ax.set_ylim(0.47, 0.80)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.55)
    ax.legend(frameon=False, loc="lower right")
    despine(ax)
    panel_label(ax, "A")


def plot_separability(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[df["panel"] == "separability"].copy()
    sub["label"] = sub["feature"].map(FEATURE_LABELS).fillna(sub["feature"])
    sub = sub.sort_values("Ratio")
    ax.hlines(sub["label"], 1.0, sub["Ratio"], color=LIGHT_GRAY, lw=1.4)
    ax.scatter(sub["Ratio"], sub["label"], color=KCH_BLUE, s=34, zorder=3)
    for _, row in sub.iterrows():
        ax.text(row["Ratio"] + 0.08, row["label"], f"{row['Ratio']:.2f}x", va="center", fontsize=7.5)
    ax.axvline(1.0, color=BASELINE_GRAY, lw=0.8)
    ax.set_xlabel("Gold / non-gold mean ratio")
    ax.set_title("Feature separability")
    ax.set_xlim(0.8, max(sub["Ratio"]) + 0.75)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.55)
    despine(ax)
    panel_label(ax, "B")


def main() -> None:
    setup_matplotlib_style()
    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.75), constrained_layout=True)
    plot_performance(axes[0], df)
    plot_separability(axes[1], df)
    save_figure(fig, "fig8_synonym_normalization")
    plt.close(fig)


if __name__ == "__main__":
    main()

