from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_memberships

from plot_style import (
    ABLATION_ORANGE,
    BASELINE_GRAY,
    GREEN,
    KCH_BLUE,
    LIGHT_GRAY,
    RED,
    despine,
    panel_label,
    save_figure,
    setup_matplotlib_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed" / "fig7_interpretability.csv"


MECH_COLORS = {
    "MeSH Hierarchy Path": KCH_BLUE,
    "Shared Entity Cluster Path": ABLATION_ORANGE,
    "PrimeKG Relation Path": GREEN,
}


def plot_dominant(ax: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[(df["panel"] == "dominant") & (df["mechanism"] != "Total")].copy()
    sub = sub.sort_values("count")
    ax.barh(
        sub["mechanism"],
        sub["count"],
        color=[MECH_COLORS.get(m, BASELINE_GRAY) for m in sub["mechanism"]],
        height=0.55,
    )
    for _, row in sub.iterrows():
        ax.text(row["count"] + 8, row["mechanism"], f"{int(row['count'])} ({row['percent']:.1f}%)", va="center", fontsize=7.5)
    ax.set_xlabel("Rescued gold passages")
    ax.set_title("Dominant rescue mechanism")
    ax.set_xlim(0, max(sub["count"]) * 1.22)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.55)
    despine(ax)
    panel_label(ax, "A")


def plot_upset_style(ax_bar: plt.Axes, ax_matrix: plt.Axes, df: pd.DataFrame) -> None:
    sub = df[df["panel"] == "cooccurrence"].copy()
    memberships = []
    for _, row in sub.iterrows():
        active = []
        if row["MeSH Hierarchy"] == 1:
            active.append("MeSH")
        if row["Entity Cluster"] == 1:
            active.append("Entity")
        if row["Relation"] == 1:
            active.append("Relation")
        memberships.append(active)
    # Keep the UpSet data transformation in the reproducible pipeline.
    _ = from_memberships(memberships, data=sub["count"].tolist())

    sub = sub.sort_values("count", ascending=False).reset_index(drop=True)
    x = range(len(sub))
    ax_bar.bar(x, sub["count"], color=BASELINE_GRAY, width=0.6)
    for idx, row in sub.iterrows():
        ax_bar.text(idx, row["count"] + 8, f"{int(row['count'])}", ha="center", va="bottom", fontsize=7.2)
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Mechanism co-occurrence")
    ax_bar.set_xticks([])
    ax_bar.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.55)
    despine(ax_bar)

    mech_rows = ["MeSH", "Entity", "Relation"]
    ax_matrix.set_xlim(-0.5, len(sub) - 0.5)
    ax_matrix.set_ylim(-0.5, len(mech_rows) - 0.5)
    for idx, row in sub.iterrows():
        active_y = []
        states = [
            row["MeSH Hierarchy"] == 1,
            row["Entity Cluster"] == 1,
            row["Relation"] == 1,
        ]
        for y, is_active in enumerate(states):
            color = KCH_BLUE if is_active else LIGHT_GRAY
            size = 35 if is_active else 18
            ax_matrix.scatter(idx, y, s=size, color=color, zorder=3)
            if is_active:
                active_y.append(y)
        if len(active_y) > 1:
            ax_matrix.plot([idx, idx], [min(active_y), max(active_y)], color=KCH_BLUE, lw=1.1, zorder=2)
    ax_matrix.set_yticks(range(len(mech_rows)))
    ax_matrix.set_yticklabels(mech_rows)
    ax_matrix.set_xticks(x)
    ax_matrix.set_xticklabels([f"I{i+1}" for i in x], fontsize=7)
    ax_matrix.set_xlabel("Intersection")
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)
    ax_matrix.spines["left"].set_visible(False)
    ax_matrix.spines["bottom"].set_visible(False)
    ax_matrix.tick_params(axis="both", length=0)
    panel_label(ax_bar, "B")


def main() -> None:
    setup_matplotlib_style()
    df = pd.read_csv(DATA)
    fig = plt.figure(figsize=(8.4, 4.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1.0, 0.58])
    ax_dom = fig.add_subplot(gs[:, 0])
    ax_up_bar = fig.add_subplot(gs[0, 1])
    ax_up_matrix = fig.add_subplot(gs[1, 1])

    plot_dominant(ax_dom, df)
    plot_upset_style(ax_up_bar, ax_up_matrix, df)
    save_figure(fig, "fig7_interpretability")
    plt.close(fig)


if __name__ == "__main__":
    main()

