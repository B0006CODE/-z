from __future__ import annotations

from pathlib import Path

import numpy as np
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
    format_metric_axis,
    panel_label,
    save_figure,
    setup_matplotlib_style,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed" / "fig2_main_results.csv"


def method_color(method: str) -> str:
    if "KCH" in method:
        return KCH_BLUE
    if "Cross-Encoder" in method:
        return GREEN
    if "Enhanced Hybrid" in method:
        return ABLATION_ORANGE
    return BASELINE_GRAY


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    metrics = ["Recall@10", "MRR@10", "nDCG@10"]
    methods = [
        "Original Hybrid RRF",
        "Enhanced Hybrid w122",
        "MedCPT Cross-Encoder",
        "Enhanced KCH-MedRank",
    ]
    offsets = np.linspace(-0.18, 0.18, len(methods))
    for i, metric in enumerate(metrics):
        vals = df[df["metric"] == metric]
        x_min = vals["value"].min()
        x_max = vals["value"].max()
        ax.hlines(i, x_min, x_max, color=LIGHT_GRAY, lw=1.2, zorder=1)
        for off, method in zip(offsets, methods):
            row = vals[vals["method"] == method]
            if row.empty:
                continue
            value = row["value"].iloc[0]
            ax.scatter(
                value,
                i + off,
                s=42 if "KCH" in method else 30,
                color=method_color(method),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
                label=method if i == 0 else None,
            )
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Metric value")
    ax.set_title("Held-out top-10 retrieval")
    ax.set_xlim(0.42, 0.82)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.5)
    despine(ax)
    format_metric_axis(ax)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderpad=0.2)
    panel_label(ax, "A")


def plot_forest(ax: plt.Axes, df: pd.DataFrame, title: str, show_ci: bool) -> None:
    metrics = ["Recall@10", "MRR@10", "nDCG@10"]
    sub = df.set_index("metric").loc[metrics].reset_index()
    y = np.arange(len(sub))
    ax.axvline(0, color=BASELINE_GRAY, lw=0.8, zorder=1)
    if show_ci:
        left = sub["delta"] - sub["ci_lower"]
        right = sub["ci_upper"] - sub["delta"]
        ax.errorbar(
            sub["delta"],
            y,
            xerr=[left, right],
            fmt="o",
            color=KCH_BLUE,
            ecolor=KCH_BLUE,
            elinewidth=1.3,
            capsize=3,
            markersize=5,
            zorder=3,
        )
    else:
        ax.scatter(sub["delta"], y, color=KCH_BLUE, s=36, zorder=3)
    for _, row in sub.iterrows():
        p = row["p-value"]
        suffix = f"p={p:.4f}" if pd.notna(p) else "p n/a"
        ax.text(row["delta"] + 0.002, row.name, suffix, va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlabel("Delta metric")
    ax.set_title(title)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.5)
    despine(ax)


def main() -> None:
    setup_matplotlib_style()
    df = pd.read_csv(DATA)
    fig = plt.figure(figsize=(10.2, 3.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, df[df["panel"] == "A"])

    ax_b = fig.add_subplot(gs[0, 1])
    plot_forest(
        ax_b,
        df[df["panel"] == "B"],
        "KCH vs. Enhanced Hybrid",
        show_ci=True,
    )
    ax_b.set_xlim(-0.01, 0.09)
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[0, 2])
    plot_forest(
        ax_c,
        df[df["panel"] == "C"],
        "KCH vs. MedCPT CE",
        show_ci=False,
    )
    ax_c.set_xlim(-0.01, 0.04)
    ax_c.text(
        0.98,
        0.04,
        "CI not reported",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=BASELINE_GRAY,
    )
    panel_label(ax_c, "C")

    save_figure(fig, "fig2_main_results")
    plt.close(fig)


if __name__ == "__main__":
    main()

