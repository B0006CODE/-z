from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "data_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)

INK = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BLUE = "#3366A8"
LIGHT_BLUE = "#8CB7DD"
TEAL = "#168C7E"
ORANGE = "#D55E00"
GOLD = "#C99A00"
RED = "#B23A48"
PURPLE = "#6F4CA4"
GREEN = "#2F7D32"


def savefig(name: str):
    plt.tight_layout(pad=0.55)
    plt.savefig(OUT / name, bbox_inches="tight", transparent=False)
    plt.close()


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(colors=INK, length=3)


def title_left(ax, title, subtitle=None):
    ax.text(
        0.0,
        1.17 if subtitle else 1.08,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=10.2,
        fontweight="bold",
    )
    if subtitle:
        ax.text(
            0.0,
            1.07,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=8,
        )


def wrap_labels(labels, width=18):
    return ["\n".join(textwrap.wrap(x, width=width, break_long_words=False)) for x in labels]


def main_results():
    methods = ["Original Hybrid", "Enhanced Hybrid", "MedCPT CE", "KCH-MedRank"]
    metrics = ["Recall@10", "MRR@10", "nDCG@10", "Recall@100"]
    values = {
        "Recall@10": [0.4636, 0.4660, 0.5172, 0.5329],
        "MRR@10": [0.7550, 0.7530, 0.7775, 0.7867],
        "nDCG@10": [0.5848, 0.5844, 0.6390, 0.6433],
        "Recall@100": [0.6077, 0.7388, 0.7388, 0.7388],
    }
    colors = [MUTED, LIGHT_BLUE, ORANGE, TEAL]
    y_pos = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.45))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        xvals = np.array(values[metric])
        ax.hlines(y_pos, xvals.min() - 0.002, xvals, color="#D1D5DB", linewidth=2.2, zorder=1)
        for i, (xv, yp) in enumerate(zip(xvals, y_pos)):
            ax.scatter(xv, yp, s=58 if i == 3 else 42, color=colors[i], edgecolor="white", linewidth=0.9, zorder=3)
            ax.text(
                xv + 0.003,
                yp,
                f"{xv:.3f}",
                va="center",
                ha="left",
                fontsize=7.8,
                color=TEAL if i == 3 else INK,
                fontweight="bold" if i == 3 else "normal",
            )
        if metric != "Recall@100":
            ax.annotate(
                f"KCH gain over enhanced Hybrid: +{xvals[-1] - xvals[1]:.3f}",
                xy=(xvals[-1], y_pos[-1]),
                xytext=(xvals.min() + 0.003, -0.78),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.8),
                color=TEAL,
                fontsize=7.8,
                ha="left",
                va="center",
            )
        ax.set_title(metric, loc="left", fontweight="bold", color=INK)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        pad = 0.018 if metric != "Recall@100" else 0.04
        ax.set_xlim(xvals.min() - pad, xvals.max() + pad)
        ax.set_ylim(-1.05, len(methods) - 0.35)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        despine(ax)
    fig.suptitle("BioASQ held-out evidence retrieval: KCH-MedRank improves top-10 ranking", x=0.03, ha="left", y=1.02, fontweight="bold", fontsize=11)
    savefig("main_results_bars.pdf")


def significance_forest():
    rows = [
        ("Enhanced Hybrid", "Recall@10", 0.0669, 0.0548, 0.0800, 0.0002, True),
        ("Enhanced Hybrid", "MRR@10", 0.0337, 0.0207, 0.0470, 0.0002, True),
        ("Enhanced Hybrid", "nDCG@10", 0.0589, 0.0495, 0.0687, 0.0002, True),
        ("MedCPT CE", "Recall@10", 0.0157, 0.0044, 0.0283, 0.0076, True),
        ("MedCPT CE", "MRR@10", 0.0093, -0.0036, 0.0254, 0.2060, False),
        ("MedCPT CE", "nDCG@10", 0.0043, -0.0042, 0.0154, 0.3840, False),
    ]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.axvline(0, color="#374151", linewidth=0.9, zorder=1)
    ax.axvspan(-0.012, 0.0, color="#F3F4F6", zorder=0)
    ax.axhspan(2.5, 5.5, color="#F8FAFC", zorder=0)
    ax.axhspan(-0.5, 2.5, color="#FFFFFF", zorder=0)
    for yi, (base, metric, d, lo, hi, p, sig) in zip(y, rows):
        color = TEAL if base == "Enhanced Hybrid" else ORANGE
        marker = "o" if sig else "s"
        ax.hlines(yi, lo, hi, color=color, linewidth=2.0)
        ax.scatter(d, yi, s=46, marker=marker, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(hi + 0.0045, yi, f"{d:+.3f}  p={p:.4f}", va="center", ha="left", fontsize=8, color=INK)
    labels = [f"{base}\n{metric}" for base, metric, *_ in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("KCH-MedRank improvement over baseline")
    ax.set_xlim(-0.012, 0.098)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    title_left(ax, "Paired bootstrap effect sizes", "Whiskers are 95% confidence intervals; square markers are non-significant.")
    despine(ax)
    savefig("significance_forest.pdf")


def hard_subset():
    methods = [
        "Enhanced Hybrid RRF",
        "Retrieval-only LambdaMART",
        "Flat knowledge LTR",
        "Pairwise graph LTR",
        "KCH-MedRank",
    ]
    recall = np.array([0.0000, 0.4411, 0.4550, 0.4508, 0.4704])
    mrr = np.array([0.0000, 0.2180, 0.2855, 0.2725, 0.2836])
    ndcg = np.array([0.0000, 0.2424, 0.2907, 0.2766, 0.2945])
    y = np.arange(len(methods))[::-1]
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.axvline(0, color="#374151", linewidth=0.8)
    ax.hlines(y, 0, recall, color="#CBD5E1", linewidth=5, zorder=1)
    colors = [MUTED, BLUE, LIGHT_BLUE, GOLD, TEAL]
    ax.scatter(recall, y, s=[42, 60, 60, 60, 78], color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    for yi, r, m, n in zip(y, recall, mrr, ndcg):
        ax.text(r + 0.015, yi, f"R={r:.3f} / MRR={m:.3f} / nDCG={n:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(-0.02, 0.56)
    ax.set_xlabel("Recall@10 on hard subset")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    title_left(ax, "Hard reranking subset: available gold evidence is recovered", "Subset: Hybrid top-100 contains gold evidence but Hybrid top-10 misses it.")
    despine(ax)
    savefig("hard_subset_recovery.pdf")


def feature_importance():
    data = [
        ("Biomedical semantic score", 455, "Semantic"),
        ("Hybrid score", 404, "Retrieval"),
        ("Dense score", 364, "Retrieval"),
        ("BM25 rank score", 351, "Retrieval"),
        ("Biomedical semantic rank score", 317, "Semantic"),
        ("Dense rank score", 220, "Retrieval"),
        ("Passage entity count", 219, "Entity"),
        ("BM25 score", 196, "Retrieval"),
        ("Hypergraph x inv-rank", 181, "Hypergraph"),
        ("Local num nodes", 157, "Structure"),
        ("Hypergraph degree centrality", 145, "Hypergraph"),
        ("Shared MeSH term cluster size", 143, "MeSH"),
        ("Shared MeSH parent cluster size", 131, "MeSH"),
        ("MeSH cluster x inv-semantic", 126, "MeSH"),
        ("Passage MeSH specificity", 118, "MeSH"),
    ][::-1]
    cmap = {
        "Retrieval": BLUE,
        "Semantic": ORANGE,
        "Entity": GREEN,
        "Hypergraph": PURPLE,
        "Structure": MUTED,
        "MeSH": TEAL,
    }
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    groups = [d[2] for d in data]
    fig, ax = plt.subplots(figsize=(7.1, 4.75))
    y = np.arange(len(data))
    ax.barh(y, values, color=[cmap[g] for g in groups], height=0.68)
    for yi, val in zip(y, values):
        ax.text(val + 7, yi, f"{val}", va="center", ha="left", fontsize=7.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 500)
    ax.set_xlabel("LightGBM split importance")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    legend_order = ["Retrieval", "Semantic", "MeSH", "Entity", "Hypergraph", "Structure"]
    handles = [Patch(facecolor=cmap[g], label=g) for g in legend_order]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower right")
    title_left(ax, "What drives the learned ranker?", "Retrieval and semantic signals dominate; biomedical structure remains visible.")
    despine(ax)
    savefig("feature_importance_barh.pdf")


def hyperparameter_heatmap():
    damping = [0.70, 0.80, 0.85, 0.90, 0.95]
    iterations = [3, 5, 7, 10]
    values = np.array(
        [
            [0.509, 0.992, 0.950, 0.836],
            [0.518, 0.999, 0.933, 0.801],
            [0.522, 1.000, 0.921, 0.784],
            [0.525, 0.999, 0.909, 0.767],
            [0.528, 0.997, 0.894, 0.751],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.75, 3.25))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.50, vmax=1.00, aspect="auto")
    ax.set_xticks(np.arange(len(iterations)))
    ax.set_xticklabels(iterations)
    ax.set_yticks(np.arange(len(damping)))
    ax.set_yticklabels([f"{d:.2f}" for d in damping])
    ax.set_xlabel("Diffusion iterations")
    ax.set_ylabel("Damping")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 0.90 else INK
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold" if values[i, j] > 0.98 else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.set_label("Pearson r")
    title_left(ax, "Diffusion scores are stable after five iterations", "Correlation against the default setting d=0.85, t=5.")
    savefig("hyperparameter_heatmap.pdf")


def structural_ablation():
    rows = [
        ("Flat knowledge over retrieval", 0.0101, 0.0080, 0.0153, True),
        ("Hypergraph no-med vs flat", -0.0044, -0.0089, -0.0074, True),
        ("KCH over no-med hypergraph", 0.0063, 0.0085, 0.0071, True),
        ("KCH over flat knowledge", 0.0020, -0.0004, -0.0003, False),
        ("KCH over pairwise graph", 0.0023, 0.0001, 0.0024, False),
    ]
    metrics = [("Recall@10", 1), ("MRR@10", 2), ("nDCG@10", 3)]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.75), sharey=True)
    for ax, (metric, idx) in zip(axes, metrics):
        vals = np.array([r[idx] for r in rows])
        ax.axvline(0, color="#374151", linewidth=0.8)
        for yi, val, row in zip(y, vals, rows):
            color = TEAL if val >= 0 else RED
            ax.hlines(yi, 0, val, color=color, linewidth=2.4)
            ax.scatter(val, yi, s=48 if row[4] else 34, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(val + (0.0007 if val >= 0 else -0.0007), yi, f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=7.3, color=INK)
        ax.set_title(metric, fontweight="bold", color=INK)
        ax.set_xlim(-0.011, 0.017)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        despine(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(wrap_labels([r[0] for r in rows], 23))
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Controlled structural ablation: knowledge helps, raw structure hurts", x=0.02, ha="left", y=1.05, fontweight="bold", fontsize=11)
    savefig("structural_ablation_deltas.pdf")


def stratified_gain():
    subsets = [
        "Overall",
        "No MeSH overlap",
        "MeSH overlap",
        "No entity overlap",
        "Entity overlap",
        "1 gold passage",
        ">=2 gold passages",
    ]
    rec5 = np.array([0.0054, 0.0098, 0.0026, 0.0046, 0.0072, 0.0098, 0.0042])
    rec10 = np.array([0.0023, 0.0032, 0.0017, 0.0043, -0.0020, 0.0098, 0.0002])
    sig5 = [True, True, False, False, True, False, True]
    y = np.arange(len(subsets))[::-1]
    fig, ax = plt.subplots(figsize=(7.05, 3.85))
    ax.axvline(0, color="#374151", linewidth=0.8)
    ax.axvspan(0, 0.0115, color="#ECFDF5", zorder=0)
    for yi, a, b, s in zip(y, rec5, rec10, sig5):
        ax.plot([a, b], [yi, yi], color="#CBD5E1", linewidth=2.0, zorder=1)
        ax.scatter(a, yi, s=58 if s else 40, color=PURPLE, edgecolor="white", linewidth=0.8, label="Recall@5" if yi == y[0] else None, zorder=3)
        ax.scatter(b, yi, s=42, color=TEAL, edgecolor="white", linewidth=0.8, label="Recall@10" if yi == y[0] else None, zorder=3)
        ax.text(max(a, b) + 0.0005, yi, f"{a:+.4f}", va="center", fontsize=7.5, color=PURPLE)
    ax.set_yticks(y)
    ax.set_yticklabels(subsets)
    ax.set_xlabel("KCH-MedRank minus Pairwise Graph LTR")
    ax.set_xlim(-0.003, 0.0125)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    title_left(ax, "Hypergraph benefit is concentrated at early ranks", "Largest gains appear when direct MeSH overlap is absent.")
    despine(ax)
    savefig("stratified_hypergraph_gain.pdf")


def interpretability():
    mechanisms = ["MeSH hierarchy", "Shared entity cluster", "PrimeKG relation"]
    pct = np.array([75.9, 23.9, 0.2])
    gain = np.array([22.1, 14.0, 25.0])
    colors = [TEAL, GREEN, MUTED]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    y = np.arange(len(mechanisms))[::-1]
    ax.barh(y, pct, color=colors, height=0.58)
    for yi, p in zip(y, pct):
        ax.text(p + 1.5, yi, f"{p:.1f}%", va="center", ha="left", fontsize=8.2, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(mechanisms, 14))
    ax.set_xlim(0, 84)
    ax.set_xlabel("Rescued passages (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    despine(ax)
    ax2 = axes[1]
    ax2.hlines(y, 0, gain, color="#D1D5DB", linewidth=2.2)
    ax2.scatter(gain, y, color=ORANGE, s=56, edgecolor="white", linewidth=0.9, zorder=3)
    for yi, g in zip(y, gain):
        ax2.text(g + 0.8, yi, f"+{g:.1f}", va="center", fontsize=8.2, color=ORANGE, fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(0, 30)
    ax2.set_xlabel("Average rank gain")
    ax2.grid(axis="x", color=GRID, linewidth=0.7)
    despine(ax2)
    fig.suptitle("Rescued evidence is mostly explained by MeSH hierarchy paths", x=0.02, ha="left", y=1.04, fontweight="bold", fontsize=11)
    savefig("interpretability_mechanisms_chart.pdf")


def failure_summary():
    fig, ax = plt.subplots(figsize=(6.7, 3.35))
    steps = [661, -307, 354]
    labels = ["Rescued\ngold passages", "Lost\ngold passages", "Net\nrescued"]
    starts = [0, 661, 0]
    colors = [TEAL, RED, BLUE]
    x = np.arange(3)
    ax.axhline(0, color="#374151", linewidth=0.8)
    for xi, start, val, label, color in zip(x, starts, steps, labels, colors):
        bottom = start if val >= 0 else start + val
        ax.bar(xi, abs(val), bottom=bottom, color=color, width=0.58)
        ax.text(xi, bottom + abs(val) + 22, f"{val:+d}" if xi > 0 else f"{val:d}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
    ax.plot([0.29, 0.71], [661, 661], color="#9CA3AF", linewidth=1.0)
    ax.plot([1.29, 1.71], [354, 354], color="#9CA3AF", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gold passages")
    ax.set_ylim(-20, 735)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    title_left(ax, "Top-10 rescue balance favors KCH-MedRank", "414 questions gain evidence, 214 questions lose evidence.")
    despine(ax)
    savefig("failure_summary_bars.pdf")


def efficiency_tradeoff():
    rows = [
        ("Retrieval-only\nLambdaMART", 0.84, 0.5208, 0.6282, BLUE, "o"),
        ("KCH no semantic", 37.50, 0.5244, 0.6330, GOLD, "o"),
        ("KCH-MedRank", 37.59, 0.5329, 0.6433, TEAL, "o"),
        ("MedCPT\nCross-Encoder", 699.46, 0.5172, 0.6390, ORANGE, "s"),
    ]
    fig, ax = plt.subplots(figsize=(6.9, 3.5))
    ax.axhspan(0.5329, 0.537, color="#ECFDF5", zorder=0)
    for name, sec, recall, _ndcg, color, marker in rows:
        ax.scatter(sec, recall, s=85, color=color, marker=marker, edgecolor="white", linewidth=0.9, zorder=3)
        offset = (1.08, 0.0007)
        if "Cross" in name:
            offset = (1.05, 0.0005)
        if "Retrieval" in name:
            offset = (1.08, -0.0001)
        ax.text(sec * offset[0], recall + offset[1], name, fontsize=8, va="center", color=INK)
    ax.annotate(
        "18.6x faster\nand higher Recall@10",
        xy=(37.59, 0.5329),
        xytext=(95, 0.5355),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=TEAL),
        color=TEAL,
        fontsize=8.2,
        ha="left",
        va="center",
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.55, 1100)
    ax.set_ylim(0.512, 0.537)
    ax.set_xlabel("Reranking time in seconds (log scale)")
    ax.set_ylabel("Recall@10")
    ax.grid(True, which="both", color=GRID, linewidth=0.7)
    title_left(ax, "Efficiency frontier: KCH-MedRank sits above the Cross-Encoder", "Same enhanced top-100 candidate pool; reranking-stage timing only.")
    despine(ax)
    savefig("efficiency_tradeoff.pdf")


def external_diagnostics():
    panels = [
        (
            "PubMedQA",
            ["BM25", "Dense", "Hybrid", "Cross-enc.", "HGB"],
            [0.7284, 0.8644, 0.8200, 0.8268, 0.8953],
            [0.7490, 0.8760, 0.8369, 0.8381, 0.9021],
        ),
        (
            "NFCorpus",
            ["BM25", "Dense", "Hybrid", "LTR"],
            [0.1379, 0.1674, 0.1676, 0.1770],
            [0.2933, 0.3496, 0.3436, 0.3627],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25))
    for ax, (title, methods, recall, ndcg) in zip(axes, panels):
        x = np.arange(len(methods))
        ax.plot(x, recall, color=TEAL, marker="o", linewidth=2.0, label="Recall@10")
        ax.plot(x, ndcg, color=ORANGE, marker="s", linewidth=2.0, label="nDCG@10")
        best = int(np.argmax(recall))
        ax.scatter(best, recall[best], s=90, facecolors="none", edgecolors=TEAL, linewidth=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title, fontweight="bold", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        despine(ax)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Secondary datasets support evidence-coverage robustness", x=0.02, ha="left", y=1.05, fontweight="bold", fontsize=11)
    savefig("external_diagnostics.pdf")


def run_all():
    main_results()
    significance_forest()
    hard_subset()
    feature_importance()
    hyperparameter_heatmap()
    structural_ablation()
    stratified_gain()
    interpretability()
    failure_summary()
    efficiency_tradeoff()
    external_diagnostics()


if __name__ == "__main__":
    run_all()
