from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "data_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 160,
    }
)

COLORS = {
    "blue": "#2A6FBB",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "gray": "#6C757D",
    "purple": "#7A4EA3",
    "green": "#4C9A2A",
    "red": "#C44E52",
}


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


def wrap_labels(labels, width=18):
    return ["\n".join(textwrap.wrap(x, width=width, break_long_words=False)) for x in labels]


def main_results():
    methods = [
        "Original Hybrid RRF",
        "Enhanced Hybrid w122",
        "MedCPT Cross-Encoder",
        "KCH-MedRank",
    ]
    metrics = ["Recall@10", "MRR@10", "nDCG@10", "Recall@100"]
    values = np.array(
        [
            [0.4636, 0.7550, 0.5848, 0.6077],
            [0.4660, 0.7530, 0.5844, 0.7388],
            [0.5172, 0.7775, 0.6390, 0.7388],
            [0.5329, 0.7867, 0.6433, 0.7388],
        ]
    )
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    palette = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["teal"]]
    for i, method in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, values[i], width, label=method, color=palette[i])
    ax.set_ylabel("Score")
    ax.set_ylim(0.40, 0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.set_title("BioASQ held-out retrieval performance")
    savefig("main_results_bars.pdf")


def significance_forest():
    rows = [
        ("vs Enhanced Hybrid", "Recall@10", 0.0669, 0.0548, 0.0800, 0.0002),
        ("vs Enhanced Hybrid", "MRR@10", 0.0337, 0.0207, 0.0470, 0.0002),
        ("vs Enhanced Hybrid", "nDCG@10", 0.0589, 0.0495, 0.0687, 0.0002),
        ("vs MedCPT CE", "Recall@10", 0.0157, 0.0044, 0.0283, 0.0076),
        ("vs MedCPT CE", "MRR@10", 0.0093, -0.0036, 0.0254, 0.2060),
        ("vs MedCPT CE", "nDCG@10", 0.0043, -0.0042, 0.0154, 0.3840),
    ]
    labels = [f"{base}: {metric}" for base, metric, *_ in rows]
    y = np.arange(len(rows))[::-1]
    delta = np.array([r[2] for r in rows])
    lo = np.array([r[3] for r in rows])
    hi = np.array([r[4] for r in rows])
    pvals = [r[5] for r in rows]
    colors = [COLORS["teal"] if "Hybrid" in r[0] else COLORS["orange"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axvline(0, color="#333333", linewidth=0.8)
    for yi, d, l, h, c, p in zip(y, delta, lo, hi, colors, pvals):
        ax.errorbar(
            d,
            yi,
            xerr=[[d - l], [h - d]],
            fmt="o",
            color=c,
            ecolor=c,
            capsize=3,
            markersize=5,
        )
        ax.text(h + 0.004, yi, f"p={p:.4f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("KCH-MedRank delta")
    ax.set_xlim(-0.012, 0.095)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_title("Paired bootstrap gains with 95% confidence intervals")
    savefig("significance_forest.pdf")


def hard_subset():
    methods = [
        "Enhanced Hybrid RRF",
        "Retrieval-only LTR",
        "Flat knowledge LTR",
        "Pairwise graph LTR",
        "KCH-MedRank",
    ]
    recall = [0.0000, 0.4411, 0.4550, 0.4508, 0.4704]
    mrr = [0.0000, 0.2180, 0.2855, 0.2725, 0.2836]
    ndcg = [0.0000, 0.2424, 0.2907, 0.2766, 0.2945]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.bar(x - width, recall, width, label="Recall@10", color=COLORS["teal"])
    ax.bar(x, mrr, width, label="MRR@10", color=COLORS["blue"])
    ax.bar(x + width, ndcg, width, label="nDCG@10", color=COLORS["orange"])
    ax.set_ylabel("Score on hard subset")
    ax.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(methods, 16))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Recovering missed top-10 evidence when gold is in top-100")
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
    color_map = {
        "Retrieval": COLORS["blue"],
        "Semantic": COLORS["orange"],
        "Entity": COLORS["green"],
        "Hypergraph": COLORS["purple"],
        "Structure": COLORS["gray"],
        "MeSH": COLORS["teal"],
    }
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [color_map[d[2]] for d in data]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(np.arange(len(data)), values, color=colors)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("LightGBM importance")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_title("Top KCH-MedRank feature importances")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=k, markerfacecolor=v, markersize=8)
        for k, v in color_map.items()
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower right")
    savefig("feature_importance_barh.pdf")


def hyperparameter_heatmap():
    damping = [0.7, 0.8, 0.85, 0.9, 0.95]
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
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    im = ax.imshow(values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(iterations)))
    ax.set_xticklabels(iterations)
    ax.set_yticks(np.arange(len(damping)))
    ax.set_yticklabels(damping)
    ax.set_xlabel("Diffusion iterations")
    ax.set_ylabel("Damping")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] < 0.78 else "black"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Hypergraph diffusion score stability")
    savefig("hyperparameter_heatmap.pdf")


def structural_ablation():
    rows = [
        ("Flat knowledge over retrieval", 0.0101, 0.0080, 0.0153),
        ("Hypergraph no-med vs flat", -0.0044, -0.0089, -0.0074),
        ("KCH over no-med hypergraph", 0.0063, 0.0085, 0.0071),
        ("KCH over flat knowledge", 0.0020, -0.0004, -0.0003),
        ("KCH over pairwise graph", 0.0023, 0.0001, 0.0024),
    ]
    labels = [r[0] for r in rows]
    rec = [r[1] for r in rows]
    mrr = [r[2] for r in rows]
    ndcg = [r[3] for r in rows]
    y = np.arange(len(rows))
    height = 0.23
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.barh(y - height, rec, height, label="Delta Recall@10", color=COLORS["teal"])
    ax.barh(y, mrr, height, label="Delta MRR@10", color=COLORS["blue"])
    ax.barh(y + height, ndcg, height, label="Delta nDCG@10", color=COLORS["orange"])
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(labels, 25))
    ax.set_xlabel("Delta")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.set_title("Structural ablation deltas on BioASQ")
    savefig("structural_ablation_deltas.pdf")


def stratified_gain():
    subsets = [
        "Overall",
        "MeSH Overlap = 0",
        "MeSH Overlap >= 1",
        "Entity Overlap = 0",
        "Entity Overlap >= 1",
        "1 gold passage",
        ">=2 gold passages",
    ]
    rec5 = [0.0054, 0.0098, 0.0026, 0.0046, 0.0072, 0.0098, 0.0042]
    rec10 = [0.0023, 0.0032, 0.0017, 0.0043, -0.0020, 0.0098, 0.0002]
    y = np.arange(len(subsets))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.barh(y - 0.18, rec5, 0.36, label="Delta Recall@5", color=COLORS["purple"])
    ax.barh(y + 0.18, rec10, 0.36, label="Delta Recall@10", color=COLORS["teal"])
    ax.set_yticks(y)
    ax.set_yticklabels(subsets)
    ax.set_xlabel("KCH-MedRank minus Pairwise Graph LTR")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Where hypergraph structure helps most")
    savefig("stratified_hypergraph_gain.pdf")


def interpretability():
    mech = ["MeSH hierarchy", "Shared entity cluster", "PrimeKG relation"]
    pct = [75.9, 23.9, 0.2]
    gain = [22.1, 14.0, 25.0]
    x = np.arange(len(mech))
    fig, ax1 = plt.subplots(figsize=(6.6, 3.6))
    bars = ax1.bar(x, pct, color=[COLORS["teal"], COLORS["green"], COLORS["gray"]], width=0.55)
    ax1.set_ylabel("Rescued passages (%)")
    ax1.set_ylim(0, 85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(wrap_labels(mech, 14))
    ax1.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2 = ax1.twinx()
    ax2.plot(x, gain, color=COLORS["orange"], marker="o", linewidth=1.8, label="Avg rank gain")
    ax2.set_ylabel("Average rank gain")
    ax2.set_ylim(0, 30)
    for b, p in zip(bars, pct):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{p:.1f}%", ha="center", fontsize=8)
    ax1.set_title("Dominant mechanisms in rescued evidence")
    savefig("interpretability_mechanisms_chart.pdf")


def failure_summary():
    labels = ["Questions with rescued evidence", "Questions with lost evidence"]
    values = [414, 214]
    passages = ["Rescued passages", "Lost passages", "Net rescued passages"]
    pvals = [661, 307, 354]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(labels, values, color=[COLORS["teal"], COLORS["red"]], width=0.55)
    axes[0].set_ylabel("Questions")
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(wrap_labels(labels, 15))
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_title("Top-10 question outcomes")
    axes[1].bar(passages, pvals, color=[COLORS["teal"], COLORS["red"], COLORS["blue"]], width=0.55)
    axes[1].set_ylabel("Gold passages")
    axes[1].set_xticks(np.arange(len(passages)))
    axes[1].set_xticklabels(wrap_labels(passages, 14))
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[1].set_title("Passage-level rescue balance")
    savefig("failure_summary_bars.pdf")


def efficiency_tradeoff():
    rows = [
        ("MedCPT Cross-Encoder", 699.46, 0.5172, 0.6390, True),
        ("KCH-MedRank", 37.59, 0.5329, 0.6433, False),
        ("KCH no semantic", 37.50, 0.5244, 0.6330, False),
        ("Retrieval-only LambdaMART", 0.84, 0.5208, 0.6282, False),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for name, sec, recall, ndcg, ce in rows:
        color = COLORS["orange"] if ce else COLORS["teal"]
        marker = "s" if ce else "o"
        ax.scatter(sec, recall, s=80, color=color, marker=marker)
        ax.text(sec * 1.08, recall + 0.001, name, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Reranking time in seconds (log scale)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.512, 0.537)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.set_title("Reranking-stage efficiency versus evidence coverage")
    savefig("efficiency_tradeoff.pdf")


def external_diagnostics():
    methods_p = ["BM25", "Dense", "Hybrid RRF", "Cross-encoder", "HGB"]
    pub_recall = [0.7284, 0.8644, 0.8200, 0.8268, 0.8953]
    pub_ndcg = [0.7490, 0.8760, 0.8369, 0.8381, 0.9021]
    methods_n = ["BM25", "Dense", "Hybrid RRF", "LTR"]
    nfc_recall = [0.1379, 0.1674, 0.1676, 0.1770]
    nfc_ndcg = [0.2933, 0.3496, 0.3436, 0.3627]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, methods, recall, ndcg, title in [
        (axes[0], methods_p, pub_recall, pub_ndcg, "PubMedQA diagnostic"),
        (axes[1], methods_n, nfc_recall, nfc_ndcg, "NFCorpus diagnostic"),
    ]:
        x = np.arange(len(methods))
        ax.bar(x - 0.18, recall, 0.36, label="Recall@10", color=COLORS["teal"])
        ax.bar(x + 0.18, ndcg, 0.36, label="nDCG@10", color=COLORS["orange"])
        ax.set_xticks(x)
        ax.set_xticklabels(wrap_labels(methods, 11))
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_title(title)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, loc="upper left")
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
