from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LATEST_ROOT = PROJECT_ROOT.parent
REPO_ROOT = LATEST_ROOT.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TABLE_DIR = REPO_ROOT / "paper_overleaf_en" / "tables"
METRIC_DIR = REPO_ROOT / "results" / "metrics"


TABLE_SOURCE_FILES = [
    "enhanced_bioasq_main_results.tex",
    "enhanced_bioasq_bootstrap_vs_hybrid.tex",
    "enhanced_bioasq_significance.tex",
    "kch_ablation_summary.tex",
    "structural_ablation.tex",
    "hypergraph_vs_pairwise_stratified.tex",
    "reranking_efficiency.tex",
    "interpretability_mechanisms.tex",
    "normalization_ablation.tex",
    "feature_enhancement_separability.tex",
]

METRIC_SOURCE_FILES = [
    "kch_medrank_enhanced_bioasq_v2_metrics.json",
    "kch_medrank_enhanced_bioasq_v2_full_kch_medrank_metrics.json",
    "medcpt_cross_encoder_enhanced_bioasq_test_top100_metrics.json",
    "enhanced_hybrid_w122_test_top100_metrics.json",
]


def strip_latex(cell: str) -> str:
    s = cell.strip()
    s = re.sub(r"\\\\\s*$", "", s)
    replacements = {
        r"\textbf{": "",
        r"\textit{": "",
        r"\hspace{6pt}": "",
        r"\%": "%",
        r"\_": "_",
        r"$\geq$": ">=",
        r"\geq": ">=",
        r"$-$": "-",
        r"$\Delta$": "Delta",
        r"\times": "x",
        r"\ ": " ",
        r"$": "",
        "{": "",
        "}": "",
        "``": '"',
        "''": '"',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = s.replace("--", "-")
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def split_row(line: str) -> list[str] | None:
    if "&" not in line:
        return None
    line = line.strip()
    if line.startswith(r"\multicolumn"):
        return None
    line = re.sub(r"\\\\\s*$", "", line)
    return [strip_latex(part) for part in line.split("&")]


def parse_float(cell: str) -> float | None:
    s = strip_latex(cell).replace(",", "")
    s = s.replace("*", "").replace("%", "").replace("x", "")
    if s in {"", "-", "--"}:
        return None
    match = re.search(r"[+-]?\d+(?:\.\d+)?", s)
    if not match:
        return None
    return float(match.group(0))


def parse_ci(cell: str) -> tuple[float | None, float | None]:
    s = strip_latex(cell)
    vals = re.findall(r"[+-]?\d+(?:\.\d+)?", s)
    if len(vals) < 2:
        return None, None
    return float(vals[0]), float(vals[1])


def read_table(name: str) -> str:
    path = TABLE_DIR / name
    return path.read_text(encoding="utf-8")


def read_json(name: str) -> dict:
    path = METRIC_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def latex_rows(tex: str) -> list[list[str]]:
    rows = []
    for line in tex.splitlines():
        row = split_row(line)
        if row and row[0] not in {"toprule", "midrule", "bottomrule"}:
            rows.append(row)
    return rows


def write_csv(name: str, rows: list[dict]) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ("source_table", "notes"):
        if col not in df.columns:
            df[col] = ""
    df.to_csv(PROCESSED_DIR / name, index=False)
    return df


def snapshot_sources() -> None:
    raw_tables = RAW_DIR / "source_tables"
    raw_metrics = RAW_DIR / "source_metrics"
    raw_tables.mkdir(parents=True, exist_ok=True)
    raw_metrics.mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename in TABLE_SOURCE_FILES:
        source = TABLE_DIR / filename
        if source.exists():
            shutil.copy2(source, raw_tables / filename)
            manifest.append(
                {
                    "kind": "latex_table",
                    "filename": filename,
                    "source_path": str(source.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "snapshot_path": str((raw_tables / filename).relative_to(PROJECT_ROOT)).replace("\\", "/"),
                }
            )
    for filename in METRIC_SOURCE_FILES:
        source = METRIC_DIR / filename
        if source.exists():
            shutil.copy2(source, raw_metrics / filename)
            manifest.append(
                {
                    "kind": "metric_json",
                    "filename": filename,
                    "source_path": str(source.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "snapshot_path": str((raw_metrics / filename).relative_to(PROJECT_ROOT)).replace("\\", "/"),
                }
            )
    pd.DataFrame(manifest).to_csv(RAW_DIR / "source_manifest.csv", index=False)


def extract_fig2() -> pd.DataFrame:
    rows: list[dict] = []
    main = latex_rows(read_table("enhanced_bioasq_main_results.tex"))
    header = main[0]
    metric_cols = {name: idx for idx, name in enumerate(header)}
    for row in main[1:]:
        method = row[0]
        for metric in ("Recall@10", "MRR@10", "nDCG@10"):
            rows.append(
                {
                    "panel": "A",
                    "comparison": "Main BioASQ result",
                    "method": method,
                    "metric": metric,
                    "value": parse_float(row[metric_cols[metric]]),
                    "baseline_value": None,
                    "kch_value": None,
                    "delta": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "p-value": None,
                    "source_table": "tab:enhanced-bioasq-main-results",
                    "notes": "",
                }
            )

    hybrid = latex_rows(read_table("enhanced_bioasq_bootstrap_vs_hybrid.tex"))
    for row in hybrid[1:]:
        metric = row[0]
        if metric not in {"Recall@10", "MRR@10", "nDCG@10"}:
            continue
        baseline = parse_float(row[1])
        kch = parse_float(row[2])
        lo, hi = parse_ci(row[3])
        rows.append(
            {
                "panel": "B",
                "comparison": "KCH-MedRank - Enhanced Hybrid RRF",
                "method": "Enhanced KCH-MedRank",
                "metric": metric,
                "value": None,
                "baseline_value": baseline,
                "kch_value": kch,
                "delta": round(kch - baseline, 4) if baseline is not None and kch is not None else None,
                "ci_lower": lo,
                "ci_upper": hi,
                "p-value": parse_float(row[4]),
                "source_table": "tab:enhanced-bioasq-bootstrap-hybrid",
                "notes": "95% CI reported in manuscript table.",
            }
        )

    ce = latex_rows(read_table("enhanced_bioasq_significance.tex"))
    for row in ce[1:]:
        metric = row[0]
        if metric not in {"Recall@10", "MRR@10", "nDCG@10"}:
            continue
        rows.append(
            {
                "panel": "C",
                "comparison": "KCH-MedRank - MedCPT Cross-Encoder",
                "method": "Enhanced KCH-MedRank",
                "metric": metric,
                "value": None,
                "baseline_value": parse_float(row[1]),
                "kch_value": parse_float(row[2]),
                "delta": parse_float(row[3]),
                "ci_lower": None,
                "ci_upper": None,
                "p-value": parse_float(row[4]),
                "source_table": "tab:enhanced-bioasq-significance",
                "notes": "95% CI unavailable in manuscript table; point estimate and p-value only.",
            }
        )
    return write_csv("fig2_main_results.csv", rows)


def extract_fig3() -> pd.DataFrame:
    report = read_json("kch_medrank_enhanced_bioasq_v2_metrics.json")
    kch = read_json("kch_medrank_enhanced_bioasq_v2_full_kch_medrank_metrics.json")
    ce = read_json("medcpt_cross_encoder_enhanced_bioasq_test_top100_metrics.json")
    enhanced_hybrid = read_json("enhanced_hybrid_w122_test_top100_metrics.json")
    method_specs = [
        ("BM25", report["baseline_metrics"]["BM25"], "results/metrics/kch_medrank_enhanced_bioasq_v2_metrics.json"),
        ("Dense", report["baseline_metrics"]["Dense"], "results/metrics/kch_medrank_enhanced_bioasq_v2_metrics.json"),
        ("Enhanced Hybrid RRF", enhanced_hybrid, "results/metrics/enhanced_hybrid_w122_test_top100_metrics.json"),
        ("MedCPT Cross-Encoder", ce, "results/metrics/medcpt_cross_encoder_enhanced_bioasq_test_top100_metrics.json"),
        ("KCH-MedRank", kch, "results/metrics/kch_medrank_enhanced_bioasq_v2_full_kch_medrank_metrics.json"),
    ]
    rows = []
    for method, metrics, source in method_specs:
        for k in metrics.get("ks", []):
            rows.append(
                {
                    "method": method,
                    "k": int(k),
                    "metric": f"Recall@{k}",
                    "Recall@k": metrics.get(f"recall@{k}"),
                    "source_table": source,
                    "notes": "Values are read from held-out BioASQ metric logs; k=10 and k=100 match manuscript main-result rows where reported.",
                }
            )
    return write_csv("fig3_recall_at_k.csv", rows)


def extract_fig4() -> pd.DataFrame:
    rows = []
    wanted = {
        "Retrieval-feature-only LambdaMART": "Retrieval-feature-only LambdaMART",
        "Flat knowledge LTR, no graph": "Flat knowledge LTR",
        "Pairwise graph LTR": "Pairwise graph LTR",
        "Hypergraph LTR without medical knowledge": "Hypergraph without medical knowledge",
        "Full KCH-MedRank": "Full KCH-MedRank",
    }
    order = {
        "Retrieval-feature-only LambdaMART": 1,
        "Flat knowledge LTR": 2,
        "Hypergraph without medical knowledge": 3,
        "Pairwise graph LTR": 4,
        "Full KCH-MedRank": 5,
    }
    table = latex_rows(read_table("kch_ablation_summary.tex"))
    header = table[0]
    idx = {name: i for i, name in enumerate(header)}
    for row in table[1:]:
        source_method = row[0]
        if source_method not in wanted:
            continue
        method = wanted[source_method]
        for metric in ("Recall@10", "MRR@10", "nDCG@10"):
            rows.append(
                {
                    "method": method,
                    "source_method": source_method,
                    "method_order": order[method],
                    "metric": metric,
                    "value": parse_float(row[idx[metric]]),
                    "source_table": "tab:kch-ablation-summary",
                    "notes": "Distinct method values used for structural-ablation visualization.",
                }
            )
    return write_csv("fig4_structural_ablation.csv", rows)


def extract_fig5() -> pd.DataFrame:
    rows = []
    current_metric = None
    tex = read_table("hypergraph_vs_pairwise_stratified.tex")
    for line in tex.splitlines():
        if r"\textbf{Recall@5}" in line:
            current_metric = "Recall@5"
            continue
        if r"\textbf{Recall@10}" in line:
            current_metric = "Recall@10"
            continue
        row = split_row(line)
        if not row or current_metric is None or row[0] == "Subset":
            continue
        if len(row) != 7:
            continue
        lo, hi = parse_ci(row[5])
        rows.append(
            {
                "metric": current_metric,
                "subset": row[0],
                "n": int(parse_float(row[1])),
                "KCH": parse_float(row[2]),
                "Pairwise graph LTR": parse_float(row[3]),
                "delta": parse_float(row[4]),
                "ci_lower": lo,
                "ci_upper": hi,
                "p-value": parse_float(row[6]),
                "source_table": "tab:hypergraph-vs-pairwise-stratified",
                "notes": "Paired bootstrap KCH-MedRank minus Pairwise Graph LTR.",
            }
        )
    return write_csv("fig5_subgroup_forest.csv", rows)


def extract_fig6() -> pd.DataFrame:
    rows = []
    table = latex_rows(read_table("reranking_efficiency.tex"))
    header = table[0]
    idx = {name: i for i, name in enumerate(header)}
    for row in table[1:]:
        if len(row) < len(header):
            continue
        rows.append(
            {
                "method": row[idx["Method"]],
                "CE score": row[idx["CE score"]],
                "Online CE": row[idx["Online CE"]],
                "Seconds": parse_float(row[idx["Seconds"]]),
                "ms/query": parse_float(row[idx["ms/query"]]),
                "Cand./s": parse_float(row[idx["Cand./s"]]),
                "Recall@10": parse_float(row[idx["Recall@10"]]),
                "MRR@10": parse_float(row[idx["MRR@10"]]),
                "nDCG@10": parse_float(row[idx["nDCG@10"]]),
                "source_table": "tab:reranking-efficiency",
                "notes": row[idx["Notes"]],
            }
        )
    return write_csv("fig6_efficiency_pareto.csv", rows)


def extract_fig7() -> pd.DataFrame:
    rows = []
    mode = None
    tex = read_table("interpretability_mechanisms.tex")
    for line in tex.splitlines():
        if "Mechanism & Count" in line:
            mode = "dominant"
            continue
        if "Mechanism Activation Statistics" in line:
            mode = None
            continue
        if "Active Mechanisms & Count" in line:
            mode = "cooccurrence"
            continue
        row = split_row(line)
        if not row or mode is None:
            continue
        if mode == "dominant" and len(row) == 4:
            rows.append(
                {
                    "panel": "dominant",
                    "mechanism": row[0],
                    "active_mechanisms": None,
                    "count": parse_float(row[1]),
                    "percent": parse_float(row[2]),
                    "Avg Rank Gain": parse_float(row[3]),
                    "MeSH Hierarchy": None,
                    "Entity Cluster": None,
                    "Relation": None,
                    "source_table": "tab:interpretability-mechanisms",
                    "notes": "Dominant mechanism classification; Total row is retained but not plotted as a mechanism.",
                }
            )
        if mode == "cooccurrence" and len(row) == 3:
            active = row[0]
            rows.append(
                {
                    "panel": "cooccurrence",
                    "mechanism": None,
                    "active_mechanisms": active,
                    "count": parse_float(row[1]),
                    "percent": parse_float(row[2]),
                    "Avg Rank Gain": None,
                    "MeSH Hierarchy": int("MeSH Hierarchy" in active),
                    "Entity Cluster": int("Entity Cluster" in active),
                    "Relation": int("Relation" in active),
                    "source_table": "tab:mech-cooccurrence",
                    "notes": "Mechanism co-occurrence table; rendered as an UpSet-style intersection plot.",
                }
            )
    return write_csv("fig7_interpretability.csv", rows)


def extract_fig8() -> pd.DataFrame:
    rows = []
    norm = latex_rows(read_table("normalization_ablation.tex"))
    header = norm[0]
    idx = {name: i for i, name in enumerate(header)}
    for row in norm[1:]:
        config = row[0]
        if config.startswith("Delta"):
            continue
        for metric in ("MRR@10", "Recall@10", "nDCG@10"):
            rows.append(
                {
                    "panel": "performance",
                    "configuration": config,
                    "metric": metric,
                    "value": parse_float(row[idx[metric]]),
                    "feature": None,
                    "Gold mean": None,
                    "Non-gold mean": None,
                    "Gold pos. rate": None,
                    "Ratio": None,
                    "source_table": "tab:normalization-ablation",
                    "notes": "Validation split; synonym-aware features are added to flat-knowledge KCH-MedRank.",
                }
            )

    sep = latex_rows(read_table("feature_enhancement_separability.tex"))
    header = sep[0]
    idx = {name: i for i, name in enumerate(header)}
    for row in sep[1:]:
        rows.append(
            {
                "panel": "separability",
                "configuration": None,
                "metric": None,
                "value": None,
                "feature": row[idx["Feature"]],
                "Gold mean": parse_float(row[idx["Gold mean"]]),
                "Non-gold mean": parse_float(row[idx["Non-gold mean"]]),
                "Gold pos. rate": parse_float(row[idx["Gold pos. rate"]]),
                "Ratio": parse_float(row[idx["Ratio"]]),
                "source_table": "tab:feature-separability",
                "notes": "Gold mean divided by non-gold mean.",
            }
        )
    return write_csv("fig8_synonym_normalization.csv", rows)


def validate_outputs(dfs: dict[str, pd.DataFrame]) -> None:
    report = []
    fig2 = dfs["fig2"]
    main = fig2[fig2["panel"] == "A"]
    forest = fig2[fig2["panel"].isin(["B", "C"])]
    for metric in ("Recall@10", "MRR@10", "nDCG@10"):
        kch_main = float(
            main[(main["method"] == "Enhanced KCH-MedRank") & (main["metric"] == metric)]["value"].iloc[0]
        )
        for _, row in forest[forest["metric"] == metric].iterrows():
            if pd.notna(row["kch_value"]) and abs(kch_main - float(row["kch_value"])) > 1e-6:
                raise ValueError(f"Mismatch for {metric}: main={kch_main}, comparison={row['kch_value']}")
    report.append("fig2: KCH values match main and comparison tables.")

    fig4 = dfs["fig4"]
    for metric in ("Recall@10", "MRR@10", "nDCG@10"):
        fig2_val = float(
            main[(main["method"] == "Enhanced KCH-MedRank") & (main["metric"] == metric)]["value"].iloc[0]
        )
        fig4_val = float(
            fig4[(fig4["method"] == "Full KCH-MedRank") & (fig4["metric"] == metric)]["value"].iloc[0]
        )
        if abs(fig2_val - fig4_val) > 1e-6:
            raise ValueError(f"Mismatch for {metric}: fig2={fig2_val}, fig4={fig4_val}")
    report.append("fig4: Full KCH-MedRank values match main results.")

    fig7 = dfs["fig7"]
    dominant_total = fig7[(fig7["panel"] == "dominant") & (fig7["mechanism"] == "Total")]["count"].sum()
    co_total = fig7[fig7["panel"] == "cooccurrence"]["count"].sum()
    if dominant_total and abs(dominant_total - co_total) > 1e-6:
        raise ValueError(f"Interpretability totals mismatch: dominant={dominant_total}, cooccurrence={co_total}")
    report.append("fig7: dominant and co-occurrence totals match.")

    (PROCESSED_DIR / "validation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    snapshot_sources()
    dfs = {
        "fig2": extract_fig2(),
        "fig3": extract_fig3(),
        "fig4": extract_fig4(),
        "fig5": extract_fig5(),
        "fig6": extract_fig6(),
        "fig7": extract_fig7(),
        "fig8": extract_fig8(),
    }
    validate_outputs(dfs)
    print("Extracted processed CSV files to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
