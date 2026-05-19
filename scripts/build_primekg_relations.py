from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
import urllib.error
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entities import normalize_text
from src.knowledge.relations import entity_name_index
from src.utils import ensure_parent, load_config, read_jsonl, write_json, write_jsonl


PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
USER_AGENT = "kc-hypergraph-rag/0.1 (+https://github.com/B0006CODE/-z)"
DEFAULT_RELATIONS = {
    "contraindication",
    "drug_effect",
    "drug_protein",
    "disease_protein",
    "indication",
    "off-label use",
    "phenotype_absent",
    "phenotype_present",
}
TYPE_PRIORITY = {
    "disease": 0,
    "drug_or_therapy": 1,
    "protein_or_pathway": 2,
    "gene_or_genetic": 3,
    "biomedical_concept": 4,
}


def parse_csv_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter PrimeKG into project-local entity relation features.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--primekg-csv", default=None, help="Local PrimeKG kg.csv path. If absent, stream from Dataverse.")
    parser.add_argument("--primekg-url", default=PRIMEKG_URL, help="PrimeKG CSV URL.")
    parser.add_argument("--entity-dictionary", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allowed-relations", default=",".join(sorted(DEFAULT_RELATIONS)))
    parser.add_argument("--allowed-node-types", default="disease,drug,gene/protein,effect/phenotype")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional CSV row limit for smoke tests.")
    parser.add_argument("--max-matches-per-name", type=int, default=3)
    return parser.parse_args()


def urlopen_with_user_agent(url: str, *, timeout: int = 120, max_retries: int = 3):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            time.sleep(2.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def iter_primekg_rows(args: argparse.Namespace) -> tuple[Iterable[dict[str, str]], Any, str]:
    if args.primekg_csv:
        handle = Path(args.primekg_csv).open("r", encoding="utf-8", newline="")
        return csv.DictReader(handle), handle, str(args.primekg_csv)

    response = urlopen_with_user_agent(args.primekg_url, timeout=120)
    text = (line.decode("utf-8") for line in response)
    return csv.DictReader(text), response, args.primekg_url


def choose_matches(matches: list[dict[str, str]], max_matches: int) -> list[dict[str, str]]:
    return sorted(
        matches,
        key=lambda row: (TYPE_PRIORITY.get(row["entity_type"], 99), row["entity_id"]),
    )[:max_matches]


def relation_rows(
    kg_rows: Iterable[dict[str, str]],
    entity_index: dict[str, list[dict[str, str]]],
    *,
    allowed_relations: set[str] | None,
    allowed_node_types: set[str] | None,
    limit_rows: int | None,
    max_matches_per_name: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_out = []
    relation_counts: Counter[str] = Counter()
    node_type_counts: Counter[str] = Counter()
    matched_relation_counts: Counter[str] = Counter()
    matched_node_type_counts: Counter[str] = Counter()
    matched_entities: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    scanned = 0
    relation_allowed = {normalize_text(item) for item in allowed_relations} if allowed_relations else None
    node_type_allowed = {item.lower() for item in allowed_node_types} if allowed_node_types else None

    for row in kg_rows:
        scanned += 1
        if limit_rows is not None and scanned > limit_rows:
            break
        relation = str(row.get("relation", "")).strip()
        display_relation = str(row.get("display_relation", relation)).strip()
        x_type = str(row.get("x_type", "")).strip()
        y_type = str(row.get("y_type", "")).strip()
        relation_counts[relation] += 1
        node_type_counts[x_type] += 1
        node_type_counts[y_type] += 1

        if relation_allowed is not None and normalize_text(relation) not in relation_allowed:
            continue
        if node_type_allowed is not None and (x_type.lower() not in node_type_allowed or y_type.lower() not in node_type_allowed):
            continue

        x_name = str(row.get("x_name", "")).strip()
        y_name = str(row.get("y_name", "")).strip()
        x_matches = choose_matches(entity_index.get(normalize_text(x_name), []), max_matches_per_name)
        y_matches = choose_matches(entity_index.get(normalize_text(y_name), []), max_matches_per_name)
        if not x_matches or not y_matches:
            continue

        for x_match in x_matches:
            for y_match in y_matches:
                if x_match["entity_id"] == y_match["entity_id"]:
                    continue
                source_id = x_match["entity_id"]
                target_id = y_match["entity_id"]
                edge_key = tuple(sorted((source_id, target_id)) + [relation])
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                matched_entities.update([source_id, target_id])
                matched_relation_counts[relation] += 1
                matched_node_type_counts[f"{x_type}->{y_type}"] += 1
                rows_out.append(
                    {
                        "source_entity_id": source_id,
                        "source_canonical": x_match["canonical"],
                        "source_project_type": x_match["entity_type"],
                        "source_name": x_name,
                        "source_type": x_type,
                        "source_id": str(row.get("x_id", "")),
                        "target_entity_id": target_id,
                        "target_canonical": y_match["canonical"],
                        "target_project_type": y_match["entity_type"],
                        "target_name": y_name,
                        "target_type": y_type,
                        "target_id": str(row.get("y_id", "")),
                        "relation": relation,
                        "display_relation": display_relation,
                        "source": "PrimeKG",
                    }
                )

    summary = {
        "num_rows_scanned": scanned,
        "num_relations": len(rows_out),
        "num_matched_entities": len(matched_entities),
        "relation_counts_seen": dict(relation_counts.most_common(30)),
        "node_type_counts_seen": dict(node_type_counts.most_common(30)),
        "matched_relation_counts": dict(matched_relation_counts.most_common()),
        "matched_node_type_counts": dict(matched_node_type_counts.most_common()),
    }
    return rows_out, summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    entity_dictionary_path = args.entity_dictionary or paths.get("entity_dictionary", "data/processed/entity_dictionary.jsonl")
    output_path = args.output or paths.get("primekg_relations", "data/external_knowledge/primekg_project_relations.jsonl")
    allowed_relations = parse_csv_set(args.allowed_relations)
    allowed_node_types = parse_csv_set(args.allowed_node_types)

    entity_rows = read_jsonl(entity_dictionary_path)
    entity_index = entity_name_index(entity_rows)
    kg_rows, handle, source = iter_primekg_rows(args)
    try:
        rows_out, summary = relation_rows(
            kg_rows,
            entity_index,
            allowed_relations=allowed_relations,
            allowed_node_types=allowed_node_types,
            limit_rows=args.limit_rows,
            max_matches_per_name=args.max_matches_per_name,
        )
    finally:
        handle.close()

    write_jsonl(output_path, rows_out)
    summary.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "primekg_source": source,
            "entity_dictionary": entity_dictionary_path,
            "output": output_path,
            "allowed_relations": sorted(allowed_relations) if allowed_relations else None,
            "allowed_node_types": sorted(allowed_node_types) if allowed_node_types else None,
            "limit_rows": args.limit_rows,
            "max_matches_per_name": args.max_matches_per_name,
            "num_dictionary_entities": len(entity_rows),
            "num_index_names": len(entity_index),
            "license_note": "PrimeKG code is MIT; dataset reuse follows the licenses of the constituent data sources.",
        }
    )
    ensure_parent(output_path)
    write_json(Path(paths["logs_dir"]) / "build_primekg_relations_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
