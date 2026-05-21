from __future__ import annotations

import argparse
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knowledge.entities import normalize_text
from src.utils import load_config, write_json, write_jsonl


DEFAULT_MESH_XML_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MeSH descriptor names, entry terms, and tree numbers.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mesh-xml-url", default=DEFAULT_MESH_XML_URL)
    parser.add_argument("--mesh-xml", default="data/external_knowledge/desc2026.xml")
    parser.add_argument("--output", default="data/external_knowledge/mesh_synonyms_2026.jsonl")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def download_if_needed(url: str, path: str | Path, *, force: bool) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return
    urllib.request.urlretrieve(url, target)


def text_of(parent: ET.Element, path: str) -> str:
    child = parent.find(path)
    return child.text.strip() if child is not None and child.text else ""


def descriptor_records(xml_path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "DescriptorRecord":
            continue
        mesh_ui = text_of(elem, "DescriptorUI")
        mesh_name = text_of(elem, "DescriptorName/String")
        tree_numbers = [
            node.text.strip()
            for node in elem.findall("TreeNumberList/TreeNumber")
            if node.text and node.text.strip()
        ]
        entry_terms = []
        seen_terms = {normalize_text(mesh_name)}
        for term in elem.findall(".//TermList/Term/String"):
            if not term.text:
                continue
            value = term.text.strip()
            normalized = normalize_text(value)
            if not normalized or normalized in seen_terms:
                continue
            seen_terms.add(normalized)
            entry_terms.append(value)
        if mesh_ui:
            records.append(
                {
                    "mesh_ui": mesh_ui,
                    "mesh_name": mesh_name,
                    "normalized": normalize_text(mesh_name),
                    "entry_terms": entry_terms,
                    "normalized_entry_terms": [normalize_text(term) for term in entry_terms],
                    "tree_numbers": tree_numbers,
                    "num_entry_terms": len(entry_terms),
                    "num_tree_numbers": len(tree_numbers),
                }
            )
        elem.clear()
    records.sort(key=lambda row: row["mesh_ui"])
    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logs_dir = Path(config.get("paths", {}).get("logs_dir", "logs"))
    download_if_needed(args.mesh_xml_url, args.mesh_xml, force=args.force_download)
    records = descriptor_records(args.mesh_xml)
    write_jsonl(args.output, records)
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source_url": args.mesh_xml_url,
        "mesh_xml": args.mesh_xml,
        "output": args.output,
        "num_descriptors": len(records),
        "descriptors_with_entry_terms": sum(1 for row in records if row["num_entry_terms"] > 0),
        "num_entry_terms": sum(row["num_entry_terms"] for row in records),
    }
    write_json(logs_dir / "build_mesh_synonyms_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
