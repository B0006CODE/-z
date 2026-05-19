from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_parent, load_config, read_jsonl, write_json


EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PubMed MeSH terms for corpus passage ids treated as PMIDs.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--corpus", default=None, help="Corpus JSONL path with passage_id PMIDs.")
    parser.add_argument("--output", default=None, help="Output JSONL path for PubMed MeSH records.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of PMIDs to fetch.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.34, help="Delay between NCBI requests without an API key.")
    parser.add_argument("--email", default=None, help="Optional contact email passed to NCBI.")
    parser.add_argument("--tool", default="kc-hypergraph-rag", help="Tool name passed to NCBI.")
    parser.add_argument("--api-key-env", default="NCBI_API_KEY", help="Environment variable containing an NCBI API key.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output instead of resuming.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def passage_pmids(corpus_path: str | Path) -> list[str]:
    pmids = []
    seen = set()
    for row in read_jsonl(corpus_path):
        pmid = str(row.get("passage_id", "")).strip()
        if pmid.isdigit() and pmid not in seen:
            seen.add(pmid)
            pmids.append(pmid)
    return pmids


def existing_pmids(output_path: str | Path) -> set[str]:
    path = Path(output_path)
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(row.get("pmid", "")).strip()
            if pmid:
                seen.add(pmid)
    return seen


def request_xml(
    pmids: list[str],
    *,
    email: str | None,
    tool: str,
    api_key: str | None,
    timeout: int,
    max_retries: int,
) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": tool,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    url = f"{EFETCH_URL}?{urllib.parse.urlencode(params)}"

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to fetch PubMed batch after {max_retries} retries: {last_error}")


def text_or_empty(element: ET.Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def parse_pubmed_xml(xml_text: str) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(xml_text)
    records: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        pmid = text_or_empty(citation.find("PMID"))
        if not pmid:
            continue
        mesh_terms = []
        for heading in citation.findall("./MeshHeadingList/MeshHeading"):
            descriptor = heading.find("DescriptorName")
            if descriptor is None:
                continue
            qualifiers = []
            for qualifier in heading.findall("QualifierName"):
                qualifiers.append(
                    {
                        "ui": qualifier.attrib.get("UI", ""),
                        "name": text_or_empty(qualifier),
                        "major_topic": qualifier.attrib.get("MajorTopicYN", "N") == "Y",
                    }
                )
            mesh_terms.append(
                {
                    "descriptor_ui": descriptor.attrib.get("UI", ""),
                    "descriptor_name": text_or_empty(descriptor),
                    "major_topic": descriptor.attrib.get("MajorTopicYN", "N") == "Y",
                    "qualifiers": qualifiers,
                }
            )
        article_title = text_or_empty(citation.find("./Article/ArticleTitle"))
        records[pmid] = {
            "pmid": pmid,
            "title": article_title,
            "mesh_terms": mesh_terms,
            "num_mesh_terms": len(mesh_terms),
        }
    return records


def write_records(output_path: str | Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(output_path)
    with Path(output_path).open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    corpus_path = args.corpus or paths["corpus"]
    output_path = args.output or paths.get("pubmed_mesh", "data/external_knowledge/pubmed_mesh.jsonl")
    api_key = os.getenv(args.api_key_env) if args.api_key_env else None

    if args.overwrite and Path(output_path).exists():
        Path(output_path).unlink()

    pmids = passage_pmids(corpus_path)
    if args.limit is not None:
        pmids = pmids[: args.limit]
    already_done = existing_pmids(output_path)
    pending = [pmid for pmid in pmids if pmid not in already_done]

    fetched = 0
    failed_batches = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        try:
            xml_text = request_xml(
                batch,
                email=args.email,
                tool=args.tool,
                api_key=api_key,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            parsed = parse_pubmed_xml(xml_text)
            records = [
                parsed.get(pmid, {"pmid": pmid, "title": "", "mesh_terms": [], "num_mesh_terms": 0, "missing": True})
                for pmid in batch
            ]
            write_records(output_path, records)
            fetched += len(records)
            print({"fetched": fetched, "remaining": len(pending) - fetched, "output": output_path})
        except Exception as exc:
            failed_batches.append({"pmids": batch, "error": str(exc)})
            print({"failed_batch_start": start, "error": str(exc)})
        time.sleep(args.sleep if not api_key else min(args.sleep, 0.11))

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": corpus_path,
        "output": output_path,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_pmids_requested": len(pmids),
        "num_existing_records": len(already_done),
        "num_pending_at_start": len(pending),
        "num_fetched_this_run": fetched,
        "num_failed_batches": len(failed_batches),
        "failed_batches": failed_batches[:10],
    }
    write_json(Path(paths["logs_dir"]) / "fetch_pubmed_mesh_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
