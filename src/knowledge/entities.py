from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
GENE_RE = re.compile(r"^[A-Z0-9-]{2,12}$")

STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "between",
    "by",
    "but",
    "can",
    "could",
    "during",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "list",
    "may",
    "more",
    "not",
    "of",
    "on",
    "or",
    "other",
    "patients",
    "reported",
    "several",
    "should",
    "show",
    "such",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "without",
}

BIOMEDICAL_TERMS = {
    "acid",
    "activation",
    "allele",
    "antibody",
    "antigen",
    "assay",
    "blood",
    "bone",
    "cancer",
    "carcinoma",
    "cell",
    "clinical",
    "deficiency",
    "disease",
    "dna",
    "dose",
    "drug",
    "enzyme",
    "expression",
    "factor",
    "gene",
    "genetic",
    "genome",
    "growth",
    "human",
    "immune",
    "infection",
    "inhibitor",
    "kinase",
    "ligand",
    "mutation",
    "neuron",
    "pathway",
    "protein",
    "receptor",
    "response",
    "rna",
    "serum",
    "signaling",
    "syndrome",
    "therapy",
    "transcription",
    "treatment",
    "tumor",
    "virus",
}

BIOMEDICAL_SUFFIXES = (
    "ase",
    "itis",
    "emia",
    "oma",
    "opathy",
    "osis",
    "genic",
    "globin",
    "kinase",
    "protein",
    "receptor",
    "syndrome",
)

LOW_INFORMATION_SINGLE_TERMS = {
    "acid",
    "activation",
    "assay",
    "background",
    "biological",
    "blood",
    "case",
    "cell",
    "clinical",
    "conclusion",
    "conclusions",
    "data",
    "decrease",
    "disease",
    "diagnosis",
    "dose",
    "drug",
    "enzyme",
    "expression",
    "factor",
    "fast",
    "gene",
    "genetic",
    "growth",
    "human",
    "immune",
    "increase",
    "involves",
    "infection",
    "method",
    "methods",
    "mutation",
    "new",
    "objective",
    "one",
    "pathway",
    "patients",
    "phase",
    "protein",
    "procedure",
    "receptor",
    "results",
    "response",
    "na",
    "serum",
    "signaling",
    "syndrome",
    "therapy",
    "treatment",
    "tumor",
    "ii",
    "iii",
}

DISEASE_HINTS = {"cancer", "carcinoma", "deficiency", "disease", "disorder", "infection", "syndrome", "tumor"}
DRUG_HINTS = {"agonist", "antagonist", "antibiotic", "antibody", "drug", "inhibitor", "therapy", "treatment"}
GENE_HINTS = {"allele", "dna", "gene", "genetic", "genome", "mutation", "rna"}
PROTEIN_HINTS = {"enzyme", "factor", "kinase", "ligand", "protein", "receptor"}

GREEK_ASCII = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
}


def normalize_text(text: str) -> str:
    normalized = text
    for source, target in GREEK_ASCII.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.lower()
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    normalized = text
    for source, target in GREEK_ASCII.items():
        normalized = normalized.replace(source, target)
    return [match.group(0) for match in TOKEN_RE.finditer(normalized)]


def normalize_entity(text: str) -> str:
    return normalize_text(text)


def is_gene_like(surface: str) -> bool:
    lower = surface.lower()
    if lower in STOPWORDS or lower in LOW_INFORMATION_SINGLE_TERMS:
        return False
    return bool(GENE_RE.match(surface)) and any(char.isalpha() for char in surface)


def has_biomedical_hint(tokens: list[str]) -> bool:
    lowered = [token.lower() for token in tokens]
    if any(token in BIOMEDICAL_TERMS for token in lowered):
        return True
    if any(token.endswith(BIOMEDICAL_SUFFIXES) for token in lowered):
        return True
    return any(is_gene_like(token) for token in tokens)


def valid_phrase(tokens: list[str]) -> bool:
    if not tokens:
        return False
    lowered = [token.lower() for token in tokens]
    if lowered[0] in STOPWORDS or lowered[-1] in STOPWORDS:
        return False
    if len(tokens) == 1:
        token = tokens[0]
        lower = token.lower()
        if len(lower) < 3 and not is_gene_like(token):
            return False
        if lower in LOW_INFORMATION_SINGLE_TERMS and not is_gene_like(token):
            return False
        return is_gene_like(token) or lower in BIOMEDICAL_TERMS or lower.endswith(BIOMEDICAL_SUFFIXES)
    if any(token in STOPWORDS for token in lowered):
        return False
    if not has_biomedical_hint(tokens):
        return False
    return sum(1 for token in lowered if token not in STOPWORDS) >= 2


def infer_entity_type(canonical: str, surfaces: list[str] | None = None) -> str:
    tokens = canonical.split()
    surface_values = surfaces or []
    if any(token in DISEASE_HINTS for token in tokens):
        return "disease"
    if any(token in DRUG_HINTS for token in tokens):
        return "drug_or_therapy"
    if any(token in GENE_HINTS for token in tokens):
        return "gene_or_genetic"
    if any(token in PROTEIN_HINTS for token in tokens):
        return "protein_or_pathway"
    if any(is_gene_like(surface) for surface in surface_values):
        return "gene_or_protein"
    return "biomedical_concept"


def candidate_phrases(text: str, max_ngram: int = 5) -> list[tuple[str, str]]:
    raw_tokens = tokenize(text)
    candidates: list[tuple[str, str]] = []
    for ngram_size in range(1, max_ngram + 1):
        for start in range(0, len(raw_tokens) - ngram_size + 1):
            phrase_tokens = raw_tokens[start : start + ngram_size]
            if not valid_phrase(phrase_tokens):
                continue
            surface = " ".join(phrase_tokens)
            canonical = normalize_entity(surface)
            if canonical:
                candidates.append((canonical, surface))
    return candidates


def build_entity_dictionary(
    records: list[dict[str, Any]],
    text_key: str,
    min_count: int = 3,
    max_terms: int = 50000,
    max_ngram: int = 5,
    required_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    surface_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        seen_in_record = set()
        for canonical, surface in candidate_phrases(str(record.get(text_key, "")), max_ngram=max_ngram):
            surface_counts[canonical][surface] += 1
            if canonical not in seen_in_record:
                counts[canonical] += 1
                seen_in_record.add(canonical)

    for term in required_terms or []:
        canonical = normalize_entity(term)
        if canonical:
            counts[canonical] = max(counts[canonical], min_count)
            surface_counts[canonical][term] += 1

    required_set = {normalize_entity(term) for term in required_terms or [] if normalize_entity(term)}
    eligible = [
        (canonical, count)
        for canonical, count in counts.items()
        if count >= min_count and len(canonical) >= 3 and canonical not in STOPWORDS
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))

    required_kept = [(canonical, counts[canonical]) for canonical in sorted(required_set) if canonical in counts]
    kept_map = {canonical: count for canonical, count in required_kept}
    for canonical, count in eligible:
        if len(kept_map) >= max_terms:
            break
        kept_map.setdefault(canonical, count)
    kept = sorted(kept_map.items(), key=lambda item: (-item[1], item[0]))

    dictionary = []
    for idx, (canonical, count) in enumerate(kept):
        surfaces = [surface for surface, _ in surface_counts[canonical].most_common(5)]
        dictionary.append(
            {
                "entity_id": f"E{idx:07d}",
                "canonical": canonical,
                "surface_forms": surfaces,
                "document_frequency": int(count),
                "entity_type": infer_entity_type(canonical, surfaces),
            }
        )
    return dictionary


def build_match_index(dictionary: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in dictionary:
        tokens = entry["canonical"].split()
        if not tokens:
            continue
        indexed = dict(entry)
        indexed["tokens"] = tokens
        index[tokens[0]].append(indexed)
    for entries in index.values():
        entries.sort(key=lambda item: -len(item["tokens"]))
    return dict(index)


def match_entities_from_index(
    text: str,
    match_index: dict[str, list[dict[str, Any]]],
    max_matches: int = 128,
) -> list[dict[str, Any]]:
    tokens = [token.lower() for token in tokenize(text)]
    matches: list[dict[str, Any]] = []
    occupied: set[int] = set()

    for start, token in enumerate(tokens):
        if start in occupied:
            continue
        for entry in match_index.get(token, []):
            term_tokens = entry["tokens"]
            end = start + len(term_tokens)
            if end > len(tokens):
                continue
            if tokens[start:end] != term_tokens:
                continue
            if any(pos in occupied for pos in range(start, end)):
                continue
            for pos in range(start, end):
                occupied.add(pos)
            matches.append(
                {
                    "entity_id": entry["entity_id"],
                    "canonical": entry["canonical"],
                    "entity_type": entry["entity_type"],
                    "start_token": start,
                    "end_token": end,
                }
            )
            break
        if len(matches) >= max_matches:
            break
    return matches


def match_entities(text: str, dictionary: list[dict[str, Any]], max_matches: int = 128) -> list[dict[str, Any]]:
    return match_entities_from_index(text, build_match_index(dictionary), max_matches=max_matches)
