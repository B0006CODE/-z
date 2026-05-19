from __future__ import annotations

import re
from typing import Iterable

from src.knowledge.entities import normalize_text


GREEK_WORD_ALIASES = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "kappa": "kappa",
    "lambda": "lambda",
    "mu": "mu",
}

ABBREVIATION_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9][A-Za-z0-9 -]{3,80}?)\s*\(([A-Z0-9-]{2,12})\)")


def singularize_token(token: str) -> str:
    """Small deterministic plural normalization for biomedical phrase matching."""
    token = token.lower()
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "ses")) and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and token[-3] in {"s", "x", "z"}:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_lightweight(text: str, *, singularize: bool = True) -> str:
    normalized = normalize_text(text.replace("/", " ").replace("-", " "))
    tokens = normalized.split()
    if singularize:
        tokens = [singularize_token(token) for token in tokens]
    return " ".join(token for token in tokens if token)


def phrase_variants(text: str) -> set[str]:
    base = normalize_text(text)
    light = normalize_lightweight(text)
    variants = {variant for variant in {base, light} if variant}
    tokens = base.split()
    light_tokens = light.split()
    if len(tokens) > 1:
        variants.add(" ".join(tokens))
    if len(light_tokens) > 1:
        variants.add(" ".join(light_tokens))
    return variants


def entity_variant_set(entity_rows: Iterable[dict[str, object]]) -> set[str]:
    variants: set[str] = set()
    for row in entity_rows:
        canonical = str(row.get("canonical", "")).strip()
        if canonical:
            variants.update(phrase_variants(canonical))
    return variants


def mesh_variant_set(mesh_rows: Iterable[dict[str, object]]) -> set[str]:
    variants: set[str] = set()
    for row in mesh_rows:
        for key in ("normalized", "mesh_name"):
            value = str(row.get(key, "")).strip()
            if value:
                variants.update(phrase_variants(value))
    return variants


def extract_abbreviations(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for long_form, short_form in ABBREVIATION_RE.findall(text):
        short_norm = normalize_lightweight(short_form)
        long_norm = normalize_lightweight(long_form)
        if short_norm and long_norm:
            pairs[short_norm] = long_norm
    return pairs


def abbreviation_set(text: str) -> set[str]:
    pairs = extract_abbreviations(text)
    return set(pairs) | set(pairs.values())


def contained_phrase_count(text: str, phrases: Iterable[str]) -> int:
    normalized = f" {normalize_lightweight(text)} "
    count = 0
    for phrase in phrases:
        phrase_norm = normalize_lightweight(phrase)
        if phrase_norm and f" {phrase_norm} " in normalized:
            count += 1
    return count
