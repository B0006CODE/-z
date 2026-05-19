from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.idf: dict[str, float] = {}
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}

    def fit(self, documents: list[dict[str, Any]]) -> "BM25Retriever":
        self.documents = documents
        self.doc_lengths = []
        term_doc_freq: Counter[str] = Counter()
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_idx, document in enumerate(documents):
            tokens = tokenize(document.get("text", ""))
            self.doc_lengths.append(len(tokens))
            counts = Counter(tokens)
            for term, tf in counts.items():
                term_doc_freq[term] += 1
                postings[term].append((doc_idx, tf))

        total_docs = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / total_docs if total_docs else 0.0
        self.idf = {
            term: math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for term, df in term_doc_freq.items()
        }
        self.inverted_index = dict(postings)
        return self

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        query_terms = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        if not query_terms or not self.documents:
            return []

        avgdl = self.avg_doc_length or 1.0
        for term, qtf in query_terms.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, tf in self.inverted_index.get(term, []):
                dl = self.doc_lengths[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_idx] += qtf * idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        results: list[dict[str, Any]] = []
        for rank, (doc_idx, score) in enumerate(ranked, start=1):
            document = self.documents[doc_idx]
            results.append(
                {
                    "passage_id": document["passage_id"],
                    "rank": rank,
                    "score": float(score),
                    "text": document.get("text", ""),
                    "title": document.get("title", ""),
                }
            )
        return results

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Index at {path} is not a BM25Retriever.")
        return obj
