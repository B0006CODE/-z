from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def passage_text(document: dict[str, Any]) -> str:
    title = document.get("title", "")
    text = document.get("text", "")
    if title:
        return f"{title}\n{text}"
    return text


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.model: SentenceTransformer | None = None
        self.passage_ids: list[str] = []
        self.embeddings: np.ndarray | None = None

    def _model(self) -> SentenceTransformer:
        if self.model is None:
            kwargs: dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            self.model = SentenceTransformer(self.model_name, **kwargs)
        return self.model

    def fit(self, documents: list[dict[str, Any]]) -> "DenseRetriever":
        texts = [passage_text(document) for document in documents]
        self.passage_ids = [str(document["passage_id"]) for document in documents]
        embeddings = self._model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        self.embeddings = embeddings.astype(np.float32, copy=False)
        return self

    def search(self, query: str, top_k: int = 100) -> list[dict[str, Any]]:
        if self.embeddings is None:
            raise ValueError("DenseRetriever has no embeddings. Fit or load an index first.")
        if not self.passage_ids:
            return []

        query_embedding = self._model().encode(
            [query],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)[0]
        scores = self.embeddings @ query_embedding
        limit = min(top_k, len(scores))
        if limit <= 0:
            return []

        candidate_idx = np.argpartition(-scores, limit - 1)[:limit]
        ranked_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
        return [
            {
                "passage_id": self.passage_ids[int(idx)],
                "rank": rank,
                "score": float(scores[int(idx)]),
            }
            for rank, idx in enumerate(ranked_idx, start=1)
        ]

    def search_many(
        self,
        queries: list[str],
        top_k: int = 100,
        score_batch_size: int = 128,
    ) -> list[list[dict[str, Any]]]:
        if self.embeddings is None:
            raise ValueError("DenseRetriever has no embeddings. Fit or load an index first.")
        if not queries:
            return []

        query_embeddings = self._model().encode(
            queries,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32, copy=False)

        all_results: list[list[dict[str, Any]]] = []
        limit = min(top_k, len(self.passage_ids))
        for start in range(0, len(queries), score_batch_size):
            batch = query_embeddings[start : start + score_batch_size]
            batch_scores = batch @ self.embeddings.T
            for scores in batch_scores:
                candidate_idx = np.argpartition(-scores, limit - 1)[:limit]
                ranked_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
                all_results.append(
                    [
                        {
                            "passage_id": self.passage_ids[int(idx)],
                            "rank": rank,
                            "score": float(scores[int(idx)]),
                        }
                        for rank, idx in enumerate(ranked_idx, start=1)
                    ]
                )
        return all_results

    def save(self, path: str | Path) -> None:
        if self.embeddings is None:
            raise ValueError("Cannot save a dense index before fitting embeddings.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings,
            passage_ids=np.array(self.passage_ids),
            model_name=np.array(self.model_name),
            normalize_embeddings=np.array(self.normalize_embeddings),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        batch_size: int = 64,
        device: str | None = None,
    ) -> "DenseRetriever":
        loaded = np.load(path, allow_pickle=False)
        model_name = str(loaded["model_name"].item())
        normalize_embeddings = bool(loaded["normalize_embeddings"].item())
        retriever = cls(
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            device=device,
        )
        retriever.embeddings = loaded["embeddings"].astype(np.float32, copy=False)
        retriever.passage_ids = [str(pid) for pid in loaded["passage_ids"].tolist()]
        return retriever
