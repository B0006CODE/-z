from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def medcpt_article_input(document: dict[str, Any]) -> list[str]:
    title = str(document.get("title", "")).strip()
    text = str(document.get("text", "")).strip()
    return [title, text]


class MedCPTRetriever:
    def __init__(
        self,
        query_model_name: str = "ncbi/MedCPT-Query-Encoder",
        article_model_name: str = "ncbi/MedCPT-Article-Encoder",
        *,
        batch_size: int = 32,
        query_max_length: int = 64,
        article_max_length: int = 512,
        normalize_embeddings: bool = True,
        device: str = "cpu",
    ) -> None:
        self.query_model_name = query_model_name
        self.article_model_name = article_model_name
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.article_max_length = article_max_length
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.query_tokenizer = None
        self.query_model = None
        self.article_tokenizer = None
        self.article_model = None
        self.passage_ids: list[str] = []
        self.embeddings: np.ndarray | None = None

    def _load_query(self) -> None:
        if self.query_model is not None:
            return
        self.query_tokenizer = AutoTokenizer.from_pretrained(self.query_model_name)
        self.query_model = AutoModel.from_pretrained(self.query_model_name).to(self.device)
        self.query_model.eval()

    def _load_article(self) -> None:
        if self.article_model is not None:
            return
        self.article_tokenizer = AutoTokenizer.from_pretrained(self.article_model_name)
        self.article_model = AutoModel.from_pretrained(self.article_model_name).to(self.device)
        self.article_model.eval()

    def _encode_articles(self, documents: list[dict[str, Any]]) -> np.ndarray:
        self._load_article()
        assert self.article_tokenizer is not None
        assert self.article_model is not None
        all_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(documents), self.batch_size):
                batch_docs = documents[start : start + self.batch_size]
                encoded = self.article_tokenizer(
                    [medcpt_article_input(document) for document in batch_docs],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.article_max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                batch_embeddings = self.article_model(**encoded).last_hidden_state[:, 0, :]
                if self.normalize_embeddings:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings.detach().cpu())
        return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32, copy=False)

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        self._load_query()
        assert self.query_tokenizer is not None
        assert self.query_model is not None
        all_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(queries), self.batch_size):
                batch_queries = queries[start : start + self.batch_size]
                encoded = self.query_tokenizer(
                    batch_queries,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.query_max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                batch_embeddings = self.query_model(**encoded).last_hidden_state[:, 0, :]
                if self.normalize_embeddings:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(batch_embeddings.detach().cpu())
        return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32, copy=False)

    def fit(self, documents: list[dict[str, Any]]) -> "MedCPTRetriever":
        self.passage_ids = [str(document["passage_id"]) for document in documents]
        self.embeddings = self._encode_articles(documents)
        return self

    def search_many(
        self,
        queries: list[str],
        *,
        top_k: int = 100,
        score_batch_size: int = 128,
    ) -> list[list[dict[str, Any]]]:
        if self.embeddings is None:
            raise ValueError("MedCPTRetriever has no article embeddings. Fit or load an index first.")
        if not queries:
            return []
        query_embeddings = self._encode_queries(queries)
        limit = min(top_k, len(self.passage_ids))
        all_results: list[list[dict[str, Any]]] = []
        for start in range(0, len(query_embeddings), score_batch_size):
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
            raise ValueError("Cannot save MedCPT index before fitting article embeddings.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings,
            passage_ids=np.array(self.passage_ids),
            query_model_name=np.array(self.query_model_name),
            article_model_name=np.array(self.article_model_name),
            query_max_length=np.array(self.query_max_length),
            article_max_length=np.array(self.article_max_length),
            normalize_embeddings=np.array(self.normalize_embeddings),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> "MedCPTRetriever":
        loaded = np.load(path, allow_pickle=False)
        retriever = cls(
            query_model_name=str(loaded["query_model_name"].item()),
            article_model_name=str(loaded["article_model_name"].item()),
            batch_size=batch_size,
            query_max_length=int(loaded["query_max_length"].item()),
            article_max_length=int(loaded["article_max_length"].item()),
            normalize_embeddings=bool(loaded["normalize_embeddings"].item()),
            device=device,
        )
        retriever.embeddings = loaded["embeddings"].astype(np.float32, copy=False)
        retriever.passage_ids = [str(pid) for pid in loaded["passage_ids"].tolist()]
        return retriever
