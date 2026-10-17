"""
Minimal, fully deterministic exact-search index.

Deliberately avoids FAISS and any managed vector database: for the corpus
sizes involved here (tens of passages per question), a flat numpy
cosine-similarity search is exact (no approximation), fully reproducible
(no randomized index structure, no ANN parameters to tune), and has zero
external service dependency — nothing to sign up for, nothing that could
silently change behavior between runs.

If corpus size ever makes this too slow, swap in faiss.IndexFlatIP behind
the same `search()` interface — exact search, so results would be
identical; only speed would change.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ScoredPassage:
    passage_id: str
    score: float
    text: str
    is_gold: bool


class FlatIndex:
    def __init__(self, passage_ids: list[str], texts: list[str], embeddings: np.ndarray, gold_ids: set[str]):
        assert embeddings.shape[0] == len(passage_ids) == len(texts), (
            "passage_ids, texts, and embeddings must be the same length"
        )
        self.passage_ids = passage_ids
        self.texts = texts
        self.embeddings = embeddings
        self.gold_ids = gold_ids

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[ScoredPassage]:
        # Cosine similarity reduces to a dot product when both sides are
        # L2-normalized (see retrieval/embeddings.py normalize=true).
        scores = self.embeddings @ query_embedding
        order = np.argsort(-scores)[:top_k]
        return [
            ScoredPassage(
                passage_id=self.passage_ids[i],
                score=float(scores[i]),
                text=self.texts[i],
                is_gold=self.passage_ids[i] in self.gold_ids,
            )
            for i in order
        ]
