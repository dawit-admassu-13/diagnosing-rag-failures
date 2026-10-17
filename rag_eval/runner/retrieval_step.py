"""
Retrieve-then-inject wiring: turns a condition's candidate corpus into the
actual passages sent to the generator, using the configured embedding model
and the deterministic FlatIndex (retrieval/embeddings.py, retrieval/index.py).

Design decision (Phase 3A, confirmed by user 2026-08-06): retrieve-then-inject
applies uniformly across all three failure modes, but what "retrieval"
determines differs by failure mode because the failure modes are not testing
the same thing:

  Retrieval Blindness / Retrieval Noise
    The condition builder (conditions/builders.py) decides corpus MEMBERSHIP
    only (whether gold is present at all; whether distractors are mixed in).
    We then embed that whole per-condition corpus and take the FlatIndex
    top-k by cosine similarity to the query. This is a literal retrieval
    step: whether gold survives into the top-k is exactly what these two
    failure modes are testing (blind: gold isn't even in the corpus, so it
    can never be retrieved; noisy: gold competes with lexically/semantically
    similar distractors for a top-k slot).

  Evidence Ignorance
    This failure mode is NOT a retrieval-ranking experiment — it is a
    controlled POSITION experiment (Liu et al. 2023, "Lost in the Middle"):
    the same fixed context is held constant and only the gold document's
    position within it is manipulated (top / middle / removed). If we let
    embedding similarity decide gold's rank here, we would destroy the
    controlled positioning that is the entire point of the condition, and
    "position in context" would become a side effect of an unrelated
    similarity score instead of an experimenter-controlled variable.

    We reconcile this with "retrieve-then-inject" the way the original Lost
    in the Middle methodology actually works: retrieval decides which
    documents make up the fixed-size context (we retrieve top-k NON-GOLD
    passages by similarity to the query — this is the "padding" set, and is
    a real retrieval step, not a no-op), and our own condition-builder logic
    then places gold at the controlled position (top/middle) or omits it
    (removed) within that retrieved padding set. Gold's own similarity score
    is still computed and logged for transparency, but it is explicitly
    marked as NOT having determined selection or position.

Every call to run_retrieval_for_condition() produces one retrieval record
(see build_retrieval_record) containing candidate corpus, ranked retrieved
passages, similarity scores, and gold/distractor labels. That record is
logged in full (retrieval/logging is per (item, condition), independent of
model/trial, since the retriever is identical across all three models per
the project's frozen-configuration requirement). The PROMPT text sent to the
generator must only ever contain passage_id + text — labels are attached
here for logging only and must never be threaded into render_prompt().
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from conditions.builders import (
    build_retrieval_blindness_corpus,
    build_retrieval_noise_corpus,
    build_evidence_ignorance_context,
)
from data.schema import Passage, QAItem
from retrieval.embeddings import Embedder
from retrieval.index import FlatIndex, ScoredPassage


@dataclass
class RetrievalOutcome:
    # Passages actually sent to the generator, in final order.
    context_passages: list[Passage]
    # Full structured record for logging (never passed to the prompt).
    record: dict


def _passage_dict(p: Passage) -> dict:
    return {
        "passage_id": p.passage_id,
        "text": p.text,
        "is_gold": p.is_gold,
        "is_distractor": p.is_distractor,
    }


def _score_passages(embedder: Embedder, query: str, passages: list[Passage]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (query_embedding, passage_embeddings). Handles empty passages."""
    query_embedding = embedder.encode([query])[0]
    if not passages:
        dim = query_embedding.shape[0]
        return query_embedding, np.zeros((0, dim))
    passage_embeddings = embedder.encode([p.text for p in passages])
    return query_embedding, passage_embeddings


def _similarity_lookup(query_embedding: np.ndarray, passages: list[Passage], embeddings: np.ndarray) -> dict[str, float]:
    if not passages:
        return {}
    scores = embeddings @ query_embedding
    return {p.passage_id: float(s) for p, s in zip(passages, scores)}


def _retrieve_topk_literal(
    item: QAItem, condition: str, corpus: list[Passage], embedder: Embedder, top_k: int, source_dataset: str,
) -> RetrievalOutcome:
    """Retrieval Blindness / Retrieval Noise: literal top-k over the full per-condition corpus."""
    query_embedding, embeddings = _score_passages(embedder, item.question, corpus)
    gold_ids = {p.passage_id for p in corpus if p.is_gold}
    index = FlatIndex(
        passage_ids=[p.passage_id for p in corpus],
        texts=[p.text for p in corpus],
        embeddings=embeddings,
        gold_ids=gold_ids,
    )
    scored: list[ScoredPassage] = index.search(query_embedding, top_k=min(top_k, len(corpus))) if corpus else []
    text_by_id = {p.passage_id: p.text for p in corpus}
    label_by_id = {p.passage_id: p for p in corpus}
    context_passages = [label_by_id[sp.passage_id] for sp in scored]

    record = {
        "source_dataset": source_dataset,
        "question_id": item.item_id,
        "condition": condition,
        "query_text": item.question,
        "top_k": top_k,
        "retrieval_mode": "similarity_topk",
        "candidate_passages": [_passage_dict(p) for p in corpus],
        "retrieved_passages": [
            {
                "rank": i + 1,
                "passage_id": sp.passage_id,
                "text": text_by_id.get(sp.passage_id, sp.text),
                "similarity_score": sp.score,
                "is_gold": sp.is_gold,
                "is_distractor": label_by_id[sp.passage_id].is_distractor,
                "score_determined_selection": True,
            }
            for i, sp in enumerate(scored)
        ],
    }
    return RetrievalOutcome(context_passages=context_passages, record=record)


def _retrieve_for_evidence_ignorance(
    item: QAItem, condition: str, embedder: Embedder, top_k: int, source_dataset: str,
) -> RetrievalOutcome:
    """
    Evidence Ignorance: retrieve top-k NON-GOLD passages by similarity (this
    is the real retrieval step — it decides the padding/context composition),
    then apply the position-controlled condition builder on top of that
    retrieved padding set. Gold's similarity score is computed for
    transparency but explicitly marked as not having driven selection/order.
    """
    non_gold_pool = [p for p in item.candidate_passages if not p.is_gold]
    query_embedding, pool_embeddings = _score_passages(embedder, item.question, non_gold_pool)
    gold_query_embedding, gold_embedding_arr = _score_passages(embedder, item.question, [item.gold_passage])
    gold_score = float(gold_embedding_arr[0] @ query_embedding) if gold_embedding_arr.shape[0] else None

    index = FlatIndex(
        passage_ids=[p.passage_id for p in non_gold_pool],
        texts=[p.text for p in non_gold_pool],
        embeddings=pool_embeddings,
        gold_ids=set(),
    )
    k = min(top_k, len(non_gold_pool))
    scored_padding: list[ScoredPassage] = index.search(query_embedding, top_k=k) if non_gold_pool else []
    label_by_id = {p.passage_id: p for p in non_gold_pool}
    retrieved_padding = [label_by_id[sp.passage_id] for sp in scored_padding]

    # Position control happens on the RETRIEVED padding set, not the full pool.
    context_passages = build_evidence_ignorance_context(item, condition, retrieved_padding)

    # Build the retrieved_passages record in FINAL PROMPT ORDER (not raw
    # similarity order) so downstream retrieval metrics (gold_rank,
    # precision@k, nDCG) reflect what the generator actually received.
    score_by_id = {sp.passage_id: sp.score for sp in scored_padding}
    retrieved_passages_record = []
    for i, p in enumerate(context_passages):
        if p.is_gold:
            gold_position = "top" if condition == "top" else "middle_of_retrieved_padding"
            retrieved_passages_record.append({
                "rank": i + 1,
                "passage_id": p.passage_id,
                "text": p.text,
                "similarity_score": gold_score,
                "is_gold": True,
                "is_distractor": False,
                "score_determined_selection": False,
                "position_control_note": f"gold placed at protocol-controlled position ({gold_position}); "
                                          "similarity_score is informational only and did not determine "
                                          "inclusion, rank, or position.",
            })
        else:
            retrieved_passages_record.append({
                "rank": i + 1,
                "passage_id": p.passage_id,
                "text": p.text,
                "similarity_score": score_by_id.get(p.passage_id),
                "is_gold": False,
                "is_distractor": p.is_distractor,
                "score_determined_selection": True,
            })

    record = {
        "source_dataset": source_dataset,
        "question_id": item.item_id,
        "condition": condition,
        "query_text": item.question,
        "top_k": top_k,
        "retrieval_mode": "similarity_topk_padding_then_position_controlled",
        "candidate_passages": [_passage_dict(p) for p in ([item.gold_passage] + item.candidate_passages)],
        "retrieved_passages": retrieved_passages_record,
    }
    return RetrievalOutcome(context_passages=context_passages, record=record)


def run_retrieval_for_condition(
    item: QAItem, condition: str, embedder: Embedder, top_k: int,
) -> RetrievalOutcome:
    """
    Single entry point used by the runner for every (item, condition) pair.
    Retrieval happens once per (item, condition) — independent of model and
    trial — since the retriever must be identical across all three models.
    """
    if item.failure_mode == "retrieval_blindness":
        corpus = build_retrieval_blindness_corpus(item, condition)
        return _retrieve_topk_literal(item, condition, corpus, embedder, top_k, item.source_dataset)
    if item.failure_mode == "retrieval_noise":
        corpus = build_retrieval_noise_corpus(item, condition)
        return _retrieve_topk_literal(item, condition, corpus, embedder, top_k, item.source_dataset)
    if item.failure_mode == "evidence_ignorance":
        return _retrieve_for_evidence_ignorance(item, condition, embedder, top_k, item.source_dataset)
    raise ValueError(f"Unknown failure_mode: {item.failure_mode}")
