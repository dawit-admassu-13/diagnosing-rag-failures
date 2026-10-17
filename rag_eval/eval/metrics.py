"""
Evaluation metrics for the RAG failure-mode study.

Implemented (per the confirmed provisional decisions):
  - retrieval recall
  - precision@k
  - nDCG@k
  - gold_rank (1-indexed rank of gold in the retrieved list, or None)
  - answer correctness (exact match + token-F1)
  - context-ablation sensitivity

Deliberately NOT implemented yet: faithfulness / groundedness scoring and
any LLM-as-judge metric. That is still an open decision (Phase 1 audit,
"Decisions" #7 — proxy metric vs. judge model, and which judge if so).
Do not add a placeholder faithfulness score here; an unresolved metric that
looks implemented is worse than one that is visibly absent.
"""
from __future__ import annotations
import math
import re
from typing import Optional


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(prediction: str, reference: str) -> float:
    """
    Strict full-string exact match (standard SQuAD/NQ-style definition),
    kept deliberately un-relaxed (decision confirmed 2026-08-17) so the
    protocol is identical and pre-specified across all three models.

    KNOWN LIMITATION: all three target models answer in full sentences
    rather than bare short-answer spans, so this reads as ~0% across nearly
    every condition — including clean/evidence-present ones where the model
    is factually correct. That is expected behavior of strict EM against
    generative answers, not a bug and not evidence of high failure rates.
    Treat `token_f1` as the primary correctness signal; report this
    alongside it but do not over-interpret it alone. Must be stated as a
    limitation in the paper (see README.md § Known metric limitations).
    """
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_correctness(prediction: str, reference: str) -> dict:
    """Report both exact-match and token-F1 — do not collapse to a single number."""
    return {"exact_match": exact_match(prediction, reference), "token_f1": token_f1(prediction, reference)}


def retrieval_recall(retrieved_ids: list[str], gold_id: str) -> float:
    """Did the gold passage appear anywhere in the retrieved top-k? (binary, single-gold case)."""
    return 1.0 if gold_id in retrieved_ids else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for pid in top_k if pid in relevant_ids) / len(top_k)


def ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """Standard nDCG@k. `relevance` maps passage_id -> graded relevance (e.g. 1.0 for gold, 0.0 otherwise)."""
    def dcg(ids: list[str]) -> float:
        return sum(relevance.get(pid, 0.0) / math.log2(i + 2) for i, pid in enumerate(ids))

    top_k = retrieved_ids[:k]
    actual_dcg = dcg(top_k)
    ideal_order = sorted(relevance.keys(), key=lambda pid: -relevance.get(pid, 0.0))[:k]
    ideal_dcg = dcg(ideal_order)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def gold_rank(retrieved_ids: list[str], gold_id: str) -> Optional[int]:
    """1-indexed position of the gold passage in the ranked retrieval list, or None if absent."""
    for i, pid in enumerate(retrieved_ids):
        if pid == gold_id:
            return i + 1
    return None


def context_ablation_sensitivity(correctness_with_evidence: float, correctness_without_evidence: float) -> float:
    """
    Drop in answer correctness when gold evidence is removed.
    Large positive value = model relies on evidence (expected/healthy).
    Near-zero value = candidate Evidence Ignorance signal.
    """
    return correctness_with_evidence - correctness_without_evidence
