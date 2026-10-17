"""
Condition builders for the three failure-mode experiments.

Each function takes a QAItem (see data/schema.py) and returns the list of
Passage objects that make up that condition's context/corpus. Nothing here
calls a model or an embedding model — these functions only construct
context deterministically. Downstream code (retrieval/, runner/) embeds,
indexes, retrieves from, or directly inserts this context into the prompt
depending on the failure mode.

Condition definitions (must match config/experiment.yaml and the Phase 1
plan exactly — do not add or rename conditions without updating both):

  Retrieval Blindness:
    - clean:  gold passage present in the corpus (retrievable)
    - blind:  gold passage withheld from the corpus entirely

  Retrieval Noise:
    - clean:  gold passage + neutral background passages only
    - noisy:  gold passage + distractor passages injected

  Evidence Ignorance:
    - top:     gold passage placed at the START of the context
    - middle:  gold passage placed in the MIDDLE of a fixed-length context
    - removed: gold passage NOT included in context at all
"""
from __future__ import annotations

from data.schema import Passage, QAItem

RETRIEVAL_BLINDNESS_CONDITIONS = ("clean", "blind")
RETRIEVAL_NOISE_CONDITIONS = ("clean", "noisy")
EVIDENCE_IGNORANCE_CONDITIONS = ("top", "middle", "removed")


def build_retrieval_blindness_corpus(item: QAItem, condition: str) -> list[Passage]:
    if condition not in RETRIEVAL_BLINDNESS_CONDITIONS:
        raise ValueError(f"Unknown Retrieval Blindness condition: {condition!r}")
    background = list(item.candidate_passages)
    if condition == "clean":
        return [item.gold_passage] + background
    # blind: gold withheld entirely from the corpus
    return [p for p in background if not p.is_gold]


def build_retrieval_noise_corpus(item: QAItem, condition: str) -> list[Passage]:
    if condition not in RETRIEVAL_NOISE_CONDITIONS:
        raise ValueError(f"Unknown Retrieval Noise condition: {condition!r}")
    background = [p for p in item.candidate_passages if not p.is_distractor]
    if condition == "clean":
        return [item.gold_passage] + background
    distractors = [p for p in item.candidate_passages if p.is_distractor]
    return [item.gold_passage] + distractors + background


def build_evidence_ignorance_context(item: QAItem, condition: str, padding: list[Passage]) -> list[Passage]:
    if condition not in EVIDENCE_IGNORANCE_CONDITIONS:
        raise ValueError(f"Unknown Evidence Ignorance condition: {condition!r}")
    if condition == "removed":
        return list(padding)
    if condition == "top":
        return [item.gold_passage] + list(padding)
    # middle
    half = len(padding) // 2
    return padding[:half] + [item.gold_passage] + padding[half:]


CONDITIONS_BY_MODE = {
    "retrieval_blindness": list(RETRIEVAL_BLINDNESS_CONDITIONS),
    "retrieval_noise": list(RETRIEVAL_NOISE_CONDITIONS),
    "evidence_ignorance": list(EVIDENCE_IGNORANCE_CONDITIONS),
}
