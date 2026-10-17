"""
Invariant checks for condition corpora, run both as unit tests
(tests/test_conditions.py) and ad hoc against the real sampled data.

Per the Phase 3A spec:
  - blind conditions must contain ZERO gold passages.
  - noisy conditions must contain EXACTLY ONE gold passage, plus whatever
    distractors the loader attached (documented, not asserted to a fixed
    count here since RGB's "negative" list length varies per item).
"""
from __future__ import annotations
from data.schema import Passage, QAItem
from .builders import build_retrieval_blindness_corpus, build_retrieval_noise_corpus


def validate_retrieval_blindness_corpus(item: QAItem, condition: str, corpus: list[Passage]) -> None:
    gold_count = sum(1 for p in corpus if p.is_gold)
    if condition == "blind" and gold_count != 0:
        raise AssertionError(f"{item.item_id}: blind condition must have 0 gold passages, found {gold_count}")
    if condition == "clean" and gold_count != 1:
        raise AssertionError(f"{item.item_id}: clean condition must have exactly 1 gold passage, found {gold_count}")


def validate_retrieval_noise_corpus(item: QAItem, condition: str, corpus: list[Passage]) -> None:
    gold_count = sum(1 for p in corpus if p.is_gold)
    distractor_count = sum(1 for p in corpus if p.is_distractor)
    if gold_count != 1:
        raise AssertionError(f"{item.item_id}: {condition} condition must have exactly 1 gold passage, found {gold_count}")
    if condition == "clean" and distractor_count != 0:
        raise AssertionError(f"{item.item_id}: clean condition must have 0 distractors, found {distractor_count}")
    if condition == "noisy" and distractor_count == 0:
        raise AssertionError(f"{item.item_id}: noisy condition must have >=1 distractor, found 0")


def validate_evidence_ignorance_context(item: QAItem, condition: str, context_passages: list[Passage]) -> None:
    gold_count = sum(1 for p in context_passages if p.is_gold)
    if condition == "removed" and gold_count != 0:
        raise AssertionError(f"{item.item_id}: removed condition must have 0 gold passages, found {gold_count}")
    if condition in ("top", "middle") and gold_count != 1:
        raise AssertionError(f"{item.item_id}: {condition} condition must have exactly 1 gold passage, found {gold_count}")
    if condition == "top" and context_passages and not context_passages[0].is_gold:
        raise AssertionError(f"{item.item_id}: top condition must place gold passage first")


def validate_no_label_leak_in_ids(passages: list[Passage]) -> None:
    """Passage IDs must not encode gold/distractor role via naming convention."""
    banned_substrings = ("gold", "distractor", "positive", "negative")
    for p in passages:
        lowered = p.passage_id.lower()
        for term in banned_substrings:
            if term in lowered:
                raise AssertionError(f"Passage ID {p.passage_id!r} leaks role via substring {term!r}")


def validate_all_conditions_for_item(item: QAItem, check_neutral_ids: bool = False) -> None:
    """
    Runs every applicable corpus-shape check for one QAItem across all its
    conditions. check_neutral_ids defaults to False because this function is
    also exercised against tests/fixtures.py, whose deliberately readable
    IDs (e.g. "fixture_rb_gold") exist purely for debugging a synthetic
    smoke test and are never real experimental data. Real loader output
    (data/rgb_loader.py, data/lost_in_the_middle_loader.py) IS checked for
    neutral IDs, in tests/test_loaders.py, since that's the data that
    actually reaches a prompt or a results table.
    """
    if item.failure_mode == "retrieval_blindness":
        for condition in ("clean", "blind"):
            corpus = build_retrieval_blindness_corpus(item, condition)
            validate_retrieval_blindness_corpus(item, condition, corpus)
            if check_neutral_ids:
                validate_no_label_leak_in_ids(corpus)
    elif item.failure_mode == "retrieval_noise":
        for condition in ("clean", "noisy"):
            corpus = build_retrieval_noise_corpus(item, condition)
            validate_retrieval_noise_corpus(item, condition, corpus)
            if check_neutral_ids:
                validate_no_label_leak_in_ids(corpus)
    elif item.failure_mode == "evidence_ignorance":
        if check_neutral_ids:
            validate_no_label_leak_in_ids([item.gold_passage] + item.candidate_passages)
    else:
        raise ValueError(f"Unknown failure_mode: {item.failure_mode}")
