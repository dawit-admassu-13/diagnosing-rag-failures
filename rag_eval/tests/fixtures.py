"""
Tiny synthetic fixtures for --dry-run pipeline testing ONLY.

These are NOT the RGB / Lost in the Middle datasets and must never be used
as experimental data, cited in results, or copied into data/. They exist
solely to verify that runner/run_generation.py, conditions/builders.py, and
eval/run_eval.py execute correctly end-to-end with zero API calls and zero
cost. Every string below is prefixed "[FIXTURE]" so it can never be
mistaken for real content if it leaks into a log file.
"""
from __future__ import annotations
from data.schema import Passage, QAItem


def build_fixture_items() -> list[QAItem]:
    items: list[QAItem] = []

    # --- Retrieval Blindness fixture ---
    gold_rb = Passage("fixture_rb_gold", "[FIXTURE] Placeholder gold passage for Retrieval Blindness.", is_gold=True)
    background_rb = [
        Passage(f"fixture_rb_bg_{i}", f"[FIXTURE] Placeholder background passage {i}.")
        for i in range(2)
    ]
    items.append(QAItem(
        item_id="fixture_rb_1",
        question="[FIXTURE] Placeholder Retrieval Blindness question?",
        reference_answer="[FIXTURE] Placeholder answer.",
        gold_passage=gold_rb,
        candidate_passages=background_rb,
        failure_mode="retrieval_blindness",
        source_dataset="fixture",
    ))

    # --- Retrieval Noise fixture ---
    gold_rn = Passage("fixture_rn_gold", "[FIXTURE] Placeholder gold passage for Retrieval Noise.", is_gold=True)
    distractors_rn = [
        Passage(f"fixture_rn_dist_{i}", f"[FIXTURE] Placeholder distractor passage {i}.", is_distractor=True)
        for i in range(2)
    ]
    items.append(QAItem(
        item_id="fixture_rn_1",
        question="[FIXTURE] Placeholder Retrieval Noise question?",
        reference_answer="[FIXTURE] Placeholder answer.",
        gold_passage=gold_rn,
        candidate_passages=distractors_rn,
        failure_mode="retrieval_noise",
        source_dataset="fixture",
    ))

    # --- Evidence Ignorance fixture ---
    gold_ei = Passage("fixture_ei_gold", "[FIXTURE] Placeholder gold passage for Evidence Ignorance.", is_gold=True)
    padding_ei = [
        Passage(f"fixture_ei_pad_{i}", f"[FIXTURE] Placeholder padding passage {i}.")
        for i in range(4)
    ]
    items.append(QAItem(
        item_id="fixture_ei_1",
        question="[FIXTURE] Placeholder Evidence Ignorance question?",
        reference_answer="[FIXTURE] Placeholder answer.",
        gold_passage=gold_ei,
        candidate_passages=padding_ei,
        failure_mode="evidence_ignorance",
        source_dataset="fixture",
    ))

    return items
