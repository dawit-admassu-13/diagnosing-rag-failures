"""
Common data model that every dataset loader must populate.

Retrieval, condition-building, and evaluation code depend only on this
schema — never on a dataset's native file format. This is what lets the
same pipeline run over RGB and Lost in the Middle without special-casing
either one downstream of the loaders.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Passage:
    passage_id: str
    text: str
    is_gold: bool = False
    is_distractor: bool = False


@dataclass
class QAItem:
    item_id: str
    question: str
    reference_answer: str
    gold_passage: Passage
    # Background / distractor pool available for corpus construction.
    # Retrieval Noise items should mark their distractors with is_distractor=True.
    candidate_passages: list[Passage] = field(default_factory=list)
    failure_mode: str = ""      # "retrieval_blindness" | "retrieval_noise" | "evidence_ignorance"
    source_dataset: str = ""    # "rgb" | "lost_in_the_middle" | "fixture"
