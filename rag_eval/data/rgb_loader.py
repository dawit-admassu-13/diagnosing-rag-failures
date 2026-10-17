"""
Loader for RGB (Retrieval-Augmented Generation Benchmark).

RGB source: https://github.com/chen700564/RGB (commit 65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615)
Paper: Chen et al. 2023, "Benchmarking Large Language Models in
Retrieval-Augmented Generation", https://arxiv.org/abs/2309.01431

Real schema (inspected directly from the downloaded data/en_refine.json,
2026-08-06 — this is JSON-LINES, one JSON object per line, NOT a JSON array):
  {
    "id": 0,
    "query": "...",
    "answer": [["accepted answer string", "synonym", ...], [...alt answer...]],
    "positive": ["gold-supporting passage text", ...],   # >=5 per item, all 300 items
    "negative": ["distractor/background passage text", ...]  # >=5 per item, all 300 items
  }

RGB's own evaluation harness (evalue.py / reject_evalue.py) does not ship
separate "noise-robustness" and "negative-rejection" files — both tasks are
built from this SAME en_refine.json, distinguished only by which of
positive/negative go into the corpus (RGB's `noise_rate` CLI flag: 1.0 = all
negative, i.e. negative rejection; between 0 and 1 = mixed, i.e. noise
robustness). We reproduce that same distinction here at the loader level:

  failure_mode="retrieval_blindness" (RGB's negative rejection):
    gold_passage <- one passage from "positive" (the item's gold-supporting
      evidence — must be present for the "clean" condition, withheld for
      the "blind" condition by conditions/builders.py).
    candidate_passages <- all "negative" passages, is_distractor=False
      (they are background/noise for retrieval, not literally the injected
      "noise" condition — the "blind" condition uses these alone).

  failure_mode="retrieval_noise" (RGB's noise robustness):
    gold_passage <- one passage from "positive" (same role).
    candidate_passages <- all "negative" passages, is_distractor=True
      (these ARE the injected noise for the "noisy" condition).

We refine.json is used (not en.json) per RGB's own README: "we refine the
retrieved documents and some answers of en.json ... Removing incorrect
positive and negative documents / Adding some positive documents /
Correcting some inaccurate answers."

Passage IDs are neutral by construction: `rgb-<item_id>-<sha1(text)[:10]>` —
derived from a content hash, never from "positive"/"negative"/"gold" role,
so a passage ID alone reveals nothing about gold/distractor status.

Sampling: exactly `n_questions` items are drawn with
random.Random(f"{seed}-{failure_mode}") — the failure_mode is folded into
the seed string so the two RGB-backed failure modes draw independent (not
identical) 30-item samples from the shared 300-item pool, while both remain
fully reproducible for a fixed global seed. Each item's single "gold"
passage is likewise chosen deterministically among the (>=5) available
positives with a per-item seeded RNG, rather than always taking positive[0].
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

from .schema import Passage, QAItem


def _neutral_id(item_id, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"rgb-{item_id}-{digest}"


def _flatten_answers(raw_answer) -> list[str]:
    """Flatten RGB's `answer` field into a flat list of full answer strings.

    BUG FIX (2026-08-17): en_refine.json's `answer` field is NOT uniformly
    list-of-lists as the module docstring above assumed. Empirically, of the
    300 records: 209 (70%) store `answer` as a FLAT list of strings, e.g.
    ["medical"] or ["Kentucky"], and only 91 (30%) use the documented
    list-of-alias-groups form, e.g. [["January 2 2022", "Jan 2, 2022", ...]].
    The original code (`[a for group in obj["answer"] for a in group]`)
    silently iterated over the CHARACTERS of each flat-form string, so
    reference_answer ended up as a single character (e.g. "F" instead of
    "Francis Lawrence") for those 209 records. This corrupted downstream
    exact_match/token_f1 scoring for every retrieval_blindness/retrieval_noise
    item drawn from a flat-form record — it did NOT affect retrieval, gold
    passage selection, or the prompt/answer the model saw, only the gold
    string used for correctness scoring after generation.
    """
    answers = raw_answer or []
    flat: list[str] = []
    for group in answers:
        if isinstance(group, str):
            flat.append(group)
        else:
            flat.extend(group)
    return flat


def _parse_raw_rgb(raw_dir: Path, source_file: str) -> list[dict]:
    path = raw_dir / source_file
    if not path.exists():
        raise FileNotFoundError(
            f"RGB source file not found at {path.resolve()}.\n"
            "Download from https://github.com/chen700564/RGB, copy data/en_refine.json into "
            "data/raw/rgb/, then re-run."
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: malformed JSON on line {line_no}: {e}") from e
            records.append(obj)
    return records


def _validity_issues(obj: dict) -> list[str]:
    issues = []
    if not obj.get("query", "").strip():
        issues.append("empty query")
    if not obj.get("positive"):
        issues.append("no positive (gold-supporting) passages")
    if not obj.get("negative"):
        issues.append("no negative (background) passages")
    flat_answers = _flatten_answers(obj.get("answer", []))
    if not any(a.strip() for a in flat_answers):
        issues.append("no non-empty answer strings")
    return issues


def load_rgb_subset(
    raw_dir: str, failure_mode: str, n_questions: int, seed: int, source_file: str = "en_refine.json",
) -> list[QAItem]:
    """
    failure_mode: "retrieval_noise" (RGB noise-robustness) or
                  "retrieval_blindness" (RGB negative-rejection).
    Returns exactly n_questions QAItems, or raises if fewer than n_questions
    valid items are available (never silently returns a smaller sample).
    """
    if failure_mode not in ("retrieval_noise", "retrieval_blindness"):
        raise ValueError(f"RGB loader does not support failure_mode={failure_mode!r}")

    raw_path = Path(raw_dir)
    records = _parse_raw_rgb(raw_path, source_file)

    valid_records = []
    for obj in records:
        issues = _validity_issues(obj)
        if not issues:
            valid_records.append(obj)

    if len(valid_records) < n_questions:
        raise ValueError(
            f"Only {len(valid_records)} valid RGB records available in {source_file}, "
            f"need {n_questions} for failure_mode={failure_mode!r}."
        )

    rng = random.Random(f"{seed}-{failure_mode}")
    sampled = rng.sample(valid_records, n_questions)

    items: list[QAItem] = []
    for obj in sampled:
        original_id = obj["id"]
        item_id = ("rgb-rn-" if failure_mode == "retrieval_noise" else "rgb-rb-") + str(original_id)

        gold_rng = random.Random(f"{seed}-{failure_mode}-{original_id}-goldpick")
        gold_text = gold_rng.choice(obj["positive"])
        gold_passage = Passage(passage_id=_neutral_id(original_id, gold_text), text=gold_text, is_gold=True)

        is_distractor = failure_mode == "retrieval_noise"
        candidate_passages = [
            Passage(passage_id=_neutral_id(original_id, text), text=text, is_gold=False, is_distractor=is_distractor)
            for text in obj["negative"]
        ]

        flat_answers = [a for a in _flatten_answers(obj.get("answer", [])) if a.strip()]
        reference_answer = flat_answers[0]

        items.append(QAItem(
            item_id=item_id,
            question=obj["query"],
            reference_answer=reference_answer,
            gold_passage=gold_passage,
            candidate_passages=candidate_passages,
            failure_mode=failure_mode,
            source_dataset="rgb",
        ))
    return items
