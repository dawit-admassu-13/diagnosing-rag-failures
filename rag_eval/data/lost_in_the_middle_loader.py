"""
Loader for "Lost in the Middle" multi-document QA data.

Source: https://github.com/nelson-liu/lost-in-the-middle (commit 29b8a6d042ce29abccee3db1a73171a107d7e6af)
Paper: Liu et al. 2023 (TACL), https://arxiv.org/abs/2307.03172

Real schema (inspected directly from the downloaded
qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl.gz,
2026-08-06 — gzip-compressed JSON-lines, one JSON object per line):
  {
    "question": "...",
    "answers": ["accepted answer string", ...],
    "nq_annotated_gold": {...},   # NQ long/short annotated answer, unused here
    "ctxs": [
      {
        "id": "628725",           # native Wikipedia passage id — already neutral
        "title": "...",
        "text": "...",
        "score": "1.6234328",     # original retriever's similarity score (unused;
                                   # we compute our own with the configured embedder)
        "hasanswer": false,
        "original_retrieval_index": 0,
        "isgold": false           # exactly one true per item
      },
      ... 10 total ctxs per item ...
    ]
  }

We use the "10_total_documents" bucket (1 gold + 9 distractor/padding
passages per question) — the smallest available bucket, closest in scale to
the other two failure modes' corpora. We deliberately use only the
"gold_at_4" file, not all three position variants (gold_at_0/4/9): our own
retrieval + condition-builder pipeline (runner/retrieval_step.py,
conditions/builders.py) re-derives which documents are used and where gold
is placed, so the position label baked into a specific file's ctxs ordering
is not load-bearing here — any one of the three files gives the identical
underlying (1 gold + 9 distractor) document set for a given question; which
one we pick doesn't change the sample.

Passage IDs: non-gold ctxs already carry a neutral native Wikipedia passage
id (e.g. "628725") in their own "id" field — we prefix it (`litm-<id>`) for
cross-dataset uniqueness. The single gold ctx per item, however, does NOT
carry an "id" field in the raw data (it was manually spliced in by the Lost
in the Middle authors from NQ's annotated gold passage, not sourced from
the same retrieval pass as the other 9 ctxs — verified directly against the
downloaded file, 2026-08-06: every ctx with isgold=true is missing "id",
"score", and "original_retrieval_index"). For the gold passage we instead
derive a content-hash id, `litm-<sha1(text)[:12]>` — the same neutral,
role-free scheme RGB uses, and guaranteed not to contain the substring
"gold" since sha1 hex digests only ever contain [0-9a-f].

Sampling: exactly n_questions items are drawn with
random.Random(f"{seed}-evidence_ignorance") from items with a valid
single-gold ctxs list and a non-empty answers list. item_id is a content
hash of the question text (`litm-<sha1(question)[:12]>`), not the file's
line offset, so identity is stable even if a future re-download changes
row order.
"""
from __future__ import annotations
import gzip
import hashlib
import json
import random
from pathlib import Path

from .schema import Passage, QAItem


def _neutral_item_id(question: str) -> str:
    return f"litm-{hashlib.sha1(question.encode('utf-8')).hexdigest()[:12]}"


def _neutral_passage_id(ctx: dict) -> str:
    """Non-gold ctxs carry a native "id"; gold ctxs don't, so hash the text instead."""
    if ctx.get("id"):
        return f"litm-{ctx['id']}"
    digest = hashlib.sha1(ctx["text"].encode("utf-8")).hexdigest()[:12]
    return f"litm-{digest}"


def _parse_raw_lost_in_the_middle(raw_dir: Path, source_file: str) -> list[dict]:
    path = raw_dir / source_file
    if not path.exists():
        raise FileNotFoundError(
            f"Lost in the Middle source file not found at {path.resolve()}.\n"
            "Download from https://github.com/nelson-liu/lost-in-the-middle, copy "
            "qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl.gz into "
            "data/raw/lost_in_the_middle/, then re-run."
        )
    records = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
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
    if not obj.get("question", "").strip():
        issues.append("empty question")
    if not any(a.strip() for a in obj.get("answers", [])):
        issues.append("no non-empty answers")
    ctxs = obj.get("ctxs", [])
    gold_ctxs = [c for c in ctxs if c.get("isgold")]
    if len(gold_ctxs) != 1:
        issues.append(f"expected exactly 1 isgold ctx, found {len(gold_ctxs)}")
    if len(ctxs) < 2:
        issues.append("fewer than 2 ctxs (need gold + at least 1 distractor)")
    return issues


def load_lost_in_the_middle_subset(
    raw_dir: str, n_questions: int, seed: int,
    source_file: str = "nq-open-10_total_documents_gold_at_4.jsonl.gz",
) -> list[QAItem]:
    raw_path = Path(raw_dir)
    records = _parse_raw_lost_in_the_middle(raw_path, source_file)

    valid_records = [obj for obj in records if not _validity_issues(obj)]
    if len(valid_records) < n_questions:
        raise ValueError(
            f"Only {len(valid_records)} valid Lost in the Middle records available in {source_file}, "
            f"need {n_questions}."
        )

    rng = random.Random(f"{seed}-evidence_ignorance")
    sampled = rng.sample(valid_records, n_questions)

    items: list[QAItem] = []
    for obj in sampled:
        ctxs = obj["ctxs"]
        gold_ctx = next(c for c in ctxs if c.get("isgold"))
        other_ctxs = [c for c in ctxs if not c.get("isgold")]

        gold_passage = Passage(
            passage_id=_neutral_passage_id(gold_ctx),
            text=f"{gold_ctx.get('title', '')}: {gold_ctx['text']}",
            is_gold=True,
        )
        candidate_passages = [
            Passage(
                passage_id=_neutral_passage_id(c),
                text=f"{c.get('title', '')}: {c['text']}",
                is_gold=False,
                is_distractor=False,  # EI "padding", not injected noise
            )
            for c in other_ctxs
        ]

        reference_answer = next(a for a in obj["answers"] if a.strip())

        items.append(QAItem(
            item_id=_neutral_item_id(obj["question"]),
            question=obj["question"],
            reference_answer=reference_answer,
            gold_passage=gold_passage,
            candidate_passages=candidate_passages,
            failure_mode="evidence_ignorance",
            source_dataset="lost_in_the_middle",
        ))
    return items
