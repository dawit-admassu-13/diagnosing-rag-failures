"""
Aggregates raw call logs + retrieval logs into results_aggregated.csv, plus a
secondary results_ablation_sensitivity.csv for the cross-condition
context-ablation-sensitivity metric (see note below on why it is not a
per-row column).

Reads only from logs/raw_calls/ (real runs) or logs/dry_run/ (--dry-run) —
never hand-edited. Re-run this script any time to regenerate the aggregated
CSV from the raw logs; it never writes to the raw logs.

Each raw call JSON is self-contained (source_dataset, question, reference_answer,
retrieval_log_path are written into it by runner/run_generation.py), so this
script never needs to re-run the dataset loaders.

Usage:
  python -m eval.run_eval --dry-run
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from eval.metrics import answer_correctness, retrieval_recall, precision_at_k, ndcg_at_k, gold_rank, \
    context_ablation_sensitivity

REQUIRED_COLUMNS = [
    "question_id", "source_dataset", "model", "condition", "trial",
    "answer_text",
    "retrieval_recall", "precision_at_k", "ndcg_at_k", "gold_rank",
    "exact_match", "token_f1",
    "raw_call_path",
]


def iter_raw_calls(raw_dir: Path):
    if not raw_dir.exists():
        return
    for model_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for condition_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for item_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
                for trial_file in sorted(item_dir.glob("trial_*.json")):
                    with open(trial_file) as f:
                        call = json.load(f)
                    yield model_dir.name, condition_dir.name, item_dir.name, trial_file.stem, call, trial_file


_retrieval_log_cache: dict[str, dict] = {}


def _load_retrieval_log(path_str: str) -> dict | None:
    if not path_str:
        return None
    if path_str in _retrieval_log_cache:
        return _retrieval_log_cache[path_str]
    path = Path(path_str)
    if not path.exists():
        _retrieval_log_cache[path_str] = None
        return None
    with open(path) as f:
        record = json.load(f)
    _retrieval_log_cache[path_str] = record
    return record


def _retrieval_metrics_for_call(retrieval_record: dict | None, top_k: int) -> dict:
    if retrieval_record is None:
        return {"retrieval_recall": None, "precision_at_k": None, "ndcg_at_k": None, "gold_rank": None}

    retrieved = retrieval_record.get("retrieved_passages", [])
    retrieved_ids = [r["passage_id"] for r in sorted(retrieved, key=lambda r: r.get("rank") or 10**9)]
    gold_ids = {c["passage_id"] for c in retrieval_record.get("candidate_passages", []) if c.get("is_gold")}
    gold_id = next(iter(gold_ids), None)

    if gold_id is None:
        # Conditions that deliberately withhold gold (e.g. "blind") have no
        # gold_id to score against — recall/precision/ndcg/gold_rank are all
        # trivially 0/None by construction, not missing data.
        return {
            "retrieval_recall": 0.0,
            "precision_at_k": precision_at_k(retrieved_ids, set(), top_k),
            "ndcg_at_k": 0.0,
            "gold_rank": None,
        }

    relevance = {gold_id: 1.0}
    return {
        "retrieval_recall": retrieval_recall(retrieved_ids, gold_id),
        "precision_at_k": precision_at_k(retrieved_ids, {gold_id}, top_k),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, relevance, top_k),
        "gold_rank": gold_rank(retrieved_ids, gold_id),
    }


def build_rows(raw_dir: Path, top_k: int) -> list[dict]:
    rows = []
    for model_key, condition, item_id, trial, call, trial_file in iter_raw_calls(raw_dir):
        reference = call.get("reference_answer") or ""
        prediction = call.get("text", "") or ""
        correctness = answer_correctness(prediction, reference) if reference else {"exact_match": None, "token_f1": None}
        retrieval_record = _load_retrieval_log(call.get("retrieval_log_path"))
        r_metrics = _retrieval_metrics_for_call(retrieval_record, top_k)

        rows.append({
            "question_id": call.get("item_id", item_id),
            "source_dataset": call.get("source_dataset"),
            "model": model_key,
            "condition": condition,
            "trial": trial,
            "answer_text": prediction,
            "retrieval_recall": r_metrics["retrieval_recall"],
            "precision_at_k": r_metrics["precision_at_k"],
            "ndcg_at_k": r_metrics["ndcg_at_k"],
            "gold_rank": r_metrics["gold_rank"],
            "exact_match": correctness["exact_match"],
            "token_f1": correctness["token_f1"],
            "raw_call_path": str(trial_file),
            # Extra columns beyond the required set — kept for traceability.
            "failure_mode": call.get("failure_mode"),
            "question": call.get("question"),
            "reference_answer": reference,
            "model_id": call.get("model_id"),
            "is_mock": call.get("is_mock"),
            "error": call.get("error"),
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "latency_ms": call.get("latency_ms"),
            "retrieval_log_path": call.get("retrieval_log_path"),
        })
    return rows


# Which condition is the "with evidence" baseline for each failure mode's
# "without evidence" stress condition — used only for context_ablation_sensitivity.
ABLATION_PAIRS = {
    "retrieval_blindness": ("clean", "blind"),
    "retrieval_noise": ("clean", "noisy"),
    "evidence_ignorance": ("top", "removed"),
}


def build_ablation_sensitivity_rows(rows: list[dict]) -> list[dict]:
    """
    context_ablation_sensitivity is a PAIRED metric (correctness with
    evidence minus correctness without evidence) — it is not well-defined
    for a single row, so it is reported here as its own table keyed by
    (model, question_id), one row per failure mode's clean/stress pair,
    rather than forced into a per-row column of results_aggregated.csv.
    Trials within a condition are averaged first (mean exact_match) so a
    stress condition run 3x is compared fairly against a clean condition run once.
    """
    by_key: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple, dict] = {}
    for row in rows:
        fm = row.get("failure_mode")
        if fm not in ABLATION_PAIRS or row.get("exact_match") is None:
            continue
        key = (row["model"], row["question_id"])
        by_key[key][row["condition"]].append(row["exact_match"])
        meta[key] = {"model": row["model"], "question_id": row["question_id"], "source_dataset": row["source_dataset"], "failure_mode": fm}

    out = []
    for key, cond_scores in by_key.items():
        fm = meta[key]["failure_mode"]
        with_cond, without_cond = ABLATION_PAIRS[fm]
        if with_cond not in cond_scores or without_cond not in cond_scores:
            continue
        with_score = sum(cond_scores[with_cond]) / len(cond_scores[with_cond])
        without_score = sum(cond_scores[without_cond]) / len(cond_scores[without_cond])
        out.append({
            **meta[key],
            "with_evidence_condition": with_cond,
            "without_evidence_condition": without_cond,
            "correctness_with_evidence": with_score,
            "correctness_without_evidence": without_score,
            "context_ablation_sensitivity": context_ablation_sensitivity(with_score, without_score),
        })
    return out


def write_csv(rows: list[dict], out_path: Path, priority_columns: list[str] | None = None) -> None:
    if not rows:
        print(f"No rows to write for {out_path} — no raw calls found yet.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    priority_columns = priority_columns or []
    fieldnames = [c for c in priority_columns if c in rows[0]] + [k for k in rows[0].keys() if k not in priority_columns]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    top_k = config["retrieval"]["top_k"]

    if args.dry_run:
        raw_dir = Path(config["output"]["dry_run_dir"])
        out_path = raw_dir / "results_aggregated_dryrun.csv"
        ablation_path = raw_dir / "results_ablation_sensitivity_dryrun.csv"
    else:
        raw_dir = Path(config["output"]["raw_calls_dir"])
        out_path = Path(config["output"]["results_aggregated_path"])
        ablation_path = out_path.parent / "results_ablation_sensitivity.csv"

    if not raw_dir.exists():
        print(f"No raw call directory found at {raw_dir} — run the runner first.")
        return

    rows = build_rows(raw_dir, top_k)
    write_csv(rows, out_path, priority_columns=REQUIRED_COLUMNS)
    ablation_rows = build_ablation_sensitivity_rows(rows)
    write_csv(ablation_rows, ablation_path)


if __name__ == "__main__":
    main()
