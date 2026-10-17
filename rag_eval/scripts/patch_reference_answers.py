"""
One-time repair script (2026-08-17): patches the `reference_answer` field
baked into already-completed raw_calls JSON logs, after fixing the RGB
answer-flattening bug in data/rgb_loader.py (see its _flatten_answers
docstring for the bug description).

This does NOT re-run any model or retrieval calls. It only recomputes the
correct gold reference_answer per item_id (deterministic given seed) and
overwrites the `reference_answer` field in each raw call JSON that used the
old, corrupted value. All other fields (text, raw_request, raw_response,
tokens, latency, timestamps, errors, retrieved passages) are left untouched.

Usage:
    python3 -m scripts.patch_reference_answers --config config/experiment.yaml [--dry-run]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import yaml

from data.load_items import load_all_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    ap.add_argument("--raw-calls-dir", default="logs/raw_calls")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    items = load_all_items(config)
    correct_by_item_id = {item.item_id: item.reference_answer for item in items}
    print(f"Loaded {len(correct_by_item_id)} items from dataset (source of truth).")

    raw_dir = Path(args.raw_calls_dir)
    changed = 0
    unchanged = 0
    unknown_item = 0
    total = 0

    for f in sorted(raw_dir.glob("**/*.json")):
        total += 1
        with open(f) as fh:
            d = json.load(fh)
        item_id = d.get("item_id")
        if item_id is None:
            unknown_item += 1
            continue
        correct = correct_by_item_id.get(item_id)
        if correct is None:
            unknown_item += 1
            continue
        old = d.get("reference_answer")
        if old == correct:
            unchanged += 1
            continue
        changed += 1
        if changed <= 15:
            print(f"  {f.name}: item_id={item_id!r} {old!r} -> {correct!r}")
        if not args.dry_run:
            d["reference_answer"] = correct
            with open(f, "w") as fh:
                json.dump(d, fh, indent=2)

    print(f"\nTotal raw call files scanned: {total}")
    print(f"Changed (bug-corrupted -> fixed): {changed}")
    print(f"Unchanged (already correct): {unchanged}")
    print(f"Skipped (no item_id / unknown item_id): {unknown_item}")
    if args.dry_run:
        print("\nDRY RUN — no files were written.")


if __name__ == "__main__":
    main()
