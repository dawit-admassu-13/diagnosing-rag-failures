"""
Generates data/dataset_manifest.json: full provenance for the two public
datasets plus the exact 90 selected item IDs, so anyone can verify (or
reproduce) the sample without re-running the full pipeline.

Usage:
  python -m scripts.build_dataset_manifest
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from data.load_items import load_all_items

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(REPO_ROOT / "config" / "experiment.yaml") as f:
        config = yaml.safe_load(f)

    rgb_file = REPO_ROOT / "data" / "raw" / "rgb" / config["dataset"]["rgb_source_file"]
    litm_file = REPO_ROOT / "data" / "raw" / "lost_in_the_middle" / config["dataset"]["lost_in_the_middle_source_file"]

    items = load_all_items(config)
    by_mode: dict[str, list[str]] = {}
    for item in items:
        by_mode.setdefault(item.failure_mode, []).append(item.item_id)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "n_questions_per_mode": config["dataset"]["n_questions_per_mode"],
        "total_selected_items": len(items),
        "datasets": {
            "rgb": {
                "repository_url": "https://github.com/chen700564/RGB",
                "paper": "https://arxiv.org/abs/2309.01431",
                "commit": "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615",
                "license": "CC BY-NC-SA 4.0 (noncommercial)",
                "source_file": config["dataset"]["rgb_source_file"],
                "source_file_note": (
                    "en_refine.json is RGB's own refined/recommended variant (fixed incorrect "
                    "positive/negative documents and corrected some answers vs. en.json), per "
                    "the RGB repository's README."
                ),
                "source_file_sha256": sha256_of(rgb_file),
                "download_date_utc": "2026-08-06",
                "used_for_failure_modes": ["retrieval_blindness", "retrieval_noise"],
                "used_for_note": (
                    "RGB's own evaluation harness derives BOTH 'negative rejection' "
                    "(-> our retrieval_blindness) and 'noise robustness' (-> our retrieval_noise) "
                    "tasks from this SAME file, distinguished only by which of positive/negative "
                    "passages are included in the corpus (RGB's noise_rate parameter). We reproduce "
                    "that same distinction at the loader level rather than using separate files."
                ),
            },
            "lost_in_the_middle": {
                "repository_url": "https://github.com/nelson-liu/lost-in-the-middle",
                "paper": "https://arxiv.org/abs/2307.03172",
                "commit": "29b8a6d042ce29abccee3db1a73171a107d7e6af",
                "license": "see repository LICENSE file (copied to data/raw/lost_in_the_middle/LICENSE)",
                "source_file": config["dataset"]["lost_in_the_middle_source_file"],
                "source_file_note": (
                    "Used the '10_total_documents, gold_at_4' bucket only (1 gold + 9 distractor/"
                    "padding passages per question) out of the repo's several available buckets "
                    "(gold_at_0/4/9, 10/20/30-document variants, oracle) \u2014 our own condition-"
                    "builder logic (conditions/builders.py) re-derives which documents are used "
                    "and where gold is placed, so the position label baked into a specific file's "
                    "native ctxs ordering is not load-bearing; any of the three gold_at_* files "
                    "for the 10-document bucket yields the identical underlying document set."
                ),
                "source_file_sha256": sha256_of(litm_file),
                "download_date_utc": "2026-08-06",
                "used_for_failure_modes": ["evidence_ignorance"],
            },
        },
        "selected_item_ids": by_mode,
        "sampling_method": {
            "description": (
                "Each failure mode's 30-item sample is drawn independently with "
                "random.Random(f'{seed}-{failure_mode_key}').sample(valid_items, n_questions) "
                "over that dataset's full pool of valid items (RGB: 300 items shared by both "
                "retrieval_blindness and retrieval_noise, drawn as two independent 30-item "
                "samples from the same pool; Lost in the Middle: 2655 items for "
                "evidence_ignorance). Per-item gold-passage selection among an item's multiple "
                "eligible positives (RGB only) additionally uses a per-item seeded RNG "
                "(f'{seed}-{failure_mode}-{item_id}-goldpick') rather than always taking the "
                "first listed positive."
            ),
            "failure_mode_keys_used_in_seed_string": {
                "retrieval_blindness": "retrieval_blindness",
                "retrieval_noise": "retrieval_noise",
                "evidence_ignorance": "evidence_ignorance",
            },
        },
        "inclusion_exclusion_rules": {
            "rgb": [
                "query must be non-empty",
                "positive (gold-supporting passages) list must be non-empty",
                "negative (background/distractor passages) list must be non-empty",
                "answer field must contain at least one non-empty answer string",
            ],
            "lost_in_the_middle": [
                "question must be non-empty",
                "answers list must contain at least one non-empty string",
                "ctxs must contain exactly one entry with isgold=true",
                "ctxs must contain at least 2 entries total (gold + >=1 distractor/padding)",
            ],
        },
        "exclusions": {
            "rgb": (
                "0 of 300 en_refine.json records were excluded by the rules above \u2014 every "
                "record has a non-empty query, >=5 positive, >=5 negative, and >=1 non-empty "
                "answer string (verified directly against the downloaded file)."
            ),
            "lost_in_the_middle": (
                "0 of 2655 records in the gold_at_4/10_total_documents file were excluded by the "
                "rules above \u2014 every record has exactly one isgold=true ctx and a non-empty "
                "answers list (verified directly against the downloaded file)."
            ),
        },
        "known_limitations": [
            "RGB and Lost in the Middle are both English-language, Wikipedia/news-centric public "
            "benchmarks, not an enterprise-style corpus \u2014 results characterize failure-mode "
            "behavior on these specific public distributions, not any particular production RAG system.",
            "Lost in the Middle's reference_answer uses only answers[0] (the first of possibly "
            "several acceptable answer strings) for exact-match/F1 scoring \u2014 a documented "
            "simplification, not a defect in the source data.",
            "RGB's retrieval_blindness and retrieval_noise samples are drawn independently from "
            "the same 300-item pool (not guaranteed disjoint) \u2014 a deliberate, documented design "
            "choice, since the two failure modes are analyzed separately and never compared item-for-item.",
        ],
    }

    out_path = REPO_ROOT / "data" / "dataset_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest with {len(items)} selected items to {out_path}")
    for mode, ids in by_mode.items():
        print(f"  {mode}: {len(ids)} items")


if __name__ == "__main__":
    main()
