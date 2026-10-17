"""
Single entry point that assembles the full 90-item sample (30 per failure
mode) from the real dataset loaders. Used by:
  - runner/run_generation.py (both --dry-run and the real-run path)
  - scripts/build_dataset_manifest.py (to enumerate selected item IDs)
  - tests/test_loaders.py (invariant checks)

Keeping this in one place guarantees the runner, the manifest, and the tests
can never silently drift onto different samples.
"""
from __future__ import annotations
from pathlib import Path

from .rgb_loader import load_rgb_subset
from .lost_in_the_middle_loader import load_lost_in_the_middle_subset
from .schema import QAItem


def load_all_items(config: dict) -> list[QAItem]:
    ds = config["dataset"]
    raw_dir = Path(ds["raw_data_dir"])
    n = ds["n_questions_per_mode"]
    seed = config["seed"]

    items: list[QAItem] = []
    items += load_rgb_subset(
        str(raw_dir / "rgb"), "retrieval_blindness", n, seed,
        source_file=ds.get("rgb_source_file", "en_refine.json"),
    )
    items += load_rgb_subset(
        str(raw_dir / "rgb"), "retrieval_noise", n, seed,
        source_file=ds.get("rgb_source_file", "en_refine.json"),
    )
    items += load_lost_in_the_middle_subset(
        str(raw_dir / "lost_in_the_middle"), n, seed,
        source_file=ds.get("lost_in_the_middle_source_file", "nq-open-10_total_documents_gold_at_4.jsonl.gz"),
    )
    return items
