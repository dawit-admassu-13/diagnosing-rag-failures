"""
Real-run driver for the no_context baseline condition (see config/experiment.yaml
-> no_context_baseline: for the full rationale).

This is a TRUE no-RAG control: the prompt contains the bare question and
nothing else -- no context block, no retrieved passages, no distractors. It
is deliberately separate from runner/run_generation.py because that runner is
wired to CONDITIONS_BY_MODE (conditions/builders.py), which is inherently
per-failure-mode and always goes through the retrieval step. no_context
applies uniformly to all 90 base items regardless of failure_mode and never
calls the retriever at all.

Reuses, unmodified:
  - data/load_items.py load_all_items() -- the exact same 90 base items
    (same seed=42, same dataset sources), in the same order, as every other
    condition in this study.
  - models/anthropic_adapter.py AnthropicAdapter, models/llama_adapter.py
    LlamaAdapter -- the exact same adapters (same retry/proxy-TLS helpers)
    used for every other real call in this study.
  - config/experiment.yaml generation: block -- temperature=0.0,
    max_output_tokens=200, identical to every other condition.
  - runner/run_generation.py's _is_valid_completed_call() -- same
    resume/skip-existing semantics as the main runner, so a partially
    completed run (e.g. interrupted by a rate limit) can be safely re-run
    without re-billing already-successful calls or silently keeping a stale
    error result.

Writes one raw JSON file per (model, item) to
logs/raw_calls/{model_key}/no_context/{item_id}/trial_1.json, in the exact
same schema runner/run_generation.py uses (so eval/run_eval.py picks these up
with zero changes) -- with retrieval_log_path explicitly set to None, which
eval/run_eval.py already treats as "no retrieval metrics available" (null,
not fabricated), because retrieval never runs for this condition.

Usage:
  python -m scripts.run_no_context                # full 180-call real run
  python -m scripts.run_no_context --pilot 2       # 2 items x 2 models = 4 real calls, code-path check only
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Allow running as `python -m scripts.run_no_context` from the rag_eval/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.load_items import load_all_items
from models.anthropic_adapter import AnthropicAdapter
from models.llama_adapter import LlamaAdapter
from runner.run_generation import _is_valid_completed_call, write_raw_call


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_adapters(config: dict) -> dict:
    m = config["models"]
    g = config["generation"]
    return {
        "claude": AnthropicAdapter(
            "claude", m["claude"]["model_id"], g["temperature"], g["max_output_tokens"],
            m["claude"]["api_key_env"], m["claude"].get("base_url_env"),
        ),
        "llama": LlamaAdapter(
            "llama", m["llama"]["model_id"], g["temperature"], g["max_output_tokens"],
            m["llama"]["base_url_env"], m["llama"]["api_key_env"],
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--pilot", type=int, default=None,
                         help="Real run only: use this many items PER failure mode "
                              "(instead of the full 90) as a small code/cost-path check. "
                              "Never a basis for taxonomy claims -- the full 90-item run "
                              "is still required.")
    args = parser.parse_args()

    load_dotenv()  # picks up a local .env if present; harmless if absent
    config = load_config(args.config)
    nc_config = config["no_context_baseline"]

    items = load_all_items(config)
    if args.pilot is not None:
        n_full = config["dataset"]["n_questions_per_mode"]
        if args.pilot > n_full:
            raise ValueError(f"--pilot {args.pilot} exceeds n_questions_per_mode={n_full}")
        pilot_items = []
        for block_start in range(0, len(items), n_full):
            pilot_items.extend(items[block_start:block_start + args.pilot])
        items = pilot_items
        print(f"[PILOT] Using {len(items)} items ({args.pilot} per failure mode, out of {n_full}).")

    with open(nc_config["prompt_template_path"]) as f:
        template = f.read()

    adapters = build_adapters(config)
    model_keys = nc_config["models"]
    adapters = {k: v for k, v in adapters.items() if k in model_keys}

    out_dir = Path(config["output"]["raw_calls_dir"])
    condition = "no_context"
    trial = 1

    n_calls = 0
    n_skipped = 0
    n_errors = 0
    for item in items:
        prompt = template.format(question=item.question)
        for model_key, adapter in adapters.items():
            out_path = out_dir / model_key / condition / item.item_id / f"trial_{trial}.json"
            if out_path.exists() and _is_valid_completed_call(out_path, adapter.model_id):
                n_skipped += 1
                continue
            result = adapter.generate(prompt)
            # retrieval_log_path deliberately omitted (None) below -- no
            # retrieval step runs for this condition, so there is no real
            # retrieval log to reference. write_raw_call() takes a
            # retrieval_log_path positional arg for schema parity with every
            # other condition's raw calls; None is the honest value here,
            # and eval/run_eval.py already treats a missing/None path as
            # "no retrieval metrics for this row" (null, not fabricated).
            write_raw_call(out_dir, model_key, condition, item, trial, result, None)
            n_calls += 1
            if result.error:
                n_errors += 1
                print(f"[ERROR] {model_key}/{item.item_id}: {result.error}")

    print(
        f"[NO_CONTEXT RUN] Done. {n_calls} real generation calls written under "
        f"{out_dir}/{{model}}/no_context/, {n_skipped} skipped (already completed), "
        f"{n_errors} errored (see raw call JSON for exact error)."
    )


if __name__ == "__main__":
    main()
