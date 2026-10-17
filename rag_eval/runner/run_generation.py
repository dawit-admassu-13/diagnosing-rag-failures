"""
Main experiment runner.

Real mode (default, no --dry-run): loads the real sampled datasets (RGB /
Lost in the Middle) per config/experiment.yaml, retrieves top-k passages per
condition (runner/retrieval_step.py), calls the three real model adapters,
and writes one raw JSON file per (model, condition, item, trial) to
logs/raw_calls/, plus one retrieval-log JSON per (item, condition) to
logs/retrieval/. This mode is INTENTIONALLY DISABLED right now (see the
SystemExit guard in main()) until the Local Setup Checklist in the README is
complete — this prevents any accidental real API spend before the model ids
and API keys are actually in place. The full real-run code path below IS
implemented and uses the same functions as dry-run (trials_map, retrieval,
real loaders) so it is correct and ready the moment the guard is removed —
but main() raises SystemExit before it can be reached.

Dry-run mode (--dry-run): uses MockAdapter (zero API calls, zero cost) but,
as of Phase 3A, loads the REAL sampled RGB / Lost in the Middle items (not
synthetic fixtures) and exercises the REAL embedding + FlatIndex retrieval
step, so a dry run genuinely verifies the production pipeline shape (top-k
retrieval, retrieval logging, trial counts, all 90 items) before any paid
API call is ever made. Output goes to logs/dry_run/, never logs/raw_calls/,
and every dry-run file is stamped is_mock=True plus a literal
"[MOCK OUTPUT - NOT A REAL MODEL RESPONSE - DRY RUN ONLY]" string, so it can
never be confused with real experimental data.

The old tiny synthetic fixture set (tests/fixtures.py) still exists and is
used ONLY by the unit tests in tests/ — it is no longer used by the --dry-run
CLI path.

Usage:
  python -m runner.run_generation --dry-run
  python -m runner.run_generation            # real run — currently disabled, see guard below
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from conditions.builders import CONDITIONS_BY_MODE
from data.load_items import load_all_items
from data.schema import Passage, QAItem
from models.base import ModelAdapter
from models.mock_adapter import MockAdapter
from retrieval.embeddings import Embedder
from runner.retrieval_step import run_retrieval_for_condition


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_prompt_template(path: str) -> str:
    with open(path) as f:
        return f.read()


def render_prompt(template: str, question: str, context_passages: list[Passage]) -> str:
    # Only passage_id + text ever reach the prompt — never is_gold/is_distractor.
    context_text = "\n\n".join(f"[{p.passage_id}] {p.text}" for p in context_passages)
    return template.format(question=question, context=context_text)


def build_mock_adapters(config: dict) -> dict[str, ModelAdapter]:
    g = config["generation"]
    return {
        key: MockAdapter(key, cfg.get("model_id", f"mock-{key}"), g["temperature"], g["max_output_tokens"])
        for key, cfg in config["models"].items()
    }


def build_real_adapters(config: dict) -> dict[str, ModelAdapter]:
    """Only imported/instantiated for real runs — requires SDKs + API keys."""
    from models.anthropic_adapter import AnthropicAdapter
    from models.openai_adapter import OpenAIAdapter
    from models.llama_adapter import LlamaAdapter

    m = config["models"]
    g = config["generation"]
    return {
        "claude": AnthropicAdapter(
            "claude", m["claude"]["model_id"], g["temperature"], g["max_output_tokens"],
            m["claude"]["api_key_env"], m["claude"].get("base_url_env"),
        ),
        "gpt": OpenAIAdapter(
            "gpt", m["gpt"]["model_id"], g["temperature"], g["max_output_tokens"], m["gpt"]["api_key_env"]
        ),
        "llama": LlamaAdapter(
            "llama", m["llama"]["model_id"], g["temperature"], g["max_output_tokens"],
            m["llama"]["base_url_env"], m["llama"]["api_key_env"],
        ),
    }


def trials_map(config: dict) -> dict[str, int]:
    t = config["trials"]
    return {
        "clean": t["clean_condition"],
        "top": t["clean_condition"],
        "blind": t["stress_condition"],
        "noisy": t["stress_condition"],
        "middle": t["stress_condition"],
        "removed": t["stress_condition"],
    }


def write_retrieval_log(retrieval_dir: Path, ledger_path: Path, condition: str, item_id: str, record: dict) -> Path:
    """
    Writes the full per-(item,condition) retrieval record to its own JSON
    file (retrieval_dir/<condition>/<item_id>.json) AND appends a copy to the
    consolidated JSONL ledger (ledger_path). Retrieval is independent of
    model/trial, so this is called exactly once per (item, condition) and
    the resulting path is referenced by every raw call for that pair.
    """
    cond_dir = retrieval_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    out_path = cond_dir / f"{item_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return out_path


def write_raw_call(
    out_dir: Path, model_key: str, condition: str, item: QAItem, trial: int, result, retrieval_log_path: Path,
) -> Path:
    call_dir = out_dir / model_key / condition / item.item_id
    call_dir.mkdir(parents=True, exist_ok=True)
    out_path = call_dir / f"trial_{trial}.json"
    payload = result.to_json_dict()
    # Self-contained: eval/run_eval.py can build every required column from
    # this one file plus the referenced retrieval log, without re-running
    # the dataset loaders (which require the raw data files to be present).
    payload["item_id"] = item.item_id
    payload["source_dataset"] = item.source_dataset
    payload["failure_mode"] = item.failure_mode
    payload["question"] = item.question
    payload["reference_answer"] = item.reference_answer
    # retrieval_log_path is None for conditions that never run retrieval at
    # all (e.g. no_context, see scripts/run_no_context.py) -- store a real
    # JSON null there, not the misleading string "None".
    payload["retrieval_log_path"] = str(retrieval_log_path) if retrieval_log_path is not None else None
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


def run_with_retrieval(
    items: list[QAItem],
    adapters: dict[str, ModelAdapter],
    embedder: Embedder,
    template: str,
    out_dir: Path,
    retrieval_dir: Path,
    retrieval_ledger_path: Path,
    trials_by_condition: dict[str, int],
    top_k: int,
    skip_existing: bool,
) -> tuple[int, int]:
    """Returns (n_generation_calls, n_retrieval_calls)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_calls = 0
    n_retrievals = 0
    for item in items:
        for condition in CONDITIONS_BY_MODE[item.failure_mode]:
            trial_count = trials_by_condition.get(condition, 1)

            # Retrieval happens ONCE per (item, condition) — identical
            # retriever reused across all three models and all trials.
            outcome = run_retrieval_for_condition(item, condition, embedder, top_k)
            n_retrievals += 1
            retrieval_log_path = write_retrieval_log(
                retrieval_dir, retrieval_ledger_path, condition, item.item_id, outcome.record
            )
            prompt = render_prompt(template, item.question, outcome.context_passages)

            for trial in range(1, trial_count + 1):
                for model_key, adapter in adapters.items():
                    out_path = out_dir / model_key / condition / item.item_id / f"trial_{trial}.json"
                    if skip_existing and out_path.exists() and _is_valid_completed_call(out_path, adapter.model_id):
                        continue  # resumable: don't repeat calls that already succeeded on the SAME host/model_id
                    result = adapter.generate(prompt)
                    write_raw_call(out_dir, model_key, condition, item, trial, result, retrieval_log_path)
                    n_calls += 1
    return n_calls, n_retrievals


def _is_valid_completed_call(out_path: Path, expected_model_id: str) -> bool:
    """
    A raw-call JSON file only counts as "already done" for resume purposes if
    it recorded a successful call: no error field set, non-empty response
    text, AND its recorded model_id matches the CURRENTLY configured
    model_id for that model key. This exists because skip_existing
    previously treated file EXISTENCE alone as "done" -- so a stale file
    left over from a failed attempt (missing env vars, wrong model id,
    network error, etc.) would be skipped forever on every future resume,
    silently freezing that (model, condition, item, trial) slot at an error
    state instead of ever being retried with working credentials. See
    project integrity rules: never present non-real or errored output as
    though a real model run produced it.

    The model_id check was added after a real incident (2026-08-17): the
    Llama config was switched hosts (Groq -> DeepInfra -> OpenRouter) after
    229 real Groq-served calls already existed on disk. Without this check,
    resuming with skip_existing would have silently treated those
    Groq-served completions as "done" under the new host's config, mixing
    two different serving backends under one "Llama" label -- violating the
    project's own documented no-host-mixing rule (see config/experiment.yaml
    models.llama comment). Comparing model_id catches any host switch,
    including cases where two hosts happen to use a differently-cased or
    differently-prefixed model_id string for what is nominally "the same"
    model.

    Deliberately does NOT special-case is_mock: mock-adapter callers (e.g.
    dry-run / test harnesses that also want resumability) get the same
    checks, which are the only things that distinguish a genuinely completed
    call (on the currently configured host) from a stale failure or a
    stale different-host success, regardless of adapter type.
    """
    try:
        with open(out_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False  # unreadable/corrupt file — not a valid completed call, retry it
    if data.get("error"):
        return False
    if data.get("model_id") != expected_model_id:
        return False  # different host/model_id than currently configured — do not reuse
    text = data.get("text") or ""
    return bool(text.strip())


def run_dry_run_mode(config: dict) -> None:
    template = load_prompt_template(config["generation"]["prompt_template_path"])
    items = load_all_items(config)
    embedder = Embedder(**config["embedding"])
    adapters = build_mock_adapters(config)
    out_dir = Path(config["output"]["dry_run_dir"])
    retrieval_dir = out_dir / "retrieval"
    retrieval_ledger_path = out_dir / "retrieval_log_dryrun.jsonl"
    if retrieval_ledger_path.exists():
        retrieval_ledger_path.unlink()  # start each dry run from a clean ledger
    trials = trials_map(config)
    top_k = config["retrieval"]["top_k"]

    print(
        f"[DRY RUN] {len(items)} REAL sampled dataset items "
        "(RGB + Lost in the Middle — not synthetic fixtures), MockAdapter only "
        "— no paid API calls, no cost."
    )
    n_calls, n_retrievals = run_with_retrieval(
        items, adapters, embedder, template, out_dir, retrieval_dir, retrieval_ledger_path,
        trials, top_k, skip_existing=False,
    )
    print(
        f"[DRY RUN] Done. {n_retrievals} retrieval calls, {n_calls} mock generation calls "
        f"written under {out_dir}/ (all is_mock=True)."
    )


def run_real_mode(config: dict, model_filter: list[str] | None = None, pilot_per_mode: int | None = None) -> None:
    """
    Full real-run implementation — correct and ready, but never reached: the
    SystemExit guard in main() fires before this function is called. Left
    fully implemented (rather than a stub) so that trials_map, retrieval, and
    the real loaders are demonstrably wired the same way for dry-run and real
    runs, and so real runs require deleting exactly one line (the
    `raise SystemExit(...)` call in main()) rather than writing new code.

    model_filter: optional subset of {"claude","gpt","llama"} to run this
    invocation. Every model still uses the identical dataset, retriever,
    embedding, chunking, top-k, prompt template, temperature, max output
    length, and trial counts (trials_map/run_with_retrieval are unchanged
    and independent of which models are present) — this only lets real API
    access to different providers be obtained and run at different times
    without touching anything else about the protocol. Full parity across
    all three models is still required before any cross-model comparison or
    taxonomy claim is written up.

    pilot_per_mode: optional int. When set, takes only the first
    `pilot_per_mode` items from EACH of the three failure-mode blocks
    (retrieval_blindness, retrieval_noise, evidence_ignorance) instead of the
    full n_questions_per_mode from config — a small, stratified pilot that
    still covers all three failure modes with the identical dataset,
    retriever, prompt, and trial protocol as the full run. This is a code-
    and-cost-path check only, never a basis for taxonomy claims or a "best
    model" decision — the full n_questions_per_mode run is still required
    for every one of the three fixed models before any cross-model
    comparison is written up.
    """
    template = load_prompt_template(config["generation"]["prompt_template_path"])
    items = load_all_items(config)
    if pilot_per_mode is not None:
        n_full = config["dataset"]["n_questions_per_mode"]
        if pilot_per_mode > n_full:
            raise ValueError(
                f"--pilot {pilot_per_mode} exceeds n_questions_per_mode={n_full} in config; "
                "pilot must be a subset, not larger than the full per-mode sample."
            )
        # items is laid out as three contiguous blocks of n_full, one per
        # failure mode (see data/load_items.py) — take the first
        # pilot_per_mode of each block so all three modes stay represented.
        pilot_items = []
        for block_start in range(0, len(items), n_full):
            pilot_items.extend(items[block_start:block_start + pilot_per_mode])
        items = pilot_items
        print(
            f"[PILOT] Using {len(items)} items ({pilot_per_mode} per failure mode, "
            f"out of {n_full} per mode in the full sample). This is a code/cost-path "
            "check only — full runs are still required for taxonomy claims."
        )
    embedder = Embedder(**config["embedding"])
    adapters = build_real_adapters(config)
    if model_filter:
        unknown = set(model_filter) - set(adapters)
        if unknown:
            raise ValueError(f"Unknown model key(s) in --models: {sorted(unknown)}. Valid: {sorted(adapters)}")
        adapters = {k: v for k, v in adapters.items() if k in model_filter}
    out_dir = Path(config["output"]["raw_calls_dir"])
    retrieval_dir = Path(config["output"]["retrieval_log_path"]).parent / "retrieval"
    retrieval_ledger_path = Path(config["output"]["retrieval_log_path"])
    trials = trials_map(config)
    top_k = config["retrieval"]["top_k"]

    n_calls, n_retrievals = run_with_retrieval(
        items, adapters, embedder, template, out_dir, retrieval_dir, retrieval_ledger_path,
        trials, top_k, skip_existing=True,
    )
    print(f"[REAL RUN] Done. {n_retrievals} retrieval calls, {n_calls} generation calls written under {out_dir}/.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--dry-run", action="store_true",
                         help="Use MockAdapter over the real sampled datasets. Zero API calls, zero cost.")
    parser.add_argument("--models", default=None,
                         help="Comma-separated subset of claude,gpt,llama to run this invocation "
                              "(e.g. --models llama). Default: all three. Lets real API access to "
                              "different providers be obtained at different times without changing "
                              "the dataset/retriever/prompt/trial protocol, which stays identical "
                              "across every invocation regardless of this flag.")
    parser.add_argument("--pilot", type=int, default=None,
                         help="Real run only: use this many items PER failure mode (instead of "
                              "the full n_questions_per_mode from config) as a small stratified "
                              "code/cost-path check across all three failure modes. E.g. --pilot 3 "
                              "runs 3 items x 3 modes = 9 items (with full trial counts per "
                              "condition) instead of the full 90. Never a basis for taxonomy "
                              "claims or picking a 'best' model — full per-mode runs are still "
                              "required for every one of the three fixed models before any "
                              "cross-model comparison is written up.")
    args = parser.parse_args()
    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    load_dotenv()  # populates os.environ from a local .env, if present — needed for real adapters/API keys
    config = load_config(args.config)

    if args.dry_run:
        if args.pilot is not None:
            raise SystemExit("--pilot is only meaningful for real runs; --dry-run already covers "
                              "all 90 real items at zero cost via MockAdapter.")
        run_dry_run_mode(config)
        return

    # Real-run guard removed 2026-08-14 with explicit user approval, after all
    # Phase 3B decisions were locked: model IDs pinned (claude-sonnet-4-6,
    # gpt-4.1, meta-llama/Llama-3.3-70B-Instruct via DeepInfra), trial counts
    # confirmed (clean=1, stress=3), prompt template confirmed as-is, metric
    # scope confirmed as proxy-only (no LLM-judge), error-analysis sample set
    # to 30 cases. This function still requires a real .env with valid API
    # keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, LLAMA_API_KEY, LLAMA_BASE_URL)
    # to do anything — it will error out on missing/invalid credentials
    # rather than silently falling back to mock behavior.
    run_real_mode(config, model_filter=model_filter, pilot_per_mode=args.pilot)


if __name__ == "__main__":
    main()
