# RAG Failure Modes — Empirical Validation Runner (Phase 3A)

Scope of this repo right now: **the real public datasets (RGB, Lost in the
Middle) are downloaded and parsed, the full retrieve-then-inject pipeline
(embedding + FlatIndex retrieval, retrieval logging, trials, evaluation) is
wired end-to-end, and everything has been verified with `--dry-run` against
the real 90-item sample using `MockAdapter` — zero paid API calls, zero
cost, zero real model output.** No real model API has been called and no
experimental results exist yet. The real-run path in
`runner/run_generation.py` is intentionally guarded with a `SystemExit`
until the checklist at the bottom of this file is complete — this prevents
any accidental API spend.

See `data/dataset_manifest.json` for full dataset provenance (repo URLs,
commit hashes, file hashes, download date, the exact 90 selected item IDs,
and inclusion/exclusion rules).

## Repository structure

```
rag_eval/
  config/
    experiment.yaml          # all frozen settings — single source of truth
    prompt_template.txt      # verbatim prompt template used by every model
  data/
    schema.py                 # common Passage / QAItem data model
    rgb_loader.py              # real RGB loader (en_refine.json)
    lost_in_the_middle_loader.py  # real Lost in the Middle loader (gold_at_4, 10 docs)
    load_items.py              # single entry point: loads all 90 sampled items
    dataset_manifest.json      # full provenance: repos, commits, hashes, selected item IDs
    raw/rgb/                   # downloaded RGB files (gitignored — not ours to redistribute)
    raw/lost_in_the_middle/    # downloaded LitM files (gitignored)
  retrieval/
    embeddings.py              # embedding wrapper (sentence-transformers)
    index.py                   # exact, deterministic flat cosine-sim index
  conditions/
    builders.py                 # the 7 condition-corpus builders (3 modes)
    validate.py                  # corpus/condition invariant checks (gold presence, neutral IDs)
  runner/
    run_generation.py            # main orchestrator, resumable, --dry-run flag
    retrieval_step.py             # retrieve-then-inject wiring (embedder + FlatIndex per condition)
  models/
    base.py                     # ModelAdapter interface + GenerationResult
    anthropic_adapter.py          # Claude Sonnet 4.6 (real calls, billed)
    openai_adapter.py             # GPT-4.1 (real calls, billed)
    llama_adapter.py              # Llama 3.3 70B Instruct (real calls, billed)
    mock_adapter.py                # dry-run only, zero cost
  eval/
    metrics.py                    # recall, precision@k, nDCG, gold_rank, correctness, ablation sensitivity
    run_eval.py                   # aggregates raw calls + retrieval logs -> results_aggregated.csv
  scripts/
    build_dataset_manifest.py      # regenerates data/dataset_manifest.json from the real loaders
  error_analysis/
    error_log_template.csv         # empty template — no labels filled in
    labeling_rubric.md              # label definitions + rater instructions
  tests/
    fixtures.py                    # synthetic dry-run-only fixture data (unit tests only)
    test_conditions.py, test_metrics.py, test_loaders.py, test_dry_run_runner.py
  logs/
    raw_calls/                     # real-run output (empty until a real run happens)
    retrieval/                      # real-run per-(item,condition) retrieval logs
    dry_run/                       # dry-run output (mock calls + retrieval logs), committed as evidence
  requirements.txt
  .env.example
  .gitignore
```

## Embedding model choice

**`sentence-transformers/all-MiniLM-L6-v2`**, set in
`config/experiment.yaml -> embedding.model_name` (the only place it's
configured — nothing else hardcodes a model name).

Why this one for a preliminary study:
- Runs entirely locally on CPU, no API key, no per-call cost, no network
  dependency once weights are downloaded once.
- Fixed weights at inference time (no dropout/sampling) → deterministic
  embeddings for the same text, which matters for exact reproducibility.
- Small (~80 MB), fast, and one of the most widely used sentence-embedding
  models in RAG tooling and prior work, so the choice is easy to justify
  and reproduce for a reviewer.
- Swappable: change one config field to try a different embedding model
  (e.g. `BAAI/bge-small-en-v1.5`) without touching any other code.

## Vector index choice

`retrieval/index.py` implements a **flat, exact cosine-similarity search
in numpy** (`FlatIndex`) rather than FAISS or any managed vector database.
For the corpus sizes here (tens of passages per question), exact search is
fast enough, has zero approximation error, and has no external service or
account to depend on — every run is byte-for-byte reproducible given the
same embeddings. If corpus size later makes this too slow, swap in
`faiss.IndexFlatIP` behind the same `search()` interface; being exact, it
would return identical results, only faster.

## Claude host setup (Claude Sonnet 4.6)

`models/anthropic_adapter.py` supports two modes, selected purely by
environment variables (no code change needed to switch):

**Native mode (default)** — direct Anthropic API:
```
ANTHROPIC_API_KEY=<your anthropic key>
```
Leave `CLAUDE_BASE_URL` unset. `models.claude.model_id` in
`config/experiment.yaml` must be Anthropic's own model-id string (verify at
https://platform.claude.com/docs/en/about-claude/models at setup time).

**Pass-through mode** — via any OpenAI-compatible gateway, e.g. OpenRouter:
```
CLAUDE_BASE_URL=https://openrouter.ai/api/v1
CLAUDE_API_KEY=<your openrouter key>
```
Adopted 2026-08-17 because direct Anthropic console billing (Stripe)
repeatedly failed and, on one retry, requested an SSN/TIN unrelated to a
plain API-credit purchase. OpenRouter proxies the exact same Claude Sonnet
4.6 model with no per-token markup — you pay a one-time ~5.5% fee only on
credit top-ups (see https://openrouter.ai/pricing). `models.claude.model_id`
must then be OpenRouter's slug, `anthropic/claude-sonnet-4.6` (confirmed at
https://openrouter.ai/anthropic/claude-sonnet-4.6/api), NOT Anthropic's
native model-id — the two differ and are not interchangeable.

Pick **one** host for the *entire* Claude leg and pin its exact model-id
string — do not mix native and pass-through calls under the same "Claude"
label, and do not implement fallback across hosts (same "no automatic
best-model selection" rule as Llama, below). Document in the paper's
methodology section which host actually served the calls, since this is a
real deviation from calling Anthropic's API directly and must be disclosed.

## Llama host setup (Llama 3.3 70B Instruct)

`models/llama_adapter.py` talks to **any OpenAI-compatible chat-completions
endpoint**, configured entirely through environment variables — no code
changes needed to switch hosts.

**Host history for this run (see `config/experiment.yaml` `models.llama`
comment for full detail):** started on Groq's free tier, which hit a hard
100,000-tokens/day cap after 228/450 real calls, and by 2026-08-17 had also
fully removed the `llama-3.3-70b-versatile` model from its catalog entirely
(confirmed via `GET /openai/v1/models`) — so Groq is no longer usable at
all for this model. DeepInfra was the next candidate but requires a manual
account balance top-up. **Current host: OpenRouter**, the same
already-funded account used for Claude, which also serves this exact model:
```
LLAMA_BASE_URL=https://openrouter.ai/api/v1
LLAMA_API_KEY=<your openrouter key>
```
with `models.llama.model_id: "meta-llama/llama-3.3-70b-instruct"` in
`config/experiment.yaml`. Pricing confirmed 2026-08-17 via
`GET https://openrouter.ai/api/v1/models`: ~$0.10/M input + $0.32/M output
tokens — a full 450-call run costs on the order of a few cents.

Other possible hosts (same adapter, just change `.env`) if OpenRouter ever
becomes unavailable:
- Together AI — `https://api.together.xyz/v1`
- Fireworks — `https://api.fireworks.ai/inference/v1`
- DeepInfra — `https://api.deepinfra.com/v1/openai` (requires balance top-up)

Set `models.llama.model_id` to whatever exact model string that host's
account model list shows for Llama 3.3 70B Instruct (verify at setup time —
strings vary by host and can change).

Pick **one** host and pin its exact model-id string — do not implement
fallback across hosts; that would violate the "no automatic best-model
selection" rule from the project brief. The 229 real completions
previously obtained from Groq (2026-08-15/16) were moved to
`logs/raw_calls_archive/llama_groq_2026-08-15_16/` and are NOT reused or
counted as evidence — that model no longer exists on Groq, and mixing
serving backends under one "Llama" label would violate the identical-
protocol requirement. `runner/run_generation.py`'s `_is_valid_completed_call`
now checks the recorded `model_id` against the currently configured one
before treating any on-disk file as "already done", precisely to prevent
this kind of silent host-mixing on resume (see `tests/test_run_generation.py`).

## Local setup

```bash
cd rag_eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your real API keys
```

## Datasets (already downloaded, parsed, and sampled)

Both datasets are already downloaded into `data/raw/` (gitignored — you
will need to re-download them yourself on a fresh checkout; see
`data/dataset_manifest.json` for the exact repo URLs and commit hashes) and
both loaders (`data/rgb_loader.py`, `data/lost_in_the_middle_loader.py`)
are fully implemented against the real, inspected schemas — no stubs, no
`NotImplementedError`. `data/load_items.py` loads all 90 sampled items
(30 per failure mode, seed 42) in one call; `data/dataset_manifest.json`
records full provenance (repo URLs, commit hashes, file hashes, download
date, exact selected item IDs, and inclusion/exclusion rules) and can be
regenerated any time with:
```bash
python -m scripts.build_dataset_manifest
```

If you need to re-download the raw files:
- **RGB**: [github.com/chen700564/RGB](https://github.com/chen700564/RGB)
  @ `65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615` — copy `data/en_refine.json`
  into `data/raw/rgb/`.
- **Lost in the Middle**: [github.com/nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle)
  @ `29b8a6d042ce29abccee3db1a73171a107d7e6af` — copy
  `qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl.gz`
  into `data/raw/lost_in_the_middle/`.

## Known metric limitations (decision confirmed 2026-08-17)

**`exact_match` (`eval/metrics.py`) is defined as strict full-string equality**
between the normalized model output and the normalized gold answer
(`normalize_text(prediction) == normalize_text(reference)`). This is the
standard SQuAD/NQ-style definition, kept as-is (not relaxed to a substring
check) so the protocol stays identical, simple, and pre-specified across all
three models — no metric changes mid-study.

The known consequence: all three target models answer in full sentences
(e.g. "Based on the available context, ... **Kentucky**.") rather than bare
short-answer spans (e.g. "Kentucky"), so `exact_match` reads as ~0% across
essentially every condition, including clean/evidence-present conditions
where the model is factually correct. **This is expected behavior of the
strict-EM definition given generative, non-extractive answers — it is not a
bug, and it is not evidence that the models are failing at high rates.**
`token_f1` is the metric that carries real signal in this study and should
be treated as primary for correctness comparisons; `exact_match` is reported
alongside it for completeness/reproducibility but should not be
over-interpreted on its own. This must be stated explicitly as a limitation
in the paper's Evaluation Protocol and Limitations sections — do not present
`exact_match` results without this caveat.

## Dry-run verification (no API calls, no cost)

This is the only mode you should run right now. It uses the REAL 90 sampled
items and REAL embedding-based retrieval (sentence-transformers +
FlatIndex) — only the generator call is mocked.

```bash
python -m runner.run_generation --dry-run
python -m eval.run_eval --dry-run
```

Verified output in this environment (2026-08-06):
```
[DRY RUN] 90 REAL sampled dataset items (RGB + Lost in the Middle — not synthetic fixtures), MockAdapter only — no paid API calls, no cost.
[DRY RUN] Done. 210 retrieval calls, 1350 mock generation calls written under logs/dry_run/ (all is_mock=True).
Wrote 1350 rows to logs/dry_run/results_aggregated_dryrun.csv
Wrote 270 rows to logs/dry_run/results_ablation_sensitivity_dryrun.csv
```
210 retrieval calls = one per (item, condition): 30×2 (blindness) + 30×2
(noise) + 30×3 (evidence ignorance) = 210. 1350 mock generation calls =
trials × 3 models summed over conditions/items: blindness 30×(1+3)×3=360,
noise 30×(1+3)×3=360, evidence ignorance 30×(1+3+3)×3=630 → 1350. Every
raw-call row's `answer_text` reads
`[MOCK OUTPUT - NOT A REAL MODEL RESPONSE - DRY RUN ONLY]` and `is_mock` is
`True` — this output must never be copied into `logs/raw_calls/` or cited
anywhere as a result. Unit tests: `python -m unittest discover tests`
(34 tests, all passing as of this delivery).

## What real runs will require and cost

Not part of this Phase 2 delivery, for reference only: the Phase 1 audit
estimated total external API spend for a run of this size at roughly
$10–$20 across all three models combined (input/output token pricing per
provider), which is not the binding constraint — your time wiring up the
two dataset loaders and confirming the checklist below is.

## Local setup checklist — complete all of this before a real run

Datasets, loaders, retrieval, logging, trial wiring, and all three
model IDs are already done. What's left is your API keys and a final
decision sign-off:

1. [x] Download RGB and Lost in the Middle raw files into
       `data/raw/rgb/` and `data/raw/lost_in_the_middle/`. Done — see
       `data/dataset_manifest.json` for provenance.
2. [x] Implement the parsing logic in `data/rgb_loader.py` and
       `data/lost_in_the_middle_loader.py`, sampling exactly 30 items per
       mode with `seed=42`. Done — 90/90 items load and pass validation.
3. [x] Pin the exact Claude Sonnet 4.6 model-id string in
       `config/experiment.yaml -> models.claude.model_id`. Done (2026-08-14):
       `claude-sonnet-4-6`, confirmed against
       [platform.claude.com/docs/en/about-claude/models/model-ids-and-versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions).
       Re-verify at setup time in case Anthropic changes this later.
4. [x] Choose your Llama 3.3 70B host and pin the exact `models.llama.model_id`
       string for that host. Done (2026-08-14): defaulted to **DeepInfra**,
       `meta-llama/Llama-3.3-70B-Instruct` (the standard, non-Turbo/non-quantized
       variant, matching the paper's stated model name), confirmed against
       [deepinfra.com/meta-llama/Llama-3.3-70B-Instruct/api](https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct/api).
       You still need to set `LLAMA_BASE_URL=https://api.deepinfra.com/v1/openai`
       and `LLAMA_API_KEY` in `.env`. If you switch hosts (Together/Groq/Fireworks),
       update `models.llama.model_id` too — the string differs per host.
5. [x] Verify the GPT-4.1 output-length parameter name against current
       OpenAI docs. Done (2026-08-14): GPT-4.1 requires `max_completion_tokens`,
       not `max_tokens` (confirmed against OpenAI's official Help Center article
       ["Controlling the length of OpenAI model responses"](https://help.openai.com/en/articles/5072518)) —
       `models/openai_adapter.py` has been updated accordingly.
6. [x] Trial counts confirmed (2026-08-14): `clean_condition: 1`,
       `stress_condition: 3`. No change from the provisional Phase 1 value.
7. [x] Prompt template confirmed as-is (2026-08-14) — no changes to
       `config/prompt_template.txt`.
8. [x] Faithfulness/groundedness scope confirmed (2026-08-14): **proxy
       metrics only** for this v2 (exact match, token F1, retrieval
       recall/precision@k/nDCG/gold_rank, ablation sensitivity). LLM-as-judge
       explicitly deferred, not implemented — do not claim a faithfulness
       result exists in this version.
9. [x] Manual error-analysis sample size confirmed (2026-08-14): 30 cases
       (10 per failure mode). Assistant drafts labels per
       `error_analysis/labeling_rubric.md` from the real failed cases;
       final labels require the paper author's review and sign-off before
       use in the paper.
10. [x] Real-run guard removed (2026-08-14, explicit user approval) —
        `runner/run_generation.py main()` now calls `run_real_mode(config)`
        directly. It still hard-fails with a clear `RuntimeError` (not a
        silent fallback) if any of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
        or `LLAMA_API_KEY` is missing — see `models/anthropic_adapter.py` /
        `models/openai_adapter.py` / `models/llama_adapter.py`.
11. [ ] **The only step left that requires you**: `cp .env.example .env`
        and fill in real `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
        `LLAMA_API_KEY` (never commit `.env`) — `load_dotenv()` is already
        wired into `runner/run_generation.py main()`.
12. [ ] `pip install -r requirements.txt`, then run **on your own
        machine** (real paid API calls must not run from Computer, per
        project rules):
        ```bash
        python -m runner.run_generation   # real calls to all 3 models
        python -m eval.run_eval           # builds results_aggregated.csv
                                           # and results_ablation_sensitivity.csv
        ```
        Estimated cost across all 3 models at the confirmed design (90
        items, 450 calls/model): roughly $3–$10 total, based on
        [Anthropic](https://www.finout.io/blog/anthropic-api-pricing),
        [OpenAI](https://developers.openai.com/api/docs/models/gpt-4.1), and
        [DeepInfra](https://www.requesty.ai/models/deepinfra/meta-llama-llama-3.3-70b-instruct)
        pricing checked 2026-08-14 and real context sizes from the Phase 3A
        dry-run retrieval logs. Send back `logs/raw_calls/`,
        `logs/retrieval_log.jsonl`, and the two `results_*.csv` files
        (or sync them into the project repo) for Phase 3D.

Nothing past this checklist — real runs, analysis, and error labeling — is
in scope for this delivery. Retrieval recall/precision@k/nDCG/gold_rank are
implemented and wired; faithfulness/groundedness/LLM-as-judge is explicitly
NOT implemented, per the project's integrity rules.
