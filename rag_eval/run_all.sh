#!/usr/bin/env bash
# One-command runner: installs deps, runs all 3 real models, builds results CSVs.
# Requires: .env already filled in with real ANTHROPIC_API_KEY, OPENAI_API_KEY,
# LLAMA_API_KEY (copy .env.example to .env first if you haven't).
#
# Usage:
#   cd rag_eval
#   cp .env.example .env   # then edit .env and fill in your 3 real API keys
#   ./run_all.sh
#
# This makes REAL paid API calls to Claude Sonnet 4.6, GPT-4.1, and
# Llama 3.3 70B Instruct (via DeepInfra) — estimated $3-10 total across
# all three models for the full 90-item design. See README.md for details.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Run: cp .env.example .env, then fill in your real API keys."
  exit 1
fi

echo "== Checking required API keys are set in .env =="
set -a; source .env; set +a
missing=0
for var in ANTHROPIC_API_KEY OPENAI_API_KEY LLAMA_API_KEY; do
  if [ -z "${!var:-}" ]; then
    echo "  MISSING: $var"
    missing=1
  fi
done
if [ "$missing" -eq 1 ]; then
  echo "ERROR: fill in the missing key(s) above in .env before running."
  exit 1
fi
echo "  All 3 keys present."

echo "== Installing dependencies =="
pip install --quiet -r requirements.txt

echo "== Running real generation (Claude Sonnet 4.6, GPT-4.1, Llama 3.3 70B Instruct) =="
echo "   This makes real, billed API calls. Ctrl+C now if you want to stop."
sleep 3
python -m runner.run_generation

echo "== Building evaluation results (results_aggregated.csv, results_ablation_sensitivity.csv) =="
python -m eval.run_eval

echo ""
echo "== Done. Real results are in: =="
echo "   logs/raw_calls/                       (every raw model call, real)"
echo "   logs/retrieval_log.jsonl              (every retrieval, real)"
echo "   logs/results_aggregated.csv           (per-item metrics)"
echo "   logs/results_ablation_sensitivity.csv (condition-ablation deltas)"
echo ""
echo "Send/sync the logs/ folder back for Phase 3D evaluation and error analysis."
