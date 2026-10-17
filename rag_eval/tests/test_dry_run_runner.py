"""
End-to-end smoke test for the dry-run pipeline shape: retrieval ->
render_prompt -> MockAdapter -> raw-call + retrieval-log files on disk.

Uses the tiny synthetic fixtures (tests/fixtures.py) and a FakeEmbedder
(deterministic hash-based vectors, no sentence-transformers model download,
no network) rather than the real 90-item dataset + real MiniLM model — that
full-scale, real-embedder verification is what
`python -m runner.run_generation --dry-run` itself performs (see README
"Verification" and deliverable C), and is deliberately kept out of the unit
test suite so `python -m unittest discover tests` stays fast and offline.
This test instead checks that the plumbing (retrieval logging, trial
counts, MockAdapter labeling, self-contained raw-call fields) is wired
correctly, independent of which embedder or dataset is used.
"""
from __future__ import annotations
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from conditions.builders import CONDITIONS_BY_MODE
from runner.run_generation import build_mock_adapters, trials_map, load_prompt_template, run_with_retrieval
from tests.fixtures import build_fixture_items


class FakeEmbedder:
    """Deterministic hash-based stand-in for retrieval.embeddings.Embedder — no model download."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            v = rng.rand(8).astype(np.float64)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.array(vectors)


FAKE_CONFIG = {
    "generation": {"prompt_template_path": "config/prompt_template.txt", "temperature": 0.0, "max_output_tokens": 200},
    "models": {
        "claude": {"model_id": "fake-claude"},
        "gpt": {"model_id": "fake-gpt"},
        "llama": {"model_id": "fake-llama"},
    },
    "trials": {"clean_condition": 1, "stress_condition": 2},
    "retrieval": {"top_k": 3},
}


class TestDryRunRunnerShape(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.out_dir = self.tmp_dir / "raw"
        self.retrieval_dir = self.tmp_dir / "retrieval"
        self.ledger_path = self.tmp_dir / "retrieval_log.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_with_retrieval_produces_expected_files(self):
        items = build_fixture_items()
        adapters = build_mock_adapters(FAKE_CONFIG)
        embedder = FakeEmbedder()
        template = load_prompt_template(FAKE_CONFIG["generation"]["prompt_template_path"])
        trials = trials_map(FAKE_CONFIG)
        top_k = FAKE_CONFIG["retrieval"]["top_k"]

        n_calls, n_retrievals = run_with_retrieval(
            items, adapters, embedder, template, self.out_dir, self.retrieval_dir, self.ledger_path,
            trials, top_k, skip_existing=False,
        )

        # Expected retrieval calls: one per (item, condition) across 3 fixture items.
        expected_retrievals = sum(len(CONDITIONS_BY_MODE[item.failure_mode]) for item in items)
        self.assertEqual(n_retrievals, expected_retrievals)

        # Expected generation calls: trials(condition) x 3 models, summed over conditions/items.
        expected_calls = 0
        for item in items:
            for condition in CONDITIONS_BY_MODE[item.failure_mode]:
                expected_calls += trials[condition] * len(adapters)
        self.assertEqual(n_calls, expected_calls)

        # Retrieval ledger has one line per (item, condition).
        with open(self.ledger_path) as f:
            ledger_lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(ledger_lines), expected_retrievals)

        # Every raw call file exists, is stamped is_mock, and is self-contained.
        for item in items:
            for condition in CONDITIONS_BY_MODE[item.failure_mode]:
                for model_key in adapters:
                    for trial in range(1, trials[condition] + 1):
                        call_path = self.out_dir / model_key / condition / item.item_id / f"trial_{trial}.json"
                        self.assertTrue(call_path.exists(), f"missing {call_path}")
                        with open(call_path) as f:
                            call = json.load(f)
                        self.assertTrue(call["is_mock"])
                        self.assertIn("MOCK OUTPUT", call["text"])
                        self.assertEqual(call["item_id"], item.item_id)
                        self.assertEqual(call["source_dataset"], item.source_dataset)
                        self.assertEqual(call["failure_mode"], item.failure_mode)
                        self.assertEqual(call["reference_answer"], item.reference_answer)
                        self.assertTrue(Path(call["retrieval_log_path"]).exists())

    def test_top_k_is_respected(self):
        items = build_fixture_items()
        adapters = build_mock_adapters(FAKE_CONFIG)
        embedder = FakeEmbedder()
        template = load_prompt_template(FAKE_CONFIG["generation"]["prompt_template_path"])
        trials = trials_map(FAKE_CONFIG)
        top_k = 1

        run_with_retrieval(
            items, adapters, embedder, template, self.out_dir, self.retrieval_dir, self.ledger_path,
            trials, top_k, skip_existing=False,
        )
        with open(self.ledger_path) as f:
            for line in f:
                record = json.loads(line)
                if record["condition"] in ("clean", "blind", "noisy"):
                    self.assertLessEqual(len(record["retrieved_passages"]), top_k)

    def test_skip_existing_avoids_duplicate_calls(self):
        items = build_fixture_items()[:1]
        adapters = build_mock_adapters(FAKE_CONFIG)
        embedder = FakeEmbedder()
        template = load_prompt_template(FAKE_CONFIG["generation"]["prompt_template_path"])
        trials = trials_map(FAKE_CONFIG)
        top_k = FAKE_CONFIG["retrieval"]["top_k"]

        n_calls_first, _ = run_with_retrieval(
            items, adapters, embedder, template, self.out_dir, self.retrieval_dir, self.ledger_path,
            trials, top_k, skip_existing=True,
        )
        n_calls_second, _ = run_with_retrieval(
            items, adapters, embedder, template, self.out_dir, self.retrieval_dir, self.ledger_path,
            trials, top_k, skip_existing=True,
        )
        self.assertGreater(n_calls_first, 0)
        self.assertEqual(n_calls_second, 0, "second run with skip_existing=True should make zero new calls")

    def test_skip_existing_retries_stale_error_files(self):
        """
        Regression test for a real bug found 2026-08-15: skip_existing used to
        check only file EXISTENCE, so a stale raw-call file left over from a
        failed attempt (missing credentials, wrong model id, network error --
        i.e. an `error` field set and empty `text`) was skipped forever on
        every future resume instead of being retried once working credentials
        were available. This silently froze that (model, condition, item,
        trial) slot at an error state. A resumable run must retry any
        existing file that does not represent a genuine successful call.
        """
        items = build_fixture_items()[:1]
        adapters = build_mock_adapters(FAKE_CONFIG)
        embedder = FakeEmbedder()
        template = load_prompt_template(FAKE_CONFIG["generation"]["prompt_template_path"])
        trials = trials_map(FAKE_CONFIG)
        top_k = FAKE_CONFIG["retrieval"]["top_k"]

        item = items[0]
        condition = CONDITIONS_BY_MODE[item.failure_mode][0]
        model_key = next(iter(adapters))
        stale_path = self.out_dir / model_key / condition / item.item_id / "trial_1.json"
        stale_path.parent.mkdir(parents=True, exist_ok=True)
        stale_path.write_text(json.dumps({
            "model_key": model_key,
            "text": "",
            "error": "RuntimeError('Missing API_KEY.')",
            "is_mock": False,
        }))

        n_calls, _ = run_with_retrieval(
            items, adapters, embedder, template, self.out_dir, self.retrieval_dir, self.ledger_path,
            trials, top_k, skip_existing=True,
        )

        self.assertGreater(n_calls, 0, "a stale error file must be retried, not silently skipped forever")
        written = json.loads(stale_path.read_text())
        self.assertFalse(written.get("error"), "the retried call should overwrite the stale error record")


if __name__ == "__main__":
    unittest.main()
