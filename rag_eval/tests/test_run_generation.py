"""
Regression tests for `runner.run_generation._is_valid_completed_call`.

Covers a real incident (2026-08-17): the Llama config was switched hosts
(Groq -> DeepInfra -> OpenRouter) after 229 real Groq-served calls already
existed on disk under the old model_id. Before this fix, skip_existing only
checked "no error + non-empty text" and would have silently treated those
stale, different-host completions as "done" under the newly configured
model_id -- mixing two serving backends under one model label, violating
the project's no-host-mixing rule. These tests pin the fix: a raw-call file
is only a valid "already done" match when its recorded model_id equals the
model_id currently configured for that model key.
"""
from __future__ import annotations
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from runner.run_generation import _is_valid_completed_call


class TestIsValidCompletedCall(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write(self, name: str, payload: dict) -> Path:
        path = self.tmp_dir / name
        path.write_text(json.dumps(payload))
        return path

    def test_matching_model_id_no_error_nonempty_text_is_valid(self):
        path = self._write(
            "ok.json",
            {"model_id": "meta-llama/llama-3.3-70b-instruct", "text": "Paris", "error": None},
        )
        self.assertTrue(_is_valid_completed_call(path, "meta-llama/llama-3.3-70b-instruct"))

    def test_different_model_id_same_host_family_is_rejected(self):
        """Simulates the real incident: an old Groq-served completion on disk,
        but the currently configured model_id points elsewhere (OpenRouter)."""
        path = self._write(
            "stale_groq.json",
            {"model_id": "llama-3.3-70b-versatile", "text": "Paris", "error": None},
        )
        self.assertFalse(_is_valid_completed_call(path, "meta-llama/llama-3.3-70b-instruct"))

    def test_error_present_is_rejected_even_if_model_id_matches(self):
        path = self._write(
            "errored.json",
            {"model_id": "meta-llama/llama-3.3-70b-instruct", "text": "", "error": "rate_limited"},
        )
        self.assertFalse(_is_valid_completed_call(path, "meta-llama/llama-3.3-70b-instruct"))

    def test_empty_text_is_rejected_even_if_model_id_matches(self):
        path = self._write(
            "empty_text.json",
            {"model_id": "meta-llama/llama-3.3-70b-instruct", "text": "   ", "error": None},
        )
        self.assertFalse(_is_valid_completed_call(path, "meta-llama/llama-3.3-70b-instruct"))

    def test_missing_model_id_field_is_rejected(self):
        path = self._write("no_model_id.json", {"text": "Paris", "error": None})
        self.assertFalse(_is_valid_completed_call(path, "meta-llama/llama-3.3-70b-instruct"))

    def test_corrupt_json_is_rejected(self):
        path = self.tmp_dir / "corrupt.json"
        path.write_text("{not valid json")
        self.assertFalse(_is_valid_completed_call(path, "meta-llama/llama-3.3-70b-instruct"))

    def test_missing_file_is_rejected(self):
        path = self.tmp_dir / "does_not_exist.json"
        self.assertFalse(_is_valid_completed_call(path, "meta-llama/llama-3.3-70b-instruct"))


if __name__ == "__main__":
    unittest.main()
