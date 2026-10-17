"""
Mock adapter for dry-run pipeline testing ONLY.

Returns a fixed, clearly-labeled placeholder string instead of calling any
API — zero cost, zero network access. Never used for real experiments;
runner/run_generation.py only instantiates this when --dry-run is passed,
and dry-run output is written to logs/dry_run/ (never logs/raw_calls/), so
it can never be mistaken for real model output or cited as a result.
"""
from __future__ import annotations
from .base import ModelAdapter


class MockAdapter(ModelAdapter):
    provider = "mock"

    def _call(self, prompt: str):
        raw_request = {"model": self.model_id, "prompt": prompt, "mock": True}
        text = "[MOCK OUTPUT - NOT A REAL MODEL RESPONSE - DRY RUN ONLY]"
        raw_response = {"mock": True, "note": "No API call was made."}
        return text, raw_request, raw_response, len(prompt.split()), len(text.split()), None

    def generate(self, prompt: str):
        result = super().generate(prompt)
        result.is_mock = True
        return result
