"""
Adapter for GPT-4.1 via the OpenAI API.

Requires: pip install openai
Requires env var: OPENAI_API_KEY (see .env.example)
Requires: config/experiment.yaml -> models.gpt.model_id (default "gpt-4.1";
confirm this is still the exact current identifier at
https://platform.openai.com/docs/models at setup time).

This adapter makes a REAL, BILLED API call every time .generate() runs. It
is never invoked by --dry-run (see models/mock_adapter.py instead).

Output-length parameter (confirmed 2026-08-14 against OpenAI's official
Help Center article "Controlling the length of OpenAI model responses",
https://help.openai.com/en/articles/5072518, and the Chat Completions API
reference at https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create):
GPT-4.1 requires `max_completion_tokens`; the older `max_tokens` field is
deprecated and GPT-4.1 rejects it outright ("Unsupported parameter:
'max_tokens' is not supported with this model. Use 'max_completion_tokens'
instead."). Re-verify against current docs at setup time in case this
changes again before you run.
"""
from __future__ import annotations
import os

from .base import ModelAdapter
from ._proxy_http import build_httpx_client
from ._retry import call_with_retry


class OpenAIAdapter(ModelAdapter):
    provider = "openai"

    def __init__(self, model_key: str, model_id: str, temperature: float, max_output_tokens: int,
                 api_key_env: str = "OPENAI_API_KEY"):
        super().__init__(model_key, model_id, temperature, max_output_tokens)
        self.api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # local import: only required for real calls
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing {self.api_key_env}. Copy .env.example to .env and set it before real runs."
                )
            http_client = build_httpx_client()
            kwargs = {"api_key": api_key}
            if http_client is not None:
                kwargs["http_client"] = http_client
            self._client = OpenAI(**kwargs)
        return self._client

    def _call(self, prompt: str):
        raw_request = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_output_tokens,  # GPT-4.1 requires this name, not max_tokens — see module docstring
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            client = self._get_client()
            response, attempts = call_with_retry(lambda: client.chat.completions.create(**raw_request))
            raw_request["retry_attempts"] = attempts
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) if usage else None
            out_tok = getattr(usage, "completion_tokens", None) if usage else None
            raw_response = response.model_dump() if hasattr(response, "model_dump") else str(response)
            return text, raw_request, raw_response, in_tok, out_tok, None
        except Exception as e:  # noqa: BLE001 — log the real provider error, never swallow it
            return "", raw_request, None, None, None, repr(e)
