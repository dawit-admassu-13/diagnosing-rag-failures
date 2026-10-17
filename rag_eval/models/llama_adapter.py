"""
Adapter for Llama 3.3 70B Instruct via any OpenAI-compatible chat-completions
host (Deepinfra, Together, Groq, Fireworks all work — see README "Llama
host setup" for exact base URLs and model-id strings per host).

Requires: pip install openai (used here only as an OpenAI-compatible HTTP
client — no OpenAI account or key is used by this adapter).
Requires env vars: LLAMA_BASE_URL, LLAMA_API_KEY (see .env.example).
Requires: config/experiment.yaml -> models.llama.model_id pinned to the
exact string your chosen host expects for Llama 3.3 70B Instruct.

This adapter makes a REAL, BILLED API call every time .generate() runs. It
is never invoked by --dry-run (see models/mock_adapter.py instead). No
automatic host/model fallback is implemented — if your configured host or
model_id is wrong, this raises rather than silently trying another one.
"""
from __future__ import annotations
import os

from .base import ModelAdapter
from ._proxy_http import build_httpx_client
from ._retry import call_with_retry


class LlamaAdapter(ModelAdapter):
    provider = "llama_openai_compatible"

    def __init__(self, model_key: str, model_id: str, temperature: float, max_output_tokens: int,
                 base_url_env: str = "LLAMA_BASE_URL", api_key_env: str = "LLAMA_API_KEY"):
        super().__init__(model_key, model_id, temperature, max_output_tokens)
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # local import: only required for real calls
            base_url = os.environ.get(self.base_url_env)
            api_key = os.environ.get(self.api_key_env)
            if not base_url or not api_key:
                raise RuntimeError(
                    f"Missing {self.base_url_env} and/or {self.api_key_env}. "
                    "Copy .env.example to .env and set them before real runs."
                )
            http_client = build_httpx_client()
            kwargs = {"base_url": base_url, "api_key": api_key}
            if http_client is not None:
                kwargs["http_client"] = http_client
            self._client = OpenAI(**kwargs)
        return self._client

    def _call(self, prompt: str):
        raw_request = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
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
