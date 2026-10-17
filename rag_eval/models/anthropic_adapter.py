"""
Adapter for Claude Sonnet 4.6, either via the native Anthropic API or via any
OpenAI-compatible pass-through host (e.g. OpenRouter), selected purely by
which env vars are set — no code change needed to switch. See README
"Claude host setup" for details on when/why OpenRouter was added.

Native mode (default, no base_url_env configured):
  Requires: pip install anthropic
  Requires env var: ANTHROPIC_API_KEY (see .env.example)

Pass-through mode (base_url_env configured, e.g. CLAUDE_BASE_URL set):
  Requires: pip install openai (used only as a generic OpenAI-compatible
  HTTP client — this does NOT mean OpenAI's own models are being called)
  Requires env vars: <base_url_env>, <api_key_env>
  model_id must be the exact string the pass-through host expects for
  Claude Sonnet 4.6 (e.g. "anthropic/claude-sonnet-4.6" on OpenRouter —
  NOT the native Anthropic model-id string, which differs by host).

Requires: config/experiment.yaml -> models.claude.model_id pinned to the
exact current model-id string for whichever host is active (check
https://platform.claude.com/docs/en/about-claude/models for native Anthropic,
or the host's own model catalog for pass-through mode, at setup time).

This adapter makes a REAL, BILLED API call every time .generate() runs. It
is never invoked by --dry-run (see models/mock_adapter.py instead). Required
packages/keys are only required/imported at call time, so importing this
module for inspection never requires either.
"""
from __future__ import annotations
import os

from .base import ModelAdapter
from ._proxy_http import build_httpx_client
from ._retry import call_with_retry


class AnthropicAdapter(ModelAdapter):
    provider = "anthropic"

    def __init__(self, model_key: str, model_id: str, temperature: float, max_output_tokens: int,
                 api_key_env: str = "ANTHROPIC_API_KEY", base_url_env: str | None = None):
        super().__init__(model_key, model_id, temperature, max_output_tokens)
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self._client = None
        self._pass_through = False

    def _get_client(self):
        if self._client is None:
            base_url = os.environ.get(self.base_url_env) if self.base_url_env else None
            api_key = os.environ.get(self.api_key_env)
            if base_url:
                # Pass-through mode: any OpenAI-compatible host (e.g. OpenRouter).
                from openai import OpenAI  # local import: only required for real calls
                if not api_key:
                    raise RuntimeError(
                        f"Missing {self.api_key_env}. Copy .env.example to .env and set it before real runs."
                    )
                http_client = build_httpx_client()
                kwargs = {"base_url": base_url, "api_key": api_key}
                if http_client is not None:
                    kwargs["http_client"] = http_client
                self._client = OpenAI(**kwargs)
                self._pass_through = True
            else:
                # Native mode: direct Anthropic API.
                import anthropic  # local import: only required for real calls
                if not api_key:
                    raise RuntimeError(
                        f"Missing {self.api_key_env}. Copy .env.example to .env and set it before real runs."
                    )
                http_client = build_httpx_client()
                kwargs = {"api_key": api_key}
                if http_client is not None:
                    kwargs["http_client"] = http_client
                self._client = anthropic.Anthropic(**kwargs)
                self._pass_through = False
        return self._client

    def _call(self, prompt: str):
        # Peek at base_url_env without instantiating a client yet, so both
        # branches build their own raw_request dict before any network call
        # or key-lookup error — matching the other two adapters' convention
        # of always returning a loggable raw_request even on failure.
        pass_through = bool(self.base_url_env and os.environ.get(self.base_url_env))
        if pass_through:
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
        else:
            raw_request = {
                "model": self.model_id,
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            try:
                client = self._get_client()
                response, attempts = call_with_retry(lambda: client.messages.create(**raw_request))
                raw_request["retry_attempts"] = attempts
                text = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                )
                usage = getattr(response, "usage", None)
                in_tok = getattr(usage, "input_tokens", None) if usage else None
                out_tok = getattr(usage, "output_tokens", None) if usage else None
                raw_response = response.model_dump() if hasattr(response, "model_dump") else str(response)
                return text, raw_request, raw_response, in_tok, out_tok, None
            except Exception as e:  # noqa: BLE001 — log the real provider error, never swallow it
                return "", raw_request, None, None, None, repr(e)
