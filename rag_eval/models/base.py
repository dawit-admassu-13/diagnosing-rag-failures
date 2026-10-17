"""
Common interface every model adapter must implement.

Design rules (do not violate — see project integrity rules):
  - No adapter may silently fall back to a different model.
  - No adapter, and no code that instantiates adapters, may implement an
    automatic "best model" selector.
  - Every real call must return enough metadata to reconstruct exactly
    what was sent and received: raw request, raw response, token usage,
    latency, UTC timestamp, and the exact model-id string used.
  - On failure, adapters must return the error verbatim — never swallow
    it, never substitute a fabricated/simulated answer.
"""
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class GenerationResult:
    model_key: str            # config key, e.g. "claude", "gpt", "llama"
    model_id: str              # exact API model identifier used for the call
    provider: str
    prompt: str
    text: str                  # generated text ("" on error)
    raw_request: dict
    raw_response: Any          # raw provider response (json-able) or None on error
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: float
    timestamp_utc: str
    temperature: float
    max_output_tokens: int
    error: Optional[str] = None
    is_mock: bool = False

    def to_json_dict(self) -> dict:
        return dict(self.__dict__)


class ModelAdapter(ABC):
    provider: str = "unset"

    def __init__(self, model_key: str, model_id: str, temperature: float, max_output_tokens: int):
        self.model_key = model_key
        self.model_id = model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @abstractmethod
    def _call(self, prompt: str):
        """
        Perform the actual API call.
        Must return a 6-tuple:
          (text, raw_request, raw_response, input_tokens, output_tokens, error)
        On failure: ("", raw_request, None, None, None, error_message_str).
        """
        raise NotImplementedError

    def generate(self, prompt: str) -> GenerationResult:
        start = time.monotonic()
        text, raw_request, raw_response, in_tok, out_tok, error = self._call(prompt)
        latency_ms = (time.monotonic() - start) * 1000.0
        return GenerationResult(
            model_key=self.model_key,
            model_id=self.model_id,
            provider=self.provider,
            prompt=prompt,
            text=text,
            raw_request=raw_request,
            raw_response=raw_response,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            error=error,
            is_mock=False,
        )
