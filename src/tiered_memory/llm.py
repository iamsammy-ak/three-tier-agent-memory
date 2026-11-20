"""Optional LLM access through OpenRouter, used only when ``OPENROUTER_API_KEY`` is set.

Without a key every component falls back to its offline stand-in (TextRank compression, rule-based fact extraction,
heuristic importance), so the library, the tests and the benchmark never need the network.

The client tries a list of free models in order, skips a busy model instead of retrying it, caches every reply on
disk keyed by a hash of the request, and validates JSON replies against a Pydantic schema with one repair round-trip.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = (
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.2-flash:free",
    "google/gemma-4-31b-it:free",
)
DEFAULT_CACHE_DIR = Path(".cache") / "llm"
T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


class LLMError(RuntimeError):
    """No configured model produced a usable answer."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    models: tuple[str, ...]


def load_settings() -> LLMSettings:
    """Read ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODELS`` from the environment or a local ``.env`` file."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # python-dotenv is a core dependency, but stay usable without it
        pass
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip() or None
    raw = (os.environ.get("OPENROUTER_MODELS") or "").strip()
    models = tuple(m.strip() for m in raw.split(",") if m.strip()) or DEFAULT_MODELS
    return LLMSettings(api_key=key, models=models)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (handles code fences, prose and reasoning tags)."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in the model reply") from None
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("the model reply is JSON but not an object")
    return value


class LLM(Protocol):
    """Minimal provider interface: free text plus schema-validated JSON."""

    name: str

    def complete(
        self, messages: Messages, *, json_mode: bool = False, temperature: float = 0.2, max_tokens: int = 1200
    ) -> str: ...

    def complete_json(self, messages: Messages, schema: type[T], *, temperature: float = 0.1) -> T: ...


class BaseLLM:
    """Shared JSON handling: ask for a schema, validate, and repair once if the reply is unusable."""

    name = "base"

    def complete(
        self, messages: Messages, *, json_mode: bool = False, temperature: float = 0.2, max_tokens: int = 1200
    ) -> str:
        raise NotImplementedError

    def forget(self, messages: Messages, *, json_mode: bool, temperature: float) -> None:
        """Drop a cached reply that turned out to be unusable (no-op unless the provider caches)."""

    def complete_json(self, messages: Messages, schema: type[T], *, temperature: float = 0.1) -> T:
        hint = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        system = {
            "role": "system",
            "content": "Reply with one JSON object only, no prose and no code fences. "
            f"It must validate against this JSON Schema:\n{hint}",
        }
        first = [system, *messages]
        raw = self.complete(first, json_mode=True, temperature=temperature)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as error:
            self.forget(first, json_mode=True, temperature=temperature)
            repair = [
                *first,
                {"role": "assistant", "content": raw[:4000]},
                {
                    "role": "user",
                    "content": f"That reply could not be used ({str(error)[:400]}). Return the corrected JSON only.",
                },
            ]
            raw = self.complete(repair, json_mode=True, temperature=0.0)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                self.forget(repair, json_mode=True, temperature=0.0)
                raise LLMError(f"the model reply did not match the expected schema: {exc}") from exc


class OpenRouterLLM(BaseLLM):
    """OpenRouter's OpenAI-compatible API with model fallback and an on-disk reply cache."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        timeout: float = 60.0,
    ) -> None:
        self.settings = settings or load_settings()
        if not self.settings.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.settings.api_key,
            default_headers={"X-Title": "tiered-memory"},
            timeout=timeout,
            max_retries=0,
        )
        self.cache_dir = cache_dir
        self.name = f"openrouter:{self.settings.models[0]}"
        self.last_errors: list[str] = []

    def _cache_file(self, messages: Messages, json_mode: bool, temperature: float) -> Path | None:
        if self.cache_dir is None:
            return None
        key = {"models": list(self.settings.models), "messages": messages, "json": json_mode, "t": temperature}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:40]
        return self.cache_dir / f"{digest}.json"

    def complete(
        self, messages: Messages, *, json_mode: bool = False, temperature: float = 0.2, max_tokens: int = 1200
    ) -> str:
        path = self._cache_file(messages, json_mode, temperature)
        if path and path.exists():
            return str(json.loads(path.read_text(encoding="utf-8"))["content"])
        content, model = self._call_with_fallback(messages, json_mode, temperature, max_tokens)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model, "content": content}), encoding="utf-8")
        return content

    def forget(self, messages: Messages, *, json_mode: bool, temperature: float) -> None:
        path = self._cache_file(messages, json_mode, temperature)
        if path and path.exists():
            path.unlink()

    def _call_with_fallback(
        self, messages: Messages, json_mode: bool, temperature: float, max_tokens: int
    ) -> tuple[str, str]:
        """Try each model once; busy models (HTTP 429, 5xx, timeouts) get one more pass after a short pause."""
        from openai import APIConnectionError, APIStatusError, RateLimitError

        self.last_errors = []
        retryable: list[str] = []
        for attempt in range(2):
            candidates = list(self.settings.models) if attempt == 0 else retryable
            if attempt == 1:
                if not retryable:
                    break
                time.sleep(5)
            retryable = []
            for model in candidates:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    response = self._client.chat.completions.create(**kwargs)
                except RateLimitError as exc:
                    if "per-day" in str(exc) or "per day" in str(exc):
                        raise LLMError("OpenRouter's daily free-model limit is used up") from exc
                    self.last_errors.append(f"{model}: busy (HTTP 429)")
                    retryable.append(model)
                    continue
                except APIConnectionError as exc:  # includes timeouts
                    self.last_errors.append(f"{model}: {type(exc).__name__}")
                    retryable.append(model)
                    continue
                except APIStatusError as exc:
                    if exc.status_code == 401:
                        raise LLMError("OpenRouter rejected the API key (HTTP 401)") from exc
                    self.last_errors.append(f"{model}: HTTP {exc.status_code}")
                    if exc.status_code >= 500:
                        retryable.append(model)
                    continue
                choice = response.choices[0] if response.choices else None
                content = ((choice.message.content if choice else None) or "").strip()
                if content:
                    return content, model
                self.last_errors.append(f"{model}: empty reply")
        raise LLMError("no model returned an answer: " + "; ".join(self.last_errors[-6:]))


def get_llm(offline: bool = False) -> LLM | None:
    """OpenRouter client when a key is configured and ``offline`` is False, otherwise None (offline stand-ins)."""
    if offline:
        return None
    settings = load_settings()
    if not settings.api_key:
        return None
    try:
        return OpenRouterLLM(settings)
    except ImportError:  # the optional ``openai`` package is not installed
        return None
