"""Gemini integration: turns retrieved KB entries into a grounded troubleshooting answer.

Grounding is enforced in three layers:
1. The prompt only contains the retrieved KB entries and tells the model to use nothing else.
2. The model must return JSON where every step cites the KB id it came from.
3. `validate_answer` drops any step whose citation is not one of the retrieved entries.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an IT troubleshooting assistant for L1/L2 technical support engineers.

Strict rules:
1. Use ONLY the knowledge-base (KB) entries in the CONTEXT block. Do not add causes, commands,
   tools or fixes from your own general knowledge.
2. Every item in "steps" and "fixes" must be supported by one KB entry; put that entry's id in "source_id".
3. If none of the KB entries addresses the engineer's problem, set "sufficient_context" to false and
   return empty "steps" and "fixes". Do not guess.
4. Present steps as checks to perform, and fixes as actions to take only if a check confirms the cause.
   Never state that the issue is resolved.
5. Never ask for, repeat or suggest sharing passwords, API keys, tokens or other secrets.
6. Order steps the way an engineer should run them: quickest and most likely first. Maximum 7 steps and 4 fixes.
7. You may tailor wording to the engineer's description (e.g. mention "payment API" if they did), but not the facts.
8. "follow_up_questions": up to 3 short questions whose answers would narrow the problem down."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sufficient_context": {"type": "boolean"},
        "likely_issue": {"type": "string"},
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "source_id": {"type": "string"}},
                "required": ["text", "source_id"],
            },
        },
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "source_id": {"type": "string"}},
                "required": ["text", "source_id"],
            },
        },
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient_context", "likely_issue", "summary", "steps", "fixes", "follow_up_questions"],
}


@dataclass
class Step:
    text: str
    source_id: str


@dataclass
class GeneratedAnswer:
    likely_issue: str
    summary: str
    steps: list[Step]
    fixes: list[Step]
    follow_up_questions: list[str] = field(default_factory=list)
    sufficient_context: bool = True
    dropped_items: int = 0  # items removed because they cited an unknown source


def format_context(entries: list[dict[str, Any]]) -> str:
    """Render KB entries as compact JSON for the prompt (keywords are for search only)."""
    trimmed = [{k: v for k, v in e.items() if k != "keywords"} for e in entries]
    return json.dumps(trimmed, indent=2, ensure_ascii=False)


def build_prompt(query: str, entries: list[dict[str, Any]]) -> str:
    return (
        f"CONTEXT (knowledge-base entries, most relevant first):\n{format_context(entries)}\n\n"
        f"ENGINEER'S PROBLEM:\n{query}\n\n"
        "Answer using only the CONTEXT, following the rules."
    )


def validate_answer(raw: dict[str, Any], allowed_ids: set[str]) -> GeneratedAnswer:
    """Parse the model's JSON and drop anything that cites a KB id we did not provide."""
    dropped = 0

    def keep(items: list[dict[str, Any]] | None) -> list[Step]:
        nonlocal dropped
        kept: list[Step] = []
        for item in items or []:
            text = str(item.get("text", "")).strip()
            source = str(item.get("source_id", "")).strip()
            if text and source in allowed_ids:
                kept.append(Step(text=text, source_id=source))
            else:
                dropped += 1
        return kept

    steps = keep(raw.get("steps"))
    fixes = keep(raw.get("fixes"))
    return GeneratedAnswer(
        likely_issue=str(raw.get("likely_issue", "")).strip(),
        summary=str(raw.get("summary", "")).strip(),
        steps=steps,
        fixes=fixes,
        follow_up_questions=[str(q).strip() for q in raw.get("follow_up_questions", []) if str(q).strip()][:3],
        sufficient_context=bool(raw.get("sufficient_context", False)) and bool(steps),
        dropped_items=dropped,
    )


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiEngine:
    """Thin wrapper around the google-genai client with a model fallback chain."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = config.GEMINI_MODEL,
        fallback_models: list[str] | None = None,
        timeout_seconds: float = config.GEMINI_TIMEOUT_SECONDS,
    ):
        from google import genai
        from google.genai import types

        api_key = api_key or config.gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.models = [model] + [m for m in (fallback_models or config.GEMINI_FALLBACK_MODELS) if m != model]
        self.model = model  # primary model, shown in the UI
        self.last_model_used: str | None = None
        self._cooldown_until: dict[str, float] = {}  # model -> time.monotonic() when it may be tried again
        # Fail fast on one model and move to the next instead of letting the SDK retry for minutes.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def generate(self, query: str, entries: list[dict[str, Any]]) -> GeneratedAnswer:
        from google.genai import errors, types

        # Temperature is left at the model default (recommended for Gemini 3); grounding is enforced
        # by the prompt, the JSON schema and validate_answer() rather than by sampling settings.
        request_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level=config.GEMINI_THINKING_LEVEL) if config.GEMINI_THINKING_LEVEL else None,
        )
        prompt = build_prompt(query, entries)
        last_error: Exception | None = None
        for model in self._models_in_try_order():
            try:
                response = self._client.models.generate_content(model=model, contents=prompt, config=request_config)
            except errors.APIError as exc:
                last_error = exc
                if exc.code in RETRYABLE_STATUS:
                    logger.warning("Gemini model %s unavailable (%s), trying next model", model, exc.code)
                    self._cool_down(model)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 - network timeouts etc.: try the next model
                last_error = exc
                logger.warning("Gemini model %s failed (%s), trying next model", model, type(exc).__name__)
                self._cool_down(model)
                continue
            self._cooldown_until.pop(model, None)
            self.last_model_used = model
            raw = json.loads(response.text or "{}")
            return validate_answer(raw, {e["id"] for e in entries})
        raise RuntimeError(f"All Gemini models failed: {last_error}")

    def _cool_down(self, model: str) -> None:
        self._cooldown_until[model] = time.monotonic() + config.GEMINI_COOLDOWN_SECONDS

    def _models_in_try_order(self) -> list[str]:
        """Healthy models first (in configured order), then the ones still cooling down."""
        now = time.monotonic()
        healthy = [m for m in self.models if self._cooldown_until.get(m, 0) <= now]
        cooling = [m for m in self.models if m not in healthy]
        return healthy + cooling


def create_engine() -> GeminiEngine | None:
    """Return a Gemini engine when a key is configured, otherwise None (KB-only mode)."""
    if not config.gemini_api_key():
        return None
    try:
        return GeminiEngine()
    except Exception as exc:  # noqa: BLE001 - any SDK/config problem means we fall back to KB-only mode
        logger.warning("Gemini unavailable, using KB-only mode: %s", exc)
        return None
