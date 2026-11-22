"""Central configuration. Values can be overridden with environment variables or `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

KB_PATH = PROJECT_ROOT / "knowledge_base" / "technical_issues.json"

# Sentence Transformers model used for semantic search (small, CPU-friendly, 384-dim).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Gemini model used to phrase grounded answers. If it is overloaded (429/5xx) the engine tries
# the fallback models in order, then falls back to KB-only answers.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.6-flash").split(",") if m.strip()
]
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20"))
# The answer is a reformulation of retrieved text, so deep reasoning is not needed; "low" keeps latency down.
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")
# A model that just failed (overloaded / timed out) is skipped for this many seconds.
GEMINI_COOLDOWN_SECONDS = float(os.getenv("GEMINI_COOLDOWN_SECONDS", "120"))

# Cosine-similarity thresholds (embeddings are L2-normalised, so inner product == cosine).
# Below MIN_CONFIDENCE the assistant refuses to answer; at or above HIGH_CONFIDENCE it is a strong match.
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.50"))
HIGH_CONFIDENCE = float(os.getenv("HIGH_CONFIDENCE", "0.65"))

# How many knowledge-base entries are passed to the LLM as context.
TOP_K = int(os.getenv("TOP_K", "3"))


def gemini_api_key() -> str | None:
    """Return the Gemini key from the environment, or None when the user has not provided one."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None
