"""Redacts secrets (passwords, API keys, tokens) before a query is searched, logged or sent to the LLM."""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    # "password is hunter2", "api_key=abc123", "token: xyz"
    re.compile(
        r"(?i)\b(password|passwd|pwd|passcode|secret|api[\s_-]?key|access[\s_-]?key|token|client[\s_-]?secret)"
        r"(\s*(?:is|=|:)\s*)(\S+)"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{8,}=*"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),  # Google API key
    re.compile(r"\bAQ\.[0-9A-Za-z_\-]{30,}"),  # Google API key (newer format)
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # "sk-" style secret key
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"),  # GitHub tokens
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
    re.compile(r"(?<=://)[^\s:/@]+:[^\s@/]+(?=@)"),  # user:password@ in connection strings
]


def redact_secrets(text: str) -> tuple[str, bool]:
    """Return (redacted_text, found_secret)."""
    found = False

    def _sub(pattern: re.Pattern[str], value: str) -> str:
        nonlocal found

        def repl(match: re.Match[str]) -> str:
            nonlocal found
            found = True
            if pattern.groups >= 3:  # keep the label ("password is"), hide only the value
                return f"{match.group(1)}{match.group(2)}{REDACTED}"
            return REDACTED

        return pattern.sub(repl, value)

    for pattern in _SECRET_PATTERNS:
        text = _sub(pattern, text)
    return text, found
