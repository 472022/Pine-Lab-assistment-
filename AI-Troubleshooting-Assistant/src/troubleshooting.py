"""The RAG pipeline: question -> guardrails -> retrieval -> confidence check -> grounded answer -> report.

    1. Redact secrets from the question.
    2. Ask for more details if the question is too vague to search.
    3. Retrieve the closest KB entries (semantic search).
    4. Refuse ("insufficient information") if the best match is below the confidence threshold.
    5. Generate the answer with Gemini from the retrieved entries only, or compose it straight
       from the KB when Gemini is not configured / unavailable.
    6. Attach escalation advice, sources and confidence.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src import config
from src.ai_engine import GeminiEngine, GeneratedAnswer, Step, build_prompt
from src.escalation import EscalationSuggestion, suggest_escalation
from src.retriever import KnowledgeBaseRetriever, SearchResult
from src.security import redact_secrets

ANSWERED = "answered"
NEEDS_DETAILS = "needs_details"
UNKNOWN = "unknown"

# Words that carry no information about *what* is broken. A question made only of these is too vague.
_VAGUE_WORDS = {
    "a", "an", "the", "i", "im", "me", "my", "we", "our", "you", "your", "it", "its", "this", "that", "there",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "doesnt", "dont", "isnt", "cant",
    "can", "could", "will", "would", "should", "have", "has", "had", "got", "getting", "get", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "but", "so", "not", "no", "any", "some", "what", "why", "how",
    "when", "please", "pls", "plz", "help", "hi", "hello", "hey", "need", "issue", "issues", "problem",
    "problems", "error", "errors", "broken", "fix", "fixed", "working", "work", "works", "worked", "wrong",
    "something", "anything", "thing", "stuff", "urgent", "asap", "again", "still", "now", "today", "check",
    "failing", "failed", "fail", "fails", "stopped", "up", "down", "system", "happening", "going", "on",
}

GENERIC_FOLLOW_UPS = [
    "Which system, application or service is affected?",
    "What is the exact error message or code?",
    "When did it start, and did anything change just before (deployment, config, network)?",
    "Is it affecting one user, a team, or everyone?",
]


@dataclass
class TroubleshootingReport:
    query: str  # the (redacted) question as interpreted by the assistant
    status: str  # answered | needs_details | unknown
    likely_issue: str
    summary: str
    steps: list[Step] = field(default_factory=list)
    fixes: list[Step] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    sources: list[SearchResult] = field(default_factory=list)  # KB entries the answer is grounded in
    candidates: list[SearchResult] = field(default_factory=list)  # everything retrieved, for transparency
    confidence: float = 0.0
    confidence_label: str = "None"
    escalation: EscalationSuggestion | None = None
    generated_by: str = ""
    warnings: list[str] = field(default_factory=list)
    prompt: str = ""  # the grounded prompt built from the retrieved entries (shown in the UI for transparency)
    elapsed_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def primary_source(self) -> SearchResult | None:
        return self.sources[0] if self.sources else None


def needs_more_details(query: str, top_score: float, high_confidence: float = config.HIGH_CONFIDENCE) -> bool:
    """True when the question is too vague to troubleshoot reliably."""
    words = re.findall(r"[a-z0-9]+", query.lower().replace("'", ""))
    meaningful = [w for w in words if w not in _VAGUE_WORDS]
    if not meaningful:
        return True
    # Very short questions are only accepted when they match one KB article strongly ("server down", "VPN failing").
    return len(words) <= 2 and top_score < high_confidence


def compose_from_kb(sources: list[SearchResult]) -> GeneratedAnswer:
    """Deterministic answer built only from KB text (used when Gemini is off or unavailable)."""
    primary = sources[0].entry
    causes = "; ".join(primary["possible_causes"][:3])
    summary = f"This matches {primary['id']} ({primary['title']}). Common causes: {causes}."
    if len(sources) > 1:
        other = sources[1].entry
        summary += f" If the checks rule these out, review {other['id']} ({other['title']}) as well."
    return GeneratedAnswer(
        likely_issue=primary["title"],
        summary=summary,
        steps=[Step(text=c, source_id=primary["id"]) for c in primary["diagnostic_checks"]],
        fixes=[Step(text=r, source_id=primary["id"]) for r in primary["resolution"]],
        follow_up_questions=GENERIC_FOLLOW_UPS[:3],
    )


def confidence_label(score: float, min_confidence: float, high_confidence: float) -> str:
    if score >= high_confidence:
        return "High"
    if score >= min_confidence:
        return "Medium"
    return "Low"


class TroubleshootingAssistant:
    def __init__(
        self,
        retriever: KnowledgeBaseRetriever,
        engine: GeminiEngine | None = None,
        min_confidence: float = config.MIN_CONFIDENCE,
        high_confidence: float = config.HIGH_CONFIDENCE,
        top_k: int = config.TOP_K,
    ):
        self.retriever = retriever
        self.engine = engine
        self.min_confidence = min_confidence
        self.high_confidence = high_confidence
        self.top_k = top_k

    @property
    def mode(self) -> str:
        return f"Gemini ({self.engine.model})" if self.engine else "Knowledge base only (no AI key)"

    def ask(self, question: str, previous: TroubleshootingReport | None = None) -> TroubleshootingReport:
        started = time.perf_counter()
        report = self._run(question, previous)
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report

    def _run(self, question: str, previous: TroubleshootingReport | None) -> TroubleshootingReport:
        query, had_secret = redact_secrets(question.strip())
        warnings: list[str] = []
        if had_secret:
            warnings.append(
                "Sensitive values (passwords, keys or tokens) were removed from your message. "
                "Never share secrets in this chat; rotate any credential that was exposed."
            )

        # If the last answer asked for details, treat this message as those details.
        follow_up: str | None = None
        if previous is not None and previous.status == NEEDS_DETAILS and query:
            follow_up = query
            query = f"{previous.query}. {query}"

        candidates = self.retriever.search(query, self.top_k) if query else []
        if follow_up:
            # A vague first message ("it's not working") can dilute the combined text; keep whichever
            # search, combined or follow-up alone, matches the knowledge base better.
            alone = self.retriever.search(follow_up, self.top_k)
            if alone and (not candidates or alone[0].score > candidates[0].score):
                candidates = alone
        top_score = candidates[0].score if candidates else 0.0
        label = confidence_label(top_score, self.min_confidence, self.high_confidence)

        if needs_more_details(query, top_score, self.high_confidence):
            return self._needs_details(query, candidates, top_score, label, warnings)

        if top_score < self.min_confidence:
            return self._unknown(query, candidates, top_score, warnings)

        # Ground the answer in every entry that is above the threshold and close to the best match.
        sources = [c for c in candidates if c.score >= self.min_confidence and c.score >= top_score - 0.15]
        prompt = build_prompt(query, [s.entry for s in sources])

        answer: GeneratedAnswer | None = None
        generated_by = "Knowledge base (template, no AI)"
        if self.engine is not None:
            try:
                answer = self.engine.generate(query, [s.entry for s in sources])
                generated_by = f"Gemini ({self.engine.last_model_used or self.engine.model}) grounded on KB"
            except Exception:  # noqa: BLE001 - outage, quota, bad JSON: degrade to KB-only answer
                warnings.append("AI service unavailable right now; showing guidance taken directly from the knowledge base.")
                answer = None

            if answer is not None and not answer.sufficient_context:
                # Retrieval found something, but the model judged it does not cover this problem. Trust the refusal.
                report = self._unknown(
                    query,
                    candidates,
                    top_score,
                    warnings,
                    reason=(
                        f"The closest article ({sources[0].kb_id}: {sources[0].entry['title']}) does not cover this "
                        "problem closely enough, so I won't present it as the answer."
                    ),
                )
                report.prompt = prompt
                return report
            if answer is not None and answer.dropped_items:
                warnings.append(
                    f"{answer.dropped_items} AI-suggested item(s) were discarded because they did not cite a retrieved KB article."
                )

        if answer is None:
            answer = compose_from_kb(sources)

        if label == "Medium":
            warnings.append(
                f"Partial match ({top_score:.0%}). Confirm the symptoms match {sources[0].kb_id} before following these steps."
            )

        return TroubleshootingReport(
            query=query,
            status=ANSWERED,
            likely_issue=answer.likely_issue or sources[0].entry["title"],
            summary=answer.summary,
            steps=answer.steps,
            fixes=answer.fixes,
            follow_up_questions=answer.follow_up_questions,
            sources=sources,
            candidates=candidates,
            confidence=top_score,
            confidence_label=label,
            escalation=suggest_escalation(sources[0].entry, query),
            generated_by=generated_by,
            warnings=warnings,
            prompt=prompt,
        )

    def _needs_details(
        self, query: str, candidates: list[SearchResult], score: float, label: str, warnings: list[str]
    ) -> TroubleshootingReport:
        possible = [c for c in candidates if c.score >= self.min_confidence]
        summary = "I need a few more details before I can troubleshoot this reliably."
        if possible:
            titles = ", ".join(f"{c.kb_id} ({c.entry['title']})" for c in possible)
            summary += f" It might be related to: {titles}."
        return TroubleshootingReport(
            query=query,
            status=NEEDS_DETAILS,
            likely_issue="More information needed",
            summary=summary,
            follow_up_questions=GENERIC_FOLLOW_UPS,
            candidates=candidates,
            confidence=score,
            confidence_label=label,
            escalation=None,
            generated_by="Assistant guardrail",
            warnings=warnings,
        )

    def _unknown(
        self,
        query: str,
        candidates: list[SearchResult],
        score: float,
        warnings: list[str],
        reason: str | None = None,
    ) -> TroubleshootingReport:
        summary = (
            "I could not find sufficient information in the knowledge base to troubleshoot this issue, "
            "so I won't guess at a cause or a fix."
        )
        if reason:
            summary += " " + reason
        return TroubleshootingReport(
            query=query,
            status=UNKNOWN,
            likely_issue="Not covered by the knowledge base",
            summary=summary,
            follow_up_questions=GENERIC_FOLLOW_UPS,
            candidates=candidates,
            confidence=score,
            confidence_label=confidence_label(score, self.min_confidence, self.high_confidence),
            escalation=suggest_escalation(None, query),
            generated_by="Assistant guardrail",
            warnings=warnings,
        )


def build_assistant(engine: Any = "auto") -> TroubleshootingAssistant:
    """Convenience factory: loads the KB, builds the index and (optionally) connects Gemini."""
    from src.ai_engine import create_engine
    from src.retriever import load_knowledge_base

    retriever = KnowledgeBaseRetriever(load_knowledge_base())
    return TroubleshootingAssistant(retriever, create_engine() if engine == "auto" else engine)
