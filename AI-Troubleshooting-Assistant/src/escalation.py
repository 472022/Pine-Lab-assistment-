"""Decides which support team should own an issue and how urgently.

The owning team comes from the matched KB entry (a curated fact, not an LLM guess).
Priority comes from transparent impact keywords in the engineer's description.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TEAMS: dict[str, dict[str, str]] = {
    "L1 Service Desk": {
        "tier": "L1",
        "scope": "First contact: account unlocks, password resets, basic connectivity, ticket triage.",
    },
    "L2 Application Support": {
        "tier": "L2",
        "scope": "Application errors, crashes, performance and certificate problems.",
    },
    "Backend / API Support Team": {
        "tier": "L3",
        "scope": "API services, integrations, deployments and server-side application code.",
    },
    "Database Support Team": {
        "tier": "L3",
        "scope": "Database availability, performance, access and data integrity (DBA).",
    },
    "Network Support Team": {
        "tier": "L3",
        "scope": "LAN/WAN, internet links, DNS, VPN, firewalls and Wi-Fi.",
    },
    "Infrastructure / Server Team": {
        "tier": "L3",
        "scope": "Physical and virtual servers, OS, storage and capacity.",
    },
}

DEFAULT_TEAM = "L1 Service Desk"

# Words that signal business impact. Matching is whole-word / whole-phrase and case-insensitive.
CRITICAL_SIGNALS = [
    "production", "prod", "outage", "all users", "everyone", "entire company", "whole office",
    "customers", "customer-facing", "payment", "payments", "checkout", "revenue", "data loss",
    "security", "breach", "critical", "sev1", "p1",
]
HIGH_SIGNALS = [
    "multiple users", "several users", "many users", "team", "department", "site", "intermittent",
    "degraded", "urgent", "sla",
]


@dataclass
class EscalationSuggestion:
    team: str
    tier: str
    priority: str  # P1 / P2 / P3
    reason: str
    guidance: str
    team_scope: str


def _find_signals(text: str, signals: list[str]) -> list[str]:
    lowered = text.lower()
    return [s for s in signals if re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", lowered)]


def assess_priority(query: str) -> tuple[str, list[str]]:
    """Return (priority, matched_signals) based on impact words in the query."""
    critical = _find_signals(query, CRITICAL_SIGNALS)
    if critical:
        return "P1", critical
    high = _find_signals(query, HIGH_SIGNALS)
    if high:
        return "P2", high
    return "P3", []


def suggest_escalation(entry: dict[str, Any] | None, query: str) -> EscalationSuggestion:
    priority, signals = assess_priority(query)

    if entry is None:
        team = DEFAULT_TEAM
        reason = "No matching knowledge-base article, so the issue needs manual triage."
        guidance = (
            "Log a ticket with the system name, exact error message, start time and number of users affected, "
            "then route it after triage. Consider writing a new KB article once it is solved."
        )
    else:
        team = entry["escalation_team"] if entry["escalation_team"] in TEAMS else DEFAULT_TEAM
        reason = f"{entry['id']} ({entry['category']}) is owned by the {team}."
        if priority == "P1":
            guidance = f"Business-critical impact: open a P1 incident and engage the {team} now, in parallel with the checks."
        elif TEAMS[team]["tier"] == "L1":
            guidance = "L1 can usually resolve this. Escalate to L2 Application Support if the checks do not fix it."
        else:
            guidance = (
                f"Run the diagnostic checks first. Escalate to the {team} if a check confirms a "
                f"{entry['category'].lower()} fault or the issue persists."
            )

    if signals:
        reason += f" Impact signals in the description: {', '.join(signals)}."

    return EscalationSuggestion(
        team=team,
        tier=TEAMS[team]["tier"],
        priority=priority,
        reason=reason,
        guidance=guidance,
        team_scope=TEAMS[team]["scope"],
    )
