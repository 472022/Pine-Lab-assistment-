"""AI layer (Gemini) with a rule-based offline fallback. Separates Evidence / Hypothesis / Recommendation."""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _client(api_key=None):
    """A key entered in the UI takes priority over the one in .env."""
    key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY")
    if not key or key == "your_key_here":
        return None
    from google import genai
    return genai.Client(api_key=key)


def _facts_for_prompt(facts: dict, sev: dict) -> dict:
    return {
        "total_lines": facts["total_lines"], "errors": facts["errors"], "warnings": facts["warnings"],
        "services_affected": facts["services_affected"], "service_issue_counts": facts["service_issue_counts"],
        "top_patterns": [{"message": p["message"], "count": p["count"], "level": p["level"], "service": p["service"]}
                         for p in facts["patterns"][:10]],
        "unavailable_events": facts["unavailable_events"], "timeout_events": facts["timeout_events"],
        "auth_failures": facts["auth_failures"], "connection_failures": facts["connection_failures"],
        "max_response_ms": facts["max_response_ms"], "duration_sec": facts["duration_sec"],
        "rule_based_severity": sev["severity"], "severity_reasons": sev["reasons"],
    }


PROMPT = """You are a senior site-reliability engineer analysing an application incident.
Facts computed by Python (trustworthy) and the raw logs are below. Return ONLY JSON with this schema:
{{"summary": str,
 "evidence": [str],
 "hypotheses": [{{"cause": str, "confidence": "low|medium|high", "reasoning": str}}],
 "impact": str,
 "recommendations": [str],
 "severity_review": {{"agrees": bool, "suggested": "Low|Medium|High|Critical", "justification": str}}}}
Definitions: "evidence" = things literally present in the logs, with counts. "hypotheses" = possible
causes, NOT confirmed. "recommendations" = ordered troubleshooting steps and escalation.
Rules: never state a hypothesis as a confirmed cause; evidence must come only from the logs; if you
disagree with the rule-based severity, justify it with specific log evidence.

FACTS:
{facts}

LOGS:
{logs}
"""


def analyze_with_ai(raw_text: str, facts: dict, sev: dict, api_key=None) -> dict:
    client = _client(api_key)
    if client is None:
        return {**fallback_analysis(facts, sev), "source": "rule-based (no API key set)"}
    try:
        from google.genai import types
        resp = client.models.generate_content(
            model=MODEL,
            contents=PROMPT.format(facts=json.dumps(_facts_for_prompt(facts, sev), default=str, indent=1),
                                   logs="\n".join(raw_text.splitlines()[:400])),
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
        )
        data = json.loads(re.sub(r"^```json|```$", "", resp.text.strip()))
        data["source"] = f"Gemini ({MODEL})"
        return data
    except Exception as exc:  # network, quota, bad JSON: degrade gracefully
        return {**fallback_analysis(facts, sev), "source": f"rule-based (AI call failed: {type(exc).__name__})"}


def fallback_analysis(facts: dict, sev: dict) -> dict:
    """Deterministic analysis used when no AI key is available."""
    ev = [f"{p['count']}x [{p['level']}] {p['message']}" for p in facts["patterns"][:6]]
    if facts["max_response_ms"]:
        ev.append(f"Peak response time of {int(facts['max_response_ms'])}ms observed.")
    hyps, recs = [], []
    svcs = facts["service_issue_counts"]
    if facts["connection_failures"] and ("payment" in svcs or "application" in svcs):
        hyps.append({"cause": "Database connectivity problems may be causing downstream API timeouts/failures.",
                     "confidence": "medium", "reasoning": "Connection failures and API timeouts occur in the same window. This is a hypothesis, not a confirmed root cause."})
    elif facts["connection_failures"]:
        hyps.append({"cause": "Database or backend connection pool problems (exhaustion, network, or DB down).",
                     "confidence": "medium", "reasoning": "Repeated connection failures in the logs."})
    if facts["timeout_events"] and not facts["connection_failures"]:
        hyps.append({"cause": "A dependency is slow or unreachable, causing request timeouts.",
                     "confidence": "low", "reasoning": "Timeouts present without other correlated failures."})
    if facts["auth_failures"]:
        hyps.append({"cause": "Authentication issue: expired credentials/tokens, misconfigured identity provider, or a brute-force attempt.",
                     "confidence": "medium" if facts["auth_failures"] >= 3 else "low",
                     "reasoning": f"{facts['auth_failures']} authentication failure line(s)."})
    if facts["max_response_ms"] >= 3000:
        hyps.append({"cause": "Performance degradation (resource saturation or slow queries).",
                     "confidence": "low", "reasoning": "High response times recorded."})
    if facts["unavailable_events"]:
        hyps.append({"cause": "A service or dependency is down or crashed.",
                     "confidence": "medium", "reasoning": f"{facts['unavailable_events']} unavailability message(s)."})
    if not hyps:
        hyps.append({"cause": "No clear cause identifiable from the logs provided.", "confidence": "low",
                     "reasoning": "Insufficient error evidence."})
    if facts["connection_failures"]:
        recs += ["Check database connectivity and service health.", "Inspect connection pool usage and limits."]
    if facts["timeout_events"] or facts["max_response_ms"] >= 3000:
        recs.append("Check upstream API/dependency response times and recent deployments.")
    if facts["auth_failures"]:
        recs += ["Verify credential/token validity and identity provider status.", "Review auth logs for suspicious source IPs."]
    if facts["unavailable_events"]:
        recs.append("Restart or fail over the affected service and confirm health checks pass.")
    recs.append("Escalate to the backend/database/on-call team if failures continue." if sev["severity"] in ("High", "Critical")
                else "Continue monitoring; no escalation needed unless errors increase.")
    return {"summary": f"{facts['errors']} error(s) and {facts['warnings']} warning(s) across "
                       f"{', '.join(facts['services_affected']) or 'no identified services'}. Severity: {sev['severity']}.",
            "evidence": ev, "hypotheses": hyps,
            "impact": f"Services affected: {', '.join(facts['services_affected']) or 'none identified'}.",
            "recommendations": recs,
            "severity_review": {"agrees": True, "suggested": sev["severity"], "justification": "Rule-based mode: severity taken from Python rules."}}


def answer_question(question: str, df, facts: dict, raw_text: str, api_key=None) -> str:
    """Chat with logs: Gemini if available, else keyword retrieval of matching log lines."""
    client = _client(api_key)
    note = ""
    if client is not None:
        try:
            prompt = (f"Answer the engineer's question using ONLY these logs. Cite line numbers. If the logs do not "
                      f"contain the answer, say so. Label guesses as hypotheses.\n\nQUESTION: {question}\n\nLOGS:\n"
                      + "\n".join(f"{r.line_no}: {r.raw}" for r in df.head(400).itertuples()))
            return client.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as exc:
            note = f"(AI unavailable: {type(exc).__name__}; showing keyword matches)\n\n"
    else:
        note = "(Offline mode: no API key; showing keyword matches)\n\n"
    stop = {"why", "did", "the", "what", "how", "was", "and", "for", "with", "happened", "fail", "failed"}
    words = set(re.findall(r"[a-z]{3,}", question.lower())) - stop
    scored = [(sum(w in r.message.lower() or w == r.service for w in words), r) for r in df.itertuples()]
    hits = [r for s, r in sorted(scored, key=lambda x: -x[0]) if s > 0][:8]
    if not hits:
        return note + "I couldn't find log lines related to that question."
    return note + "Relevant log lines:\n" + "\n".join(f"- line {r.line_no}: [{r.level}] {r.message}" for r in sorted(hits, key=lambda r: r.line_no))
