"""Scores the assistant on a labelled set of questions and prints a results table.

    python evaluate.py          # KB-only mode (deterministic, no API calls)
    python evaluate.py --ai     # use Gemini as configured in .env
"""

from __future__ import annotations

import argparse
import statistics

from src.ai_engine import create_engine
from src.retriever import KnowledgeBaseRetriever, load_knowledge_base
from src.troubleshooting import ANSWERED, NEEDS_DETAILS, UNKNOWN, TroubleshootingAssistant

# (question, expected status, expected KB id for answered questions)
SCENARIOS = [
    ("The payment API is returning a 500 error.", ANSWERED, "KB-001"),  # example from the brief
    ("My payment API is returning a 500 Internal Server Error. What should I check?", ANSWERED, "KB-001"),
    ("Application cannot connect to the database", ANSWERED, "KB-009"),
    ("VPN not connecting", ANSWERED, "KB-015"),
    ("The application is running slowly", ANSWERED, "KB-019"),
    ("Our checkout endpoint keeps timing out with 504 errors", ANSWERED, "KB-003"),
    ("Partner integration suddenly gets 401 Unauthorized", ANSWERED, "KB-004"),
    ("server down", ANSWERED, "KB-005"),
    ("CPU is pegged at 100% on the web server", ANSWERED, "KB-006"),
    ("SSH session to the app server freezes, swap is full", ANSWERED, "KB-007"),
    ("Report query is cancelled after 30 seconds", ANSWERED, "KB-010"),
    ("Whole office lost internet access", ANSWERED, "KB-013"),
    ("Video calls lag and ping shows packet loss", ANSWERED, "KB-016"),
    ("Users cannot log in to the HR portal", ANSWERED, "KB-017"),
    ("browser says the certificate is not secure on our website", ANSWERED, "KB-020"),
    ("My office printer keeps jamming paper", UNKNOWN, None),
    ("Bluetooth headset will not pair with my laptop", UNKNOWN, None),
    ("Excel macro gives a runtime error 1004", UNKNOWN, None),
    ("It's not working, please help", NEEDS_DETAILS, None),
    ("error", NEEDS_DETAILS, None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai", action="store_true", help="use Gemini (needs GEMINI_API_KEY in .env)")
    args = parser.parse_args()

    engine = create_engine() if args.ai else None
    if args.ai and engine is None:
        raise SystemExit("GEMINI_API_KEY is not set; run without --ai or add the key to .env")
    assistant = TroubleshootingAssistant(KnowledgeBaseRetriever(load_knowledge_base()), engine)

    rows, passed, latencies = [], 0, []
    for question, want_status, want_kb in SCENARIOS:
        report = assistant.ask(question)
        got_kb = report.primary_source.kb_id if report.primary_source else "-"
        ok = report.status == want_status and (want_kb is None or got_kb == want_kb)
        passed += ok
        latencies.append(report.elapsed_ms)
        rows.append((ok, question, want_status, want_kb or "-", report.status, got_kb, report.confidence, report.elapsed_ms))

    print(f"\nMode: {assistant.mode}\n")
    print(f"{'':2} {'Question':58} {'Expected':22} {'Got':22} {'Score':>6} {'ms':>6}")
    for ok, q, ws, wk, gs, gk, score, ms in rows:
        q = q if len(q) <= 58 else q[:55] + "..."
        print(f"{'✔' if ok else '✘':2} {q:58} {ws + ' ' + wk:22} {gs + ' ' + gk:22} {score:6.2f} {ms:6}")

    print(f"\nPassed {passed}/{len(rows)} ({passed / len(rows):.0%}) · median latency {statistics.median(latencies):.0f} ms")


if __name__ == "__main__":
    main()
