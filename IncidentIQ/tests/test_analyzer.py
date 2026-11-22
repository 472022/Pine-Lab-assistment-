import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("GEMINI_API_KEY", None)

from modules.ai_analyzer import analyze_with_ai, answer_question  # noqa: E402
from modules.error_detector import analyze_logs  # noqa: E402
from modules.log_parser import parse_logs  # noqa: E402
from modules.report_generator import generate_pdf  # noqa: E402
from modules.severity_classifier import classify_severity  # noqa: E402

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_logs")


def run(name):
    text = open(os.path.join(LOGS, name)).read()
    df = parse_logs(text)
    facts = analyze_logs(df)
    return text, df, facts, classify_severity(facts)


@pytest.mark.parametrize("name,expected", [
    ("payment_failure.log", "High"),
    ("single_timeout.log", "Medium"),
    ("minor_warnings.log", "Low"),
    ("service_outage.log", "Critical"),
    ("database_error.log", "High"),
    ("authentication_error.log", "High"),
])
def test_severity_scenarios(name, expected):
    assert run(name)[3]["severity"] == expected


def test_parser_formats():
    df = parse_logs("2024-05-01 10:01:12 ERROR Payment API timeout\nERROR: Database connection failed\nWARNING: slow 8500ms\njunk line\nValueError: bad")
    assert list(df["level"]) == ["ERROR", "ERROR", "WARN", "OTHER", "ERROR"]
    assert df["timestamp"].notna().sum() == 1
    assert df.loc[0, "service"] == "payment" and df.loc[1, "service"] == "database"


def test_payment_facts():
    _, _, f, _ = run("payment_failure.log")
    assert f["errors"] == 5 and f["warnings"] == 1
    assert f["repeated_patterns"] == 2
    assert f["max_response_ms"] == 8500 and f["duration_sec"] == 18


def test_empty_input():
    f = analyze_logs(parse_logs(""))
    assert classify_severity(f)["severity"] == "Low"


def test_offline_ai_and_report_and_chat():
    text, df, facts, sev = run("payment_failure.log")
    ai = analyze_with_ai(text, facts, sev)
    assert ai["hypotheses"] and ai["recommendations"] and ai["evidence"]
    assert generate_pdf(facts, sev, ai)[:4] == b"%PDF"
    assert "line" in answer_question("Why did the payment API fail?", df, facts, text)
