"""Scenario and unit tests. They run offline: Gemini is replaced by fakes, so results are deterministic.

Run:  pytest -q
"""

from __future__ import annotations

import pytest

from src.ai_engine import GeneratedAnswer, Step, build_prompt, validate_answer
from src.escalation import assess_priority, suggest_escalation
from src.retriever import REQUIRED_FIELDS, KnowledgeBaseRetriever, load_knowledge_base
from src.security import REDACTED, redact_secrets
from src.troubleshooting import ANSWERED, NEEDS_DETAILS, UNKNOWN, TroubleshootingAssistant, needs_more_details


@pytest.fixture(scope="session")
def kb():
    return load_knowledge_base()


@pytest.fixture(scope="session")
def retriever(kb):
    return KnowledgeBaseRetriever(kb)


@pytest.fixture()
def assistant(retriever):
    return TroubleshootingAssistant(retriever, engine=None)  # KB-only mode


# ------------------------------------------------------------------ knowledge base
def test_kb_has_expected_size_and_fields(kb):
    assert 15 <= len(kb) <= 20
    for entry in kb:
        for field in REQUIRED_FIELDS:
            assert entry[field], f"{entry['id']} missing {field}"
    assert {e["category"] for e in kb} == {"Server", "API", "Database", "Network", "Application"}


# ------------------------------------------------------------------ assignment scenarios
def test_assignment_example_use_case(assistant):
    """The example from the assignment brief, word for word."""
    report = assistant.ask("The payment API is returning a 500 error.")
    assert report.status == ANSWERED
    assert report.primary_source.kb_id == "KB-001", "shows which KB entry informed the answer"
    steps = " ".join(s.text.lower() for s in report.steps)
    assert "application logs" in steps and "dependencies" in steps
    assert report.escalation.team == "Backend / API Support Team"


@pytest.mark.parametrize(
    "query, expected_kb, expected_team",
    [
        ("My payment API is returning a 500 Internal Server Error. What should I check?", "KB-001", "Backend / API Support Team"),
        ("Application cannot connect to the database", "KB-009", "Database Support Team"),
        ("VPN not connecting", "KB-015", "Network Support Team"),
        ("The application is running slowly", "KB-019", "L2 Application Support"),
        # extra coverage across every category, including paraphrases with few shared keywords
        ("server down", "KB-005", "Infrastructure / Server Team"),
        ("CPU is pegged at 100% on the web server", "KB-006", "Infrastructure / Server Team"),
        ("getting 404 on the /v2/orders endpoint", "KB-002", "Backend / API Support Team"),
        ("Users cannot log in to the HR portal", "KB-017", "L1 Service Desk"),
        ("nslookup fails for our intranet hostname", "KB-014", "Network Support Team"),
        ("Access denied for user app_user when connecting to MySQL", "KB-011", "Database Support Team"),
        ("no space left on device when writing logs", "KB-008", "Infrastructure / Server Team"),
        ("deadlock detected in the orders table", "KB-012", "Database Support Team"),
        ("browser says the certificate is not secure on our website", "KB-020", "L2 Application Support"),
        ("the desktop app closes unexpectedly when I export a report", "KB-018", "L2 Application Support"),
    ],
)
def test_known_issues_are_answered_from_the_right_article(assistant, query, expected_kb, expected_team):
    report = assistant.ask(query)
    assert report.status == ANSWERED
    assert report.primary_source.kb_id == expected_kb
    assert report.escalation.team == expected_team
    assert report.steps, "should return troubleshooting steps"
    assert all(step.source_id == expected_kb for step in report.steps), "every step cites its source"


@pytest.mark.parametrize(
    "query",
    [
        "My office printer keeps jamming paper",
        "How do I bake a chocolate cake?",
        "Bluetooth headset will not pair with my laptop",
        "Excel macro gives a runtime error 1004",
    ],
)
def test_unknown_issue_reports_insufficient_information(assistant, query):
    report = assistant.ask(query)
    assert report.status == UNKNOWN
    assert "could not find sufficient information" in report.summary
    assert report.steps == [] and report.fixes == [], "must not invent steps"
    assert report.sources == []


@pytest.mark.parametrize("query", ["it is not working", "help", "error", "Something is broken, please fix ASAP", ""])
def test_incomplete_question_asks_for_details(assistant, query):
    report = assistant.ask(query)
    assert report.status == NEEDS_DETAILS
    assert report.follow_up_questions
    assert report.steps == []


@pytest.mark.parametrize(
    "vague, details, expected_kb",
    [
        ("it is not working", "the VPN client fails to connect from home", "KB-015"),
        # regression: the vague first message used to dilute the combined search below the threshold
        ("It's not working, please help", "the CRM app is very slow for all users", "KB-019"),
    ],
)
def test_details_after_clarification_are_answered(assistant, vague, details, expected_kb):
    first = assistant.ask(vague)
    assert first.status == NEEDS_DETAILS
    second = assistant.ask(details, previous=first)
    assert second.status == ANSWERED
    assert second.primary_source.kb_id == expected_kb


def test_semantic_search_matches_meaning_without_shared_keywords(retriever):
    """'quitting by itself' shares no words with 'Application crash / closes unexpectedly'."""
    top = retriever.search("the app keeps quitting by itself", top_k=1)[0]
    assert top.kb_id == "KB-018" and top.score >= 0.5


def test_short_but_specific_question_is_answered(assistant):
    assert not needs_more_details("server down", top_score=0.85)
    assert needs_more_details("payments failing", top_score=0.55)


# ------------------------------------------------------------------ confidence & sources
def test_confidence_labels_follow_thresholds(assistant):
    strong = assistant.ask("My payment API is returning a 500 Internal Server Error")
    assert strong.confidence_label == "High" and strong.confidence >= assistant.high_confidence
    weak = assistant.ask("My office printer keeps jamming paper")
    assert weak.confidence < assistant.min_confidence


def test_sources_are_transparent(assistant):
    report = assistant.ask("Application cannot connect to the database")
    source = report.primary_source
    assert source.entry["resolution"], "the matched KB content is available to show"
    assert source.matched_text, "we can show which text matched"
    assert report.candidates[0].kb_id == source.kb_id


# ------------------------------------------------------------------ AI grounding (with fake engines)
class FakeEngine:
    model = "fake-model"

    def __init__(self, answer: GeneratedAnswer | None = None, error: Exception | None = None):
        self.answer, self.error, self.calls = answer, error, []
        self.last_model_used = self.model

    def generate(self, query, entries):
        self.calls.append((query, [e["id"] for e in entries]))
        if self.error:
            raise self.error
        return self.answer


def test_validate_answer_drops_uncited_or_hallucinated_steps():
    raw = {
        "sufficient_context": True,
        "likely_issue": "Payment API failure",
        "summary": "s",
        "steps": [
            {"text": "Check logs", "source_id": "KB-001"},
            {"text": "Reinstall the operating system", "source_id": "KB-999"},  # not retrieved
            {"text": "Restart everything", "source_id": ""},  # no citation
        ],
        "fixes": [],
        "follow_up_questions": ["q1", "q2", "q3", "q4"],
    }
    answer = validate_answer(raw, {"KB-001"})
    assert [s.text for s in answer.steps] == ["Check logs"]
    assert answer.dropped_items == 2
    assert len(answer.follow_up_questions) == 3


def test_validate_answer_without_valid_steps_is_insufficient():
    raw = {"sufficient_context": True, "likely_issue": "", "summary": "", "steps": [{"text": "x", "source_id": "KB-777"}],
           "fixes": [], "follow_up_questions": []}
    assert validate_answer(raw, {"KB-001"}).sufficient_context is False


def test_ai_answer_is_used_and_grounded_on_retrieved_entries(retriever):
    answer = GeneratedAnswer("Payment API 500", "summary", [Step("Check logs", "KB-001")], [Step("Roll back", "KB-001")])
    engine = FakeEngine(answer)
    report = TroubleshootingAssistant(retriever, engine).ask("payment API returns HTTP 500")
    assert report.status == ANSWERED and report.generated_by.startswith("Gemini")
    assert engine.calls and "KB-001" in engine.calls[0][1]


def test_ai_refusal_is_respected(retriever):
    refusal = GeneratedAnswer("", "", [], [], sufficient_context=False)
    report = TroubleshootingAssistant(retriever, FakeEngine(refusal)).ask("payment API returns HTTP 500")
    assert report.status == UNKNOWN and report.steps == []


def test_ai_not_called_for_unknown_issues(retriever):
    engine = FakeEngine(GeneratedAnswer("x", "x", [Step("x", "KB-001")], []))
    TroubleshootingAssistant(retriever, engine).ask("My office printer keeps jamming paper")
    assert engine.calls == [], "low-relevance questions never reach the LLM"


def test_ai_outage_falls_back_to_kb(retriever):
    report = TroubleshootingAssistant(retriever, FakeEngine(error=RuntimeError("503"))).ask("VPN not connecting")
    assert report.status == ANSWERED
    assert report.primary_source.kb_id == "KB-015"
    assert "Knowledge base" in report.generated_by
    assert any("AI service unavailable" in w for w in report.warnings)


def test_prompt_contains_only_retrieved_context(kb):
    prompt = build_prompt("vpn issue", [e for e in kb if e["id"] == "KB-015"])
    assert "KB-015" in prompt and "KB-001" not in prompt


# ------------------------------------------------------------------ escalation
@pytest.mark.parametrize(
    "query, priority",
    [
        ("payment API is down in production", "P1"),
        ("all users report the app is slow", "P1"),
        ("several users cannot reach the VPN", "P2"),
        ("my VPN won't connect", "P3"),
    ],
)
def test_priority_from_impact(query, priority):
    assert assess_priority(query)[0] == priority


def test_unknown_issue_goes_to_service_desk():
    suggestion = suggest_escalation(None, "printer jam")
    assert suggestion.team == "L1 Service Desk"


def test_signal_matching_is_whole_word():
    # "prod" must not fire inside "product", "site" not inside "website"
    assert assess_priority("product page on the website has a typo")[0] == "P3"


# ------------------------------------------------------------------ security
@pytest.mark.parametrize(
    "text, secret",
    [
        ("my password is Hunter2!", "Hunter2!"),
        ("api_key=abcd1234efgh", "abcd1234efgh"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9"),
        ("postgres://app:SuperSecret@db.internal:5432/app", "SuperSecret"),
        ("key AKIAABCDEFGHIJKLMNOP leaked", "AKIAABCDEFGHIJKLMNOP"),
    ],
)
def test_secrets_are_redacted(text, secret):
    redacted, found = redact_secrets(text)
    assert found and secret not in redacted and REDACTED in redacted


def test_plain_text_is_not_redacted():
    text = "Users cannot reset their password from the portal"
    assert redact_secrets(text) == (text, False)


def test_assistant_redacts_before_answering(assistant):
    report = assistant.ask("database authentication failed, password is S3cret!")
    assert "S3cret!" not in report.query
    assert report.warnings and "Sensitive values" in report.warnings[0]
