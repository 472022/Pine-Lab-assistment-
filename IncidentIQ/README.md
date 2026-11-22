# IncidentIQ - AI-Powered Incident Analyzer (Prototype)

IncidentIQ reads application logs (upload or paste), detects errors and repeated failures, classifies
severity with transparent Python rules, and uses Gemini to propose **evidence-based hypotheses and
troubleshooting steps**. It also generates a PDF incident report and lets you chat with the logs.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # or edit .env, then set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)
streamlit run app.py
```
You can also paste your own key into the **Gemini API key** box in the app sidebar and click **Apply key** (session only, never saved to disk; it overrides `.env`).
Without an API key the app still works in **rule-based mode** (the sidebar/analysis header shows which engine ran).
Run tests: `python -m pytest -q`

## AI model
Google Gemini (`gemini-2.5-flash` by default, override with `GEMINI_MODEL` in `.env`) via the `google-genai` SDK,
temperature 0.2, JSON-only output.

## Approach (hybrid)
1. `log_parser.py` - regex parsing of timestamp, level (ERROR/WARN/FATAL/INFO...), service and normalised message. Handles `10:01:12 ERROR ...`, `ERROR: ...`, full dates and bare exception lines.
2. `error_detector.py` - counts, repeated patterns, time window, timeouts, connection failures, auth failures, unavailability, response times (all measurable facts).
3. `severity_classifier.py` - rule-based severity with a written reason for every rule that fires.
4. `ai_analyzer.py` - Gemini receives the facts + logs and must return **Evidence** (present in logs), **Hypotheses** (not confirmed, with confidence), **Impact**, **Recommendations**, and a **severity review**. If it disagrees with Python's severity it must justify it; the dashboard shows this as a suggestion and does **not** silently override the rules.
5. `report_generator.py` - PDF report (reportlab). `app.py` - Streamlit dashboard with Plotly charts and chat.

### Severity rules
| Level | Triggers (any) |
|---|---|
| Critical | FATAL entries; 2+ unavailability events; 3+ services failing with 6+ errors |
| High | Errors in 2+ services (3+ errors); repeated error pattern with 4+ errors; unavailability + 3+ errors; response time >= 5000ms with errors |
| Medium | Any error below High thresholds; warning-heavy logs (5+ warnings or >= 5000ms) |
| Low | Only occasional warnings / no errors |

Severity depends on scope and impact (services affected, persistence, unavailability), not raw ERROR count.

## Testing
`tests/test_analyzer.py` covers the four required scenarios plus database and authentication logs, parser formats, empty input, offline AI, PDF generation and chat.

| Sample log | Expected | Result |
|---|---|---|
| payment_failure.log | High | High |
| single_timeout.log | Medium | Medium |
| minor_warnings.log | Low | Low |
| service_outage.log | Critical | Critical |
| database_error.log / authentication_error.log | High | High |

## Limitations
- Rules are heuristic keyword/regex based; unusual log formats or non-English messages may be misclassified.
- Service detection is keyword-based, not read from service metadata.
- AI output can be wrong: it is labelled as hypothesis, and the evidence/severity reasons let an engineer verify it.
- Only the first 400 lines are sent to the AI; no multi-file correlation, streaming logs or persistence.

## Future improvements
Support JSON/structured logs, time-window correlation across services, anomaly detection on error rates, integration with monitoring/ticketing (Grafana, PagerDuty, Jira), feedback loop on AI answers, authentication and log redaction for production.
