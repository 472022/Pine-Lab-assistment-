"""Deterministic, explainable severity rules. The AI may comment on these, never silently override them."""

LEVELS = ["Low", "Medium", "High", "Critical"]


def classify_severity(facts: dict) -> dict:
    errors, warns = facts["errors"], facts["warnings"]
    services = len(facts["services_affected"])
    unavail, repeated = facts["unavailable_events"], facts["repeated_patterns"]
    max_rt = facts["max_response_ms"]
    reasons, rank = [], 0  # rank indexes LEVELS

    if errors == 0 and warns == 0:
        reasons.append("No errors or warnings found.")
    elif errors == 0:
        reasons.append(f"{warns} warning(s) and no errors: degraded signals only.")
        if warns >= 5 or max_rt >= 5000:
            rank = 1
            reasons.append("Warning volume or response time is high enough to watch closely (Medium).")
    else:
        rank = 1
        reasons.append(f"{errors} error line(s) detected (baseline Medium).")
        if services >= 2 and errors >= 3:
            rank = max(rank, 2)
            reasons.append(f"Errors span {services} services ({', '.join(facts['services_affected'])}): wider impact (High).")
        if repeated >= 1 and errors >= 4:
            rank = max(rank, 2)
            reasons.append(f"{repeated} error pattern(s) repeat; failures are persistent, not one-off (High).")
        if unavail >= 1 and errors >= 3:
            rank = max(rank, 2)
            reasons.append("Service unavailability reported alongside other errors (High).")
        if max_rt >= 5000 and errors >= 2:
            rank = max(rank, 2)
            reasons.append(f"Response time reached {int(max_rt)}ms alongside errors (High).")
        if facts["fatal"] >= 1:
            rank = 3
            reasons.append("FATAL/CRITICAL level entries present (Critical).")
        if unavail >= 2:
            rank = 3
            reasons.append(f"{unavail} unavailability events: service appears down (Critical).")
        if services >= 3 and errors >= 6:
            rank = 3
            reasons.append("Failures across 3+ services at volume: likely platform-wide impact (Critical).")

    severity = LEVELS[rank]
    status = {"Critical": "Escalate", "High": "Investigate", "Medium": "Monitor", "Low": "No action"}[severity]
    return {"severity": severity, "reasons": reasons, "status": status}
