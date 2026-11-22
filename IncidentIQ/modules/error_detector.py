"""Error and pattern detection: computes measurable facts from the parsed logs."""
import re
import pandas as pd

UNAVAILABLE_RE = re.compile(r"unavailable|outage|is down|service down|not responding|unreachable|refused|shut ?down", re.I)
TIMEOUT_RE = re.compile(r"time[d ]?-?out", re.I)
AUTH_FAIL_RE = re.compile(r"(?:auth\w*|login|credential|token|password).*(?:fail|denied|invalid|expired)|unauthorized", re.I)
CONN_RE = re.compile(r"connection (?:failed|refused|lost|reset|pool)|cannot connect|pool exhausted", re.I)
RT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ms\b", re.I)


def analyze_logs(df: pd.DataFrame) -> dict:
    """Return a dict of evidence-level facts. No interpretation happens here."""
    facts = {"total_lines": int(len(df))}
    if df.empty:
        return {**facts, "level_counts": {}, "errors": 0, "warnings": 0, "fatal": 0, "exceptions": 0,
                "services_affected": [], "service_issue_counts": {}, "patterns": [], "repeated_patterns": 0,
                "unavailable_events": 0, "timeout_events": 0, "auth_failures": 0, "connection_failures": 0,
                "max_response_ms": 0, "response_times": [], "start": None, "end": None, "duration_sec": 0,
                "peak_bucket": None, "bucket_freq": "1min"}
    problems = df[df["level"].isin(["ERROR", "FATAL", "WARN"])]
    err = df[df["level"].isin(["ERROR", "FATAL"])]
    facts["level_counts"] = df["level"].value_counts().to_dict()
    facts["errors"] = int(len(err))
    facts["warnings"] = int((df["level"] == "WARN").sum())
    facts["fatal"] = int((df["level"] == "FATAL").sum())
    facts["exceptions"] = int(df["is_exception"].sum())

    svc = problems[problems["service"] != "unknown"]["service"].value_counts()
    facts["service_issue_counts"] = svc.to_dict()
    facts["services_affected"] = list(err[err["service"] != "unknown"]["service"].unique())

    patterns = []
    for _, g in problems.groupby("pattern"):
        patterns.append({
            "pattern": g["pattern"].iloc[0], "message": g["message"].iloc[0], "count": int(len(g)),
            "level": g["level"].iloc[0], "service": g["service"].iloc[0],
            "first_seen": g["timestamp"].min(), "last_seen": g["timestamp"].max(),
            "lines": g["line_no"].tolist(),
        })
    patterns.sort(key=lambda p: (-p["count"], p["lines"][0]))
    facts["patterns"] = patterns
    facts["repeated_patterns"] = sum(1 for p in patterns if p["count"] >= 2 and p["level"] in ("ERROR", "FATAL"))

    msgs = problems["message"]
    facts["unavailable_events"] = int(msgs.str.contains(UNAVAILABLE_RE).sum())
    facts["timeout_events"] = int(msgs.str.contains(TIMEOUT_RE).sum())
    facts["auth_failures"] = int(msgs.str.contains(AUTH_FAIL_RE).sum())
    facts["connection_failures"] = int(msgs.str.contains(CONN_RE).sum())

    rts = [float(x) for m in df["message"] for x in RT_RE.findall(m)]
    facts["response_times"] = rts
    facts["max_response_ms"] = max(rts) if rts else 0

    ts = df["timestamp"].dropna()
    facts["bucket_freq"] = "1min"
    if len(ts):
        facts["start"], facts["end"] = ts.min(), ts.max()
        facts["duration_sec"] = int((ts.max() - ts.min()).total_seconds())
        pts = problems.dropna(subset=["timestamp"])
        facts["bucket_freq"] = "1min" if facts["duration_sec"] > 600 else "10s"
        facts["peak_bucket"] = pts["timestamp"].dt.floor(facts["bucket_freq"]).value_counts().idxmax() if len(pts) else None
    else:
        facts.update({"start": None, "end": None, "duration_sec": 0, "peak_bucket": None})
    return facts
