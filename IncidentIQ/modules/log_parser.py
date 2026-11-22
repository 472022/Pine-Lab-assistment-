"""Log parser: turns raw log text into a structured pandas DataFrame."""
import re
from datetime import datetime
import pandas as pd

LINE_RE = re.compile(
    r"^\s*(?:(?P<date>\d{4}-\d{2}-\d{2})[T ])?"
    r"(?:(?P<time>\d{2}:\d{2}:\d{2})(?:[.,]\d+)?\s+)?"
    r"\[?(?P<level>CRITICAL|FATAL|ERROR|WARNING|WARN|INFO|DEBUG)\]?:?\s*(?P<msg>.*)$",
    re.IGNORECASE,
)
EXCEPTION_RE = re.compile(r"(exception|traceback|\w+error:|stack trace|segfault)", re.IGNORECASE)

# Ordered: first match wins as the "primary" service of a line.
SERVICE_KEYWORDS = {
    "payment": ["payment", "checkout", "billing", "transaction"],
    "database": ["database", "db ", "sql", "query", "connection pool", "postgres", "mysql", "deadlock"],
    "authentication": ["auth", "login", "token", "password", "credential", "unauthorized"],
    "cache": ["cache", "redis", "memcache"],
    "network": ["network", "dns", "socket", "unreachable", "gateway"],
    "storage": ["disk", "storage", "filesystem", "s3"],
    "application": ["api", "service", "server", "application", "request"],
}
LEVEL_MAP = {"WARNING": "WARN", "FATAL": "FATAL", "CRITICAL": "FATAL"}


def detect_service(message: str) -> str:
    low = message.lower() + " "
    for service, words in SERVICE_KEYWORDS.items():
        if any(w in low for w in words):
            return service
    return "unknown"


def normalize_message(message: str) -> str:
    """Collapse numbers/ids so 'timeout after 30s' and '... 45s' group together."""
    msg = re.sub(r"\b[0-9a-f]{8,}\b", "#", message.lower())
    msg = re.sub(r"\d+(?:\.\d+)*", "#", msg)
    return re.sub(r"\s+", " ", msg).strip(" .:;")


def parse_logs(text: str) -> pd.DataFrame:
    rows = []
    for i, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        m = LINE_RE.match(raw)
        ts, is_exc = pd.NaT, False
        if m:
            level = m.group("level").upper()
            level = LEVEL_MAP.get(level, level)
            msg = m.group("msg").strip()
            if m.group("time"):
                date = m.group("date") or "1900-01-01"
                ts = pd.Timestamp(datetime.strptime(f"{date} {m.group('time')}", "%Y-%m-%d %H:%M:%S"))
            if EXCEPTION_RE.search(msg):
                is_exc = True
        else:
            msg = raw.strip()
            is_exc = bool(EXCEPTION_RE.search(msg))
            level = "ERROR" if is_exc else "OTHER"
        rows.append({"line_no": i, "timestamp": ts, "level": level, "service": detect_service(msg),
                     "message": msg, "pattern": normalize_message(msg), "is_exception": is_exc, "raw": raw.strip()})
    cols = ["line_no", "timestamp", "level", "service", "message", "pattern", "is_exception", "raw"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
