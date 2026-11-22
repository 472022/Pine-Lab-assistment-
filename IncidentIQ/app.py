"""IncidentIQ - AI-Powered Incident Analyzer (Streamlit)."""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.ai_analyzer import analyze_with_ai, answer_question
from modules.error_detector import analyze_logs
from modules.log_parser import parse_logs
from modules.report_generator import generate_pdf
from modules.severity_classifier import classify_severity

st.set_page_config(page_title="IncidentIQ", page_icon="🚨", layout="wide")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_logs")
SEV_COLOR = {"Critical": "#ff8f8f", "High": "#ffb36b", "Medium": "#ffd75e", "Low": "#8fe0a5"}
LEVEL_COLOR = {"FATAL": "#e04f5f", "ERROR": "#ff7a85", "WARN": "#ffc247", "INFO": "#6ea8ff", "DEBUG": "#b8b3d6", "OTHER": "#d4d0e6"}


CLAY_CSS = """
<style>
:root { --ink:#2d2a4a; --clay-shadow: 10px 10px 22px rgba(150,128,200,.35), -8px -8px 18px rgba(255,255,255,.95),
        inset -5px -5px 10px rgba(0,0,0,.06), inset 5px 5px 10px rgba(255,255,255,.65); }
.stApp { background: #f3eefc; color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #e9e1fb; border-right: 0; box-shadow: 8px 0 20px rgba(150,128,200,.2); }
h1 { color: #5b3fd1 !important; font-weight: 800 !important; }
h2, h3, h4, p, li, label, span { color: var(--ink); }
/* clay cards */
[data-testid="stVerticalBlockBorderWrapper"] { background: #f9f6ff; border: 0 !important; border-radius: 30px; box-shadow: var(--clay-shadow); padding: 6px; }
.kpi { border-radius: 30px; padding: 20px 22px; height: 100%; box-shadow: var(--clay-shadow); color: var(--ink); }
.kpi .label { font-size: .8rem; letter-spacing: .1em; text-transform: uppercase; font-weight: 700; opacity: .75; }
.kpi .value { font-size: 2.2rem; font-weight: 800; line-height: 1.15; margin-top: 4px; color: var(--ink); }
.kpi.big .value { font-size: 3.2rem; letter-spacing: .03em; }
.kpi .sub { font-size: .82rem; margin-top: 6px; opacity: .75; }
div.stButton > button[kind="primary"], .stDownloadButton > button { background: #7c5cff; color: #fff; border: 0; border-radius: 20px; padding: .6rem 1.2rem; font-weight: 700; box-shadow: var(--clay-shadow); }
[data-testid="stExpander"] { background: #f9f6ff; border: 0; border-radius: 24px; box-shadow: var(--clay-shadow); }
[data-testid="stChatInput"] { border-radius: 24px; }
</style>
"""
st.markdown(CLAY_CSS, unsafe_allow_html=True)


def kpi(label, value, bg="#ffffff", sub="", big=False):
    st.markdown(f"<div class='kpi {'big' if big else ''}' style='background:{bg}'><div class='label'>{label}</div>"
                f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def style_fig(fig, height=300):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#2d2a4a"), height=height, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
    return fig

st.title("🚨 IncidentIQ")
st.caption("AI-Powered Incident Analyzer · Python parses the facts, rules set the severity, AI explains the hypotheses")

# ---------------- Input ----------------
with st.sidebar:
    st.header("1. Log input")
    mode = st.radio("Source", ["Sample logs", "Upload file", "Paste logs"])
    raw_text, source_name = "", ""
    if mode == "Sample logs":
        files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".log"))
        choice = st.selectbox("Sample", files)
        source_name = choice
        raw_text = open(os.path.join(SAMPLE_DIR, choice)).read()
    elif mode == "Upload file":
        up = st.file_uploader("Upload .txt / .log", type=["txt", "log"])
        if up:
            raw_text, source_name = up.getvalue().decode("utf-8", errors="replace"), up.name
    else:
        raw_text = st.text_area("Paste logs", height=220, placeholder="10:01:12 ERROR Payment API request timeout")
        source_name = "pasted logs"
    st.header("2. Gemini API key")
    env_key = os.getenv("GEMINI_API_KEY", "")
    has_env = bool(env_key) and env_key != "your_key_here"
    key_in = st.text_input("Your Gemini API key", type="password", placeholder="AIza...", key="key_input",
                           help="Used only in this browser session. It is never saved to disk or shown again.")
    k1, k2 = st.columns(2)
    if k1.button("Apply key", width="stretch"):
        if key_in.strip():
            st.session_state["api_key"] = key_in.strip()
        else:
            st.warning("Paste a key first.")
    if k2.button("Remove", width="stretch"):
        st.session_state.pop("api_key", None)
    if st.session_state.get("api_key"):
        st.success("Using your key (this session)")
    elif has_env:
        st.info("Using the key from .env")
    else:
        st.warning("No key set: rule-based mode")
    st.markdown("[Get a free key](https://aistudio.google.com/apikey)")
    st.header("3. Options")
    use_ai = st.toggle("AI enabled (Gemini)", value=True, help="Falls back to rule-based analysis if no API key is set.")
    go = st.button("Analyze Incident", type="primary", width="stretch")

if go:
    if not raw_text.strip():
        st.warning("Please provide some logs first.")
    else:
        df = parse_logs(raw_text)
        facts = analyze_logs(df)
        sev = classify_severity(facts)
        if use_ai:
            with st.spinner("Analyzing incident..."):
                ai = analyze_with_ai(raw_text, facts, sev, st.session_state.get("api_key"))
        else:
            from modules.ai_analyzer import fallback_analysis
            ai = {**fallback_analysis(facts, sev), "source": "rule-based (AI disabled)"}
        st.session_state.update(df=df, facts=facts, sev=sev, ai=ai, raw=raw_text, src=source_name, chat=[])

if "facts" not in st.session_state:
    st.info("Choose a log source in the sidebar and click **Analyze Incident**. Try `payment_failure.log` first.")
    st.stop()

df, facts, sev, ai = (st.session_state[k] for k in ("df", "facts", "sev", "ai"))

df, facts, sev, ai = (st.session_state[k] for k in ("df", "facts", "sev", "ai"))

# ---------------- Dashboard (bento grid) ----------------
accent = SEV_COLOR[sev["severity"]]
st.caption(f"Analysis engine: {ai.get('source')} · Source: {st.session_state['src']} · {facts['total_lines']} lines")

# Row 1: one big glowing severity card + four smaller KPI cards
r1 = st.columns([2.2, 1, 1, 1, 1.2])
with r1[0]:
    kpi("Severity", sev["severity"].upper(), accent, f"Rule-based · AI review: {'agrees' if (ai.get('severity_review') or {}).get('agrees', True) else 'differs'}", big=True)
with r1[1]:
    kpi("Errors detected", facts["errors"], "#ffb3b3", f"{facts['fatal']} fatal")
with r1[2]:
    kpi("Warnings", facts["warnings"], "#ffe08a", f"peak {int(facts['max_response_ms'])}ms" if facts["max_response_ms"] else "")
with r1[3]:
    kpi("Repeated failures", facts["repeated_patterns"], "#a8e6f0", "recurring patterns")
with r1[4]:
    kpi("Incident status", sev["status"], "#cdb8ff", f"{facts['duration_sec']}s window")

st.write("")
# Row 2: three cards of different widths
r2 = st.columns([1, 1.4, 1])
with r2[0]:
    with st.container(border=True):
        st.markdown("#### 🔎 Evidence")
        for e in ai.get("evidence", []):
            st.write("• " + str(e))
        st.markdown("**Impact**")
        st.write(ai.get("impact", ""))
with r2[1]:
    with st.container(border=True):
        st.markdown("#### 🧠 Likely root cause")
        st.caption("Hypotheses based on the logs, not confirmed causes")
        for h in ai.get("hypotheses", []):
            st.write(f"• **{h['cause']}** _(confidence: {h['confidence']})_  \n{h['reasoning']}")
        rv = ai.get("severity_review") or {}
        if rv and not rv.get("agrees", True):
            st.warning(f"AI suggests **{rv.get('suggested')}**: {rv.get('justification', '')} (rule-based level is not auto-overridden)")
with r2[2]:
    with st.container(border=True):
        st.markdown("#### ✅ Recommended actions")
        for i, r in enumerate(ai.get("recommendations", []), 1):
            st.write(f"{i}. {r}")

with st.expander("Why this severity? (rule-based reasoning)"):
    for r in sev["reasons"]:
        st.write("• " + r)

st.write("")
# Row 3: charts in uneven bento tiles
r3 = st.columns([1, 1.6, 1.4])
lc = pd.DataFrame({"level": list(facts["level_counts"]), "count": list(facts["level_counts"].values())})
with r3[0]:
    with st.container(border=True):
        st.plotly_chart(style_fig(px.bar(lc, x="level", y="count", color="level", color_discrete_map=LEVEL_COLOR, title="Log levels")), width="stretch")
with r3[1]:
    with st.container(border=True):
        if facts["patterns"]:
            pdf_ = pd.DataFrame(facts["patterns"][:8])
            st.plotly_chart(style_fig(px.bar(pdf_, x="count", y="message", orientation="h", color="level", color_discrete_map=LEVEL_COLOR, title="Most frequent problems").update_layout(yaxis_title=None)), width="stretch")
        else:
            st.info("No problem patterns found.")
with r3[2]:
    with st.container(border=True):
        tl = df[df["level"].isin(["ERROR", "FATAL", "WARN"])].dropna(subset=["timestamp"]).copy()
        if len(tl):
            tl["bucket"] = tl["timestamp"].dt.floor(facts["bucket_freq"])
            tl = tl.groupby(["bucket", "level"]).size().reset_index(name="count")
            st.plotly_chart(style_fig(px.bar(tl, x="bucket", y="count", color="level", color_discrete_map=LEVEL_COLOR, title="Problems over time")), width="stretch")
        else:
            st.info("No timestamps found for a timeline.")
if facts["peak_bucket"] is not None:
    st.caption(f"Peak problem window starts at {facts['peak_bucket']:%H:%M:%S}.")

with st.expander("Detected error patterns"):
    st.dataframe(pd.DataFrame(facts["patterns"]).drop(columns=["pattern"], errors="ignore"), width="stretch")
with st.expander("Parsed log lines"):
    st.dataframe(df.drop(columns=["pattern", "raw"]), width="stretch")

# ---------------- Report + Chat ----------------
st.download_button("📄 Download incident report (PDF)", generate_pdf(facts, sev, ai, st.session_state["src"]),
                   file_name="incident_report.pdf", mime="application/pdf")
st.markdown("### 💬 Chat with logs")
for role, msg in st.session_state["chat"]:
    st.chat_message(role).write(msg)
if q := st.chat_input("Ask e.g. 'Why did the payment API fail?'"):
    st.chat_message("user").write(q)
    a = answer_question(q, df, facts, st.session_state["raw"], st.session_state.get("api_key"))
    st.chat_message("assistant").write(a)
    st.session_state["chat"] += [("user", q), ("assistant", a)]
