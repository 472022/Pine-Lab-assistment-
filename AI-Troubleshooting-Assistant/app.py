"""AI TroubleshootPro - Streamlit dashboard for the IT troubleshooting assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
from collections import Counter

import pandas as pd
import streamlit as st

from src import config
from src.ai_engine import SYSTEM_PROMPT, create_engine
from src.retriever import KnowledgeBaseRetriever, load_knowledge_base
from src.security import redact_secrets
from src.troubleshooting import ANSWERED, NEEDS_DETAILS, UNKNOWN, TroubleshootingAssistant, TroubleshootingReport

st.set_page_config(page_title="AI TroubleshootPro", page_icon=":material/support_agent:", layout="wide")

EXAMPLES = [
    "The payment API is returning a 500 error.",
    "Our application cannot connect to the database since this morning",
    "VPN is not connecting from home",
    "The CRM application is running very slowly for all users",
    "My office printer keeps jamming paper",
    "It's not working, please help",
]

CATEGORY_ICONS = {
    "API": ":material/api:",
    "Server": ":material/dns:",
    "Database": ":material/database:",
    "Network": ":material/lan:",
    "Application": ":material/apps:",
}
OUTCOME_ICONS = {"resolved": ":material/check_circle:", "escalated": ":material/arrow_upward:", "not_resolved": ":material/cancel:"}
USER_AVATAR, BOT_AVATAR = ":material/person:", ":material/support_agent:"
STATUS_BADGE = {
    ANSWERED: ("Guidance found", "green", ":material/check_circle:"),
    NEEDS_DETAILS: ("Needs more details", "orange", ":material/help:"),
    UNKNOWN: ("Not in knowledge base", "red", ":material/block:"),
}
CONFIDENCE_COLOR = {"High": "green", "Medium": "orange", "Low": "red", "None": "gray"}
PRIORITY_COLOR = {"P1": "red", "P2": "orange", "P3": "blue"}
OUTCOMES = {"resolved": "Resolved", "escalated": "Escalated", "not_resolved": "Not resolved"}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .tp-header h1 {margin-bottom: 0;}
      .tp-sub {color: rgba(128,128,128,0.95); margin-top: 0.1rem;}
      .tp-step {display:flex; gap:0.6rem; align-items:flex-start; margin:0.35rem 0;}
      .tp-num {flex:0 0 1.6rem; height:1.6rem; border-radius:50%; display:flex; align-items:center;
               justify-content:center; font-weight:600; font-size:0.85rem;
               background: rgba(37,99,235,0.15); color: rgb(37,99,235);}
      .tp-cite {font-size:0.72rem; padding:0.05rem 0.4rem; border-radius:0.4rem; margin-left:0.35rem;
                border:1px solid rgba(128,128,128,0.35); color: rgba(128,128,128,1); white-space:nowrap;}
      .tp-kv {font-size:0.92rem; margin:0.15rem 0;}
      .tp-kv b {display:inline-block; min-width:5.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- resources & state
@st.cache_resource(show_spinner="Loading knowledge base and embedding model…")
def get_assistant() -> TroubleshootingAssistant:
    retriever = KnowledgeBaseRetriever(load_knowledge_base())
    return TroubleshootingAssistant(retriever, create_engine())


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # {"role": "user"|"assistant", "content": str, "report": report, "qid": int}
    st.session_state.setdefault("history", [])  # one dict per question, newest last
    st.session_state.setdefault("pending_query", None)


def history_record(qid: int) -> dict | None:
    return next((h for h in st.session_state.history if h["id"] == qid), None)


def set_outcome(qid: int, outcome: str) -> None:
    record = history_record(qid)
    if record is not None:
        record["outcome"] = outcome


def submit(query: str) -> None:
    st.session_state.pending_query = query


# ---------------------------------------------------------------- rendering helpers
def esc(text: str) -> str:
    return html.escape(text or "")


def render_steps(steps, start: int = 1) -> None:
    rows = []
    for i, step in enumerate(steps, start=start):
        rows.append(
            f"<div class='tp-step'><div class='tp-num'>{i}</div>"
            f"<div>{esc(step.text)}<span class='tp-cite'>{esc(step.source_id)}</span></div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_kb_entry(entry: dict, compact: bool = False) -> None:
    st.markdown(
        f"<div class='tp-kv'><b>Issue</b> {esc(entry['title'])}</div>"
        f"<div class='tp-kv'><b>Category</b> {esc(entry['category'])}</div>"
        f"<div class='tp-kv'><b>Escalation</b> {esc(entry['escalation_team'])}</div>",
        unsafe_allow_html=True,
    )
    sections = [("Symptoms", "symptoms"), ("Possible causes", "possible_causes")]
    if not compact:
        sections += [("Diagnostic checks", "diagnostic_checks"), ("Known resolution", "resolution")]
    for label, key in sections:
        st.markdown(f"**{label}**\n" + "\n".join(f"- {item}" for item in entry[key]))


def render_report(report: TroubleshootingReport, qid: int) -> None:
    label, color, icon = STATUS_BADGE[report.status]
    with st.container(horizontal=True):
        st.badge(label, color=color, icon=icon)
        if report.candidates:
            st.badge(
                f"Confidence: {report.confidence_label} ({report.confidence:.0%})",
                color=CONFIDENCE_COLOR[report.confidence_label],
                icon=":material/insights:",
            )
        if report.escalation and report.status == ANSWERED:
            st.badge(report.escalation.priority, color=PRIORITY_COLOR[report.escalation.priority], icon=":material/flag:")

    st.markdown(f"#### {report.likely_issue}")
    st.write(report.summary)
    for warning in report.warnings:
        st.warning(warning, icon=":material/shield:")

    if report.status == ANSWERED:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("**Recommended troubleshooting steps**")
            render_steps(report.steps)
            if report.fixes:
                st.markdown("**If a check confirms the cause**")
                render_steps(report.fixes, start=len(report.steps) + 1)
        with right:
            esc_ = report.escalation
            with st.container(border=True):
                st.markdown("**Suggested escalation**")
                st.markdown(f"### {esc_.team}")
                st.caption(f"Tier {esc_.tier} · Priority {esc_.priority} · {esc_.team_scope}")
                st.write(esc_.guidance)
                st.caption(esc_.reason)
            if report.follow_up_questions:
                with st.container(border=True):
                    st.markdown("**Ask the user**")
                    st.markdown("\n".join(f"- {q}" for q in report.follow_up_questions))

        with st.expander(
            f"Knowledge base source · {', '.join(s.kb_id for s in report.sources)} matched",
            expanded=False,
            icon=":material/menu_book:",
        ):
            for source in report.sources:
                st.markdown(f"**{source.kb_id} matched** · similarity {source.score:.0%}")
                st.caption(f"Best-matching text: “{source.matched_text}”")
                render_kb_entry(source.entry)
                st.divider()
    else:
        cols = st.columns([3, 2], gap="large")
        with cols[0]:
            st.markdown("**Please share**" if report.status == NEEDS_DETAILS else "**To log a useful ticket, capture**")
            st.markdown("\n".join(f"- {q}" for q in report.follow_up_questions))
        if report.escalation:
            with cols[1], st.container(border=True):
                st.markdown("**Suggested next step**")
                st.markdown(f"### {report.escalation.team}")
                st.write(report.escalation.guidance)

    if report.candidates:
        with st.expander("Retrieval details (semantic search)", expanded=False, icon=":material/manage_search:"):
            st.caption(
                f"Answers require similarity ≥ {get_assistant().min_confidence:.0%}; "
                f"≥ {get_assistant().high_confidence:.0%} is a strong match."
            )
            for cand in report.candidates:
                st.progress(
                    max(0.0, min(1.0, cand.score)),
                    text=f"{cand.kb_id} · {cand.entry['title']} — {cand.score:.0%}",
                )

    if report.prompt:
        title = "Prompt sent to Gemini" if get_assistant().engine else "Grounding prompt (KB-only mode, not sent)"
        with st.expander(title, expanded=False, icon=":material/psychology:"):
            st.caption("System instruction (the rules), then the user message: retrieved KB entries + the question.")
            st.code(SYSTEM_PROMPT, language=None, wrap_lines=True)
            st.code(report.prompt, language=None, wrap_lines=True)

    st.caption(
        f"{report.generated_by} · {report.elapsed_ms} ms · Recommended guidance only — "
        "the engineer confirms whether the issue is resolved."
    )

    if report.status != NEEDS_DETAILS:
        record = history_record(qid)
        current = record.get("outcome") if record else None
        with st.container(horizontal=True):
            if current:
                st.badge(f"Outcome: {OUTCOMES[current]}", color="gray", icon=OUTCOME_ICONS[current])
            for key, text in OUTCOMES.items():
                st.button(
                    text,
                    icon=OUTCOME_ICONS[key],
                    key=f"outcome_{key}_{qid}",
                    on_click=set_outcome,
                    args=(qid, key),
                    type="primary" if current == key else "secondary",
                    disabled=current == key,
                )


def render_metrics() -> None:
    history = st.session_state.history
    total = len(history)
    answered = sum(h["status"] == ANSWERED for h in history)
    resolved = sum(h.get("outcome") == "resolved" for h in history)
    escalated = sum(h.get("outcome") == "escalated" for h in history)
    unanswered = total - answered
    scores = [h["confidence"] for h in history if h["status"] == ANSWERED]
    cols = st.columns(5)
    cols[0].metric("Total queries", total, border=True)
    cols[1].metric("Guidance provided", answered, border=True, help="Questions answered from the knowledge base")
    cols[2].metric("Confirmed resolved", resolved, border=True, help="Marked resolved by the engineer, not by the AI")
    cols[3].metric("Escalated", escalated, border=True)
    cols[4].metric(
        "Avg. confidence",
        f"{sum(scores) / len(scores):.0%}" if scores else "—",
        border=True,
        help=f"{unanswered} question(s) were unknown or needed more details",
    )


# ---------------------------------------------------------------- pages
def page_assistant(assistant: TroubleshootingAssistant) -> None:
    render_metrics()

    if not st.session_state.messages:
        st.markdown("##### Try an example")
        cols = st.columns(3)
        for i, example in enumerate(EXAMPLES):
            cols[i % 3].button(example, key=f"example_{i}", on_click=submit, args=(example,), width="stretch")

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else BOT_AVATAR):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_report(message["report"], message["qid"])

    typed = st.chat_input("Describe your technical issue… (never paste passwords or API keys)")
    query = typed or st.session_state.pending_query
    st.session_state.pending_query = None
    if not query:
        return

    previous = st.session_state.messages[-1]["report"] if st.session_state.messages else None
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)
    with st.chat_message("assistant", avatar=BOT_AVATAR), st.spinner("Searching the knowledge base…"):
        report = assistant.ask(query, previous=previous)

    qid = len(st.session_state.history) + 1
    # Store the redacted query so secrets never stay in the session.
    shown_query, _ = redact_secrets(query)
    st.session_state.messages.append({"role": "user", "content": shown_query})
    st.session_state.messages.append({"role": "assistant", "report": report, "qid": qid})
    top = report.primary_source or (report.candidates[0] if report.candidates else None)
    st.session_state.history.append(
        {
            "id": qid,
            "time": report.timestamp,
            "query": shown_query,
            "status": report.status,
            "kb_id": report.primary_source.kb_id if report.primary_source else "",
            "category": top.entry["category"] if top and report.status == ANSWERED else "",
            "confidence": round(report.confidence, 3),
            "team": report.escalation.team if report.escalation else "",
            "priority": report.escalation.priority if report.escalation and report.status == ANSWERED else "",
            "answered_by": report.generated_by,
            "outcome": None,
        }
    )
    st.rerun()


def page_knowledge_base(assistant: TroubleshootingAssistant) -> None:
    entries = assistant.retriever.entries
    counts = Counter(e["category"] for e in entries)
    cols = st.columns(len(counts) + 1)
    cols[0].metric("KB articles", len(entries), border=True)
    for col, (cat, n) in zip(cols[1:], sorted(counts.items())):
        col.metric(f"{CATEGORY_ICONS.get(cat, '')} {cat}", n, border=True)

    tab_browse, tab_search = st.tabs(["Browse articles", "Test semantic search"])
    with tab_browse:
        f1, f2 = st.columns([2, 3])
        categories = f1.multiselect("Category", sorted(counts), placeholder="All categories")
        text = f2.text_input("Filter by keyword", placeholder="e.g. timeout, VPN, 500")
        filtered = [
            e
            for e in entries
            if (not categories or e["category"] in categories)
            and (not text or text.lower() in " ".join(str(v) for v in e.values()).lower())
        ]
        st.dataframe(
            pd.DataFrame(
                [{"ID": e["id"], "Issue": e["title"], "Category": e["category"], "Escalation team": e["escalation_team"]} for e in filtered]
            ),
            hide_index=True,
            width="stretch",
        )
        for entry in filtered:
            with st.expander(f"{entry['id']} · {entry['title']}", icon=CATEGORY_ICONS.get(entry["category"])):
                render_kb_entry(entry)
    with tab_search:
        st.caption(
            "Semantic search compares meaning, not exact words. Try a phrasing that shares no keywords with the "
            "article, e.g. “the app keeps quitting by itself” (finds Application crash)."
        )
        probe = st.text_input("Search query", key="probe")
        if probe:
            for result in assistant.retriever.search(probe, top_k=5):
                ok = result.score >= assistant.min_confidence
                st.progress(
                    max(0.0, min(1.0, result.score)),
                    text=f"{'Above' if ok else 'Below'} threshold · {result.kb_id} · {result.entry['title']} — {result.score:.0%}",
                )
                st.caption(f"Matched on: “{result.matched_text}”")


def page_history() -> None:
    history = st.session_state.history
    if not history:
        st.info("No questions yet. Ask something on the Assistant page.")
        return
    df = pd.DataFrame(history)
    df["outcome"] = df["outcome"].map(lambda o: OUTCOMES.get(o, "Pending"))
    df["status"] = df["status"].map(lambda s: STATUS_BADGE[s][0])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Queries by result**")
        st.bar_chart(df["status"].value_counts(), horizontal=True, x_label="Queries", y_label="")
    with c2:
        st.markdown("**Answered queries by category**")
        cats = df.loc[df["category"] != "", "category"].value_counts()
        if len(cats):
            st.bar_chart(cats, horizontal=True, x_label="Queries", y_label="")
        else:
            st.caption("No answered queries yet.")

    st.markdown("**Recent queries**")
    st.dataframe(
        df.iloc[::-1][["time", "query", "status", "kb_id", "confidence", "team", "priority", "outcome", "answered_by"]],
        hide_index=True,
        width="stretch",
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="percent"),
            "kb_id": "KB",
        },
    )
    st.download_button("Download history (CSV)", df.to_csv(index=False), file_name="query_history.csv", mime="text/csv")


PIPELINE_DOT = """
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#6366f1", fontname="Helvetica", fontsize=11];
  edge [color="#64748b", fontname="Helvetica", fontsize=9];
  q [label="Engineer's question"];
  g [label="Guardrails\\nredact secrets\\nvague-question check"];
  r [label="Semantic retrieval\\nMiniLM embeddings + FAISS"];
  c [label="Confidence check\\nsimilarity ≥ threshold?", shape=diamond, fillcolor="#fef3c7", color="#d97706"];
  u [label="Not in KB\\n(no guess)", fillcolor="#fee2e2", color="#dc2626"];
  l [label="Gemini\\nprompt = retrieved KB only\\nJSON with source_id per step"];
  v [label="Validator\\ndrop uncited steps\\nrespect AI refusal"];
  e [label="Escalation rules\\nteam from KB\\npriority from impact"];
  out [label="Report\\nsteps · team · sources · confidence", fillcolor="#dcfce7", color="#16a34a"];
  q -> g -> r -> c;
  c -> u [label="no"];
  c -> l [label="yes"];
  l -> v -> e -> out;
}
"""


def page_how_it_works(assistant: TroubleshootingAssistant) -> None:
    st.markdown("#### Retrieval-Augmented Generation (RAG) pipeline")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.caption(
        "Retrieval finds the relevant KB articles · Augmentation puts only those articles into the prompt · "
        "Generation has Gemini write the answer from that context and cite it."
    )

    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(border=True):
        st.markdown("**AI models and tools**")
        engine = assistant.engine
        st.markdown(
            f"- **Embeddings:** `{config.EMBEDDING_MODEL}` (Sentence Transformers, 384-dim, runs locally on CPU)\n"
            f"- **Vector search:** FAISS `IndexFlatIP` (cosine similarity on normalised vectors)\n"
            f"- **LLM:** Google Gemini, `{engine.models[0] if engine else config.GEMINI_MODEL}`"
            + (f", with fallbacks {', '.join(f'`{m}`' for m in engine.models[1:])}" if engine and len(engine.models) > 1 else "")
            + "\n- **Output:** structured JSON (`response_json_schema`) so every step's citation can be checked\n"
            "- **UI:** Streamlit · **Data:** JSON knowledge base (sample data only)"
        )
    with c2, st.container(border=True):
        st.markdown("**How hallucination is prevented**")
        st.markdown(
            f"1. Questions below **{assistant.min_confidence:.0%}** similarity never reach the LLM.\n"
            "2. The prompt contains only the retrieved articles, and the rules forbid outside knowledge.\n"
            "3. Every step must cite a KB ID; steps citing anything else are discarded.\n"
            "4. If Gemini says the context is insufficient, the assistant reports *unknown*.\n"
            "5. The escalation team comes from the KB article, not from the LLM.\n"
            "6. Answers are labelled *recommended guidance*; only the engineer marks an issue resolved."
        )

    st.markdown("#### Prompt design")
    st.caption("System instruction sent with every request. Open “Prompt sent to Gemini” under any answer to see the full prompt.")
    st.code(SYSTEM_PROMPT, language=None, wrap_lines=True)

    st.markdown("#### Decision rules")
    st.dataframe(
        pd.DataFrame(
            [
                {"Situation": "Similarity ≥ " + f"{assistant.high_confidence:.0%}", "Behaviour": "Answer (High confidence)"},
                {"Situation": f"{assistant.min_confidence:.0%} – {assistant.high_confidence:.0%}", "Behaviour": "Answer with “confirm symptoms match” warning (Medium)"},
                {"Situation": f"Similarity < {assistant.min_confidence:.0%}", "Behaviour": "Refuse: knowledge base has insufficient information"},
                {"Situation": "Vague question / ≤ 2 words without a strong match", "Behaviour": "Ask for system, error, start time, users affected"},
                {"Situation": "Gemini unavailable or no API key", "Behaviour": "Answer straight from the KB article (template)"},
                {"Situation": "Production / outage / all users / payment", "Behaviour": "Priority P1: engage owning team now"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    a, b = st.columns(2, gap="large")
    with a, st.container(border=True):
        st.markdown("**Assumptions**")
        st.markdown(
            "- Users are L1/L2 engineers describing issues in English.\n"
            "- The KB is curated and correct; the assistant is only as good as its articles.\n"
            "- Sample/simulated data only; no access to production systems or tickets.\n"
            "- The engineer, not the AI, decides when an issue is resolved."
        )
    with b, st.container(border=True):
        st.markdown("**Known limitations**")
        st.markdown(
            "- Only 20 sample articles; anything else is (correctly) refused.\n"
            "- Thresholds were calibrated on a small question set.\n"
            "- Priority detection is keyword-based.\n"
            "- Session-only history; no login or ticketing integration.\n"
            "- Gemini wording can vary between runs; facts are constrained, phrasing is not."
        )


# ---------------------------------------------------------------- layout
def main() -> None:
    init_state()
    assistant = get_assistant()

    with st.sidebar:
        st.markdown("## :material/support_agent: AI TroubleshootPro")
        st.caption("Intelligent IT support assistant for L1/L2 engineers")
        page = st.radio(
            "Navigate", ["Assistant", "Knowledge base", "Query history", "How it works"], label_visibility="collapsed"
        )
        st.divider()
        st.markdown("**System status**")
        if assistant.engine:
            st.badge(f"AI: {assistant.engine.model}", color="green", icon=":material/smart_toy:")
        else:
            st.badge("AI: off — KB-only answers", color="orange", icon=":material/smart_toy:")
            st.caption("Set GEMINI_API_KEY in .env to enable AI-written answers.")
        st.badge(f"KB: {len(assistant.retriever.entries)} articles", color="blue", icon=":material/menu_book:")
        st.caption(
            f"Search: {config.EMBEDDING_MODEL.split('/')[-1]} + FAISS · "
            f"answer threshold {assistant.min_confidence:.0%}"
        )
        st.divider()
        if st.button("New conversation", icon=":material/refresh:", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        st.caption(":material/lock: Never share passwords, API keys or tokens here. Detected secrets are redacted automatically.")

    st.markdown(
        "<div class='tp-header'><h1>AI Troubleshooting Assistant</h1></div>"
        "<p class='tp-sub'>Answers come only from the support knowledge base, with sources, confidence and escalation.</p>",
        unsafe_allow_html=True,
    )

    if page == "Assistant":
        page_assistant(assistant)
    elif page == "Knowledge base":
        page_knowledge_base(assistant)
    elif page == "Query history":
        page_history()
    else:
        page_how_it_works(assistant)


main()
