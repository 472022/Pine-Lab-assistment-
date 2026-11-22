# Live demo script (about 3 minutes)

**Before the presentation**
- Start the app: `streamlit run app.py` and open http://localhost:8501.
- Ask one throwaway question so the embedding model is already loaded. Then click **New conversation**.
- Backup if Wi-Fi or Gemini fails: the app keeps working in KB-only mode (answers come straight from the KB). Say so.
  If you have to use it, remove `GEMINI_API_KEY` from `.env` and restart the app.

| # | Type / click | What to point out |
|---|---|---|
| 1 | Example button **"The payment API is returning a 500 error."** (the brief's example) | Likely issue, ordered checks (logs, dependencies, deployments, config), **Backend / API Support Team**, confidence badge. Open **Knowledge base source** to show KB-001 was used. |
| 2 | Under the same answer, open **Prompt sent to Gemini** | Prompt engineering: rules plus *only* the retrieved entries; JSON output with a `source_id` per step. |
| 3 | `Our database connection keeps failing in production` | Different category → Database Support Team. "production" raises priority to **P1**. |
| 4 | `VPN is not connecting from home` | Network troubleshooting → Network Support Team. |
| 5 | `My office printer keeps jamming paper` | **Not in KB**: no steps invented, the LLM is not even called, ticket checklist instead. Open **Retrieval details** to show scores below the 50% threshold. |
| 6 | `It's not working, please help` → then `the CRM app is very slow for all users` | Vague question → asks for details. The follow-up is combined and answered (slow application, P1). |
| 7 | `DB login fails, my password is Test123` | Password is redacted with a warning; never stored or sent to the AI. |
| 8 | Click **Resolved** on one answer → sidebar **Query history** | Engineer confirms the outcome (AI never claims "resolved"); dashboard counts and history table. |
| 9 | Sidebar **Knowledge base** → *Test semantic search*: `the app keeps quitting by itself` | Finds **KB-018 Application crash** (~62%) although they share no words: semantic vs keyword search. |
| 10 | Sidebar **How it works** | Pipeline diagram, guardrails, decision rules, assumptions, limitations. Good page to keep open for Q&A. |

**Likely Q&A, one-line answers**
- *What is RAG?* Retrieve relevant KB entries, give only those to the LLM as context, generate an answer from them.
- *How do you prevent hallucination?* Confidence threshold before the LLM; context-only prompt; required citations, checked in code; AI refusal respected; team taken from the KB.
- *Keyword vs semantic search?* Keyword needs shared words; embeddings match meaning ("the app keeps quitting by itself" → Application crash).
- *How is the team chosen?* From the matched KB entry; priority from impact words.
- *What if the KB has no answer?* Similarity below 0.50 → "insufficient information" + what to capture for a ticket.
- *How did you test accuracy?* 51 automated tests + 20 labelled scenarios, all correct in KB-only and Gemini modes; thresholds calibrated from score gaps (known 0.54–0.93 vs unrelated ≤ 0.42).
- *Limitations?* Small KB, thresholds tuned on few questions, keyword-based priority, session-only history, LLM wording varies.
