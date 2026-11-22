# AI Troubleshooting Assistant for L1/L2 Support

An AI-powered technical support assistant built using **Python, Streamlit, FAISS, Sentence Transformers, and Google Gemini**. It uses Retrieval-Augmented Generation (RAG) to identify technical issues, retrieve relevant knowledge-base solutions, provide step-by-step troubleshooting guidance, and recommend the appropriate escalation team.

The system includes semantic search, confidence-based filtering, source citations, secure handling of sensitive information, and a KB-only fallback to ensure reliable, grounded responses.

**Key Features:**
- Semantic search using FAISS and Sentence Transformers.
- AI-generated troubleshooting recommendations using Google Gemini.
- Knowledge-base citations for transparent and grounded answers.
- Automatic escalation team and priority recommendations.
- Unknown issue detection to prevent hallucinated solutions.
- Clarifying questions for incomplete problem descriptions.
- Sensitive information detection and redaction.
- KB-only fallback when Gemini is unavailable.
- Interactive Streamlit dashboard and troubleshooting history.
- Automated testing and evaluation.

**Tech Stack:** Python, Streamlit, Google Gemini, FAISS, Sentence Transformers, PyTorch, RAG, JSON, Pytest.

# IncidentIQ – AI-Powered Incident Analyzer

IncidentIQ is an AI-powered application designed to help Technology Operations and IT Support teams investigate application incidents by analyzing application logs. It automatically identifies errors, detects repeated failure patterns, classifies incident severity, and suggests potential root causes with practical troubleshooting recommendations.

The application combines Python-based log processing with AI-powered analysis to generate structured, evidence-based incident reports, helping support engineers understand issues faster and make better troubleshooting decisions.

### Key Features
- **Log Upload & Input:** Upload log files or paste application logs directly.
- **Error Detection:** Identify errors, warnings, exceptions, and repeated failures.
- **Pattern Recognition:** Detect recurring errors and analyze log patterns.
- **AI Root Cause Analysis:** Suggest possible causes based on the available log evidence.
- **Incident Severity Classification:** Categorize incidents as Critical, High, Medium, or Low.
- **Troubleshooting Recommendations:** Provide actionable investigation steps and escalation suggestions.
- **Incident Summary:** Generate concise reports for operations and engineering teams.

### Technology Stack
- Python
- Streamlit
- Pandas
- Regular Expressions (Regex)
- Google Gemini API
- Plotly

### Objective
To simplify incident investigation, reduce manual log analysis efforts, and support faster incident triage through AI-assisted technical analysis.

**Note:** This project uses sample and simulated application logs and is intended as a prototype for incident investigation and technical operations support.
