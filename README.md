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
