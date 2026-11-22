"""Semantic retrieval over the knowledge base with Sentence Transformers + FAISS.

Each KB entry is split into several short "views" (its title, each symptom paired with the
title, and its keywords). Every view is embedded and stored in a FAISS inner-product index.
At query time we score every view and keep the best score per entry, so a query that
matches just one symptom still finds the right article.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import config

REQUIRED_FIELDS = (
    "id",
    "title",
    "category",
    "symptoms",
    "possible_causes",
    "diagnostic_checks",
    "resolution",
    "escalation_team",
)


def load_knowledge_base(path: Path | str = config.KB_PATH) -> list[dict[str, Any]]:
    """Load and validate the JSON knowledge base."""
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)

    seen: set[str] = set()
    for entry in entries:
        missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
        if missing:
            raise ValueError(f"KB entry {entry.get('id', '?')} is missing fields: {missing}")
        if entry["id"] in seen:
            raise ValueError(f"Duplicate KB id: {entry['id']}")
        seen.add(entry["id"])
    return entries


def entry_views(entry: dict[str, Any]) -> list[str]:
    """Short texts that represent one KB entry in the vector index."""
    title = entry["title"]
    views = [f"{title} ({entry['category']} issue)"]
    views += [f"{title}: {symptom}" for symptom in entry["symptoms"]]
    if entry.get("keywords"):
        views.append(f"{title}. Keywords: {', '.join(entry['keywords'])}")
    return views


@dataclass
class SearchResult:
    entry: dict[str, Any]
    score: float  # cosine similarity of the best-matching view, roughly 0..1 for real queries
    matched_text: str  # which view of the entry matched best (useful for explaining a match)

    @property
    def kb_id(self) -> str:
        return self.entry["id"]


class KnowledgeBaseRetriever:
    """Embeds the KB once and answers nearest-neighbour queries."""

    def __init__(self, entries: list[dict[str, Any]], model: Any = None, model_name: str = config.EMBEDDING_MODEL):
        import faiss

        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self.model = model
        self.entries = entries

        self._view_texts: list[str] = []
        self._view_owner: list[int] = []  # view index -> entry index
        for idx, entry in enumerate(entries):
            for view in entry_views(entry):
                self._view_texts.append(view)
                self._view_owner.append(idx)

        vectors = self._embed(self._view_texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def search(self, query: str, top_k: int = config.TOP_K) -> list[SearchResult]:
        """Return up to `top_k` distinct KB entries, best first."""
        query = query.strip()
        if not query:
            return []

        scores, ids = self.index.search(self._embed([query]), len(self._view_texts))
        best: dict[int, SearchResult] = {}
        for score, view_id in zip(scores[0], ids[0]):
            if view_id < 0:
                continue
            owner = self._view_owner[view_id]
            if owner not in best:  # results come back sorted, so the first hit per entry is its best
                best[owner] = SearchResult(
                    entry=self.entries[owner],
                    score=float(score),
                    matched_text=self._view_texts[view_id],
                )
            if len(best) >= top_k:
                break
        return sorted(best.values(), key=lambda r: r.score, reverse=True)
