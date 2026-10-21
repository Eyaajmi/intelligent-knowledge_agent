"""Outils de l'agent (hors recherche) : lister et résumer les documents."""
from __future__ import annotations

import numpy as np

from .store import IndexStore

NO_DOCS = "Aucun document n'est indexé pour le moment. Ajoute un PDF pour commencer."


def list_documents(store: IndexStore) -> str:
    docs = store.documents()
    if not docs:
        return NO_DOCS
    lines = [f"- {d['name']} ({d['pages']} pages, {d['chunks']} passages)" for d in docs]
    return "Documents indexés :\n" + "\n".join(lines)


def summary_passages(store: IndexStore, source: str, max_chunks: int = 12) -> list[dict]:
    """Passages répartis uniformément dans le document, dans l'ordre de lecture."""
    snap = store.snapshot()
    idx = [i for i, c in enumerate(snap.chunks) if c["source"] == source]
    idx.sort(key=lambda i: (snap.chunks[i]["page"], snap.chunks[i]["chunk_id"]))
    if len(idx) > max_chunks:
        picks = np.unique(np.linspace(0, len(idx) - 1, max_chunks).round().astype(int))
        idx = [idx[p] for p in picks]
    return [snap.chunks[i] for i in idx]
