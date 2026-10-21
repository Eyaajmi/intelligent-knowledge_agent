"""Indexation d'un PDF : lecture -> découpage -> embeddings -> store."""
from __future__ import annotations

from . import ollama_client
from .chunking import split_text
from .loader import NoExtractableTextError, load_pdf
from .store import IndexStore, canonical_source


def index_pdf(store: IndexStore, path, source: str | None = None) -> dict:
    """Indexe `path` sous l'identifiant `source` (par défaut : le chemin canonique)."""
    source = canonical_source(source or path)
    pages = load_pdf(path)

    items = [
        {"text": chunk, "page": page["page"]}
        for page in pages
        for chunk in split_text(page["text"])
    ]
    if not items:
        raise NoExtractableTextError("Aucun passage exploitable n'a pu être extrait de ce PDF.")

    embeddings = ollama_client.embed_documents([it["text"] for it in items])
    added = store.add_document(source, items, embeddings)
    return {"source": source, "chunks": added, "pages": len(pages)}
