"""Lecture des PDF."""
from __future__ import annotations

try:
    import pymupdf
except ImportError:  # anciennes versions de PyMuPDF
    import fitz as pymupdf  # type: ignore


class InvalidPdfError(ValueError):
    """Le fichier n'est pas un PDF lisible."""


class NoExtractableTextError(ValueError):
    """Le PDF ne contient pas de texte (probablement un scan : il faut un OCR)."""


MIN_TOTAL_CHARS = 30


def load_pdf(path) -> list[dict]:
    """Retourne [{page, text}] ; lève une erreur explicite si le PDF est illisible ou scanné."""
    try:
        document = pymupdf.open(str(path))
    except Exception as exc:  # pymupdf lève plusieurs types d'erreurs
        raise InvalidPdfError(f"PDF illisible : {exc}") from exc

    try:
        pages = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(document)]
    finally:
        document.close()

    if sum(len(p["text"].strip()) for p in pages) < MIN_TOTAL_CHARS:
        raise NoExtractableTextError(
            "Aucun texte exploitable : ce PDF est probablement un scan. "
            "Passe-le d'abord dans un OCR (ex. `ocrmypdf entree.pdf sortie.pdf`)."
        )
    return pages
