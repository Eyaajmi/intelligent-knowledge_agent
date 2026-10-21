"""Découpage du texte en passages (unique implémentation)."""
from __future__ import annotations

import re

from .config import settings

# Fin de phrase suivie d'une majuscule, d'un chiffre, d'une puce ou d'un guillemet ouvrant.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ0-9«\"“(•\-–])")


def _paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # mots coupés en fin de ligne
    parts = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in parts if p]


def _hard_split(sentence: str, size: int) -> list[str]:
    """Découpe sur les mots une « phrase » plus longue que `size`."""
    if len(sentence) <= size:
        return [sentence]
    out, cur = [], ""
    for word in sentence.split():
        if cur and len(cur) + 1 + len(word) > size:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def _joined_len(sentences: list[str]) -> int:
    return sum(len(s) for s in sentences) + max(len(sentences) - 1, 0)


def split_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[str]:
    """Regroupe des phrases entières en passages de `chunk_size` caractères au plus,
    avec un chevauchement d'au plus `overlap` caractères (phrases complètes)."""
    chunk_size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    min_chars = settings.min_chunk_chars if min_chars is None else min_chars

    units: list[str] = []
    for paragraph in _paragraphs(text):
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.extend(_hard_split(sentence, chunk_size))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and _joined_len(current) + 1 + len(unit) > chunk_size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for sentence in reversed(current):  # chevauchement par phrases entières
                if _joined_len([sentence] + tail) > overlap:
                    break
                tail.insert(0, sentence)
            current = tail
            while current and _joined_len(current) + 1 + len(unit) > chunk_size:
                current.pop(0)
        current.append(unit)
    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c) >= min_chars]
