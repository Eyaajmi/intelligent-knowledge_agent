"""Recherche hybride : sémantique (matrice numpy) + mots-clés (TF-IDF), fusion RRF."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import ollama_client
from .config import settings
from .store import IndexStore, normalize_text

RRF_K = 60
MODES = ("hybrid", "semantic", "keyword")


def _rank(scores: np.ndarray, mask: np.ndarray, k: int, positive_only: bool = False) -> list[int]:
    """Indices des k meilleurs scores parmi les passages autorisés par `mask`."""
    allowed = int(mask.sum())
    k = min(k, allowed)
    if k <= 0:
        return []
    masked = np.where(mask, scores, -np.inf)
    top = np.argpartition(-masked, k - 1)[:k]
    top = top[np.argsort(-masked[top])]
    return [int(i) for i in top if not positive_only or masked[i] > 0]


def retrieve(
    store: IndexStore,
    query: str,
    *,
    top_k: int | None = None,
    source: str | None = None,
    mode: str = "hybrid",
    query_vector: list[float] | None = None,
) -> list[dict]:
    """Retourne les meilleurs passages, sans doublons de texte.

    Chaque résultat contient `semantic_score` (cosinus) et `keyword_score` (TF-IDF),
    calculés pour TOUS les candidats, ce qui permet un garde-fou fiable avant le LLM.
    """
    if mode not in MODES:
        raise ValueError(f"mode inconnu : {mode}")
    top_k = top_k or settings.top_k
    snap = store.snapshot()
    if snap.size == 0:
        return []
    mask = snap.mask(source)
    if not mask.any():
        return []

    semantic = keyword = None
    rankings: list[list[int]] = []
    if mode in ("hybrid", "semantic"):
        vector = query_vector if query_vector is not None else ollama_client.embed_query(query)
        semantic = snap.semantic_scores(vector)
        rankings.append(_rank(semantic, mask, settings.candidate_k))
    if mode in ("hybrid", "keyword"):
        keyword = snap.keyword_scores(query)
        rankings.append(_rank(keyword, mask, settings.candidate_k, positive_only=True))

    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for position, i in enumerate(ranking, start=1):
            fused[i] += 1.0 / (RRF_K + position)

    results, seen = [], set()
    for i in sorted(fused, key=lambda i: -fused[i]):
        chunk = snap.chunks[i]
        key = normalize_text(chunk["text"])
        if key in seen:  # filet de sécurité contre d'éventuels doublons
            continue
        seen.add(key)
        results.append(
            {
                **chunk,
                "score": fused[i],
                "semantic_score": float(semantic[i]) if semantic is not None else None,
                "keyword_score": float(keyword[i]) if keyword is not None else None,
            }
        )
        if len(results) >= top_k:
            break
    return results
