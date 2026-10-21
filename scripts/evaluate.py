"""Évalue le VRAI pipeline de recherche (semantic vs keyword vs hybrid) et aide à calibrer le seuil.

    python -m scripts.evaluate [--index data/index.json] [--k 5]

Nécessite Ollama (embeddings des questions). Les cas sont dans tests/retrieval_cases.py.
"""
from __future__ import annotations

import argparse
import statistics
import sys

from app import ollama_client
from app.config import settings
from app.ollama_client import OllamaError
from app.search import MODES, retrieve
from app.store import IndexStore
from tests.retrieval_cases import TEST_CASES, UNANSWERABLE_CASES


def _rank_of_expected(results: list[dict], case: dict) -> int | None:
    for rank, r in enumerate(results, 1):
        if r["page"] == case["expected_page"] and (not case.get("expected_source") or case["expected_source"] in r["source"]):
            return rank
    return None


SEM_MARGIN = 0.05   # marge de sécurité sous la plus basse question valide
KW_MARGIN = 0.02
MIN_RELIABLE_QUESTIONS = 30


def passes_gate(score: dict, min_sem: float, min_kw: float) -> bool:
    """Même règle que app.agent._is_answerable."""
    return score["sem"] >= min_sem or score["kw"] >= min_kw


def gate_refusals(answerable: list[dict], unanswerable: list[dict], min_sem: float, min_kw: float) -> tuple[int, int]:
    """(questions valides refusées à tort, questions sans réponse refusées)."""
    wrongly = sum(not passes_gate(s, min_sem, min_kw) for s in answerable)
    caught = sum(not passes_gate(s, min_sem, min_kw) for s in unanswerable)
    return wrongly, caught


def suggest_thresholds(answerable: list[dict]) -> tuple[float, float]:
    """Seuils juste sous la question valide la moins bien notée, avec une marge de sécurité :
    on préfère laisser passer une question douteuse que refuser une question valide."""
    sem = max(0.0, min(s["sem"] for s in answerable) - SEM_MARGIN)
    kw = max(0.0, min(s["kw"] for s in answerable) - KW_MARGIN)
    return round(sem, 2), round(kw, 2)


def _scores(store, cases, vectors, k):
    out = []
    for c in cases:
        res = retrieve(store, c["question"], top_k=k, query_vector=vectors[c["question"]])
        out.append({
            "question": c["question"],
            "sem": max((r["semantic_score"] or 0.0 for r in res), default=0.0),
            "kw": max((r["keyword_score"] or 0.0 for r in res), default=0.0),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(settings.index_path))
    ap.add_argument("--k", type=int, default=settings.top_k)
    args = ap.parse_args()

    store = IndexStore(args.index)
    if store.snapshot().size == 0:
        print(f"Index vide ou introuvable : {args.index}")
        return 1

    questions = [c["question"] for c in TEST_CASES + UNANSWERABLE_CASES]
    try:
        vectors = dict(zip(questions, [ollama_client.embed_query(q) for q in questions]))
    except OllamaError as exc:
        print(f"Ollama indisponible : {exc}")
        return 1

    print(f"Index : {store.snapshot().size} passages, {len(store.documents())} documents | k={args.k} | {len(TEST_CASES)} questions\n")
    print(f"{'mode':<10} {'Recall@' + str(args.k):>10} {'MRR':>8}")
    for mode in MODES:
        ranks = [
            _rank_of_expected(retrieve(store, c["question"], top_k=args.k, mode=mode, query_vector=vectors[c["question"]]), c)
            for c in TEST_CASES
        ]
        recall = sum(r is not None for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        print(f"{mode:<10} {recall:>10.0%} {mrr:>8.3f}")

    ok = _scores(store, TEST_CASES, vectors, args.k)
    ko = _scores(store, UNANSWERABLE_CASES, vectors, args.k)
    cur_sem, cur_kw = settings.min_semantic_score, settings.min_keyword_score

    print("\nScores du garde-fou (meilleur score parmi les passages retournés)")
    for key, label in (("sem", "sémantique"), ("kw", "mots-clés")):
        a, b = [s_[key] for s_ in ok], [s_[key] for s_ in ko]
        print(f"  {label:<11} avec réponse : min={min(a):.3f} médiane={statistics.median(a):.3f} | "
              f"sans réponse : max={max(b):.3f} médiane={statistics.median(b):.3f}")

    wrongly, caught = gate_refusals(ok, ko, cur_sem, cur_kw)
    print(f"\nGarde-fou ACTUEL (MIN_SEMANTIC_SCORE={cur_sem:.2f}, MIN_KEYWORD_SCORE={cur_kw:.2f}) :")
    print(f"  questions valides refusées à tort : {wrongly}/{len(ok)}")
    print(f"  questions sans réponse refusées   : {caught}/{len(ko)}")
    if caught == 0:
        print("  -> ATTENTION : le garde-fou ne refuse aucune question sans réponse (il est sans effet).")

    sug_sem, sug_kw = suggest_thresholds(ok)
    wrongly_s, caught_s = gate_refusals(ok, ko, sug_sem, sug_kw)
    print(f"\nRéglage SUGGÉRÉ : MIN_SEMANTIC_SCORE={sug_sem:.2f}  MIN_KEYWORD_SCORE={sug_kw:.2f}")
    print(f"  (marges de sécurité : {SEM_MARGIN} sémantique, {KW_MARGIN} mots-clés sous la question valide la plus faible)")
    print(f"  -> questions valides refusées : {wrongly_s}/{len(ok)} | questions sans réponse refusées : {caught_s}/{len(ko)}")
    if caught_s == 0:
        print("  -> ATTENTION : ce réglage ne refuse aucune question sans réponse. Les scores ne séparent pas les cas :")
        print("     le garde-fou serait sans effet et seul le prompt protège contre l'hallucination.")

    print("\nDétail des questions sans réponse (sémantique / mots-clés / décision avec le réglage suggéré) :")
    for s_ in ko:
        verdict = "passe" if passes_gate(s_, sug_sem, sug_kw) else "REFUSÉE"
        print(f"  {s_['sem']:.3f} / {s_['kw']:.3f}  {verdict:<8} {s_['question'][:60]}")

    if len(ok) + len(ko) < MIN_RELIABLE_QUESTIONS:
        print(f"\n/!\\ Seulement {len(ok) + len(ko)} questions : ces seuils sont fragiles (vise {MIN_RELIABLE_QUESTIONS}+,")
        print("    sur plusieurs documents, avec des questions sans réponse voisines de ton corpus).")
    print("Un seuil n'attrape que les questions hors sujet : une question voisine mais sans réponse")
    print("reste gérée par le prompt. Recalibre après toute réindexation ou changement de modèle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
