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

    def best(cases, key):
        return [
            max((r[key] or 0.0 for r in retrieve(store, c["question"], top_k=args.k, query_vector=vectors[c["question"]])), default=0.0)
            for c in cases
        ]

    print("\nCalibrage du garde-fou (meilleur score parmi les passages retournés)")
    for key, setting_name in (("semantic_score", "MIN_SEMANTIC_SCORE"), ("keyword_score", "MIN_KEYWORD_SCORE")):
        ok, ko = best(TEST_CASES, key), best(UNANSWERABLE_CASES, key)
        print(f"  {key}: avec réponse  min={min(ok):.3f} médiane={statistics.median(ok):.3f} | sans réponse  max={max(ko):.3f} médiane={statistics.median(ko):.3f}")
        if max(ko) < min(ok):
            print(f"    -> séparation nette : {setting_name}={(max(ko) + min(ok)) / 2:.2f} (actuel : {getattr(settings, setting_name.lower()):.2f})")
        else:
            print(f"    -> recouvrement : ce score seul ne sépare pas les cas ; garde {setting_name} bas (actuel : {getattr(settings, setting_name.lower()):.2f})")
    print("\nNote : un seuil n'attrape que les questions hors sujet. Les questions voisines du corpus mais sans réponse")
    print("restent gérées par le prompt (réponse exacte d'abstention) : ajoute-en dans UNANSWERABLE_CASES.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
