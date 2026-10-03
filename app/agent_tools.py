from app.rag_pipeline import load_index, search
from app.keyword_search import (
    keyword_search,
    create_keyword_index
)
from app.hybrid_search import reciprocal_rank_fusion
from app.reranker import rerank


INDEX_PATH = "data/index.json"


def search_documents(query, top_k=5):
    """
    Recherche les passages les plus pertinents
    dans les documents disponibles.
    """

    index = load_index(INDEX_PATH)

    # Recherche sémantique
    semantic_results = search(
        query,
        index,
        top_k=10
    )

    # Création de l'index lexical
    vectorizer, matrix = create_keyword_index(index)

    # Recherche lexicale
    keyword_results = keyword_search(
        query,
        index,
        vectorizer,
        matrix,
        top_k=10
    )

    # Recherche hybride
    hybrid_results = reciprocal_rank_fusion(
        [
            ("semantic", semantic_results),
            ("keyword", keyword_results)
        ]
    )

    # Reranking
    reranked_results = rerank(
        query,
        hybrid_results
    )

    return reranked_results[:top_k]