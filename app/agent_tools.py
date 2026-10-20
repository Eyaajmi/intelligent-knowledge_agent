from functools import lru_cache
import hashlib
import os

from app.rag_pipeline import load_index, search
from app.keyword_search import keyword_search, create_keyword_index
from app.hybrid_search import reciprocal_rank_fusion
from app.reranker import rerank


INDEX_PATH = "data/index.json"


@lru_cache(maxsize=32)
def _get_keyword_index(index_hash, source):
    """Cache the TF-IDF index until the JSON index changes."""
    index = load_index(INDEX_PATH)

    if source:
        index = [
            item for item in index
            if item["source"].replace("\\", "/") == source
        ]

    if not index:
        return [], None, None

    vectorizer, matrix = create_keyword_index(index)
    return index, vectorizer, matrix


def search_documents(query, top_k=5, source=None):
    if not os.path.exists(INDEX_PATH):
        return []

    with open(INDEX_PATH, "rb") as file:
        index_hash = hashlib.sha256(file.read()).hexdigest()

    normalized_source = source.replace("\\", "/") if source else None

    index, vectorizer, matrix = _get_keyword_index(
        index_hash,
        normalized_source,
    )

    if not index:
        return []

    # Semantic search
    semantic_results = search(query, index, top_k=10)

    # Keyword search using the cached TF-IDF index
    keyword_results = keyword_search(
        query,
        index,
        vectorizer,
        matrix,
        top_k=10,
    )

    # Hybrid search
    hybrid_results = reciprocal_rank_fusion(
        [
            ("semantic", semantic_results),
            ("keyword", keyword_results),
        ]
    )

    # Reranking
    reranked_results = rerank(query, hybrid_results)

    return reranked_results[:top_k]
