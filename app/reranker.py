from app.rag_pipeline import get_embedding, cosine_similarity


def rerank(query, results):
    """
    Re-classe les résultats en combinant
    la similarité sémantique et la similarité lexicale.
    """

    query_embedding = get_embedding(query)

    reranked_results = []

    for result in results:

        passage = result["text"]

        # Similarité sémantique directe
        passage_embedding = get_embedding(passage)

        semantic_score = cosine_similarity(
            query_embedding,
            passage_embedding
        )

        # Similarité lexicale simple
        query_words = set(
            query.lower().split()
        )

        passage_words = set(
            passage.lower().split()
        )

        common_words = query_words.intersection(
            passage_words
        )

        if len(query_words) > 0:
            lexical_score = (
                len(common_words) / len(query_words)
            )
        else:
            lexical_score = 0

        # Score final
        rerank_score = (
            0.7 * semantic_score
            + 0.3 * lexical_score
        )

        new_result = result.copy()

        new_result["direct_semantic_score"] = semantic_score
        new_result["lexical_score"] = lexical_score
        new_result["rerank_score"] = rerank_score

        reranked_results.append(new_result)

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True
    )

    return reranked_results