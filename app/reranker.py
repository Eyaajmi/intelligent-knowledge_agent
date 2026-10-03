def normalize_scores(results, score_key):
    """
    Normalise un ensemble de scores entre 0 et 1
    avec une normalisation min-max.
    """

    scores = [
        result.get(score_key, 0)
        for result in results
    ]

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def rerank(results):
    """
    Re-classe les résultats en combinant
    les scores semantic et keyword.
    """

    semantic_scores = normalize_scores(
        results,
        "semantic_score"
    )

    keyword_scores = normalize_scores(
        results,
        "keyword_score"
    )

    reranked_results = []

    for index, result in enumerate(results):

        rerank_score = (
            0.6 * semantic_scores[index]
            + 0.4 * keyword_scores[index]
        )

        new_result = result.copy()

        new_result["rerank_score"] = rerank_score

        reranked_results.append(new_result)

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True
    )

    return reranked_results