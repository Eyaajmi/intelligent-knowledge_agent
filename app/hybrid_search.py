def reciprocal_rank_fusion(result_lists, k=60):
    
    scores = {}
    documents = {}

    for list_name, results in result_lists:

        for rank, result in enumerate(results, start=1):

            chunk_id = result["chunk_id"]

            rrf_score = 1 / (k + rank)

            if chunk_id not in scores:
                scores[chunk_id] = 0
                documents[chunk_id] = result.copy()

            scores[chunk_id] += rrf_score

            if list_name == "semantic":
                documents[chunk_id]["semantic_score"] = result["score"]

            elif list_name == "keyword":
                documents[chunk_id]["keyword_score"] = result["score"]

    ranked_chunk_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    hybrid_results = []

    for chunk_id in ranked_chunk_ids:

        result = documents[chunk_id].copy()

        result["hybrid_score"] = scores[chunk_id]

        hybrid_results.append(result)

    return hybrid_results
def group_results_by_page(results):
   
    pages = {}

    for result in results:

        page = result["page"]

        if page not in pages:
            pages[page] = result

        elif result["hybrid_score"] > pages[page]["hybrid_score"]:
            pages[page] = result

    ranked_results = sorted(
        pages.values(),
        key=lambda result: result["hybrid_score"],
        reverse=True
    )

    return ranked_results