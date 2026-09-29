from app.rag_pipeline import load_index, search
from app.keyword_search import create_keyword_index, keyword_search
from app.hybrid_search import (
    reciprocal_rank_fusion,
    group_results_by_page
)



index = load_index("data/index.json")



vectorizer, matrix = create_keyword_index(index)



query = "Combien de jours au maximum un salarié peut-il être suspendu ?"


semantic_results = search(
    query,
    index,
    top_k=10
)



keyword_results = keyword_search(
    query,
    index,
    vectorizer,
    matrix,
    top_k=10
)



hybrid_results = reciprocal_rank_fusion(
    [
        semantic_results,
        keyword_results
    ]
)
hybrid_results = group_results_by_page(
    hybrid_results
)


print("\n==============================")
print("SEMANTIC SEARCH")
print("==============================")


for rank, result in enumerate(semantic_results, start=1):

    print(
        rank,
        "| Page:",
        result["page"],
        "| Chunk:",
        result["chunk_id"],
        "| Score:",
        round(result["score"], 4)
    )


print("\n==============================")
print("KEYWORD SEARCH")
print("==============================")


for rank, result in enumerate(keyword_results, start=1):

    print(
        rank,
        "| Page:",
        result["page"],
        "| Chunk:",
        result["chunk_id"],
        "| Score:",
        round(result["score"], 4)
    )


print("\n==============================")
print("HYBRID SEARCH - RRF")
print("==============================")


for rank, result in enumerate(hybrid_results[:10], start=1):

    print(
        rank,
        "| Page:",
        result["page"],
        "| Chunk:",
        result["chunk_id"],
        "| Hybrid score:",
        round(result["hybrid_score"], 6)
    )