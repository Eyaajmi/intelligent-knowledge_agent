from app.rag_pipeline import load_index, search
from app.keyword_search import create_keyword_index, keyword_search
from app.hybrid_search import (
    reciprocal_rank_fusion,
    group_results_by_page
)
from app.reranker import rerank


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
        ("semantic", semantic_results),
        ("keyword", keyword_results)
    ]
)

hybrid_results = group_results_by_page(
    hybrid_results
)


reranked_results = rerank(
    hybrid_results
)


print("\n==============================")
print("HYBRID RESULTS")
print("==============================")

for rank, result in enumerate(hybrid_results[:10], start=1):

    print(
    rank,
    "| Page:",
    result["page"],
    "| Semantic:",
    round(result.get("semantic_score", 0), 4),
    "| Keyword:",
    round(result.get("keyword_score", 0), 4),
    "| RRF:",
    round(result["hybrid_score"], 6)
)


print("\n==============================")
print("RERANKED RESULTS")
print("==============================")

for rank, result in enumerate(reranked_results[:10], start=1):

  
 print(
    rank,
    "| Page:",
    result["page"],
    "| Semantic:",
    round(result.get("semantic_score", 0), 4),
    "| Keyword:",
    round(result.get("keyword_score", 0), 4),
    "| Rerank:",
    round(result["rerank_score"], 4)
)
    