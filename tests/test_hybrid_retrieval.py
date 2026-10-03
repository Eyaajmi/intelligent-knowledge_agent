from app.rag_pipeline import load_index, search
from app.keyword_search import create_keyword_index, keyword_search
from app.hybrid_search import (
    reciprocal_rank_fusion,
    group_results_by_page
)
from app.reranker import rerank
from tests.retrieval_cases import TEST_CASES


index = load_index("data/index.json")

vectorizer, matrix = create_keyword_index(index)


recall_at_3 = 0
recall_at_10 = 0

mrr_at_3 = 0
mrr_at_10 = 0


for case in TEST_CASES:

    question = case["question"]
    expected_page = case["expected_page"]

    semantic_results = search(
        question,
        index,
        top_k=10
    )

    keyword_results = keyword_search(
        question,
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

    retrieved_pages = [
        result["page"]
        for result in reranked_results[:10]
    ]

    # Recall@3
    if expected_page in retrieved_pages[:3]:
        recall_at_3 += 1

    # Recall@10
    if expected_page in retrieved_pages[:10]:
        recall_at_10 += 1

    # MRR@3
    for rank, page in enumerate(
        retrieved_pages[:3],
        start=1
    ):
        if page == expected_page:
            mrr_at_3 += 1 / rank
            break

    # MRR@10
    for rank, page in enumerate(
        retrieved_pages[:10],
        start=1
    ):
        if page == expected_page:
            mrr_at_10 += 1 / rank
            break

    print("\nQuestion:", question)
    print("Expected page:", expected_page)
    print("Retrieved pages:", retrieved_pages)

    if expected_page in retrieved_pages:
        rank = retrieved_pages.index(expected_page) + 1
        print("Relevant page rank:", rank)
    else:
        print("Relevant page rank: NOT FOUND")


number_of_cases = len(TEST_CASES)

recall_at_3 /= number_of_cases
recall_at_10 /= number_of_cases

mrr_at_3 /= number_of_cases
mrr_at_10 /= number_of_cases


print("\n==============================")
print("RERANKER EVALUATION")
print("==============================")

print("Recall@3:", recall_at_3)
print("Recall@10:", recall_at_10)
print("MRR@3:", mrr_at_3)
print("MRR@10:", mrr_at_10)