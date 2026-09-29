from app.rag_pipeline import load_index, search
from app.keyword_search import create_keyword_index, keyword_search
from app.hybrid_search import (
    reciprocal_rank_fusion,
    group_results_by_page
)

from tests.retrieval_cases import TEST_CASES



index = load_index("data/index.json")


vectorizer, matrix = create_keyword_index(index)


correct = 0
mrr_scores = []

evaluation_results = []


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
            semantic_results,
            keyword_results
        ]
    )
    hybrid_results = group_results_by_page(
    hybrid_results
)


    top_results = hybrid_results[:3]

    retrieved_pages = [
        result["page"]
        for result in top_results
    ]


    is_correct = expected_page in retrieved_pages

    if is_correct:
        correct += 1

   
    reciprocal_rank = 0

    for rank, page in enumerate(retrieved_pages, start=1):

        if page == expected_page:
            reciprocal_rank = 1 / rank
            break

    mrr_scores.append(reciprocal_rank)

    evaluation_results.append({
        "question": question,
        "expected_page": expected_page,
        "retrieved_pages": retrieved_pages,
        "reciprocal_rank": reciprocal_rank
    })

    print("\nQuestion:", question)
    print("Expected page:", expected_page)
    print("Retrieved pages:", retrieved_pages)

    if is_correct:
        print("PASS")
    else:
        print("FAIL")

    if reciprocal_rank > 0:
        print("Correct rank:", int(1 / reciprocal_rank))
    else:
        print("Correct rank: Not found")



recall_at_3 = correct / len(TEST_CASES)
mrr_at_3 = sum(mrr_scores) / len(mrr_scores)


print("\n==============================")
print("HYBRID RETRIEVAL EVALUATION")
print("==============================")

print("Recall@3:", recall_at_3)
print("MRR@3:", mrr_at_3)


print("\n==============================")
print("DETAILED ANALYSIS")
print("==============================")


for result in evaluation_results:

    print("\nQuestion:", result["question"])
    print("Expected:", result["expected_page"])
    print("Retrieved:", result["retrieved_pages"])
    print("Reciprocal rank:", result["reciprocal_rank"])