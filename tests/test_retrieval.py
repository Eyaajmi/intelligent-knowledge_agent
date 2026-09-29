from app.rag_pipeline import search, load_index
from tests.retrieval_cases import TEST_CASES


index = load_index("data/index.json")

correct = 0
mrr_scores = []
evaluation_results = []

print("\n==============================")
print("RETRIEVAL EVALUATION")
print("==============================")


for test_case in TEST_CASES:

    question = test_case["question"]
    expected_page = test_case["expected_page"]

    results = search(
        question,
        index,
        top_k=3
    )

    retrieved_pages = [
        result["page"]
        for result in results
    ]

    # ------------------------------
    # Recall@3
    # ------------------------------

    is_correct = expected_page in retrieved_pages

    if is_correct:
        correct += 1

    # ------------------------------
    # MRR@3
    # ------------------------------

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

    # ------------------------------
    # Display results
    # ------------------------------

    print("\nQuestion:", question)
    print("Expected page:", expected_page)
    print("Retrieved pages:", retrieved_pages)
    print("Retrieved scores:", [
    round(result["score"], 4)
    for result in results
])

    if is_correct:
        print("Result: PASS")
    else:
        print("Result: FAIL")

    if reciprocal_rank > 0:
        rank = int(1 / reciprocal_rank)
        print("Expected page rank:", rank)
    else:
        print("Expected page rank: Not found")


# ------------------------------
# Final metrics
# ------------------------------

recall_at_3 = correct / len(TEST_CASES)

mrr_at_3 = sum(mrr_scores) / len(mrr_scores)


print("\n==============================")
print("EVALUATION RESULT")
print("==============================")

print("Recall@3:", recall_at_3)
print("MRR@3:", mrr_at_3)
print("\n==============================")
print("DETAILED RETRIEVAL ANALYSIS")
print("==============================")


for result in evaluation_results:

    print("\nQuestion:", result["question"])
    print("Expected page:", result["expected_page"])
    print("Retrieved pages:", result["retrieved_pages"])

    if result["reciprocal_rank"] > 0:
        rank = int(1 / result["reciprocal_rank"])
        print("Expected page rank:", rank)
    else:
        print("Expected page rank: Not found")
