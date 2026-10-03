from app.rag_pipeline import load_index, search
from app.keyword_search import keyword_search
from app.hybrid_search import reciprocal_rank_fusion
from app.reranker import rerank
from tests.retrieval_cases import TEST_CASES
from app.keyword_search import keyword_search, create_keyword_index


def reciprocal_rank(expected_page, retrieved_pages, k):

    for rank, page in enumerate(retrieved_pages[:k], start=1):

        if page == expected_page:
            return 1 / rank

    return 0


def main():

    index = load_index("data/index.json")
    vectorizer, matrix = create_keyword_index(index)

    recall_at_3 = 0
    recall_at_10 = 0

    mrr_at_3 = 0
    mrr_at_10 = 0

    total = len(TEST_CASES)

    for case in TEST_CASES:

        question = case["question"]
        expected_page = case["expected_page"]

        # Recherche sémantique
        semantic_results = search(
            question,
            index,
            top_k=10
        )

        # Recherche lexicale
        keyword_results = keyword_search(
         question,
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
            question,
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
        mrr_at_3 += reciprocal_rank(
            expected_page,
            retrieved_pages,
            3
        )

        # MRR@10
        mrr_at_10 += reciprocal_rank(
            expected_page,
            retrieved_pages,
            10
        )

        print()
        print("Question:", question)
        print("Page attendue:", expected_page)
        print("Pages récupérées:", retrieved_pages)

    print()
    print("=" * 50)
    print("RÉSULTATS RERANKER")
    print("=" * 50)

    print(
        "Recall@3:",
        round(recall_at_3 / total, 4)
    )

    print(
        "Recall@10:",
        round(recall_at_10 / total, 4)
    )

    print(
        "MRR@3:",
        round(mrr_at_3 / total, 4)
    )

    print(
        "MRR@10:",
        round(mrr_at_10 / total, 4)
    )


if __name__ == "__main__":
    main()