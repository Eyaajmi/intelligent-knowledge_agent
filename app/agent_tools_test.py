from app.agent_tools import search_documents


query = "Quelles sont les sanctions disciplinaires ?"

results = search_documents(query)


for rank, result in enumerate(results, start=1):

    print(
        rank,
        "| Page:",
        result["page"],
        "| Score:",
        round(result["rerank_score"], 4)
    )

    print(result["text"][:300])
    print()