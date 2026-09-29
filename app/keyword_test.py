from app.rag_pipeline import load_index
from app.keyword_search import create_keyword_index, keyword_search


index = load_index("data/index.json")

vectorizer, matrix = create_keyword_index(index)


query = "Combien de jours au maximum un salarié peut-il être suspendu ?"


results = keyword_search(
    query,
    index,
    vectorizer,
    matrix,
    top_k=10
)


print("\n==============================")
print("KEYWORD SEARCH")
print("==============================")


for result in results:

    print("\nScore:", result["score"])
    print("Page:", result["page"])
    print("Chunk:", result["chunk_id"])
    print("Text:", result["text"])