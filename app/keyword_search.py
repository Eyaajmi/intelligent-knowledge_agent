from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_keyword_index(index):

    texts = [
        item["text"]
        for item in index
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def keyword_search(query, index, vectorizer, matrix, top_k=3):
   

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        matrix
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index_position in ranked_indices:

        item = index[index_position]

        results.append({
            "text": item["text"],
            "score": similarities[index_position],
            "page": item["page"],
            "source": item["source"],
            "chunk_id": item["chunk_id"]
        })

    return results
