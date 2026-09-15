import requests
import numpy as np


# ==========================================
# 1. Obtenir l'embedding d'un texte
# ==========================================

def get_embedding(text):

    url = "http://localhost:11434/api/embed"

    data = {
        "model": "nomic-embed-text",
        "input": text
    }

    response = requests.post(url, json=data)
    response.raise_for_status()

    result = response.json()

    return result["embeddings"][0]


# ==========================================
# 2. Calculer la similarité cosinus
# ==========================================

def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )


# ==========================================
# 3. Créer notre index
# ==========================================

def create_index(documents):

    index = []

    for document in documents:

        embedding = get_embedding(document)

        index.append({
            "text": document,
            "embedding": embedding
        })

    return index


# ==========================================
# 4. Rechercher les documents pertinents
# ==========================================

def search(query, index, top_k=3):

    query_embedding = get_embedding(query)

    results = []

    for item in index:

        score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        results.append({
            "text": item["text"],
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ==========================================
# 5. Documents
# ==========================================

documents = [
    "Customers can request a refund within 30 days of purchase.",
    "Refund requests must be submitted through the support portal.",
    "Employees receive 25 days of annual leave.",
    "Payments can be made by credit card or bank transfer.",
    "The company headquarters are located in Tunis."
]


# ==========================================
# 6. Création de l'index
# ==========================================

print("Creating index...")

index = create_index(documents)

print("Index created!")


# ==========================================
# 7. Recherche
# ==========================================

query = "How can I get my money back?"

results = search(
    query,
    index,
    top_k=3
)


# ==========================================
# 8. Affichage
# ==========================================

print("\nQuery:")
print(query)

print("\nTop results:")

for rank, result in enumerate(results, start=1):

    print(f"\nRank {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['text']}")