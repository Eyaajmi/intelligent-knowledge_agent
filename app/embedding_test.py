import requests
import numpy as np

url = "http://localhost:11434/api/embed"

texts = [
    "How can I get my money back?",
    "What is the weather today?"
]

embeddings = []

for text in texts:
    data = {
        "model": "nomic-embed-text",
        "input": text
    }

    response = requests.post(url, json=data)
    response.raise_for_status()

    result = response.json()

    embeddings.append(result["embeddings"][0])

vector_a = np.array(embeddings[0])
vector_b = np.array(embeddings[1])

similarity = np.dot(vector_a, vector_b) / (
    np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
)

print("Similarity:", similarity)