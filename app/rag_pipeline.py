import requests
import numpy as np


# ==========================================
# 1. Embedding
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
# 2. Similarité cosinus
# ==========================================

def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )


# ==========================================
# 3. Chunking
# ==========================================

def split_text(text, chunk_size=300, overlap_sentences=1):

   
    text = text.strip()

    paragraphs = text.split("\n\n")

    chunks = []

    current_sentences = []
    current_length = 0

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

     
        sentences = paragraph.split(". ")

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

          
            if not sentence.endswith("."):
                sentence += "."

            sentence_length = len(sentence)

          
            if current_length + sentence_length + 1 <= chunk_size:

                current_sentences.append(sentence)
                current_length += sentence_length + 1

            else:

                
                if current_sentences:

                    chunk = " ".join(current_sentences)

                    chunks.append(chunk)

                overlap = current_sentences[
                    -overlap_sentences:
                ]

                current_sentences = overlap + [sentence]

                current_length = sum(
                    len(s) + 1
                    for s in current_sentences
                )

  
    if current_sentences:

        chunk = " ".join(current_sentences)

        chunks.append(chunk)

    return chunks
# ==========================================
# 4. Créer l'index
# ==========================================

def create_index(chunks):

    index = []

    for chunk in chunks:

        embedding = get_embedding(chunk)

        index.append({
            "text": chunk,
            "embedding": embedding
        })

    return index


# ==========================================
# 5. Recherche
# ==========================================

def search(query, index, top_k=2):

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
# 6. Appel au LLM
# ==========================================

def generate_answer(query, context):

    url = "http://localhost:11434/api/chat"

    system_prompt = """
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that you do not have enough information.

Do not invent facts.
"""

    user_prompt = f"""
Context:

{context}

Question:

{query}
"""

    data = {
        "model": "qwen2.5:3b-instruct",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False
    }

    response = requests.post(url, json=data)
    response.raise_for_status()

    result = response.json()

    return result["message"]["content"]


# ==========================================
# 7. Document
# ==========================================

document = """
Our company offers a 30-day refund policy for all eligible purchases.

Customers who want to request a refund must contact the support team
through the online support portal. The customer should provide the
order number and proof of purchase.

Refunds are normally processed within five business days after the
request has been approved.

Employees receive 25 days of annual leave every year. Annual leave
requests must be submitted to the employee's manager.

Payments can be made using a credit card, debit card, or bank transfer.

The company's headquarters are located in Tunis.
"""


# ==========================================
# 8. Question
# ==========================================

query = "How can I get my money back?"


# ==========================================
# 9. Chunking
# ==========================================

chunks = split_text(document)

print("Number of chunks:", len(chunks))


# ==========================================
# 10. Embeddings + index
# ==========================================

print("Creating index...")

index = create_index(chunks)

print("Index created!")


# ==========================================
# 11. Retrieval
# ==========================================

results = search(
    query,
    index,
    top_k=2
)


# ==========================================
# 12. Construire le contexte
# ==========================================

context = "\n\n".join(
    result["text"]
    for result in results
)


print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")

print(context)


# ==========================================
# 13. Génération
# ==========================================

answer = generate_answer(
    query,
    context
)


print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)