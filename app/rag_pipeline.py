import requests
import numpy as np

from document_loader import load_pdf


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
# 4. Créer les chunks à partir du PDF
# ==========================================

def create_chunks_from_pdf(path):

    pages = load_pdf(path)

    chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = split_text(page["text"])

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": path,
                "chunk_id": chunk_id
            })

            chunk_id += 1

    return chunks


# ==========================================
# 5. Créer l'index
# ==========================================

def create_index(chunks):

    index = []

    for chunk in chunks:

        embedding = get_embedding(chunk["text"])

        index.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "embedding": embedding
        })

    return index


# ==========================================
# 6. Recherche
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
            "score": score,
            "page": item["page"],
            "source": item["source"],
            "chunk_id": item["chunk_id"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ==========================================
# 7. Appel au LLM
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

    response = requests.post(
        url,
        json=data
    )

    response.raise_for_status()

    result = response.json()

    return result["message"]["content"]


# ==========================================
# 8. Programme principal
# ==========================================

pdf_path = "data/test_document.pdf"

chunks = create_chunks_from_pdf(pdf_path)

print("Number of chunks:", len(chunks))


# ==========================================
# 9. Embeddings + index
# ==========================================

print("Creating index...")

index = create_index(chunks)

print("Index created!")


# ==========================================
# 10. Question
# ==========================================

query = "Quelles sont les sanctions disciplinaires prévues par le règlement intérieur ?"


# ==========================================
# 11. Retrieval
# ==========================================

results = search(
    query,
    index,
    top_k=3
)


# ==========================================
# 12. Afficher les résultats
# ==========================================

print("\n==============================")
print("SEARCH RESULTS")
print("==============================")


for result in results:

    print("\nScore:", result["score"])

    print("Source:", result["source"])

    print("Page:", result["page"])

    print("Chunk:", result["chunk_id"])

    print("Text:", result["text"])




# ==========================================
# 13. Construire le contexte
# ==========================================

context = "\n\n".join(
    f"Source: {result['source']}\n"
    f"Page: {result['page']}\n"
    f"Chunk: {result['chunk_id']}\n\n"
    f"{result['text']}"
    for result in results
)

print("\n==============================")
print("CONTEXT SENT TO LLM")
print("==============================")
print(context)

# ==========================================
# 14. Générer la réponse
# ==========================================

answer = generate_answer(
    query,
    context
)

print("\n==============================")
print("FINAL ANSWER")
print("==============================")
print(answer)

