import json
import os

from app.document_loader import load_pdf
from app.rag_pipeline import (
    get_embeddings,
    split_text
)

INDEX_PATH = "data/index.json"


def load_existing_index():
    if not os.path.exists(INDEX_PATH):
        return []

    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_index(index):
    with open(INDEX_PATH, "w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)


def index_document(path):
    pages = load_pdf(path)
    index = load_existing_index()

    chunks_to_index = []

    for page in pages:

        chunks = split_text(page["text"])

        for chunk in chunks:

            chunks_to_index.append({
                "text": chunk,
                "page": page["page"],
                "source": path
            })

    texts = [
        chunk["text"]
        for chunk in chunks_to_index
    ]

    embeddings = get_embeddings(texts)

    start_id = len(index)

    new_chunks = []

    for i, chunk in enumerate(chunks_to_index):

        new_chunks.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "chunk_id": start_id + i,
            "embedding": embeddings[i]
        })

    index.extend(new_chunks)

    save_index(index)

    return len(new_chunks)