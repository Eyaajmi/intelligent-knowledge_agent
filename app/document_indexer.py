
import json
import os
import tempfile

from app.document_loader import load_pdf
from app.rag_pipeline import get_embeddings, split_text

INDEX_PATH = "data/index.json"


def load_existing_index():
    if not os.path.exists(INDEX_PATH):
        return []

    with open(INDEX_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_index(index):
    """Sauvegarde l'index via un fichier temporaire."""
    directory = os.path.dirname(INDEX_PATH) or "."
    os.makedirs(directory, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            suffix=".tmp",
            delete=False
        ) as file:
            temp_path = file.name
            json.dump(index, file, ensure_ascii=False, indent=2)

        os.replace(temp_path, INDEX_PATH)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_source(path):
    """Retourne un chemin comparable sous Windows."""
    return os.path.normcase(os.path.abspath(path))


def index_document(path):
    pages = load_pdf(path)

    chunks_to_index = []

    for page in pages:
        for chunk in split_text(page["text"]):
            chunks_to_index.append({
                "text": chunk,
                "page": page["page"],
                "source": os.path.normpath(path).replace("\\", "/")
            })

    if not chunks_to_index:
        raise ValueError(
            f"Aucun texte exploitable trouvé dans le document : {path}"
        )

    embeddings = get_embeddings(
        [chunk["text"] for chunk in chunks_to_index]
    )

    if len(embeddings) != len(chunks_to_index):
        raise RuntimeError(
            "Le nombre d'embeddings ne correspond pas au nombre de chunks."
        )

    index = load_existing_index()
    target_source = normalize_source(path)

    # Retirer toutes les anciennes versions de ce document,
    # même si le chemin utilise des slashs différents.
    index = [
        chunk for chunk in index
        if normalize_source(chunk["source"]) != target_source
    ]

    # Garder les anciens identifiants et attribuer des IDs uniques
    # aux nouveaux chunks.
    existing_ids = [
        chunk["chunk_id"]
        for chunk in index
        if isinstance(chunk.get("chunk_id"), int)
    ]
    next_id = max(existing_ids, default=-1) + 1

    new_chunks = []

    for i, chunk in enumerate(chunks_to_index):
        new_chunks.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "chunk_id": next_id + i,
            "embedding": embeddings[i]
        })

    updated_index = index + new_chunks
    save_index(updated_index)

    return len(new_chunks)
