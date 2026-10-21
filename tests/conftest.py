"""Fixtures : faux Ollama déterministe, store temporaire, client API."""
import hashlib
import re

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import ollama_client
from app.config import settings
from app.store import STOP_WORDS, IndexStore

DIM = 256
_STOP = set(STOP_WORDS)


def fake_vec(text: str) -> np.ndarray:
    """Embedding « sac de mots » par hachage : textes proches => cosinus élevé."""
    v = np.zeros(DIM, dtype=np.float32)
    for word in re.findall(r"\w+", text.lower()):
        if word in _STOP or len(word) < 3:
            continue
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeOllama:
    def __init__(self):
        self.chat_calls: list[list[dict]] = []
        self.rewrite_to = "Quelle est la durée maximale de la suspension de travail ?"
        self.answer = "La suspension ne peut excéder huit jours (règlement.pdf, p. 16)."

    def embed(self, texts, prefix="", batch_size=32):
        return [fake_vec(t).tolist() for t in texts]

    def chat(self, messages, temperature=0.1):
        self.chat_calls.append(messages)
        if "Réécris la dernière question" in messages[0]["content"]:
            return self.rewrite_to
        return self.answer

    def chat_stream(self, messages, temperature=0.1):
        self.chat_calls.append(messages)
        for piece in self.answer.split(" "):
            yield piece + " "


@pytest.fixture
def fake_ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(ollama_client, "embed", fake.embed)
    monkeypatch.setattr(ollama_client, "chat", fake.chat)
    monkeypatch.setattr(ollama_client, "chat_stream", fake.chat_stream)
    monkeypatch.setattr(ollama_client, "is_available", lambda: True)
    return fake


def add(store: IndexStore, source: str, pages: dict[int, str]) -> None:
    items = [{"text": t, "page": p} for p, t in pages.items()]
    store.add_document(source, items, [fake_vec(i["text"]).tolist() for i in items])


@pytest.fixture
def store(tmp_path):
    return IndexStore(tmp_path / "index.json")


@pytest.fixture
def populated(store, fake_ollama):
    add(store, "data/uploads/reglement.pdf", {
        16: "Les sanctions disciplinaires sont l'avertissement, le blâme et la mise à pied. "
            "La suspension de travail ne peut excéder huit jours.",
        11: "Le refus du contrôle médical entraîne la suspension du salaire pendant l'absence.",
    })
    add(store, "data/uploads/societe.pdf", {1: "Le siège social de la société est situé à Tunis, avenue principale."})
    return store


def make_pdf(path, pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch, fake_ollama):
    from app.main import app

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    with TestClient(app) as c:
        c.data_dir = tmp_path / "data"
        yield c
