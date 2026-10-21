"""Client Ollama : timeouts, erreurs explicites, streaming."""
from __future__ import annotations

import json
import logging
from typing import Iterator

import requests

from .config import settings

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Ollama est injoignable ou a renvoyé une erreur."""


def _post(path: str, payload: dict, stream: bool = False) -> requests.Response:
    url = f"{settings.ollama_url}{path}"
    try:
        response = requests.post(url, json=payload, timeout=settings.request_timeout, stream=stream)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise OllamaError(
            f"Ollama est indisponible ou a renvoyé une erreur ({path}) : {exc}. "
            f"Vérifie que `ollama serve` tourne et que les modèles sont installés."
        ) from exc


def embed(texts: list[str], prefix: str = "", batch_size: int = 32) -> list[list[float]]:
    """Embeddings par lots. `prefix` est ajouté devant chaque texte."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i : i + batch_size]]
        data = _post("/api/embed", {"model": settings.embed_model, "input": batch}).json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise OllamaError("Réponse d'embedding invalide (nombre de vecteurs inattendu).")
        out.extend(embeddings)
    return out


def embed_query(text: str) -> list[float]:
    return embed([text], settings.query_prefix)[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embed(texts, settings.document_prefix)


def _chat_payload(messages: list[dict], temperature: float, stream: bool) -> dict:
    return {
        "model": settings.chat_model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature, "num_ctx": settings.num_ctx},
    }


def chat(messages: list[dict], temperature: float = 0.1) -> str:
    data = _post("/api/chat", _chat_payload(messages, temperature, False)).json()
    try:
        return data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise OllamaError("Réponse de chat invalide.") from exc


def chat_stream(messages: list[dict], temperature: float = 0.1) -> Iterator[str]:
    response = _post("/api/chat", _chat_payload(messages, temperature, True), stream=True)
    try:
        for line in response.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                raise OllamaError(str(obj["error"]))
            piece = obj.get("message", {}).get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                break
    except requests.RequestException as exc:
        raise OllamaError(f"Flux interrompu : {exc}") from exc
    finally:
        response.close()


def is_available() -> bool:
    try:
        return requests.get(f"{settings.ollama_url}/api/tags", timeout=2).ok
    except requests.RequestException:
        return False
