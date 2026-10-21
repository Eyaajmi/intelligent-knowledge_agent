import json

from app import ollama_client
from app.config import settings
from app.ollama_client import OllamaError
from tests.conftest import make_pdf

REGLEMENT = [
    "Les sanctions disciplinaires sont l'avertissement, le blâme et la mise à pied. "
    "La suspension de travail ne peut excéder huit jours.",
    "Le refus du contrôle médical entraîne la suspension du salaire pendant l'absence.",
]


def _upload(client, tmp_path, name="reglement.pdf", pages=REGLEMENT):
    data = make_pdf(tmp_path / "src.pdf", pages)
    return client.post("/documents/upload", files={"file": (name, data, "application/pdf")})


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "ollama": True, "documents": 0, "chunks": 0}


def test_upload_then_list(client, tmp_path):
    r = _upload(client, tmp_path)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "reglement.pdf" and body["chunks_added"] >= 2 and body["pages"] == 2
    docs = client.get("/documents").json()["documents"]
    assert [d["name"] for d in docs] == ["reglement.pdf"]
    assert (client.data_dir / "uploads" / "reglement.pdf").is_file()
    assert (client.data_dir / "index.json").is_file()


def test_reuploading_same_name_does_not_duplicate(client, tmp_path):
    first = _upload(client, tmp_path).json()["chunks_added"]
    _upload(client, tmp_path)
    assert client.get("/health").json()["chunks"] == first


def test_upload_rejects_non_pdf_extension(client, tmp_path):
    r = client.post("/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_upload_rejects_fake_pdf(client):
    r = client.post("/documents/upload", files={"file": ("faux.pdf", b"ceci n'est pas un pdf", "application/pdf")})
    assert r.status_code == 400 and "PDF valide" in r.json()["detail"]
    assert list((client.data_dir / "uploads").iterdir()) == []


def test_upload_path_traversal_is_neutralised(client, tmp_path):
    r = _upload(client, tmp_path, name="../../evil.pdf")
    assert r.status_code == 200
    assert not (tmp_path / "evil.pdf").exists() and not (tmp_path.parent / "evil.pdf").exists()
    assert (client.data_dir / "uploads" / "evil.pdf").is_file()


def test_scanned_pdf_gives_422_not_500(client, tmp_path):
    r = _upload(client, tmp_path, name="scan.pdf", pages=["", ""])
    assert r.status_code == 422 and "OCR" in r.json()["detail"]
    assert client.get("/health").json()["chunks"] == 0
    assert list((client.data_dir / "uploads").iterdir()) == []


def test_upload_size_limit(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0.0001)
    assert _upload(client, tmp_path).status_code == 413
    assert list((client.data_dir / "uploads").iterdir()) == []


def test_upload_when_ollama_is_down_returns_503_and_cleans_up(client, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OllamaError("Ollama est indisponible")
    monkeypatch.setattr(ollama_client, "embed", boom)
    r = _upload(client, tmp_path)
    assert r.status_code == 503
    assert list((client.data_dir / "uploads").iterdir()) == []


def test_ask(client, tmp_path):
    _upload(client, tmp_path)
    r = client.post("/ask", json={"question": "Quelle est la durée maximale de la suspension de travail ?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["sources"][0]["page"] == 1 and "snippet" in body["sources"][0]


def test_ask_validation_and_unknown_document(client, tmp_path):
    _upload(client, tmp_path)
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    r = client.post("/ask", json={"question": "Bonjour ?", "source": "data/uploads/fantome.pdf"})
    assert r.status_code == 404


def test_ask_when_ollama_is_down(client, tmp_path, monkeypatch):
    _upload(client, tmp_path)
    monkeypatch.setattr(ollama_client, "embed", lambda *a, **k: (_ for _ in ()).throw(OllamaError("hors ligne")))
    r = client.post("/ask", json={"question": "suspension de travail"})
    assert r.status_code == 503 and "hors ligne" in r.json()["detail"]


def test_ask_stream(client, tmp_path):
    _upload(client, tmp_path)
    r = client.post("/ask/stream", json={"question": "durée maximale de la suspension de travail"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]
    assert events[0]["type"] == "meta" and events[-1]["type"] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token").strip()


def test_delete_document(client, tmp_path):
    doc_id = _upload(client, tmp_path).json()["id"]
    r = client.delete("/documents", params={"source": doc_id})
    assert r.status_code == 200 and r.json()["chunks_removed"] >= 2
    assert client.get("/documents").json()["documents"] == []
    assert not (client.data_dir / "uploads" / "reglement.pdf").exists()
    assert client.delete("/documents", params={"source": doc_id}).status_code == 404


def test_serve_document_and_block_traversal(client, tmp_path):
    doc_id = _upload(client, tmp_path).json()["id"]
    r = client.get("/documents/file", params={"source": doc_id})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    assert client.get("/documents/file", params={"source": "../../etc/passwd"}).status_code == 404
    assert client.get("/documents/file", params={"source": "data/uploads/inconnu.pdf"}).status_code == 404
