"""API FastAPI de l'Intelligent Knowledge Agent."""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from . import agent, ollama_client
from .config import settings
from .indexing import index_pdf
from .loader import InvalidPdfError, NoExtractableTextError
from .ollama_client import OllamaError
from .store import IndexStore, StoreError, canonical_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = IndexStore(settings.index_path)
    log.info("Index chargé : %d passages, %d documents", app.state.store.snapshot().size, len(app.state.store.documents()))
    yield


app = FastAPI(
    title="Intelligent Knowledge Agent",
    description="Questions/réponses sur des documents PDF (RAG hybride, 100 % local avec Ollama).",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(OllamaError)
async def _ollama_error(_: Request, exc: OllamaError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(StoreError)
async def _store_error(_: Request, exc: StoreError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


def _store(request: Request) -> IndexStore:
    return request.app.state.store


# --------------------------------------------------------------------------- #
# Modèles
# --------------------------------------------------------------------------- #
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(max_length=2000)
    source: str | None = None
    history: list[Turn] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La question ne peut pas être vide.")
        return value


# --------------------------------------------------------------------------- #
# Santé & documents
# --------------------------------------------------------------------------- #
@app.get("/health")
def health(request: Request):
    store = _store(request)
    return {
        "status": "ok",
        "ollama": ollama_client.is_available(),
        "documents": len(store.documents()),
        "chunks": store.snapshot().size,
    }


@app.get("/documents")
def list_documents(request: Request):
    return {"documents": _store(request).documents()}


def _resolve_document_path(source: str) -> Path:
    """Chemin du PDF d'un document connu ; 404 sinon (protège contre le path traversal)."""
    root = Path(settings.data_dir).resolve()
    path = Path(canonical_source(source)).resolve()
    if root not in path.parents or path.suffix.lower() != ".pdf" or not path.is_file():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return path


@app.get("/documents/file")
def get_document_file(request: Request, source: str = Query(...)):
    if canonical_source(source) not in _store(request).document_ids():
        raise HTTPException(status_code=404, detail="Document inconnu.")
    path = _resolve_document_path(source)
    return FileResponse(path, media_type="application/pdf", filename=path.name, content_disposition_type="inline")


@app.delete("/documents")
def delete_document(request: Request, source: str = Query(...)):
    store = _store(request)
    source = canonical_source(source)
    removed = store.remove_document(source)
    if not removed:
        raise HTTPException(status_code=404, detail="Document inconnu.")
    try:
        _resolve_document_path(source).unlink(missing_ok=True)
    except HTTPException:
        pass  # fichier déjà absent : l'index est de toute façon nettoyé
    return {"message": "Document supprimé.", "id": source, "chunks_removed": removed}


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
def _safe_filename(raw: str | None) -> str:
    name = Path((raw or "").replace("\\", "/")).name  # retire tout chemin (../ inclus)
    name = re.sub(r"[^\w.\-() ']", "_", name, flags=re.UNICODE).strip(" .")
    if not name.lower().endswith(".pdf") or len(name) <= 4:
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés.")
    return name[:150]


@app.post("/documents/upload")
def upload_document(request: Request, file: UploadFile = File(...)):
    store = _store(request)
    filename = _safe_filename(file.filename)
    upload_dir = settings.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    final_path = upload_dir / filename
    tmp_path = upload_dir / f".{uuid.uuid4().hex}.part"
    limit = int(settings.max_upload_mb * 1024 * 1024)

    try:
        size = 0
        with open(tmp_path, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (max {settings.max_upload_mb:g} Mo).")
                out.write(chunk)

        with open(tmp_path, "rb") as check:
            if check.read(5) != b"%PDF-":
                raise HTTPException(status_code=400, detail="Ce fichier n'est pas un PDF valide.")

        source = canonical_source(final_path)
        try:
            info = index_pdf(store, tmp_path, source=source)
        except InvalidPdfError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except NoExtractableTextError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        os.replace(tmp_path, final_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "message": "Document indexé avec succès.",
        "id": info["source"],
        "name": filename,
        "pages": info["pages"],
        "chunks_added": info["chunks"],
    }


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #
def _prepare(request: Request, body: AskRequest) -> agent.Prepared:
    store = _store(request)
    if body.source and canonical_source(body.source) not in store.document_ids():
        raise HTTPException(status_code=404, detail="Le document sélectionné n'existe plus.")
    history = [t.model_dump() for t in body.history]
    return agent.prepare(body.question, store, source=body.source, history=history)


@app.post("/ask")
def ask(request: Request, body: AskRequest):
    return agent.run(_prepare(request, body))


@app.post("/ask/stream")
def ask_stream(request: Request, body: AskRequest):
    """Server-Sent Events : `meta` (sources), `token`*, puis `done` (ou `error`)."""
    prepared = _prepare(request, body)  # erreurs (404/503) levées AVANT d'ouvrir le flux

    def events():
        for event in agent.stream(prepared):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
