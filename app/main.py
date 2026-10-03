from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
from app.rag_pipeline import load_index
from app.document_indexer import index_document

from app.agent import ask_agent

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="Intelligent Knowledge Agent",
    description="API for document-based question answering",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"


class QuestionRequest(BaseModel):
    question: str
    source: str | None = None

@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }

@app.get("/documents")
def get_documents():
    index = load_index("data/index.json")

    documents = sorted(
        set(
            item["source"].replace("\\", "/")
            for item in index
        )
    )

    return {
        "documents": documents
    }
@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = ask_agent(
        request.question,
        source=request.source
    )
    return result

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    number_of_chunks = index_document(
        file_path
    )

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": file.filename,
        "chunks_added": number_of_chunks
    }