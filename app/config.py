"""Configuration centralisée (variables d'environnement, voir .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Charge un fichier `.env` (KEY=VALUE) sans écraser les variables déjà définies."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]  # les guillemets permettent de garder un espace final
        os.environ.setdefault(key.strip(), value)


def _list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else default


@dataclass
class Settings:
    # --- Ollama ---
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:3b-instruct"
    embed_model: str = "nomic-embed-text"
    # nomic-embed-text est entraîné avec des préfixes de tâche. Pour les activer :
    # QUERY_PREFIX="search_query: "  DOCUMENT_PREFIX="search_document: "  (puis réindexer)
    query_prefix: str = ""
    document_prefix: str = ""
    request_timeout: float = 120.0
    num_ctx: int = 4096

    # --- Données ---
    data_dir: Path = Path("data")
    max_upload_mb: float = 50.0

    # --- Découpage ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_chars: int = 40

    # --- Recherche ---
    top_k: int = 5
    candidate_k: int = 20
    # Garde-fou avant le LLM : en dessous de ces seuils on répond « introuvable »
    # sans appeler le modèle. Valeurs de départ prudentes (peu de faux refus) :
    # À RECALIBRER sur ton corpus avec `python -m scripts.evaluate`.
    # (min_keyword_score=0.15 : mesuré sur 10 questions valides >= 0.17 et 4 sans réponse <= 0.14.)
    min_semantic_score: float = 0.30
    min_keyword_score: float = 0.15
    history_turns: int = 4

    # --- API ---
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:4200"])

    @property
    def upload_dir(self) -> Path:
        return Path(self.data_dir) / "uploads"

    @property
    def index_path(self) -> Path:
        return Path(self.data_dir) / "index.json"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        g = os.getenv
        return cls(
            ollama_url=g("OLLAMA_URL", d.ollama_url).rstrip("/"),
            chat_model=g("CHAT_MODEL", d.chat_model),
            embed_model=g("EMBED_MODEL", d.embed_model),
            query_prefix=g("QUERY_PREFIX", d.query_prefix),
            document_prefix=g("DOCUMENT_PREFIX", d.document_prefix),
            request_timeout=float(g("REQUEST_TIMEOUT", d.request_timeout)),
            num_ctx=int(g("NUM_CTX", d.num_ctx)),
            data_dir=Path(g("DATA_DIR", str(d.data_dir))),
            max_upload_mb=float(g("MAX_UPLOAD_MB", d.max_upload_mb)),
            chunk_size=int(g("CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(g("CHUNK_OVERLAP", d.chunk_overlap)),
            min_chunk_chars=int(g("MIN_CHUNK_CHARS", d.min_chunk_chars)),
            top_k=int(g("TOP_K", d.top_k)),
            candidate_k=int(g("CANDIDATE_K", d.candidate_k)),
            min_semantic_score=float(g("MIN_SEMANTIC_SCORE", d.min_semantic_score)),
            min_keyword_score=float(g("MIN_KEYWORD_SCORE", d.min_keyword_score)),
            history_turns=int(g("HISTORY_TURNS", d.history_turns)),
            cors_origins=_list("CORS_ORIGINS", d.cors_origins),
        )


load_env_file(Path(__file__).resolve().parent.parent / ".env")
settings = Settings.from_env()
