"""Index en mémoire (matrice numpy + TF-IDF), persisté en JSON.

Corrige les défauts de l'ancienne version :
- une source est *canonique* (slashs normalisés) : plus de doublons `a/b.pdf` vs `a\\b.pdf` ;
- l'index est chargé UNE fois ; les lectures sont sans verrou (snapshot immuable) ;
- la recherche sémantique est un produit matriciel (≈0,2 ms au lieu de ≈130 ms) ;
- les écritures sont atomiques et sérialisées par un verrou.

Le format du fichier reste compatible avec l'ancien `index.json`
(liste de {text, page, source, chunk_id, embedding}).
"""
from __future__ import annotations

import json
import logging
import os
import posixpath
import re
import tempfile
import threading
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)


class StoreError(RuntimeError):
    """Index illisible ou incohérent."""


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def canonical_source(source) -> str:
    """Forme unique d'un chemin : slashs `/`, sans `./` ni `//`."""
    return posixpath.normpath(str(source).replace("\\", "/"))


def doc_name(source: str) -> str:
    return posixpath.basename(source)


def normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Mots vides FR (+ quelques EN), sans accents car le vectoriseur retire les accents.
_STOPWORDS = """
le la les un une des du de et ou en au aux ce ces cet cette son sa ses leur leurs mon ma mes
ton ta tes notre nos votre vos qui que quoi dont est sont etre ete etait etaient sera seront
ont avoir avait pour par sur sous dans avec sans ne pas plus se il elle ils elles on nous vous
je tu lors comme mais donc or ni car si ainsi aussi alors entre vers chez tout tous toute toutes
the of and to in is are for on with by at an be as it that this was were from
quel quels quelle quelles comment combien quand pourquoi lequel laquelle lesquels lesquelles
peut peuvent doit doivent faut fait font faire
"""
STOP_WORDS = sorted({_strip_accents(w) for w in _STOPWORDS.split()})
_STOP_SET = frozenset(STOP_WORDS)
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Mots-clés d'un texte (déjà en minuscules et sans accents) :
    mots vides retirés, pluriel simple normalisé (« blâmes » -> « blame »)."""
    tokens = []
    for token in _TOKEN_RE.findall(text):
        if token in _STOP_SET:
            continue
        if len(token) > 4 and token[-1] in "sx":
            token = token[:-1]
        tokens.append(token)
    return tokens

_EMPTY = np.zeros((0, 0), dtype=np.float32)


def normalize_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Canonise les sources, retire les doublons exacts, garantit des chunk_id uniques."""
    seen: set[tuple] = set()
    out: list[dict] = []
    report = {"input": len(records), "renamed_sources": 0, "duplicates_removed": 0, "invalid_skipped": 0}
    for record in records:
        if not record.get("text") or not record.get("embedding") or "page" not in record:
            report["invalid_skipped"] += 1
            continue
        source = canonical_source(record.get("source", ""))
        if source != record.get("source"):
            report["renamed_sources"] += 1
        key = (source, record["page"], normalize_text(record["text"]))
        if key in seen:
            report["duplicates_removed"] += 1
            continue
        seen.add(key)
        out.append({**record, "source": source})

    ids = [r.get("chunk_id") for r in out]
    if len(set(ids)) != len(ids) or not all(isinstance(i, int) for i in ids):
        for n, record in enumerate(out):
            record["chunk_id"] = n
        report["ids_reassigned"] = True
    report["output"] = len(out)
    return out, report


def _unit_rows(rows) -> np.ndarray:
    try:
        matrix = np.asarray(rows, dtype=np.float32)
    except ValueError as exc:  # vecteurs de tailles différentes
        raise StoreError("Embeddings de dimensions différentes : réindexe tous les documents.") from exc
    if matrix.ndim != 2:
        raise StoreError("Embeddings invalides.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# --------------------------------------------------------------------------- #
# Snapshot immuable (lu sans verrou)
# --------------------------------------------------------------------------- #
class Snapshot:
    def __init__(self, chunks: list[dict], matrix: np.ndarray):
        self.chunks = chunks  # sans embeddings : {text, page, source, chunk_id}
        self.matrix = matrix  # (N, d) float32, lignes normalisées
        self.sources = np.array([c["source"] for c in chunks], dtype=object)
        self._vectorizer = None
        self._tfidf = None
        if chunks:
            try:
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    tokenizer=tokenize,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                )
                self._tfidf = vectorizer.fit_transform([c["text"] for c in chunks])
                self._vectorizer = vectorizer
            except ValueError:  # vocabulaire vide
                log.warning("Index mots-clés vide (aucun terme exploitable).")

    @property
    def size(self) -> int:
        return len(self.chunks)

    def mask(self, source: str | None) -> np.ndarray:
        if not source:
            return np.ones(self.size, dtype=bool)
        return self.sources == canonical_source(source)

    def semantic_scores(self, query_vector) -> np.ndarray:
        """Cosinus entre la requête et TOUS les passages (vecteur de taille N)."""
        if self.size == 0:
            return np.zeros(0, dtype=np.float32)
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape[0] != self.matrix.shape[1]:
            raise StoreError(
                f"Dimension d'embedding {q.shape[0]} ≠ {self.matrix.shape[1]} (index). "
                "Le modèle d'embedding a changé : réindexe les documents."
            )
        norm = np.linalg.norm(q)
        return self.matrix @ (q / norm if norm else q)

    def keyword_scores(self, query: str) -> np.ndarray:
        """Similarité TF-IDF (insensible aux accents/casse) avec TOUS les passages."""
        if self._vectorizer is None or self._tfidf is None:
            return np.zeros(self.size, dtype=np.float32)
        qv = self._vectorizer.transform([query])
        return (self._tfidf @ qv.T).toarray().ravel()


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class IndexStore:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state = Snapshot([], _EMPTY)
        self.report: dict = {}
        self.load()

    # -- lecture ---------------------------------------------------------- #
    def snapshot(self) -> Snapshot:
        return self._state

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self._state.chunks:
            d = docs.setdefault(c["source"], {"id": c["source"], "name": doc_name(c["source"]), "chunks": 0, "pages": 0})
            d["chunks"] += 1
            d["pages"] = max(d["pages"], c["page"])
        return sorted(docs.values(), key=lambda d: d["name"].lower())

    def document_ids(self) -> set[str]:
        return {c["source"] for c in self._state.chunks}

    # -- chargement / sauvegarde ----------------------------------------- #
    def load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._state = Snapshot([], _EMPTY)
                self.report = {"input": 0, "output": 0, "duplicates_removed": 0}
                return
            try:
                records = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StoreError(f"Index illisible ({self.path}) : {exc}") from exc
            if not isinstance(records, list):
                raise StoreError(f"Format d'index inattendu : {self.path}")

            records, self.report = normalize_records(records)
            if self.report["duplicates_removed"] or self.report["renamed_sources"]:
                log.warning(
                    "Index nettoyé en mémoire (%s). Lance `python -m scripts.migrate_index` pour le persister.",
                    self.report,
                )
            matrix = _unit_rows([r["embedding"] for r in records]) if records else _EMPTY
            chunks = [{k: v for k, v in r.items() if k != "embedding"} for r in records]
            self._state = Snapshot(chunks, matrix)

    def save(self, state: Snapshot | None = None) -> None:
        state = state or self._state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False
            ) as f:
                tmp_name = f.name
                f.write("[")
                for i, (chunk, row) in enumerate(zip(state.chunks, state.matrix)):
                    if i:
                        f.write(",")
                    embedding = np.round(row.astype(np.float64), 6).tolist()
                    json.dump({**chunk, "embedding": embedding}, f, ensure_ascii=False, separators=(",", ":"))
                f.write("]")
            os.chmod(tmp_name, 0o644)  # NamedTemporaryFile crée en 0600
            os.replace(tmp_name, self.path)
            tmp_name = None
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)

    # -- écriture --------------------------------------------------------- #
    def add_document(self, source: str, items: list[dict], embeddings: list[list[float]]) -> int:
        """Remplace TOUTES les anciennes versions de `source` par `items` ({text, page})."""
        if len(items) != len(embeddings):
            raise StoreError("Le nombre d'embeddings ne correspond pas au nombre de passages.")
        source = canonical_source(source)
        new_matrix = _unit_rows(embeddings)
        with self._lock:
            old = self._state
            keep = old.sources != source
            kept_chunks = [c for c, k in zip(old.chunks, keep) if k]
            kept_matrix = old.matrix[keep] if old.size else _EMPTY
            if kept_matrix.shape[0] and kept_matrix.shape[1] != new_matrix.shape[1]:
                raise StoreError(
                    "Dimension d'embedding différente de celle de l'index : réindexe tous les documents."
                )
            next_id = max((c["chunk_id"] for c in kept_chunks), default=-1) + 1
            new_chunks = [
                {"text": it["text"], "page": it["page"], "source": source, "chunk_id": next_id + i}
                for i, it in enumerate(items)
            ]
            matrix = np.vstack([kept_matrix, new_matrix]) if kept_matrix.shape[0] else new_matrix
            state = Snapshot(kept_chunks + new_chunks, matrix)
            self.save(state)  # si l'écriture échoue, l'état en mémoire n'est pas modifié
            self._state = state
            return len(new_chunks)

    def remove_document(self, source: str) -> int:
        source = canonical_source(source)
        with self._lock:
            old = self._state
            keep = old.sources != source
            removed = int((~keep).sum()) if old.size else 0
            if not removed:
                return 0
            matrix = old.matrix[keep]
            state = Snapshot([c for c, k in zip(old.chunks, keep) if k], matrix if matrix.shape[0] else _EMPTY)
            self.save(state)
            self._state = state
            return removed
