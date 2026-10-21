"""Agent documentaire.

Étapes : routage -> (réécriture si historique) -> recherche -> garde-fou de pertinence
-> génération (avec ou sans streaming).

Le routeur est volontairement simple et déterministe (expressions régulières) : un modèle 3B
est peu fiable pour choisir un outil. Il se remplace plus tard par du tool calling.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from . import ollama_client, tools
from .config import settings
from .ollama_client import OllamaError
from .search import retrieve
from .store import IndexStore, canonical_source, doc_name

log = logging.getLogger(__name__)

ABSTAIN = "Je ne trouve pas cette information dans les documents indexés."
SNIPPET_CHARS = 300

SYSTEM_PROMPT = f"""Tu es un assistant documentaire. Tu réponds UNIQUEMENT à partir des extraits fournis.

Règles :
- Réponds dans la langue de la question, de façon claire et concise.
- Appuie chaque affirmation sur un extrait et cite-le ainsi : (nom_du_document, p. N).
- Si les extraits ne contiennent pas la réponse, réponds exactement : "{ABSTAIN}"
- N'invente rien et n'utilise pas de connaissances extérieures aux extraits."""

SUMMARY_PROMPT = """Tu es un assistant documentaire. À partir des extraits fournis (échantillon réparti dans tout le document), \
rédige un résumé structuré en quelques points. Précise que ce résumé est fondé sur un échantillon d'extraits. \
Cite les pages entre parenthèses (p. N). N'invente rien."""

REWRITE_PROMPT = """Réécris la dernière question de l'utilisateur pour qu'elle soit compréhensible sans la conversation \
(remplace les pronoms et références par leur sens). Ne réponds pas à la question. \
Réponds uniquement par la question réécrite, sans guillemets ni commentaire."""

_LIST_RE = re.compile(
    r"^\s*(quels?|quelles?|liste\w*|affiche\w*|montre\w*|which|list|show)\b.{0,40}?"
    r"\b(documents?|fichiers?|pdfs?|files?)\s*(disponibles?|index[ée]s?|charg[ée]s?|available|indexed)?\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(r"\b(r[ée]sum[eé]\w*|synth[eè]se|synth[ée]tise\w*|summari[sz]e|summary)\b", re.IGNORECASE)
_DOC_WORD_RE = re.compile(r"\b(document|doc|pdf|fichier|rapport|texte|file)\b", re.IGNORECASE)


@dataclass
class Prepared:
    tool: str
    answerable: bool
    messages: list[dict] | None = None
    direct_answer: str | None = None
    sources: list[dict] = field(default_factory=list)
    rewritten_question: str | None = None


# --------------------------------------------------------------------------- #
# Étapes
# --------------------------------------------------------------------------- #
def rewrite_question(question: str, history: list[dict]) -> str:
    """Transforme une question de suivi en question autonome (no-op sans historique)."""
    if not history:
        return question
    turns = history[-settings.history_turns * 2 :]
    convo = "\n".join(
        f"{'Utilisateur' if t['role'] == 'user' else 'Assistant'} : {t['content'][:500]}" for t in turns
    )
    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Conversation :\n{convo}\n\nNouvelle question : {question}\n\nQuestion autonome :"},
    ]
    try:
        rewritten = ollama_client.chat(messages, temperature=0).strip().strip('"«» ')
    except OllamaError:
        return question
    return rewritten if 3 <= len(rewritten) <= 500 else question


def _is_answerable(results: list[dict]) -> bool:
    """Garde-fou : écarte les requêtes sans aucun passage un tant soit peu pertinent."""
    best_sem = max((r["semantic_score"] or 0.0 for r in results), default=0.0)
    best_kw = max((r["keyword_score"] or 0.0 for r in results), default=0.0)
    return best_sem >= settings.min_semantic_score or best_kw >= settings.min_keyword_score


def _to_sources(results: list[dict]) -> list[dict]:
    """Une entrée par (document, page), avec extrait vérifiable."""
    best: dict[tuple, dict] = {}
    for r in results:
        key = (r["source"], r["page"])
        if key not in best or (r["semantic_score"] or 0) > (best[key]["score"] or 0):
            text = " ".join(r["text"].split())
            best[key] = {
                "source": r["source"],
                "name": doc_name(r["source"]),
                "page": r["page"],
                "snippet": text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…",
                "score": round(r["semantic_score"], 3) if r["semantic_score"] is not None else None,
            }
    return list(best.values())


def _context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {doc_name(r['source'])} — page {r['page']}\n{r['text']}" for i, r in enumerate(results, 1)
    )


def _summary_target(question: str, source: str | None, store: IndexStore) -> str | None:
    """Document à résumer si la demande est un résumé *global* ; sinon None (-> recherche)."""
    if not _SUMMARY_RE.search(question):
        return None
    ids = sorted(store.document_ids())
    named = [d for d in ids if len(doc_name(d)) > 4 and doc_name(d).rsplit(".", 1)[0].lower() in question.lower()]
    generic = bool(_DOC_WORD_RE.search(question)) or len(question.split()) <= 2 or bool(named)
    if not generic:
        return None
    if source:
        return source
    if named:
        return named[0]
    return ids[0] if len(ids) == 1 else None


def prepare(
    question: str,
    store: IndexStore,
    source: str | None = None,
    history: list[dict] | None = None,
) -> Prepared:
    question = question.strip()
    source = canonical_source(source) if source else None

    if store.snapshot().size == 0:
        return Prepared(tool="none", answerable=False, direct_answer=tools.NO_DOCS)

    if _LIST_RE.match(question):
        return Prepared(tool="list_documents", answerable=True, direct_answer=tools.list_documents(store))

    target = _summary_target(question, source, store)
    if target:
        passages = tools.summary_passages(store, target)
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Extraits de « {doc_name(target)} » :\n\n{_context(passages)}\n\nRésume ce document."},
        ]
        return Prepared(tool="summarize", answerable=True, messages=messages, sources=_to_sources(
            [{**p, "semantic_score": None, "keyword_score": None} for p in passages]
        ))

    standalone = rewrite_question(question, history or [])
    rewritten = standalone if standalone != question else None
    results = retrieve(store, standalone, top_k=settings.top_k, source=source)

    if not results or not _is_answerable(results):
        log.info("Garde-fou : aucun passage pertinent pour %r", standalone)
        return Prepared(tool="search", answerable=False, direct_answer=ABSTAIN, rewritten_question=rewritten)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extraits :\n\n{_context(results)}\n\nQuestion : {standalone}"},
    ]
    return Prepared(tool="search", answerable=True, messages=messages, sources=_to_sources(results), rewritten_question=rewritten)


# --------------------------------------------------------------------------- #
# Exécution
# --------------------------------------------------------------------------- #
def _meta(prep: Prepared) -> dict:
    return {
        "tool": prep.tool,
        "answerable": prep.answerable,
        "sources": prep.sources,
        "rewritten_question": prep.rewritten_question,
    }


def run(prep: Prepared) -> dict:
    """Réponse complète (sans streaming)."""
    answer = prep.direct_answer if prep.direct_answer is not None else ollama_client.chat(prep.messages)
    return {"answer": answer.strip(), **_meta(prep)}


def stream(prep: Prepared) -> Iterator[dict]:
    """Événements : meta -> token* -> done (ou error)."""
    yield {"type": "meta", **_meta(prep)}
    if prep.direct_answer is not None:
        yield {"type": "token", "text": prep.direct_answer}
        yield {"type": "done"}
        return
    try:
        for piece in ollama_client.chat_stream(prep.messages):
            yield {"type": "token", "text": piece}
    except OllamaError as exc:
        yield {"type": "error", "detail": str(exc)}
        return
    yield {"type": "done"}


def ask(question: str, store: IndexStore, source: str | None = None, history: list[dict] | None = None) -> dict:
    return run(prepare(question, store, source=source, history=history))
