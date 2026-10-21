import pytest

from app import agent, ollama_client
from app.agent import ABSTAIN
from app.ollama_client import OllamaError
from tests.conftest import add


def test_no_documents(store, fake_ollama):
    out = agent.ask("Que dit le règlement ?", store)
    assert "Aucun document" in out["answer"] and out["answerable"] is False
    assert fake_ollama.chat_calls == []


def test_off_topic_question_abstains_without_calling_the_llm(populated, fake_ollama):
    out = agent.ask("Quelle est la couleur du logo de Maroclear ?", populated)
    assert out["answer"] == ABSTAIN and out["sources"] == [] and out["answerable"] is False
    assert fake_ollama.chat_calls == []  # pas d'appel LLM => pas d'hallucination possible


def test_answer_has_verifiable_sources(populated, fake_ollama):
    out = agent.ask("Quelle est la durée maximale de la suspension de travail ?", populated)
    assert out["answerable"] and out["tool"] == "search"
    top = out["sources"][0]
    assert top["name"] == "reglement.pdf" and top["page"] == 16
    assert "huit jours" in top["snippet"]
    system, user = fake_ollama.chat_calls[-1]
    assert "page 16" in user["content"] and "UNIQUEMENT" in system["content"]


def test_sources_are_unique_per_document_and_page(populated):
    out = agent.ask("sanctions disciplinaires suspension", populated)
    keys = [(s["source"], s["page"]) for s in out["sources"]]
    assert len(keys) == len(set(keys))


def test_follow_up_question_is_rewritten_with_history(populated, fake_ollama):
    history = [
        {"role": "user", "content": "Quelles sont les sanctions disciplinaires ?"},
        {"role": "assistant", "content": "L'avertissement, le blâme et la mise à pied."},
    ]
    out = agent.ask("Et combien de temps au maximum ?", populated, history=history)
    assert out["rewritten_question"] == fake_ollama.rewrite_to
    assert len(fake_ollama.chat_calls) == 2  # réécriture + réponse
    assert out["sources"][0]["page"] == 16


def test_rewrite_failure_falls_back_to_original_question(populated, monkeypatch):
    def boom(*a, **k):
        raise OllamaError("down")
    monkeypatch.setattr(ollama_client, "chat", boom)
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    assert agent.rewrite_question("Ma question ?", history) == "Ma question ?"


def test_list_documents_tool(populated, fake_ollama):
    out = agent.ask("Quels sont les documents disponibles ?", populated)
    assert out["tool"] == "list_documents"
    assert "reglement.pdf" in out["answer"] and "societe.pdf" in out["answer"]
    assert fake_ollama.chat_calls == []


def test_question_about_documents_content_is_not_mistaken_for_listing(populated):
    assert agent.ask("Quels documents parlent de la suspension de travail ?", populated)["tool"] == "search"


def test_summary_tool_with_selected_document(populated, fake_ollama):
    out = agent.ask("Résume ce document", populated, source="data/uploads/reglement.pdf")
    assert out["tool"] == "summarize"
    assert "reglement.pdf" in fake_ollama.chat_calls[-1][1]["content"]
    assert {s["page"] for s in out["sources"]} == {11, 16}


def test_summary_without_target_is_ambiguous_and_falls_back_to_search(populated):
    assert agent.ask("Résume ce document", populated)["tool"] == "search"  # 2 documents, aucun choisi


def test_topic_summary_uses_retrieval_not_whole_document(populated):
    out = agent.ask("résume les sanctions disciplinaires", populated, source="data/uploads/reglement.pdf")
    assert out["tool"] == "search"


def test_single_document_summary_needs_no_selection(store, fake_ollama):
    add(store, "solo.pdf", {1: "Contenu du seul document disponible dans l'index."})
    assert agent.ask("Fais un résumé du document", store)["tool"] == "summarize"


def test_stream_event_order(populated):
    events = list(agent.stream(agent.prepare("durée maximale suspension de travail", populated)))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and set(kinds[1:-1]) == {"token"}
    assert events[0]["sources"][0]["page"] == 16


def test_stream_reports_mid_stream_errors(populated, monkeypatch):
    def broken(messages, temperature=0.1):
        yield "début "
        raise OllamaError("connexion perdue")
    monkeypatch.setattr(ollama_client, "chat_stream", broken)
    events = list(agent.stream(agent.prepare("durée maximale suspension de travail", populated)))
    assert events[-1] == {"type": "error", "detail": "connexion perdue"}
