import pytest

from app.search import retrieve
from tests.conftest import add


def test_hybrid_finds_the_right_page(populated):
    results = retrieve(populated, "durée maximale de la suspension de travail")
    assert results[0]["page"] == 16 and results[0]["source"].endswith("reglement.pdf")


def test_keyword_is_accent_insensitive(store, fake_ollama):
    add(store, "a.pdf", {1: "Le règlement intérieur précise les obligations des salariés."})
    add(store, "b.pdf", {1: "Le catalogue des produits est mis à jour chaque trimestre."})
    results = retrieve(store, "reglement interieur", mode="keyword")
    assert results and results[0]["source"] == "a.pdf"


def test_source_filter(populated):
    results = retrieve(populated, "siège société Tunis", source="data/uploads/societe.pdf")
    assert results and {r["source"] for r in results} == {"data/uploads/societe.pdf"}
    assert retrieve(populated, "siège", source="data/uploads/inconnu.pdf") == []


def test_identical_texts_are_not_returned_twice(store, fake_ollama):
    text = "La politique de remboursement autorise un retour sous trente jours."
    add(store, "a.pdf", {1: text})
    add(store, "b.pdf", {1: text})
    results = retrieve(store, "politique de remboursement")
    assert len(results) == 1


def test_results_expose_scores_for_every_candidate(populated):
    for r in retrieve(populated, "sanctions disciplinaires blâme"):
        assert r["semantic_score"] is not None and r["keyword_score"] is not None


def test_modes(populated):
    sem = retrieve(populated, "suspension de travail", mode="semantic")
    kw = retrieve(populated, "suspension de travail", mode="keyword")
    assert sem[0]["keyword_score"] is None and kw[0]["semantic_score"] is None


def test_keyword_mode_ignores_zero_score_passages(populated):
    assert retrieve(populated, "xyzzy plugh", mode="keyword") == []


def test_empty_store_and_bad_mode(tmp_path, populated, fake_ollama):
    from app.store import IndexStore
    assert retrieve(IndexStore(tmp_path / "vide.json"), "n'importe quoi") == []
    with pytest.raises(ValueError):
        retrieve(populated, "q", mode="magie")


def test_top_k_is_respected(populated):
    assert len(retrieve(populated, "suspension", top_k=1)) == 1


def test_keyword_normalises_plurals_and_ignores_question_words(store, fake_ollama):
    add(store, "a.pdf", {16: "Le blâme est une sanction disciplinaire qui peut suivre un avertissement."})
    add(store, "b.pdf", {1: "Le catalogue des produits est mis à jour chaque trimestre."})
    found = retrieve(store, "Quelle sanction peut suivre plusieurs blâmes ?", mode="keyword")
    assert found and found[0]["page"] == 16  # « blâmes » (pluriel) retrouve « blâme »
    # une question composée uniquement de mots interrogatifs ne doit rien matcher
    assert retrieve(store, "Comment combien quand pourquoi quelle ?", mode="keyword") == []
