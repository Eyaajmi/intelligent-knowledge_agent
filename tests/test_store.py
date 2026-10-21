import json

import pytest

from app.store import IndexStore, StoreError, canonical_source, normalize_records
from tests.conftest import add, fake_vec


def _rec(source, text, page=1, cid=0):
    return {"text": text, "page": page, "source": source, "chunk_id": cid, "embedding": fake_vec(text).tolist()}


def test_canonical_source_unifies_separators():
    assert canonical_source("data/uploads\\a.pdf") == "data/uploads/a.pdf"
    assert canonical_source("./data//uploads/a.pdf") == "data/uploads/a.pdf"


def test_legacy_duplicates_with_backslashes_are_removed():
    """Reproduit le bug réel : même PDF indexé avec `/` puis avec `\\`."""
    records = [
        _rec("data/uploads/a.pdf", "Texte un sur la discipline.", cid=0),
        _rec("data/uploads/a.pdf", "Texte deux sur les congés.", cid=1),
        _rec("data/uploads\\a.pdf", "Texte un sur la discipline.", cid=2),
        _rec("data/uploads\\a.pdf", "Texte deux sur les congés.", cid=3),
    ]
    cleaned, report = normalize_records(records)
    assert len(cleaned) == 2
    assert report["duplicates_removed"] == 2 and report["renamed_sources"] == 2
    assert {r["source"] for r in cleaned} == {"data/uploads/a.pdf"}


def test_load_cleans_legacy_index_in_memory(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps([
        _rec("data/uploads/a.pdf", "Alpha bravo charlie.", cid=0),
        _rec("data/uploads\\a.pdf", "Alpha bravo charlie.", cid=1),
    ]), encoding="utf-8")
    store = IndexStore(path)
    assert store.snapshot().size == 1
    assert [d["id"] for d in store.documents()] == ["data/uploads/a.pdf"]


def test_duplicate_chunk_ids_are_reassigned():
    records = [_rec("a.pdf", "Un texte.", cid=0), _rec("b.pdf", "Un autre texte.", cid=0)]
    cleaned, report = normalize_records(records)
    assert sorted(r["chunk_id"] for r in cleaned) == [0, 1] and report["ids_reassigned"]


def test_re_adding_a_document_replaces_it(store):
    add(store, "data/uploads/a.pdf", {1: "Premier texte sur les congés.", 2: "Second texte sur la paie."})
    add(store, "data/uploads\\a.pdf", {1: "Premier texte sur les congés.", 2: "Second texte sur la paie."})
    assert store.snapshot().size == 2
    ids = [c["chunk_id"] for c in store.snapshot().chunks]
    assert len(set(ids)) == len(ids)


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "index.json"
    store = IndexStore(path)
    add(store, "data/uploads/a.pdf", {1: "Texte sur la discipline des employés."})
    query = fake_vec("discipline employés")
    before = store.snapshot().semantic_scores(query)
    reloaded = IndexStore(path)
    assert reloaded.snapshot().size == 1
    after = reloaded.snapshot().semantic_scores(query)
    assert abs(float(before[0]) - float(after[0])) < 1e-5  # arrondi à 6 décimales
    assert not list(tmp_path.glob("*.tmp"))


def test_remove_document(populated):
    assert populated.remove_document("data/uploads/societe.pdf") == 1
    assert populated.remove_document("data/uploads/societe.pdf") == 0
    assert [d["name"] for d in populated.documents()] == ["reglement.pdf"]


def test_remove_last_document_leaves_valid_empty_store(store):
    add(store, "a.pdf", {1: "Seul document indexé ici."})
    store.remove_document("a.pdf")
    assert store.snapshot().size == 0 and store.documents() == []


def test_mask_filters_by_source(populated):
    snap = populated.snapshot()
    assert snap.mask("data/uploads/societe.pdf").sum() == 1
    assert snap.mask(None).sum() == snap.size
    assert snap.mask("data/uploads\\societe.pdf").sum() == 1  # forme legacy acceptée


def test_embedding_dimension_mismatch_is_explicit(populated):
    with pytest.raises(StoreError, match="réindexe"):
        populated.add_document("x.pdf", [{"text": "t", "page": 1}], [[0.1, 0.2, 0.3]])
    with pytest.raises(StoreError, match="réindexe"):
        populated.snapshot().semantic_scores([0.1, 0.2])


def test_failed_save_keeps_memory_state(populated, monkeypatch):
    before = populated.snapshot().size
    monkeypatch.setattr(populated, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disque plein")))
    with pytest.raises(OSError):
        add(populated, "data/uploads/new.pdf", {1: "Nouveau document sur autre chose."})
    assert populated.snapshot().size == before


def test_corrupt_index_raises_instead_of_silently_wiping(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreError):
        IndexStore(path)
