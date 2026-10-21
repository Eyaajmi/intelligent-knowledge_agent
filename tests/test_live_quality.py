"""Tests de bout en bout (Ollama + index réel). Lancer avec : RUN_LIVE=1 pytest tests/test_live_quality.py"""
import os

import pytest

from app import agent
from app.agent import ABSTAIN
from app.config import settings
from app.store import IndexStore

pytestmark = pytest.mark.skipif(not os.getenv("RUN_LIVE"), reason="nécessite Ollama et un index réel (RUN_LIVE=1)")


@pytest.fixture(scope="module")
def store():
    return IndexStore(settings.index_path)


def test_answer_contains_expected_information(store):
    result = agent.ask("Quelle est la durée maximale de suspension de travail prévue par le règlement intérieur ?", store)
    assert "8" in result["answer"] or "huit" in result["answer"].lower()
    assert 16 in {s["page"] for s in result["sources"]}


def test_missing_information_is_acknowledged(store):
    result = agent.ask("Quelle est la couleur du logo de Maroclear ?", store)
    answer = result["answer"].lower()
    assert result["answer"] == ABSTAIN or "ne trouve pas" in answer or "pas dans" in answer
