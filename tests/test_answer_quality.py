from app.agent import ask_agent


def test_answer_contains_expected_information():
    question = (
        "Quelle est la durée maximale de suspension "
        "de travail prévue par le règlement intérieur ?"
    )

    result = ask_agent(question)

    answer = result["answer"].lower()
    cited_pages = {
        source["page"]
        for source in result["sources"]
    }

    assert "8" in answer or "huit" in answer, (
        f"Expected the answer to mention 8 days. Got: {result['answer']}"
    )

    assert 16 in cited_pages, (
        f"Expected page 16 among sources. Got: {cited_pages}"
    )

    print("PASS: expected information and source page found.")


def test_agent_handles_missing_information():
    question = "Quelle est la couleur du logo de Maroclear ?"

    result = ask_agent(question)
    answer = result["answer"].lower()

    abstention_phrases = [
        "ne trouve pas",
        "pas dans le contexte",
        "pas dans les passages",
        "information n'est pas",
        "information n’est pas",
        "je ne peux pas déterminer",
    ]

    assert any(phrase in answer for phrase in abstention_phrases), (
        "The agent should acknowledge missing information. "
        f"Got: {result['answer']}"
    )

    print("PASS: missing information handled.")