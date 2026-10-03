import requests
import re


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b-instruct"


def score_relevance(question, passage):
    """
    Demande au LLM d'évaluer la pertinence
    d'un passage par rapport à une question.

    Score :
    0 = pas pertinent
    1 = légèrement pertinent
    2 = pertinent
    3 = très pertinent
    """

    prompt = f"""
Tu es un système de reranking pour un moteur de recherche documentaire.

Évalue la pertinence du passage par rapport à la question.

Question :
{question}

Passage :
{passage}

Donne uniquement un score entier entre 0 et 3 :

0 = le passage ne répond pas à la question
1 = le passage est légèrement lié à la question
2 = le passage est pertinent
3 = le passage répond directement à la question

Réponds uniquement avec le nombre.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
    )

    response.raise_for_status()

    result = response.json()

    answer = result["message"]["content"].strip()

    match = re.search(r"[0-3]", answer)

    if match is None:
        return 0

    return int(match.group())


def rerank(question, results):
    """
    Re-classe les passages selon leur pertinence
    évaluée par le LLM.
    """

    reranked_results = []

    for result in results:

        score = score_relevance(
            question,
            result["text"]
        )

        new_result = result.copy()

        new_result["rerank_score"] = score

        reranked_results.append(new_result)

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True
    )

    return reranked_results