import requests

from app.agent_tools import search_documents


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b-instruct"


def generate_answer(question, results):

    context = ""

    for result in results:

        context += (
            f"[Page {result['page']}]\n"
            f"{result['text']}\n\n"
        )

    prompt = f"""
Tu es un assistant documentaire.

Réponds à la question uniquement à partir des passages
fournis dans le contexte.

Si l'information n'est pas présente dans le contexte,
dis clairement que tu ne trouves pas l'information.

Ne fabrique aucune information.

Indique les pages utilisées dans ta réponse.

Question :
{question}

Contexte :
{context}
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

    return result["message"]["content"]


def ask_agent(question, source=None):
    results = search_documents(
        question,
        top_k=5,
        source=source
    )

    answer = generate_answer(
        question,
        results
    )

    return {
        "answer": answer,
        "sources": [
            {
                "page": result["page"],
                "source": result["source"]
            }
            for result in results
        ]
    }