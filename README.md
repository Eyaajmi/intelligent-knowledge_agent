# Intelligent Knowledge Agent

Questions/réponses sur des documents PDF, **100 % local** (Ollama). Recherche hybride
(sémantique + mots-clés), réponses en streaming, sources citées avec extrait et lien vers la page du PDF.

## Démarrage

Prérequis : Python 3.10+, Node 18+, [Ollama](https://ollama.com).

```bash
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text

python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

cd knowledge-agent-frontend
npm install
npm start                          # http://localhost:4200
```

Configuration optionnelle : copie `.env.example` en `.env` (chargé automatiquement).

## Mise à jour depuis la v1

1. Sauvegarde ton projet (surtout `data/`).
2. Remplace `app/` et `tests/` par ceux de cette version. Les anciens fichiers disparaissent :
   `rag_pipeline.py`, `semantic_search.py`, `keyword_search.py`, `hybrid_search.py`, `reranker.py`,
   `document_indexer.py`, `document_loader.py`, `agent_tools.py` et tous les `app/*_test.py`.
3. Remplace `knowledge-agent-frontend/src/` et copie `angular.json`, `package.json`, `karma.ci.conf.js`.
4. **Nettoie ton index** (sans Ollama, sauvegarde `.bak` automatique) :
   ```bash
   python -m scripts.migrate_index --dry-run   # voir ce qui sera fait
   python -m scripts.migrate_index
   ```
   Sur l'index d'origine : 1 438 → 847 passages (591 doublons supprimés), 23 Mo → 6,6 Mo.
   Même sans cette étape, l'API nettoie l'index en mémoire au démarrage.
5. *Optionnel* : `python -m scripts.reindex` pour réindexer avec le nouveau découpage
   (800 caractères avec chevauchement, fragments < 40 caractères ignorés). Nécessite Ollama.

## Architecture

| Module | Rôle |
|---|---|
| `config.py` | Réglages par variables d'environnement / `.env` |
| `ollama_client.py` | Embeddings et chat : timeouts, erreurs explicites, streaming |
| `loader.py`, `chunking.py`, `indexing.py` | PDF → passages (phrases entières, chevauchement) → embeddings |
| `store.py` | Index en mémoire (matrice numpy + TF-IDF), sources canoniques, écriture atomique |
| `search.py` | Fusion RRF sémantique + mots-clés, dédoublonnage, modes `hybrid`/`semantic`/`keyword` |
| `tools.py`, `agent.py` | Routage (recherche / liste / résumé), historique, garde-fou, prompts |
| `main.py` | API FastAPI |

## API

| Route | Description |
|---|---|
| `GET /health` | État du serveur, d'Ollama et de l'index |
| `GET /documents` | `{documents: [{id, name, pages, chunks}]}` |
| `POST /documents/upload` | PDF uniquement, limité en taille ; `422` si PDF scanné, `503` si Ollama est hors ligne |
| `DELETE /documents?source=<id>` | Supprime le document, ses passages et son fichier |
| `GET /documents/file?source=<id>` | Sert le PDF (ouvrir à `#page=N`) |
| `POST /ask` | `{question, source?, history?}` → `{answer, sources[], tool, answerable}` |
| `POST /ask/stream` | Même entrée, flux SSE : `meta` (sources) → `token`* → `done` / `error` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                              # 62 tests, sans Ollama (faux Ollama déterministe)
RUN_LIVE=1 pytest tests/test_live_quality.py        # bout en bout, avec Ollama et ton index

cd knowledge-agent-frontend && npm run test:ci      # 13 tests Karma (nécessite Chrome / CHROME_BIN)
```

## Évaluer et calibrer

```bash
python -m scripts.evaluate
```

Compare `semantic` / `keyword` / `hybrid` (Recall@k, MRR) sur `tests/retrieval_cases.py` et affiche la
distribution des scores des questions avec et sans réponse pour choisir `MIN_SEMANTIC_SCORE` et
`MIN_KEYWORD_SCORE`. **Ajoute tes propres questions**, en particulier des questions voisines de ton
corpus mais sans réponse.

## Limites connues

- **Seuils non calibrés** : les valeurs par défaut sont prudentes. Le garde-fou n'attrape que les questions
  hors sujet ; une question voisine mais sans réponse reste gérée par le prompt du modèle.
- **Préfixes `nomic-embed-text`** (`search_query:` / `search_document:`) : recommandés par l'auteur du modèle,
  désactivés par défaut car ils obligent à réindexer. À tester avec `scripts.evaluate`.
- **Routeur à règles** : volontairement simple (un modèle 3B choisit mal ses outils). Étape suivante : tool calling.
- **Pas d'OCR** : un PDF scanné est refusé avec un message clair (`ocrmypdf` avant l'upload).
- **Pas d'authentification** : ne pas exposer l'API telle quelle hors de ta machine.
- **Index JSON en mémoire** : adapté à quelques dizaines de milliers de passages ; au-delà, passer à FAISS / Qdrant.
- Le faux reranker de la v1 a été retiré : il recalculait le même cosinus que la recherche. Un vrai
  cross-encoder (ex. `bge-reranker-v2-m3`) est la bonne amélioration à mesurer ensuite.
