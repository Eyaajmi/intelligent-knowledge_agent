"""Réindexe tous les PDF de data/uploads (nouveau découpage, nouveau modèle ou préfixes d'embedding).

    python -m scripts.reindex

Nécessite Ollama. L'ancien index est conservé en `.bak` ; le nouveau remplace l'ancien seulement
si TOUT s'est bien passé.
"""
from __future__ import annotations

import os
import shutil
import sys

from app.config import settings
from app.indexing import index_pdf
from app.loader import InvalidPdfError, NoExtractableTextError
from app.ollama_client import OllamaError
from app.store import IndexStore, canonical_source


def main() -> int:
    pdfs = sorted(settings.upload_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Aucun PDF dans {settings.upload_dir}")
        return 1

    new_path = settings.index_path.with_suffix(".json.new")
    new_path.unlink(missing_ok=True)
    store = IndexStore(new_path)
    failures = 0
    for pdf in pdfs:
        try:
            info = index_pdf(store, pdf, source=canonical_source(pdf))
            print(f"OK   {pdf.name}: {info['chunks']} passages / {info['pages']} pages")
        except (InvalidPdfError, NoExtractableTextError) as exc:
            failures += 1
            print(f"SKIP {pdf.name}: {exc}")
        except OllamaError as exc:
            print(f"ECHEC {pdf.name}: {exc}\nAbandon : l'index actuel n'a pas été modifié.")
            new_path.unlink(missing_ok=True)
            return 1

    if settings.index_path.exists():
        shutil.copy2(settings.index_path, settings.index_path.with_suffix(".json.bak"))
    os.replace(new_path, settings.index_path)
    print(f"\nIndex reconstruit : {store.snapshot().size} passages ({failures} fichier(s) ignoré(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
