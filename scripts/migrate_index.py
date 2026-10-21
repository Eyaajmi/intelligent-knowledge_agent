"""Nettoie un ancien index : sources canoniques, doublons supprimés, ids uniques.

    python -m scripts.migrate_index [--index data/index.json] [--dry-run]

Ne nécessite pas Ollama. Une sauvegarde `index.json.bak` est créée avant toute écriture.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from app.config import settings
from app.store import IndexStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(settings.index_path))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.index)
    if not path.exists():
        print(f"Introuvable : {path}")
        return 1

    store = IndexStore(path)  # la normalisation se fait au chargement
    r = store.report
    print(f"Passages avant : {r['input']} | après : {r['output']}")
    print(f"  sources renommées (\\ -> /) : {r['renamed_sources']}")
    print(f"  doublons supprimés        : {r['duplicates_removed']}")
    print(f"  entrées invalides ignorées: {r['invalid_skipped']}")
    for d in store.documents():
        print(f"  - {d['name']}: {d['chunks']} passages, {d['pages']} pages")

    if r["duplicates_removed"] == 0 and r["renamed_sources"] == 0 and r["invalid_skipped"] == 0:
        print("Rien à nettoyer.")
        return 0
    if args.dry_run:
        print("(dry-run : rien n'a été écrit)")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    store.save()
    print(f"Index nettoyé écrit. Sauvegarde : {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
