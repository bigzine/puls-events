"""
Teste manuellement la qualité de la recherche sémantique sur l'index FAISS
construit par build_index.py.

Usage :
    python scripts/search_test.py
    python scripts/search_test.py "concert de jazz à Paris"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

sys.path.append(str(Path(__file__).resolve().parent))
from embeddings_factory import EmbeddingConfigError, get_embeddings  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index" / "faiss_events"
INDEX_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index_metadata.json"

SAMPLE_QUERIES = [
    "concert de musique classique",
    "exposition d'art contemporain à Paris",
    "activité culturelle gratuite pour les enfants",
    "festival en Seine-Saint-Denis",
    "spectacle de danse",
]


def load_index() -> FAISS:
    if not INDEX_DIR.exists():
        raise SystemExit(
            f"Index introuvable : {INDEX_DIR}\nLancez d'abord : python scripts/build_index.py"
        )

    provider = "mistral"
    if INDEX_METADATA_PATH.exists():
        with open(INDEX_METADATA_PATH, encoding="utf-8") as f:
            provider = json.load(f).get("provider", "mistral")

    try:
        embeddings = get_embeddings(provider)
    except EmbeddingConfigError as exc:
        raise SystemExit(f"ERREUR de configuration des embeddings : {exc}")

    # allow_dangerous_deserialization : sûr ici car l'index est généré par
    # nos propres scripts (build_index.py), jamais depuis une source externe.
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def print_results(query: str, vectorstore: FAISS, k: int = 3) -> None:
    print(f"\n{'=' * 70}\nRequête : {query!r}\n{'=' * 70}")
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        print("  Aucun résultat.")
        return
    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"\n  #{rank} (score de distance : {score:.4f})")
        print(f"     Titre     : {meta.get('title')}")
        print(f"     Ville     : {meta.get('city')}")
        print(f"     Dates     : {meta.get('date_start')} -> {meta.get('date_end')}")
        print(f"     URL       : {meta.get('url')}")
        print(f"     Extrait   : {doc.page_content[:150]}...")


def main() -> None:
    load_dotenv()
    vectorstore = load_index()

    queries = sys.argv[1:] if len(sys.argv) > 1 else SAMPLE_QUERIES
    for query in queries:
        print_results(query, vectorstore)


if __name__ == "__main__":
    main()