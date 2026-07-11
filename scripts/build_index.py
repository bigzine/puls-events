"""
Découpage (chunking) des événements nettoyés et construction de l'index
vectoriel FAISS, avec conservation des métadonnées (dates, lieu, url...).

Usage :
    python scripts/build_index.py

Entrée  : data/processed/events_clean.json  (produit par clean_data.py)
Sortie  : data/index/faiss_events/           (index FAISS persistant)
          data/index/index_metadata.json     (infos de reproductibilité : provider, dimension, date de build)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).resolve().parent))
from embeddings_factory import EmbeddingConfigError, get_embeddings  # noqa: E402

EVENTS_CLEAN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "events_clean.json"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index" / "faiss_events"
INDEX_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index_metadata.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_BATCH_SIZE = 64


def load_clean_events(path: Path = EVENTS_CLEAN_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Fichier introuvable : {path}\n"
            "Lancez d'abord : python scripts/fetch_open_agenda.py puis python scripts/clean_data.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def event_to_text(event: dict) -> str:
    """Construit le texte source à vectoriser pour un événement.

    On inclut le titre, la ville et les mots-clés en plus de la description :
    cela permet à la recherche sémantique de matcher aussi sur des requêtes
    du type "concert à Nanterre" ou "exposition gratuite", pas uniquement sur
    le contenu narratif de la description.
    """
    parts = [event.get("title", "")]
    if event.get("city"):
        parts.append(f"Lieu : {event['city']}")
    if event.get("keywords"):
        parts.append("Mots-clés : " + ", ".join(event["keywords"]))
    parts.append(event.get("description_full", ""))
    return "\n".join(p for p in parts if p)


def chunk_events(events: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Découpe chaque événement en un ou plusieurs chunks (Document LangChain),
    en conservant les métadonnées métier sur chaque chunk (nécessaire pour
    que la recherche renvoie des informations exploitables, pas juste du texte)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for event in events:
        text = event_to_text(event)
        if not text.strip():
            continue

        chunks = splitter.split_text(text)
        for i, chunk_text in enumerate(chunks):
            metadata = {
                "uid": event["uid"],
                "title": event["title"],
                "city": event.get("city"),
                "address": event.get("address"),
                "postal_code": event.get("postal_code"),
                "date_start": event.get("date_start"),
                "date_end": event.get("date_end"),
                "url": event.get("url"),
                "source_agenda_title": event.get("source_agenda_title"),
                "chunk_index": i,
                "chunk_count": len(chunks),
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))

    return documents


def build_faiss_index(
    documents: list[Document],
    embeddings: Embeddings,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> FAISS:
    """Construit l'index FAISS par lots (batches), puis fusionne les lots.

    Le traitement par lots évite d'envoyer des milliers de textes en un seul
    appel API (risque de dépassement de limite de requête / timeout), et
    permet de suivre la progression sur un grand volume d'événements.
    """
    if not documents:
        raise ValueError("Aucun document à indexer.")

    vectorstore: FAISS | None = None
    total = len(documents)

    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        batch_store = FAISS.from_documents(batch, embeddings)

        if vectorstore is None:
            vectorstore = batch_store
        else:
            vectorstore.merge_from(batch_store)

        done = min(start + batch_size, total)
        print(f"  Indexation : {done}/{total} chunks vectorisés")

    return vectorstore


def main() -> None:
    load_dotenv()

    events = load_clean_events()
    print(f"Événements chargés : {len(events)}")

    documents = chunk_events(events)
    print(f"Chunks générés : {len(documents)}")

    try:
        embeddings = get_embeddings()
    except EmbeddingConfigError as exc:
        print(f"ERREUR de configuration des embeddings : {exc}", file=sys.stderr)
        sys.exit(1)

    provider = os.getenv("EMBEDDING_PROVIDER", "mistral")
    print(f"Fournisseur d'embeddings : {provider}")

    start_time = time.time()
    vectorstore = build_faiss_index(documents, embeddings)
    elapsed = time.time() - start_time

    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    metadata = {
        "provider": provider,
        "num_events": len(events),
        "num_chunks": len(documents),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "build_duration_seconds": round(elapsed, 1),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nIndex FAISS sauvegardé dans {INDEX_DIR}")
    print(f"Métadonnées de build sauvegardées dans {INDEX_METADATA_PATH}")
    print(f"Durée de construction : {elapsed:.1f}s pour {len(documents)} chunks")


if __name__ == "__main__":
    main()