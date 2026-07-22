"""
Affiche, pour chaque question du jeu de test, les chunks EXACTEMENT
récupérés par le système (même logique que answer_question : marge élargie
puis filtrage des événements passés) — pour réécrire les réponses de
référence à partir de faits que le système peut réellement retrouver,
plutôt que d'une vérité choisie indépendamment.

C'est la bonne pratique standard pour un jeu de test Ragas : une référence
qui décrit un fait hors de portée du retrieval fait mécaniquement chuter
context_recall, même si le système fonctionne correctement par ailleurs.

Usage :
    python eval/inspect_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from embeddings_factory import get_embeddings  # noqa: E402
from rag_chain import filter_future_events, retrieve_events  # noqa: E402

QA_DATASET_PATH = Path(__file__).resolve().parent / "qa_dataset.json"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index" / "faiss_events"
INDEX_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index_metadata.json"


def main() -> None:
    load_dotenv()

    provider = "mistral"
    if INDEX_METADATA_PATH.exists():
        with open(INDEX_METADATA_PATH, encoding="utf-8") as f:
            provider = json.load(f).get("provider", "mistral")

    embeddings = get_embeddings(provider)
    vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    with open(QA_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    for case in data["cases"]:
        question = case["question"]
        if not question.strip() or case.get("expects_refusal"):
            continue  # inutile pour les cas vides ou de refus attendu

        print(f"\n{'=' * 70}\n{case['id']} : {question}\n{'=' * 70}")

        # Même logique que answer_question : marge élargie puis filtre.
        fetch_k = max(4 * 15, 60)
        candidates = retrieve_events(vectorstore, question, k=fetch_k)
        documents = filter_future_events(candidates)[:4]

        if not documents:
            print("  (aucun résultat après filtrage)")
            continue

        for i, doc in enumerate(documents, start=1):
            meta = doc.metadata
            print(f"\n  [{i}] {meta.get('title')}")
            print(f"      Ville : {meta.get('city')} | Dates : {meta.get('date_start')} -> {meta.get('date_end')}")
            print(f"      URL   : {meta.get('url')}")
            print(f"      Texte : {doc.page_content[:200]}")


if __name__ == "__main__":
    main()
