"""
Chatbot en ligne de commande pour tester manuellement le pipeline RAG
(recherche FAISS + génération Mistral) avant de l'exposer via l'API.

Usage :
    python scripts/chatbot.py
    (tapez 'quit' ou 'exit' pour arrêter)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

sys.path.append(str(Path(__file__).resolve().parent))
from embeddings_factory import EmbeddingConfigError, get_embeddings  # noqa: E402
from rag_chain import EventChatbotError, answer_question, get_llm  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index" / "faiss_events"
INDEX_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index_metadata.json"


def load_vectorstore() -> FAISS:
    if not INDEX_DIR.exists():
        raise SystemExit(
            f"Index introuvable : {INDEX_DIR}\nLancez d'abord : python scripts/build_index.py"
        )
    provider = "mistral"
    if INDEX_METADATA_PATH.exists():
        with open(INDEX_METADATA_PATH, encoding="utf-8") as f:
            provider = json.load(f).get("provider", "mistral")

    embeddings = get_embeddings(provider)
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def main() -> None:
    load_dotenv()

    print("Chargement de l'index FAISS...")
    try:
        vectorstore = load_vectorstore()
    except EmbeddingConfigError as exc:
        raise SystemExit(f"ERREUR de configuration des embeddings : {exc}")

    try:
        llm = get_llm()
    except EventChatbotError as exc:
        raise SystemExit(f"ERREUR : {exc}")

    print("Chatbot Puls-Events prêt. Posez votre question (ou 'quit' pour quitter).\n")

    while True:
        try:
            question = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir !")
            break
        if not question:
            continue

        try:
            result = answer_question(vectorstore, llm, question)
        except EventChatbotError as exc:
            print(f"Erreur : {exc}\n")
            continue

        print(f"\nAssistant > {result['answer']}\n")
        if result["sources"]:
            print(f"({result['num_events_found']} événement(s) source) :")
            for source in result["sources"]:
                print(f"  - {source['title']} ({source['city']}) — {source['url']}")
        print()


if __name__ == "__main__":
    main()