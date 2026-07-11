"""
Chaîne RAG (Retrieval-Augmented Generation) : recherche sémantique dans
l'index FAISS puis génération de réponse en langage naturel avec Mistral.

Conformément au brief, le POC est STATELESS : aucun historique de
conversation n'est conservé entre les questions (chaque appel à
`answer_question` est indépendant).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore

DEFAULT_MISTRAL_CHAT_MODEL = "mistral-small-latest"
DEFAULT_TOP_K = 4

SYSTEM_PROMPT = """Tu es l'assistant culturel de Puls-Events. Tu aides les utilisateurs \
à trouver des événements culturels en Île-de-France (concerts, expositions, spectacles, \
festivals...).

Règles impératives :
- Réponds UNIQUEMENT à partir des événements fournis dans le contexte ci-dessous. \
N'invente jamais d'événement, de date, de lieu ou de détail qui n'y figure pas.
- Si aucun événement du contexte ne correspond à la question, dis-le clairement \
et propose à l'utilisateur de reformuler sa recherche (ne propose jamais un \
événement non pertinent juste pour répondre quelque chose).
- Mentionne systématiquement le titre, la ville et la date de chaque événement \
que tu recommandes.
- Réponds en français, de façon claire et conviviale, en 3 à 5 phrases maximum \
sauf si la question demande explicitement une liste plus longue.
"""


class EventChatbotError(Exception):
    """Erreur levée lors d'un échec du pipeline RAG."""


def get_llm(model: str | None = None, temperature: float = 0.2) -> BaseChatModel:
    """Instancie le modèle de chat Mistral utilisé pour la génération."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EventChatbotError(
            "MISTRAL_API_KEY est absente. Ajoutez votre clé Mistral dans .env "
            "pour activer la génération de réponses."
        )
    from langchain_mistralai import ChatMistralAI

    model = model or os.getenv("MISTRAL_CHAT_MODEL", DEFAULT_MISTRAL_CHAT_MODEL)
    return ChatMistralAI(model=model, api_key=api_key, temperature=temperature)


def format_context(documents: list[Document]) -> str:
    """Formate les chunks récupérés en un bloc de contexte lisible par le LLM,
    avec les métadonnées essentielles (titre, lieu, dates) à côté du texte."""
    if not documents:
        return "Aucun événement trouvé."

    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        block = (
            f"[Événement {i}]\n"
            f"Titre : {meta.get('title', 'N/A')}\n"
            f"Ville : {meta.get('city', 'N/A')}\n"
            f"Dates : {meta.get('date_start', 'N/A')} -> {meta.get('date_end', 'N/A')}\n"
            f"URL : {meta.get('url', 'N/A')}\n"
            f"Description : {doc.page_content}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def retrieve_events(vectorstore: VectorStore, question: str, k: int = DEFAULT_TOP_K) -> list[Document]:
    """Recherche sémantique dans l'index FAISS."""
    return vectorstore.similarity_search(question, k=k)


def deduplicate_sources(documents: list[Document]) -> list[dict[str, Any]]:
    """Construit la liste des sources citées (un événement = une source),
    en dédupliquant par uid pour éviter de citer 3 fois le même événement
    si plusieurs de ses chunks ont été récupérés."""
    seen_uids: set[Any] = set()
    sources: list[dict[str, Any]] = []
    for doc in documents:
        uid = doc.metadata.get("uid")
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        sources.append({
            "uid": uid,
            "title": doc.metadata.get("title"),
            "city": doc.metadata.get("city"),
            "date_start": doc.metadata.get("date_start"),
            "date_end": doc.metadata.get("date_end"),
            "url": doc.metadata.get("url"),
        })
    return sources


def answer_question(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    question: str,
    k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Pipeline RAG complet : recherche puis génération.

    Retourne un dict {answer, sources, num_events_found} plutôt qu'une simple
    chaîne de caractères, pour que l'API (étape suivante) puisse exposer les
    sources utilisées séparément de la réponse en langage naturel.
    """
    if not question or not question.strip():
        raise EventChatbotError("La question ne peut pas être vide.")

    documents = retrieve_events(vectorstore, question, k=k)

    if not documents:
        return {
            "answer": (
                "Je n'ai trouvé aucun événement correspondant à votre recherche. "
                "Essayez de reformuler votre question, par exemple en précisant "
                "un type d'événement ou une ville d'Île-de-France."
            ),
            "sources": [],
            "num_events_found": 0,
        }

    context = format_context(documents)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Contexte (événements disponibles) :\n\n{context}\n\nQuestion : {question}"),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": deduplicate_sources(documents),
        "num_events_found": len(deduplicate_sources(documents)),
    }