"""
Chaîne RAG (Retrieval-Augmented Generation) : recherche sémantique dans
l'index FAISS puis génération de réponse en langage naturel avec Mistral.

Conformément au brief, le POC est STATELESS : aucun historique de
conversation n'est conservé entre les questions (chaque appel à
`answer_question` est indépendant).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore

DEFAULT_MISTRAL_CHAT_MODEL = "mistral-small-latest"
DEFAULT_TOP_K = 4

JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def current_date_str() -> str:
    """Date du jour en français, ex. 'vendredi 11 juillet 2026'.

    Le LLM n'a aucune notion fiable de la date actuelle par lui-même : sans
    cette info explicite dans le prompt, il peut halluciner une date pour
    interpréter des expressions relatives comme "ce week-end" ou "demain".
    """
    now = datetime.now()
    return f"{JOURS_FR[now.weekday()]} {now.day} {MOIS_FR[now.month - 1]} {now.year}"


def build_system_prompt() -> str:
    return f"""Tu es l'assistant culturel de Puls-Events. Tu aides les utilisateurs \
à trouver des événements culturels en Île-de-France (concerts, expositions, spectacles, \
festivals...).

Nous sommes aujourd'hui le {current_date_str()}. Utilise cette date UNIQUEMENT si la \
question contient une expression temporelle explicite (ex. "ce week-end", "demain", \
"le mois prochain", "aujourd'hui") — ne l'utilise jamais autrement.

IMPORTANT : tous les événements fournis dans le contexte ci-dessous ont déjà été \
vérifiés comme étant en cours ou à venir (jamais terminés). Si la question de \
l'utilisateur NE PRÉCISE AUCUNE période ou date (ex. "un concert à Paris", "une \
exposition d'art"), NE LIMITE PAS ta réponse aux événements du jour même : tout \
événement du contexte qui correspond au thème demandé est une réponse valide, quelle \
que soit sa date exacte. N'applique un filtre de date strict QUE si l'utilisateur en \
demande une explicitement.

Règles impératives :
- Réponds UNIQUEMENT à partir des événements fournis dans le contexte ci-dessous. \
N'invente jamais d'événement, de date, de lieu ou de détail qui n'y figure pas.
- Si aucun événement du contexte ne correspond au thème ou au lieu demandé (ou à la \
période, si une période a été explicitement demandée), dis-le clairement et propose \
à l'utilisateur de reformuler sa recherche (ne propose jamais un événement non \
pertinent juste pour répondre quelque chose).
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


def format_context(documents: list[Document], reference_time: datetime | None = None) -> str:
    """Formate les chunks récupérés en un bloc de contexte lisible par le LLM,
    avec les métadonnées essentielles (titre, lieu, dates) à côté du texte.

    Pour un événement récurrent ou de longue durée (ex. saison de concerts),
    la date de début peut être dans le passé alors que l'événement est
    toujours d'actualité (sa date de fin est future). Sans clarification,
    le LLM interprète parfois une date de début passée comme signifiant que
    l'événement est terminé. On l'annote donc explicitement dans ce cas.
    """
    if not documents:
        return "Aucun événement trouvé."

    reference_time = reference_time or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        date_note = ""
        date_start = parse_event_datetime(meta.get("date_start"))
        if date_start:
            if date_start.tzinfo is None:
                date_start = date_start.replace(tzinfo=timezone.utc)
            if date_start < reference_time:
                date_note = (
                    " (ATTENTION : la date de début est déjà passée, mais cet événement "
                    "est encore d'actualité — il s'agit probablement d'un événement "
                    "récurrent ou se déroulant sur une longue période. Ne le considère "
                    "PAS comme terminé.)"
                )

        block = (
            f"[Événement {i}]\n"
            f"Titre : {meta.get('title', 'N/A')}\n"
            f"Ville : {meta.get('city', 'N/A')}\n"
            f"Dates : {meta.get('date_start', 'N/A')} -> {meta.get('date_end', 'N/A')}{date_note}\n"
            f"URL : {meta.get('url', 'N/A')}\n"
            f"Description : {doc.page_content}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def retrieve_events(vectorstore: VectorStore, question: str, k: int = DEFAULT_TOP_K) -> list[Document]:
    """Recherche sémantique dans l'index FAISS."""
    return vectorstore.similarity_search(question, k=k)


def parse_event_datetime(value: str | None) -> datetime | None:
    """Parse une date ISO8601 issue des métadonnées OpenAgenda (formats
    variables : suffixe 'Z' ou offset explicite '+02:00'). Retourne None si
    la valeur est absente ou illisible, plutôt que de lever une exception."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_future_events(documents: list[Document], reference_time: datetime | None = None) -> list[Document]:
    """Écarte les chunks dont l'événement est déjà terminé.

    Le LLM ne doit JAMAIS avoir à juger lui-même si un événement est passé ou
    futur : ce calcul de dates lui est peu fiable (constaté en test : un
    événement du 6 juin recommandé alors qu'on était le 11 juillet, ou une
    erreur de calcul du type "se termine demain" pour un événement déjà
    terminé depuis une semaine). On filtre donc ici, programmatiquement,
    AVANT que le contexte n'atteigne le prompt.

    Un événement sans date de fin exploitable (date manquante ou illisible)
    est conservé par prudence plutôt que silencieusement écarté.
    """
    reference_time = reference_time or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    kept: list[Document] = []
    for doc in documents:
        date_end = parse_event_datetime(doc.metadata.get("date_end"))
        if date_end is None:
            kept.append(doc)
            continue
        if date_end.tzinfo is None:
            date_end = date_end.replace(tzinfo=timezone.utc)
        if date_end >= reference_time:
            kept.append(doc)
    return kept


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

    # On récupère beaucoup plus de candidats que le k final demandé, car une
    # bonne partie sera écartée par filter_future_events (événements déjà
    # passés). En pratique, sur les données réelles, les événements passés
    # peuvent être 10x plus nombreux que les événements futurs pour un même
    # thème (ex. 273 concerts de jazz passés à Paris contre seulement 26 à
    # venir) : une marge trop faible (ex. x4) laisse alors 0 résultat après
    # filtrage alors que des événements pertinents existent bien dans l'index.
    fetch_k = max(k * 15, 60)
    candidates = retrieve_events(vectorstore, question, k=fetch_k)
    documents = filter_future_events(candidates)[:k]

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
        SystemMessage(content=build_system_prompt()),
        HumanMessage(content=f"Contexte (événements disponibles) :\n\n{context}\n\nQuestion : {question}"),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": deduplicate_sources(documents),
        "num_events_found": len(deduplicate_sources(documents)),
    }
