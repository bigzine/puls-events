"""
API REST exposant le système RAG Puls-Events.

Lancement local :
    uvicorn api.main:app --reload --port 8000

Documentation interactive (Swagger, générée automatiquement par FastAPI) :
    http://localhost:8000/docs

Toute la logique métier (recherche, génération, indexation) vit dans
scripts/ — ce module ne fait qu'exposer cette logique via HTTP, conformément
à la recommandation de séparer l'API du cœur RAG.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, status
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from build_index import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    build_faiss_index,
    chunk_events,
    load_clean_events,
)
from datetime import datetime, timezone  # noqa: E402
from embeddings_factory import EmbeddingConfigError, get_embeddings  # noqa: E402
from rag_chain import EventChatbotError, answer_question, get_llm  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("puls_events_api")

load_dotenv()

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index" / "faiss_events"
INDEX_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index_metadata.json"
REBUILD_API_KEY = os.getenv("REBUILD_API_KEY")  # clé requise pour protéger /rebuild


# ---------------------------------------------------------------------------
# État applicatif (chargé UNE FOIS au démarrage, pas à chaque requête)
# ---------------------------------------------------------------------------

class AppState:
    """Regroupe les ressources coûteuses à initialiser (index, LLM), chargées
    une seule fois au démarrage plutôt qu'à chaque appel à /ask — relancer
    tout le pipeline (chargement du modèle d'embeddings, du LLM) à chaque
    requête serait inutilement lent."""

    def __init__(self) -> None:
        self.vectorstore: FAISS | None = None
        self.llm = None
        self.embedding_provider: str | None = None
        self.rebuild_lock = threading.Lock()
        self.rebuild_in_progress = False
        self.last_rebuild_error: str | None = None

    def load(self) -> None:
        if not INDEX_DIR.exists():
            logger.warning("Index FAISS introuvable à %s. /ask échouera tant qu'aucun index n'existe.", INDEX_DIR)
            return

        provider = "mistral"
        if INDEX_METADATA_PATH.exists():
            with open(INDEX_METADATA_PATH, encoding="utf-8") as f:
                provider = json.load(f).get("provider", "mistral")
        self.embedding_provider = provider

        embeddings = get_embeddings(provider)
        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        logger.info("Index FAISS chargé (%d vecteurs, provider=%s).", self.vectorstore.index.ntotal, provider)

    def ensure_llm(self):
        if self.llm is None:
            self.llm = get_llm()
        return self.llm


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.load()
    except EmbeddingConfigError as exc:
        logger.error("Échec du chargement de l'index au démarrage : %s", exc)
    yield


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API exposant le chatbot culturel Puls-Events : recherche sémantique "
        "d'événements en Île-de-France (FAISS) et génération de réponse en "
        "langage naturel (Mistral, via LangChain)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schémas de requête / réponse
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question de l'utilisateur en langage naturel.",
        examples=["Un concert de jazz à Paris ce mois-ci ?"],
    )
    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Nombre d'événements sources à considérer pour la réponse.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La question ne peut pas être vide ou ne contenir que des espaces.")
        return v.strip()


class SourceItem(BaseModel):
    uid: Any
    title: str | None
    city: str | None
    date_start: str | None
    date_end: str | None
    url: str | None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    num_events_found: int


class RebuildStatusResponse(BaseModel):
    rebuild_in_progress: bool
    last_build: dict[str, Any] | None
    last_rebuild_error: str | None


class RebuildAcceptedResponse(BaseModel):
    status: str
    detail: str


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Sécurité minimale sur /rebuild
# ---------------------------------------------------------------------------

def verify_rebuild_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protège /rebuild : opération coûteuse (appels API payants, minutes de
    traitement) qui ne doit pas être déclenchable par n'importe qui si l'API
    venait à être exposée au-delà de localhost.

    Si REBUILD_API_KEY n'est pas configurée côté serveur, l'endpoint reste
    bloqué par défaut (échec sécurisé) plutôt que de s'ouvrir silencieusement.
    """
    if not REBUILD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "L'endpoint /rebuild est désactivé : REBUILD_API_KEY n'est pas "
                "configurée côté serveur (voir .env)."
            ),
        )
    if x_api_key != REBUILD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide ou absente (en-tête X-API-Key requis).",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Santé"], summary="Informations générales sur l'API")
def root() -> dict[str, str]:
    return {
        "name": "Puls-Events RAG API",
        "docs": "/docs",
        "ask_endpoint": "POST /ask",
        "rebuild_endpoint": "POST /rebuild (protégé)",
    }


@app.get("/health", tags=["Santé"], summary="Vérifie que l'API et l'index sont opérationnels")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if state.vectorstore is not None else "degraded",
        "index_loaded": state.vectorstore is not None,
        "num_vectors": state.vectorstore.index.ntotal if state.vectorstore else 0,
        "embedding_provider": state.embedding_provider,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Chatbot"],
    summary="Pose une question au chatbot culturel Puls-Events",
    description=(
        "Recherche sémantique dans les événements indexés (FAISS) puis "
        "génération d'une réponse en langage naturel (Mistral). Le POC est "
        "stateless : chaque question est traitée indépendamment, sans "
        "historique de conversation."
    ),
)
def ask(request: AskRequest) -> AskResponse:
    if state.vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aucun index vectoriel chargé. Appelez POST /rebuild pour en construire un.",
        )

    try:
        llm = state.ensure_llm()
    except EventChatbotError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    try:
        result = answer_question(state.vectorstore, llm, request.question, k=request.k)
    except EventChatbotError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception:
        logger.exception("Erreur inattendue lors du traitement de la question.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur interne est survenue lors de la génération de la réponse.",
        )

    return AskResponse(**result)


def _run_rebuild() -> None:
    """Reconstruit l'index à partir des données déjà nettoyées
    (data/processed/events_clean.json) — ne re-télécharge PAS depuis
    OpenAgenda, qui reste un processus séparé et manuel."""
    with state.rebuild_lock:
        state.rebuild_in_progress = True
        state.last_rebuild_error = None
    try:
        events = load_clean_events()
        documents = chunk_events(events)
        embeddings = get_embeddings()
        provider = os.getenv("EMBEDDING_PROVIDER", "mistral")
        pace_seconds = 1.1 if provider == "mistral" else 0.0

        vectorstore = build_faiss_index(
            documents,
            embeddings,
            save_dir=str(INDEX_DIR),
            resume=True,
            pace_seconds=pace_seconds,
        )
        vectorstore.save_local(str(INDEX_DIR))

        metadata = {
            "provider": provider,
            "num_events": len(events),
            "num_chunks": len(documents),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        state.vectorstore = vectorstore
        state.embedding_provider = provider
        logger.info("Reconstruction de l'index terminée (%d vecteurs).", vectorstore.index.ntotal)
    except Exception as exc:
        logger.exception("Échec de la reconstruction de l'index.")
        state.last_rebuild_error = str(exc)
    finally:
        with state.rebuild_lock:
            state.rebuild_in_progress = False


@app.post(
    "/rebuild",
    response_model=RebuildAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={401: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Administration"],
    summary="Reconstruit la base vectorielle en arrière-plan (protégé par clé API)",
    description=(
        "Relance le chunking + la vectorisation à partir des données déjà "
        "nettoyées et reconstruit l'index FAISS. Opération potentiellement "
        "longue (plusieurs dizaines de secondes à quelques minutes selon le "
        "volume et le fournisseur d'embeddings) : elle s'exécute en tâche de "
        "fond, consultez GET /rebuild/status pour suivre sa progression. "
        "Nécessite l'en-tête `X-API-Key`."
    ),
    dependencies=[Depends(verify_rebuild_api_key)],
)
def rebuild(background_tasks: BackgroundTasks) -> RebuildAcceptedResponse:
    with state.rebuild_lock:
        if state.rebuild_in_progress:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Une reconstruction est déjà en cours. Consultez GET /rebuild/status.",
            )
    background_tasks.add_task(_run_rebuild)
    return RebuildAcceptedResponse(
        status="accepted",
        detail="Reconstruction de l'index lancée en arrière-plan. Consultez GET /rebuild/status.",
    )


@app.get(
    "/rebuild/status",
    response_model=RebuildStatusResponse,
    tags=["Administration"],
    summary="Consulte l'état de la dernière reconstruction d'index",
)
def rebuild_status() -> RebuildStatusResponse:
    last_build = None
    if INDEX_METADATA_PATH.exists():
        with open(INDEX_METADATA_PATH, encoding="utf-8") as f:
            last_build = json.load(f)
    return RebuildStatusResponse(
        rebuild_in_progress=state.rebuild_in_progress,
        last_build=last_build,
        last_rebuild_error=state.last_rebuild_error,
    )
