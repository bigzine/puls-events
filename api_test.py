"""
Tests fonctionnels de l'API REST (/ask, /rebuild, /health).

Lancer avec :
    pytest api_test.py -v

Ces tests utilisent le TestClient de FastAPI (basé sur httpx) : ils
n'exigent PAS de lancer réellement `uvicorn`. Le LLM et les embeddings sont
remplacés par des versions factices (aucun appel réseau, aucune clé API
requise), pour tester le comportement de l'API elle-même — routage, codes
HTTP, validation, sécurité — indépendamment de la qualité réelle des
réponses générées (couverte par les tests de rag_chain.py et par
evaluate_rag.py).
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))

from api.main import app, state  # noqa: E402
from build_index import build_faiss_index, chunk_events  # noqa: E402

EMBEDDING_DIM = 64


def make_event(**overrides):
    from datetime import datetime, timedelta, timezone
    base = {
        "uid": 1,
        "title": "Concert de jazz",
        "description_full": "Un concert de jazz au Sunset, avec un trio local reconnu.",
        "city": "Paris",
        "address": "60 rue des Lombards",
        "postal_code": "75001",
        "date_start": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat(),
        "date_end": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat(),
        "keywords": ["jazz", "concert"],
        "url": "https://openagenda.com/1/events/1",
        "source_agenda_title": "Culture Paris",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def reset_state():
    """Réinitialise l'état applicatif partagé avant/après chaque test pour
    éviter toute fuite entre tests (ex. un index chargé par un test qui
    fausserait le suivant)."""
    original_vectorstore = state.vectorstore
    original_llm = state.llm
    original_rebuild_in_progress = state.rebuild_in_progress
    original_error = state.last_rebuild_error
    yield
    state.vectorstore = original_vectorstore
    state.llm = original_llm
    state.rebuild_in_progress = original_rebuild_in_progress
    state.last_rebuild_error = original_error


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded_state():
    """Injecte un index FAISS factice + un LLM factice dans l'état de l'API,
    pour tester /ask sans dépendre d'un vrai index sur disque ni d'une clé
    Mistral."""
    embeddings = DeterministicFakeEmbedding(size=EMBEDDING_DIM)
    events = [
        make_event(uid=1, title="Concert de jazz", description_full="Un concert de jazz au Sunset."),
        make_event(uid=2, title="Exposition photo", description_full="Une exposition de photographie."),
    ]
    docs = chunk_events(events)
    state.vectorstore = build_faiss_index(docs, embeddings, batch_size=64)
    state.llm = FakeListChatModel(responses=["Voici un événement qui pourrait vous plaire."])
    return state


# ---------------------------------------------------------------------------
# Routes de base
# ---------------------------------------------------------------------------

class TestRootAndHealth:
    def test_root_returns_api_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()

    def test_health_degraded_when_no_index(self, client):
        state.vectorstore = None
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["index_loaded"] is False

    def test_health_ok_when_index_loaded(self, client, loaded_state):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["index_loaded"] is True
        assert response.json()["num_vectors"] > 0


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    def test_ask_returns_answer_and_sources(self, client, loaded_state):
        response = client.post("/ask", json={"question": "un concert de jazz"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Voici un événement qui pourrait vous plaire."
        assert isinstance(data["sources"], list)
        assert "num_events_found" in data

    def test_ask_empty_question_is_rejected(self, client, loaded_state):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422  # validation Pydantic

    def test_ask_whitespace_only_question_is_rejected(self, client, loaded_state):
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 422

    def test_ask_missing_question_field(self, client, loaded_state):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_ask_question_too_long_is_rejected(self, client, loaded_state):
        response = client.post("/ask", json={"question": "a" * 2000})
        assert response.status_code == 422

    def test_ask_without_index_returns_503(self, client):
        state.vectorstore = None
        response = client.post("/ask", json={"question": "un concert"})
        assert response.status_code == 503

    def test_ask_respects_custom_k(self, client, loaded_state):
        response = client.post("/ask", json={"question": "un concert", "k": 1})
        assert response.status_code == 200

    def test_ask_invalid_k_is_rejected(self, client, loaded_state):
        response = client.post("/ask", json={"question": "un concert", "k": 0})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /rebuild (protégé) et GET /rebuild/status
# ---------------------------------------------------------------------------

class TestRebuildEndpoint:
    def test_rebuild_without_configured_api_key_is_disabled(self, client, monkeypatch):
        monkeypatch.setattr("api.main.REBUILD_API_KEY", None)
        response = client.post("/rebuild", headers={"X-API-Key": "peu importe"})
        assert response.status_code == 503

    def test_rebuild_with_wrong_api_key_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr("api.main.REBUILD_API_KEY", "secret-correct")
        response = client.post("/rebuild", headers={"X-API-Key": "mauvaise-cle"})
        assert response.status_code == 401

    def test_rebuild_without_api_key_header_is_rejected(self, client, monkeypatch):
        monkeypatch.setattr("api.main.REBUILD_API_KEY", "secret-correct")
        response = client.post("/rebuild")
        assert response.status_code == 401

    def test_rebuild_with_correct_api_key_is_accepted(self, client, monkeypatch, mocker):
        monkeypatch.setattr("api.main.REBUILD_API_KEY", "secret-correct")
        # on ne veut PAS déclencher un vrai rebuild (appels API réels) dans ce test
        mocker.patch("api.main._run_rebuild")

        response = client.post("/rebuild", headers={"X-API-Key": "secret-correct"})
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"

    def test_rebuild_returns_409_if_already_in_progress(self, client, monkeypatch, mocker):
        monkeypatch.setattr("api.main.REBUILD_API_KEY", "secret-correct")
        mocker.patch("api.main._run_rebuild")
        state.rebuild_in_progress = True

        response = client.post("/rebuild", headers={"X-API-Key": "secret-correct"})
        assert response.status_code == 409

    def test_rebuild_status_reports_progress(self, client):
        state.rebuild_in_progress = True
        response = client.get("/rebuild/status")
        assert response.status_code == 200
        assert response.json()["rebuild_in_progress"] is True

    def test_rebuild_status_reports_last_error(self, client):
        state.last_rebuild_error = "Erreur simulée"
        response = client.get("/rebuild/status")
        assert response.status_code == 200
        assert response.json()["last_rebuild_error"] == "Erreur simulée"
