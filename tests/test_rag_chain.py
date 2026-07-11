"""
Tests unitaires de l'étape 4 : chaîne RAG (retrieval + génération).

Lancer avec :
    pytest tests/test_rag_chain.py -v

Le LLM est simulé (FakeListChatModel de LangChain) : aucun appel réseau,
aucune clé API, réponses reproductibles. Les embeddings sont également
simulés (DeterministicFakeEmbedding, voir tests/test_indexing.py). Ces tests
valident la MÉCANIQUE de la chaîne (récupération, formatage du contexte,
dédoublonnage des sources, gestion du cas "aucun résultat") — pas la
qualité linguistique réelle des réponses, évaluée à l'étape 5 avec le
jeu de test annoté et le vrai modèle Mistral.
"""

import sys
from pathlib import Path

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from build_index import build_faiss_index, chunk_events  # noqa: E402
from rag_chain import (  # noqa: E402
    EventChatbotError,
    answer_question,
    deduplicate_sources,
    format_context,
    retrieve_events,
)

EMBEDDING_DIM = 64


def make_event(**overrides):
    base = {
        "uid": 1,
        "title": "Concert de jazz",
        "description_full": "Un concert de jazz au Sunset, avec un trio local reconnu.",
        "city": "Paris",
        "address": "60 rue des Lombards",
        "postal_code": "75001",
        "date_start": "2026-08-01T20:00:00+02:00",
        "date_end": "2026-08-01T23:00:00+02:00",
        "keywords": ["jazz", "concert"],
        "url": "https://openagenda.com/1/events/1",
        "source_agenda_title": "Culture Paris",
    }
    base.update(overrides)
    return base


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=EMBEDDING_DIM)


@pytest.fixture
def small_vectorstore(fake_embeddings):
    events = [
        make_event(uid=1, title="Concert de jazz", description_full="Un concert de jazz au Sunset."),
        make_event(uid=2, title="Exposition photo", description_full="Une exposition de photographie contemporaine."),
        make_event(uid=3, title="Spectacle de danse", description_full="Un spectacle de danse contemporaine à Créteil.", city="Créteil"),
    ]
    docs = chunk_events(events)
    return build_faiss_index(docs, fake_embeddings, batch_size=64)


class TestFormatContext:
    def test_empty_list_returns_placeholder(self):
        assert format_context([]) == "Aucun événement trouvé."

    def test_includes_key_metadata(self):
        doc = Document(
            page_content="Un super concert.",
            metadata={"title": "Concert", "city": "Paris", "date_start": "2026-01-01", "date_end": "2026-01-01", "url": "https://x"},
        )
        context = format_context([doc])
        assert "Concert" in context
        assert "Paris" in context
        assert "https://x" in context
        assert "Un super concert." in context

    def test_multiple_documents_are_numbered(self):
        docs = [
            Document(page_content="A", metadata={"title": "Event A"}),
            Document(page_content="B", metadata={"title": "Event B"}),
        ]
        context = format_context(docs)
        assert "[Événement 1]" in context
        assert "[Événement 2]" in context


class TestDeduplicateSources:
    def test_deduplicates_by_uid(self):
        docs = [
            Document(page_content="chunk 1", metadata={"uid": 1, "title": "Event A", "chunk_index": 0}),
            Document(page_content="chunk 2", metadata={"uid": 1, "title": "Event A", "chunk_index": 1}),
            Document(page_content="chunk 3", metadata={"uid": 2, "title": "Event B", "chunk_index": 0}),
        ]
        sources = deduplicate_sources(docs)
        assert len(sources) == 2
        assert {s["uid"] for s in sources} == {1, 2}

    def test_empty_list(self):
        assert deduplicate_sources([]) == []

    def test_preserves_metadata_fields(self):
        docs = [Document(page_content="x", metadata={"uid": 1, "title": "Event", "city": "Paris", "url": "https://x"})]
        sources = deduplicate_sources(docs)
        assert sources[0]["title"] == "Event"
        assert sources[0]["city"] == "Paris"
        assert sources[0]["url"] == "https://x"


class TestRetrieveEvents:
    def test_returns_documents(self, small_vectorstore):
        docs = retrieve_events(small_vectorstore, "concert de jazz", k=2)
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_respects_k(self, small_vectorstore):
        docs = retrieve_events(small_vectorstore, "spectacle", k=1)
        assert len(docs) == 1


class TestAnswerQuestion:
    def test_raises_on_empty_question(self, small_vectorstore):
        llm = FakeListChatModel(responses=["peu importe"])
        with pytest.raises(EventChatbotError):
            answer_question(small_vectorstore, llm, "   ")

    def test_returns_answer_and_sources(self, small_vectorstore):
        llm = FakeListChatModel(responses=["Voici un concert de jazz qui pourrait vous plaire."])
        result = answer_question(small_vectorstore, llm, "un concert de jazz", k=2)

        assert result["answer"] == "Voici un concert de jazz qui pourrait vous plaire."
        assert result["num_events_found"] >= 1
        assert len(result["sources"]) == result["num_events_found"]
        assert all("title" in s and "url" in s for s in result["sources"])

    def test_sources_are_deduplicated_by_event(self, small_vectorstore):
        llm = FakeListChatModel(responses=["Réponse"])
        result = answer_question(small_vectorstore, llm, "concert", k=3)
        uids = [s["uid"] for s in result["sources"]]
        assert len(uids) == len(set(uids))

    def test_llm_receives_context_in_prompt(self, small_vectorstore, mocker):
        """Vérifie que le LLM reçoit bien le contexte récupéré (et pas
        seulement la question brute) — c'est le principe même du RAG."""
        llm = FakeListChatModel(responses=["Réponse"])
        spy = mocker.spy(FakeListChatModel, "invoke")

        answer_question(small_vectorstore, llm, "concert de jazz", k=2)

        assert spy.call_count == 1
        messages_sent = spy.call_args[0][1]
        assert len(messages_sent) == 2
        human_content = messages_sent[1].content
        assert "concert de jazz" in human_content
        assert "Sunset" in human_content


class TestNoResultsHandling:
    def test_no_documents_found_skips_llm_call(self, small_vectorstore, mocker):
        """Si la recherche ne retourne rien, on ne doit pas appeler le LLM
        et renvoyer un message explicite plutôt qu'une réponse inventée."""
        mocker.patch("rag_chain.retrieve_events", return_value=[])
        llm = FakeListChatModel(responses=[])
        spy = mocker.spy(FakeListChatModel, "invoke")

        result = answer_question(small_vectorstore, llm, "concert de jazz", k=2)

        assert spy.call_count == 0
        assert result["num_events_found"] == 0
        assert result["sources"] == []
        assert "aucun événement" in result["answer"].lower()