"""
Tests unitaires de l'étape 3 : chunking et indexation vectorielle FAISS.

Lancer avec :
    pytest tests/test_indexing.py -v

Ces tests utilisent DeterministicFakeEmbedding (LangChain) : un embedding
factice qui génère un vecteur reproductible à partir du hash du texte.
Aucun appel réseau, aucune clé API requise. Cela permet de valider la
MÉCANIQUE du pipeline (chunking, indexation, métadonnées, sauvegarde/
rechargement) indépendamment de la qualité sémantique réelle des vecteurs,
qui elle est évaluée à l'étape 5 (jeu de test annoté) avec les vrais
embeddings Mistral/HuggingFace.
"""

import sys
from pathlib import Path

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from build_index import build_faiss_index, chunk_events, event_to_text  # noqa: E402

EMBEDDING_DIM = 64


def make_event(**overrides):
    base = {
        "uid": 1,
        "title": "Exposition Monet",
        "description_short": "Une belle exposition.",
        "description_full": "Une belle exposition sur Claude Monet et l'impressionnisme à Paris.",
        "city": "Paris",
        "address": "1 rue de Rivoli",
        "postal_code": "75001",
        "date_start": "2026-08-01T10:00:00+02:00",
        "date_end": "2026-08-02T18:00:00+02:00",
        "keywords": ["exposition", "peinture"],
        "url": "https://openagenda.com/1/events/1",
        "source_agenda_title": "Culture Paris",
    }
    base.update(overrides)
    return base


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=EMBEDDING_DIM)


class TestEventToText:
    def test_includes_title_city_and_description(self):
        event = make_event()
        text = event_to_text(event)
        assert "Exposition Monet" in text
        assert "Paris" in text
        assert "impressionnisme" in text

    def test_includes_keywords(self):
        event = make_event(keywords=["jazz", "concert"])
        text = event_to_text(event)
        assert "jazz" in text
        assert "concert" in text

    def test_handles_missing_optional_fields(self):
        event = make_event(city=None, keywords=[])
        text = event_to_text(event)
        assert "Exposition Monet" in text


class TestChunkEvents:
    def test_short_event_produces_one_chunk(self):
        events = [make_event()]
        docs = chunk_events(events, chunk_size=800, chunk_overlap=100)
        assert len(docs) == 1
        assert docs[0].metadata["uid"] == 1
        assert docs[0].metadata["chunk_index"] == 0
        assert docs[0].metadata["chunk_count"] == 1

    def test_long_event_produces_multiple_chunks(self):
        long_description = "Une phrase répétée sur l'art contemporain. " * 100
        events = [make_event(description_full=long_description)]
        docs = chunk_events(events, chunk_size=200, chunk_overlap=20)
        assert len(docs) > 1
        assert all(d.metadata["uid"] == 1 for d in docs)
        assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
        assert all(d.metadata["chunk_count"] == len(docs) for d in docs)

    def test_metadata_is_preserved_on_each_chunk(self):
        events = [make_event(city="Nanterre", url="https://example.com/42")]
        docs = chunk_events(events)
        assert docs[0].metadata["city"] == "Nanterre"
        assert docs[0].metadata["url"] == "https://example.com/42"

    def test_multiple_events_produce_independent_chunks(self):
        events = [make_event(uid=1, title="Événement A"), make_event(uid=2, title="Événement B")]
        docs = chunk_events(events)
        uids = {d.metadata["uid"] for d in docs}
        assert uids == {1, 2}

    def test_event_without_exploitable_text_is_skipped(self):
        events = [make_event(title="", description_full="", city=None, keywords=[])]
        docs = chunk_events(events)
        assert docs == []

    def test_empty_events_list(self):
        assert chunk_events([]) == []


class TestBuildFaissIndex:
    def test_raises_on_empty_documents(self, fake_embeddings):
        with pytest.raises(ValueError):
            build_faiss_index([], fake_embeddings)

    def test_indexes_all_documents(self, fake_embeddings):
        events = [make_event(uid=i, title=f"Événement {i}") for i in range(5)]
        docs = chunk_events(events)
        vectorstore = build_faiss_index(docs, fake_embeddings, batch_size=64)
        assert vectorstore.index.ntotal == len(docs)

    def test_batching_produces_same_total_as_single_batch(self, fake_embeddings):
        events = [make_event(uid=i, title=f"Événement {i}") for i in range(10)]
        docs = chunk_events(events)

        vs_single_batch = build_faiss_index(docs, fake_embeddings, batch_size=100)
        vs_multi_batch = build_faiss_index(docs, fake_embeddings, batch_size=3)

        assert vs_single_batch.index.ntotal == vs_multi_batch.index.ntotal == len(docs)

    def test_exact_text_query_retrieves_matching_document(self, fake_embeddings):
        """Avec un embedding déterministe, interroger EXACTEMENT le texte
        d'un chunk doit le renvoyer en tout premier résultat (distance ~0),
        ce qui valide que l'indexation et la correspondance métadonnées
        fonctionnent correctement de bout en bout."""
        events = [
            make_event(uid=1, title="Concert de jazz", description_full="Un concert de jazz au Sunset."),
            make_event(uid=2, title="Exposition photo", description_full="Une exposition de photographie."),
        ]
        docs = chunk_events(events)
        vectorstore = build_faiss_index(docs, fake_embeddings, batch_size=64)

        target_text = docs[0].page_content
        results = vectorstore.similarity_search_with_score(target_text, k=1)
        assert len(results) == 1
        top_doc, score = results[0]
        assert top_doc.metadata["uid"] == 1
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_metadata_is_retrievable_after_search(self, fake_embeddings):
        events = [make_event(uid=7, title="Ballet", city="Créteil", url="https://example.com/7")]
        docs = chunk_events(events)
        vectorstore = build_faiss_index(docs, fake_embeddings, batch_size=64)

        results = vectorstore.similarity_search(docs[0].page_content, k=1)
        assert results[0].metadata["city"] == "Créteil"
        assert results[0].metadata["url"] == "https://example.com/7"

    def test_save_and_reload_preserves_search_results(self, fake_embeddings, tmp_path):
        events = [make_event(uid=1, title="Concert de jazz", description_full="Un concert de jazz au Sunset.")]
        docs = chunk_events(events)
        vectorstore = build_faiss_index(docs, fake_embeddings, batch_size=64)

        save_path = tmp_path / "faiss_test_index"
        vectorstore.save_local(str(save_path))

        reloaded = FAISS.load_local(
            str(save_path), fake_embeddings, allow_dangerous_deserialization=True
        )
        assert reloaded.index.ntotal == vectorstore.index.ntotal

        results = reloaded.similarity_search(docs[0].page_content, k=1)
        assert results[0].metadata["uid"] == 1