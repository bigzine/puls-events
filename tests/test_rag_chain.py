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
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from build_index import build_faiss_index, chunk_events  # noqa: E402
from rag_chain import (  # noqa: E402
    EventChatbotError,
    answer_question,
    build_system_prompt,
    current_date_str,
    deduplicate_sources,
    filter_future_events,
    format_context,
    parse_event_datetime,
    retrieve_events,
    this_weekend_str,
)

EMBEDDING_DIM = 64


class _FrozenDateTime(datetime):
    """Sous-classe de datetime dont .now() renvoie une date figée, pour
    tester le calcul du week-end sans dépendre de la date réelle
    d'exécution des tests. Hérite de datetime, donc fromisoformat() et le
    reste de l'API restent intacts pour les autres fonctions du module."""
    _frozen_now: datetime

    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return cls._frozen_now.replace(tzinfo=tz)
        return cls._frozen_now


@contextmanager
def freeze_time_weekday(dt: datetime):
    _FrozenDateTime._frozen_now = dt
    with mock.patch("rag_chain.datetime", _FrozenDateTime):
        yield


def iso_in_days(days: int) -> str:
    """Date ISO8601 relative à maintenant, pour des fixtures de test qui
    restent valides (futures ou passées) quelle que soit la date d'exécution
    des tests — contrairement à une date codée en dur qui finirait par
    devenir "passée" avec le temps et fausser silencieusement les tests."""
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def make_event(**overrides):
    base = {
        "uid": 1,
        "title": "Concert de jazz",
        "description_full": "Un concert de jazz au Sunset, avec un trio local reconnu.",
        "city": "Paris",
        "address": "60 rue des Lombards",
        "postal_code": "75001",
        "date_start": iso_in_days(30),
        "date_end": iso_in_days(30),
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


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestDateAwareness:
    """Vérifie que le LLM reçoit bien la date du jour, pour éviter qu'il
    n'hallucine une date incorrecte lors de l'interprétation d'expressions
    temporelles relatives ('ce week-end', 'demain'...)."""

    def test_current_date_str_matches_today(self):
        import datetime as dt
        today = dt.datetime.now()
        date_str = current_date_str()
        assert str(today.year) in date_str
        assert str(today.day) in date_str

    def test_system_prompt_includes_current_date(self):
        prompt = build_system_prompt()
        assert current_date_str() in prompt

    def test_llm_receives_current_date_in_system_message(self, small_vectorstore, mocker):
        llm = FakeListChatModel(responses=["Réponse"])
        spy = mocker.spy(FakeListChatModel, "invoke")

        answer_question(small_vectorstore, llm, "un concert ce week-end", k=2)

        messages_sent = spy.call_args[0][1]
        system_content = messages_sent[0].content
        assert current_date_str() in system_content


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


# ---------------------------------------------------------------------------
# deduplicate_sources
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# retrieve_events
# ---------------------------------------------------------------------------

class TestRetrieveEvents:
    def test_returns_documents(self, small_vectorstore):
        docs = retrieve_events(small_vectorstore, "concert de jazz", k=2)
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_respects_k(self, small_vectorstore):
        docs = retrieve_events(small_vectorstore, "spectacle", k=1)
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# answer_question (pipeline complet, LLM factice)
# ---------------------------------------------------------------------------

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
        assert len(uids) == len(set(uids))  # pas de doublon d'événement

    def test_llm_receives_context_in_prompt(self, small_vectorstore, mocker):
        """Vérifie que le LLM reçoit bien le contexte récupéré (et pas
        seulement la question brute) — c'est le principe même du RAG."""
        llm = FakeListChatModel(responses=["Réponse"])
        # Pydantic empêche le monkeypatch direct sur l'instance : on patch la classe.
        spy = mocker.spy(FakeListChatModel, "invoke")

        answer_question(small_vectorstore, llm, "concert de jazz", k=2)

        assert spy.call_count == 1
        messages_sent = spy.call_args[0][1]  # [0] = self, [1] = messages
        assert len(messages_sent) == 2  # SystemMessage + HumanMessage
        human_content = messages_sent[1].content
        assert "concert de jazz" in human_content  # la question est bien incluse
        assert "Sunset" in human_content  # le contexte récupéré (description de l'événement) est bien inclus


class TestNoResultsHandling:
    def test_no_documents_found_skips_llm_call(self, small_vectorstore, mocker):
        """Si la recherche ne retourne rien, on ne doit pas appeler le LLM
        et renvoyer un message explicite plutôt qu'une réponse inventée."""
        mocker.patch("rag_chain.retrieve_events", return_value=[])
        llm = FakeListChatModel(responses=[])
        spy = mocker.spy(FakeListChatModel, "invoke")

        result = answer_question(small_vectorstore, llm, "concert de jazz", k=2)

        assert spy.call_count == 0  # le LLM ne doit jamais être appelé sans contexte
        assert result["num_events_found"] == 0
        assert result["sources"] == []
        assert "aucun événement" in result["answer"].lower()


class TestDateFiltering:
    """Tests du filtre d'événements passés — corrige un bug constaté en
    conditions réelles : le LLM recommandait des événements déjà terminés
    (ex. un concert du 6 juin recommandé alors qu'on était le 11 juillet),
    ou faisait des erreurs de calcul de dates ("se termine demain" pour un
    événement en réalité terminé depuis une semaine)."""

    def test_parse_event_datetime_handles_z_suffix(self):
        assert parse_event_datetime("2026-03-08T17:00:00Z") is not None

    def test_parse_event_datetime_handles_explicit_offset(self):
        assert parse_event_datetime("2026-08-01T20:00:00+02:00") is not None

    def test_parse_event_datetime_handles_none(self):
        assert parse_event_datetime(None) is None

    def test_parse_event_datetime_handles_garbage(self):
        assert parse_event_datetime("pas une date") is None

    def test_filter_future_events_keeps_future(self):
        doc = Document(page_content="x", metadata={"date_end": iso_in_days(10)})
        assert filter_future_events([doc]) == [doc]

    def test_filter_future_events_excludes_past(self):
        doc = Document(page_content="x", metadata={"date_end": iso_in_days(-10)})
        assert filter_future_events([doc]) == []

    def test_filter_future_events_keeps_ongoing_event(self):
        """Un événement commencé il y a quelques jours mais qui se termine
        dans le futur (ex. une exposition sur plusieurs semaines) doit être
        conservé : seule la date de FIN compte, pas la date de début."""
        doc = Document(page_content="x", metadata={
            "date_start": iso_in_days(-5),
            "date_end": iso_in_days(5),
        })
        assert filter_future_events([doc]) == [doc]

    def test_filter_future_events_keeps_events_without_date(self):
        """Un événement sans date exploitable est conservé par prudence
        plutôt qu'écarté silencieusement (mieux vaut le laisser au LLM que
        de perdre un résultat potentiellement pertinent)."""
        doc = Document(page_content="x", metadata={})
        assert filter_future_events([doc]) == [doc]

    def test_filter_future_events_respects_reference_time(self):
        """Le paramètre reference_time permet de tester le filtre de façon
        déterministe, sans dépendre de la date réelle d'exécution des tests."""
        fixed_now = datetime(2026, 7, 11, tzinfo=timezone.utc)
        past_doc = Document(page_content="x", metadata={"date_end": "2026-06-06T20:00:00+02:00"})
        future_doc = Document(page_content="x", metadata={"date_end": "2026-08-01T20:00:00+02:00"})

        result = filter_future_events([past_doc, future_doc], reference_time=fixed_now)
        assert result == [future_doc]

    def test_answer_question_never_recommends_a_past_event(self, fake_embeddings):
        """Reproduit le bug observé : un événement passé et un événement
        futur remontent tous deux dans le top-k sémantique ; seul le futur
        doit atteindre le LLM."""
        past_event = make_event(uid=1, title="Concert passé", date_end=iso_in_days(-30))
        future_event = make_event(uid=2, title="Concert futur", date_end=iso_in_days(30))
        docs = chunk_events([past_event, future_event])
        vectorstore = build_faiss_index(docs, fake_embeddings, batch_size=64)

        llm = FakeListChatModel(responses=["Réponse"])
        result = answer_question(vectorstore, llm, "concert", k=4)

        uids_in_sources = [s["uid"] for s in result["sources"]]
        assert 1 not in uids_in_sources
        assert 2 in uids_in_sources

    def test_answer_question_backfills_when_top_k_is_mostly_past(self, fake_embeddings):
        """Reproduit précisément le bug signalé : si les meilleurs résultats
        sémantiques bruts sont majoritairement des événements passés, la
        réponse ne doit PAS être "aucun événement trouvé" tant qu'un
        événement futur pertinent existe dans l'index — même s'il est moins
        bien classé sémantiquement que les événements passés."""
        events = [make_event(uid=i, title=f"Concert {i}", date_end=iso_in_days(-30)) for i in range(1, 6)]
        events.append(make_event(uid=99, title="Concert 99", date_end=iso_in_days(30)))
        docs = chunk_events(events)
        vectorstore = build_faiss_index(docs, fake_embeddings, batch_size=64)

        llm = FakeListChatModel(responses=["Réponse"])
        result = answer_question(vectorstore, llm, "concert", k=4)

        assert result["num_events_found"] >= 1
        assert 99 in [s["uid"] for s in result["sources"]]


class TestFormatContextOngoingEventAnnotation:
    """Vérifie que les événements récurrents/longue durée (date de début
    passée mais événement toujours d'actualité) sont explicitement annotés
    dans le contexte, pour que le LLM ne les écarte pas à tort comme
    'terminés' — bug constaté en conditions réelles avec des concerts d'une
    saison d'orchestre récurrente."""

    def test_past_start_future_end_is_annotated(self):
        doc = Document(
            page_content="Concert récurrent",
            metadata={
                "title": "Concert - Nous sommes le Vent",
                "date_start": iso_in_days(-60),
                "date_end": iso_in_days(10),
            },
        )
        context = format_context([doc])
        assert "ATTENTION" in context
        assert "PAS" in context  # "Ne le considère PAS comme terminé"

    def test_normal_future_event_is_not_annotated(self):
        doc = Document(
            page_content="Concert classique",
            metadata={
                "title": "Concert normal",
                "date_start": iso_in_days(5),
                "date_end": iso_in_days(5),
            },
        )
        context = format_context([doc])
        assert "ATTENTION" not in context

    def test_missing_date_start_is_not_annotated(self):
        doc = Document(page_content="x", metadata={"title": "Sans date"})
        context = format_context([doc])
        assert "ATTENTION" not in context


class TestFetchKMargin:
    """Reproduit le scénario réel constaté : un thème où les événements
    passés sont très majoritaires (ex. 20 concerts de jazz passés pour
    seulement 2 à venir). Une marge de récupération trop faible masquerait
    les événements futurs pertinents ; il faut une marge suffisamment large."""

    def test_finds_future_event_even_when_vastly_outnumbered_by_past_ones(self, fake_embeddings):
        past_events = [
            make_event(uid=i, title=f"Concert de jazz passé {i}", date_end=iso_in_days(-30 - i))
            for i in range(1, 21)
        ]
        future_events = [
            make_event(uid=100, title="Concert de jazz à venir A", date_end=iso_in_days(5)),
            make_event(uid=101, title="Concert de jazz à venir B", date_end=iso_in_days(10)),
        ]
        docs = chunk_events(past_events + future_events)
        vectorstore = build_faiss_index(docs, fake_embeddings, batch_size=64)

        llm = FakeListChatModel(responses=["Réponse"])
        result = answer_question(vectorstore, llm, "concert de jazz", k=4)

        found_uids = {s["uid"] for s in result["sources"]}
        assert found_uids & {100, 101}, (
            "Aucun des événements futurs n'a été trouvé malgré une marge de "
            "récupération censée compenser la majorité d'événements passés."
        )


class TestWeekendCalculation:
    """Reproduit le bug constaté en conditions réelles : le LLM a annoncé
    'ce week-end (20 et 21 juillet 2026)' un samedi 18 juillet — alors que
    le 20 et le 21 juillet 2026 tombent un lundi et un mardi. Les dates du
    week-end sont désormais calculées en Python, pas confiées au LLM."""

    def test_saturday_returns_today_and_tomorrow(self):
        with freeze_time_weekday(datetime(2026, 7, 18)):  # samedi
            assert this_weekend_str() == "samedi 18 juillet 2026 et dimanche 19 juillet 2026"

    def test_sunday_returns_yesterday_and_today(self):
        with freeze_time_weekday(datetime(2026, 7, 19)):  # dimanche
            assert this_weekend_str() == "samedi 18 juillet 2026 et dimanche 19 juillet 2026"

    def test_weekday_returns_upcoming_weekend(self):
        with freeze_time_weekday(datetime(2026, 7, 16)):  # jeudi
            assert this_weekend_str() == "samedi 18 juillet 2026 et dimanche 19 juillet 2026"

    def test_monday_returns_next_weekend_not_the_one_just_passed(self):
        with freeze_time_weekday(datetime(2026, 7, 20)):  # lundi
            assert this_weekend_str() == "samedi 25 juillet 2026 et dimanche 26 juillet 2026"

    def test_weekend_dates_are_injected_in_system_prompt(self):
        with freeze_time_weekday(datetime(2026, 7, 18)):
            prompt = build_system_prompt()
            assert this_weekend_str() in prompt


class TestDistanceThresholdFiltering:
    """Filtre de seuil de distance (inspiré de l'exercice OpenClassrooms
    "Mettez en place un RAG pour un LLM") : réduit le bruit dans le contexte
    récupéré en écartant les résultats trop éloignés sémantiquement, même
    s'ils font partie des k plus proches voisins."""

    def test_no_threshold_preserves_default_behavior(self, small_vectorstore):
        with_threshold = retrieve_events(small_vectorstore, "concert de jazz", k=3, max_distance=None)
        without_param = retrieve_events(small_vectorstore, "concert de jazz", k=3)
        assert len(with_threshold) == len(without_param) == 3

    def test_exact_match_survives_a_strict_threshold(self, small_vectorstore, fake_embeddings):
        docs = small_vectorstore.similarity_search("concert de jazz au Sunset", k=1)
        exact_text = docs[0].page_content

        results = retrieve_events(small_vectorstore, exact_text, k=3, max_distance=1e-6)
        assert len(results) >= 1
        assert results[0].page_content == exact_text

    def test_very_strict_threshold_can_return_empty(self, small_vectorstore):
        """Un seuil extrêmement strict peut légitimement ne rien retourner :
        la fonction doit gérer ce cas sans planter (pas d'IndexError etc)."""
        results = retrieve_events(small_vectorstore, "une requête sans rapport", k=3, max_distance=-1.0)
        assert results == []

    def test_answer_question_respects_env_threshold(self, small_vectorstore, monkeypatch, mocker):
        """Vérifie que MAX_RETRIEVAL_DISTANCE, lu depuis l'environnement,
        est bien transmis jusqu'à retrieve_events."""
        monkeypatch.setenv("MAX_RETRIEVAL_DISTANCE", "0.5")
        spy = mocker.spy(sys.modules["rag_chain"], "retrieve_events")

        llm = FakeListChatModel(responses=["Réponse"])
        answer_question(small_vectorstore, llm, "concert de jazz", k=2)

        assert spy.call_count == 1
        _, kwargs = spy.call_args
        assert kwargs.get("max_distance") == 0.5

    def test_answer_question_defaults_to_no_threshold_when_env_unset(self, small_vectorstore, monkeypatch, mocker):
        monkeypatch.delenv("MAX_RETRIEVAL_DISTANCE", raising=False)
        spy = mocker.spy(sys.modules["rag_chain"], "retrieve_events")

        llm = FakeListChatModel(responses=["Réponse"])
        answer_question(small_vectorstore, llm, "concert de jazz", k=2)

        _, kwargs = spy.call_args
        assert kwargs.get("max_distance") is None