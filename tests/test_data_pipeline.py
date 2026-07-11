"""
Tests unitaires de l'étape 2 : récupération et nettoyage des données OpenAgenda.

Lancer avec :
    pytest tests/test_data_pipeline.py -v

Ces tests ne font AUCUN appel réseau réel : les réponses de l'API OpenAgenda
sont simulées (mock) pour garantir des tests rapides, déterministes et
exécutables sans clé API.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from clean_data import (  # noqa: E402
    clean_event,
    clean_events,
    extract_dates,
    extract_multilingual,
    is_in_ile_de_france,
    is_relevant_agenda,
    strip_html,
)
from openagenda_client import OpenAgendaClient, OpenAgendaError  # noqa: E402


def make_raw_event(**overrides):
    base = {
        "uid": 1001,
        "title": {"fr": "Exposition Monet"},
        "description": {"fr": "Une belle exposition sur Monet."},
        "longDescription": {"fr": "<p>Une <b>belle</b> exposition sur Claude Monet et l'impressionnisme.</p>"},
        "location": {
            "city": "Paris",
            "address": "1 rue de Rivoli",
            "postalCode": "75001",
            "department": "Paris",
            "region": "Île-de-France",
            "latitude": 48.86,
            "longitude": 2.34,
        },
        "timings": [
            {"begin": "2026-08-01T10:00:00+0200", "end": "2026-08-01T18:00:00+0200"},
            {"begin": "2026-08-02T10:00:00+0200", "end": "2026-08-02T18:00:00+0200"},
        ],
        "keywords": {"fr": ["exposition", "peinture"]},
        "_source_agenda_uid": 5555,
        "_source_agenda_title": "Culture Paris",
    }
    base.update(overrides)
    return base


class TestStripHtml:
    def test_removes_tags(self):
        assert strip_html("<p>Bonjour <b>monde</b></p>") == "Bonjour monde"

    def test_collapses_whitespace(self):
        assert strip_html("Bonjour   \n\n  monde") == "Bonjour monde"

    def test_empty_string(self):
        assert strip_html("") == ""


class TestExtractMultilingual:
    def test_prefers_french(self):
        assert extract_multilingual({"en": "Hello", "fr": "Bonjour"}) == "Bonjour"

    def test_falls_back_to_other_language(self):
        assert extract_multilingual({"en": "Hello"}) == "Hello"

    def test_handles_plain_string(self):
        assert extract_multilingual("Déjà une chaîne") == "Déjà une chaîne"

    def test_handles_none(self):
        assert extract_multilingual(None) == ""

    def test_handles_empty_dict(self):
        assert extract_multilingual({}) == ""


class TestIsInIleDeFrance:
    def test_region_match(self):
        assert is_in_ile_de_france({"region": "Île-de-France"}) is True

    def test_department_match(self):
        assert is_in_ile_de_france({"department": "Hauts-de-Seine"}) is True

    def test_postal_code_match(self):
        assert is_in_ile_de_france({"postalCode": "94200"}) is True

    def test_outside_idf(self):
        assert is_in_ile_de_france({"region": "Bretagne", "postalCode": "35000"}) is False

    def test_none_location(self):
        assert is_in_ile_de_france(None) is False

    def test_empty_location(self):
        assert is_in_ile_de_france({}) is False


class TestIsRelevantAgenda:
    def test_diocese_is_excluded(self):
        assert is_relevant_agenda("Diocèse de Paris") is False

    def test_case_insensitive(self):
        assert is_relevant_agenda("DIOCÈSE DE PARIS") is False

    def test_info_jeunes_is_excluded(self):
        assert is_relevant_agenda("Info Jeunes Hauts-de-Seine") is False

    def test_colos_apprenantes_is_excluded(self):
        assert is_relevant_agenda("Offre de \"Colos apprenantes\" - Val-de-Marne") is False

    def test_cultural_agenda_is_kept(self):
        assert is_relevant_agenda("FICEP - Forum des Instituts Culturels Étrangers à Paris") is True

    def test_none_title_is_kept(self):
        assert is_relevant_agenda(None) is True

    def test_empty_title_is_kept(self):
        assert is_relevant_agenda("") is True


class TestExtractDates:
    def test_normal_case(self):
        timings = [
            {"begin": "2026-08-01T10:00:00+0200", "end": "2026-08-01T18:00:00+0200"},
            {"begin": "2026-08-02T10:00:00+0200", "end": "2026-08-02T18:00:00+0200"},
        ]
        start, end = extract_dates(timings)
        assert start == "2026-08-01T10:00:00+0200"
        assert end == "2026-08-02T18:00:00+0200"

    def test_empty_timings(self):
        assert extract_dates([]) == (None, None)

    def test_none_timings(self):
        assert extract_dates(None) == (None, None)

    def test_missing_end_falls_back_to_begin(self):
        timings = [{"begin": "2026-08-01T10:00:00+0200"}]
        start, end = extract_dates(timings)
        assert start == "2026-08-01T10:00:00+0200"
        assert end == "2026-08-01T10:00:00+0200"


class TestCleanEvent:
    def test_valid_event_is_cleaned_correctly(self):
        raw = make_raw_event()
        cleaned = clean_event(raw)
        assert cleaned is not None
        assert cleaned["uid"] == 1001
        assert cleaned["title"] == "Exposition Monet"
        assert "Monet" in cleaned["description_full"]
        assert "<p>" not in cleaned["description_full"]
        assert cleaned["city"] == "Paris"
        assert cleaned["date_start"] == "2026-08-01T10:00:00+0200"

    def test_event_without_title_is_rejected(self):
        raw = make_raw_event(title={})
        assert clean_event(raw) is None

    def test_event_without_description_is_rejected(self):
        raw = make_raw_event(description={}, longDescription={})
        assert clean_event(raw) is None

    def test_event_outside_idf_is_rejected(self):
        raw = make_raw_event(location={
            "city": "Lyon", "postalCode": "69000", "department": "Rhône", "region": "Auvergne-Rhône-Alpes",
        })
        assert clean_event(raw) is None

    def test_event_without_uid_is_rejected(self):
        raw = make_raw_event(uid=None)
        assert clean_event(raw) is None

    def test_event_without_location_is_rejected(self):
        raw = make_raw_event(location=None)
        assert clean_event(raw) is None

    def test_event_from_irrelevant_agenda_is_rejected(self):
        raw = make_raw_event(_source_agenda_title="Diocèse de Paris")
        assert clean_event(raw) is None


class TestCleanEvents:
    def test_deduplicates_by_uid(self):
        raw_events = [make_raw_event(uid=1), make_raw_event(uid=1), make_raw_event(uid=2)]
        cleaned = clean_events(raw_events)
        uids = [e["uid"] for e in cleaned]
        assert sorted(uids) == [1, 2]

    def test_filters_out_invalid_events(self):
        raw_events = [
            make_raw_event(uid=1),
            make_raw_event(uid=2, title={}),  # invalide : pas de titre
            make_raw_event(uid=3, location={"region": "Bretagne", "postalCode": "35000"}),  # hors IDF
        ]
        cleaned = clean_events(raw_events)
        assert len(cleaned) == 1
        assert cleaned[0]["uid"] == 1

    def test_empty_input(self):
        assert clean_events([]) == []


class TestOpenAgendaClient:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            OpenAgendaClient(api_key="")

    def test_search_agendas_returns_list(self, mocker):
        client = OpenAgendaClient(api_key="fake-key")
        mock_response = mocker.Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agendas": [{"uid": 111, "title": "Culture Paris"}],
            "total": 1,
        }
        mocker.patch.object(client.session, "get", return_value=mock_response)

        agendas = client.search_agendas("Paris")
        assert len(agendas) == 1
        assert agendas[0]["uid"] == 111

    def test_get_events_paginates_with_after_cursor(self, mocker):
        client = OpenAgendaClient(api_key="fake-key")

        page_1 = mocker.Mock(ok=True, status_code=200)
        page_1.json.return_value = {
            "events": [{"uid": 1}, {"uid": 2}],
            "after": [2],
        }
        page_2 = mocker.Mock(ok=True, status_code=200)
        page_2.json.return_value = {
            "events": [{"uid": 3}],
            "after": None,
        }
        mocker.patch.object(client.session, "get", side_effect=[page_1, page_2])

        events = list(client.get_events(agenda_uid=5555))
        assert [e["uid"] for e in events] == [1, 2, 3]

    def test_get_events_respects_max_events(self, mocker):
        client = OpenAgendaClient(api_key="fake-key")
        page = mocker.Mock(ok=True, status_code=200)
        page.json.return_value = {
            "events": [{"uid": i} for i in range(10)],
            "after": [9],
        }
        mocker.patch.object(client.session, "get", return_value=page)

        events = list(client.get_events(agenda_uid=5555, max_events=3))
        assert len(events) == 3

    def test_api_error_raises_openagenda_error(self, mocker):
        client = OpenAgendaClient(api_key="fake-key")
        mock_response = mocker.Mock(ok=False, status_code=500, text="Internal error")
        mocker.patch.object(client.session, "get", return_value=mock_response)

        with pytest.raises(OpenAgendaError):
            client.search_agendas("Paris")