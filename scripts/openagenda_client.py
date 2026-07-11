"""
Client léger pour l'API OpenAgenda v2.

Documentation officielle : https://developers.openagenda.com/
- GET /v2/agendas                      -> recherche d'agendas
- GET /v2/agendas/{agendaUid}/events    -> événements d'un agenda

Note importante :
L'ancien export JSON (paramètres oaq[...]) est déprécié et retiré par OpenAgenda.
Ce client utilise exclusivement l'API v2 "officielle".

Authentification : une clé publique est nécessaire pour toute lecture.
Elle se crée gratuitement sur https://openagenda.com (espace développeur),
puis se transmet via l'en-tête HTTP "key".
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://api.openagenda.com/v2"


class OpenAgendaError(Exception):
    """Erreur levée lors d'un échec d'appel à l'API OpenAgenda."""


class OpenAgendaClient:
    def __init__(self, api_key: str, timeout: int = 15):
        if not api_key:
            raise ValueError(
                "Une clé API OpenAgenda est requise. "
                "Créez-en une sur https://openagenda.com puis renseignez-la "
                "dans la variable d'environnement OPENAGENDA_API_KEY."
            )
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"key": self.api_key})

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code == 429:
            # Trop de requêtes : on laisse tenacity retenter après backoff
            raise OpenAgendaError("Rate limit atteint (429)")
        if not response.ok:
            raise OpenAgendaError(
                f"Erreur API OpenAgenda ({response.status_code}) sur {url}: {response.text[:300]}"
            )
        return response.json()

    def search_agendas(
        self,
        search: str,
        official_only: bool = True,
        max_agendas: int = 20,
    ) -> list[dict[str, Any]]:
        """Recherche des agendas correspondant à un mot-clé (ex: 'Paris', 'Île-de-France')."""
        agendas: list[dict[str, Any]] = []
        params: dict[str, Any] = {"search": search, "size": min(max_agendas, 100)}
        if official_only:
            params["official"] = 1

        data = self._get("/agendas", params=params)
        agendas.extend(data.get("agendas", []))
        return agendas[:max_agendas]

    def get_events(
        self,
        agenda_uid: int,
        timings_gte: str | None = None,
        timings_lte: str | None = None,
        detailed: bool = True,
        page_size: int = 100,
        max_events: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Génère les événements d'un agenda, avec pagination automatique (curseur `after`).

        timings_gte / timings_lte : bornes ISO8601 (ex: "2025-07-04T00:00:00.000Z")
        """
        params: dict[str, Any] = {"size": page_size}
        if detailed:
            params["detailed"] = 1
        if timings_gte:
            params["timings[gte]"] = timings_gte
        if timings_lte:
            params["timings[lte]"] = timings_lte

        fetched = 0
        after_cursor = None
        while True:
            call_params = dict(params)
            if after_cursor:
                call_params["after"] = after_cursor

            data = self._get(f"/agendas/{agenda_uid}/events", params=call_params)
            events = data.get("events", [])
            if not events:
                break

            for event in events:
                yield event
                fetched += 1
                if max_events and fetched >= max_events:
                    return

            after_cursor = data.get("after")
            if not after_cursor:
                break
            time.sleep(0.2)