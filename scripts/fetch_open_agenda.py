"""
Récupère les événements culturels d'Île-de-France via l'API OpenAgenda,
sur une fenêtre temporelle de 12 mois glissants (passés) + événements à venir.

Usage :
    python scripts/fetch_open_agenda.py

Variables d'environnement (voir .env.example) :
    OPENAGENDA_API_KEY   Clé publique OpenAgenda (obligatoire)

Sortie :
    data/raw/events_raw.json   Liste brute des événements récupérés (non filtrés/non nettoyés)

Stratégie de ciblage géographique :
    L'API OpenAgenda v2 ne propose pas de filtre géographique direct sur les
    événements. On recherche donc des agendas pertinents pour l'Île-de-France
    (mots-clés) via /v2/agendas, puis on filtre les événements récupérés sur
    leur localisation (région ou département) au moment du nettoyage
    (voir scripts/clean_data.py).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent))
from openagenda_client import OpenAgendaClient, OpenAgendaError  # noqa: E402

# Mots-clés d'agendas ciblant l'Île-de-France. Ajustables selon les besoins métier.
SEARCH_KEYWORDS = [
    "Paris",
    "Île-de-France",
    "Hauts-de-Seine",
    "Seine-Saint-Denis",
    "Val-de-Marne",
]

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_RAW_DIR / "events_raw.json"

MAX_AGENDAS_PER_KEYWORD = 10
MAX_EVENTS_PER_AGENDA = 300


def iso_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAGENDA_API_KEY")

    if not api_key:
        print(
            "ERREUR: OPENAGENDA_API_KEY n'est pas définie.\n"
            "1. Créez un compte sur https://openagenda.com\n"
            "2. Récupérez votre clé publique dans votre espace développeur\n"
            "3. Ajoutez-la dans le fichier .env : OPENAGENDA_API_KEY=votre_cle\n",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAgendaClient(api_key=api_key)

    now = datetime.now(timezone.utc)
    one_year_ago = now - timedelta(days=365)
    timings_gte = iso_utc(one_year_ago)

    print(f"Fenêtre temporelle : depuis {timings_gte} (12 derniers mois + à venir)")

    # 1. Identifier les agendas pertinents
    agenda_uids: dict[int, str] = {}
    for keyword in SEARCH_KEYWORDS:
        try:
            agendas = client.search_agendas(keyword, official_only=True, max_agendas=MAX_AGENDAS_PER_KEYWORD)
        except OpenAgendaError as exc:
            print(f"[AVERTISSEMENT] Recherche d'agendas échouée pour '{keyword}': {exc}", file=sys.stderr)
            continue
        for agenda in agendas:
            agenda_uids[agenda["uid"]] = agenda.get("title", "")
        print(f"  '{keyword}' -> {len(agendas)} agendas trouvés")

    if not agenda_uids:
        print("ERREUR: Aucun agenda trouvé. Vérifiez la clé API et les mots-clés.", file=sys.stderr)
        sys.exit(1)

    print(f"Total agendas uniques à interroger : {len(agenda_uids)}")

    # 2. Récupérer les événements de chaque agenda
    all_events: list[dict] = []
    for uid, title in agenda_uids.items():
        try:
            events = list(
                client.get_events(
                    agenda_uid=uid,
                    timings_gte=timings_gte,
                    detailed=True,
                    max_events=MAX_EVENTS_PER_AGENDA,
                )
            )
        except OpenAgendaError as exc:
            print(f"[AVERTISSEMENT] Événements ignorés pour l'agenda {uid} ({title}): {exc}", file=sys.stderr)
            continue

        for event in events:
            event["_source_agenda_uid"] = uid
            event["_source_agenda_title"] = title
        all_events.extend(events)
        print(f"  Agenda {uid} ({title}): {len(events)} événements")

    # 3. Sauvegarde brute
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)

    print(f"\n{len(all_events)} événements bruts sauvegardés dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()