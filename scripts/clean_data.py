"""
Nettoie, filtre (Île-de-France) et structure les événements bruts OpenAgenda
en un jeu de données prêt pour la vectorisation.

Usage :
    python scripts/clean_data.py

Entrée  : data/raw/events_raw.json
Sortie  : data/processed/events_clean.json  (structuré, un objet par événement)
          data/processed/events_clean.csv   (même contenu, format tabulaire)

Règles de nettoyage appliquées :
- Déduplication par uid d'événement
- Filtrage géographique : conservation des événements dont la localisation
  est en Île-de-France (région == "Île-de-France" ou département/code postal IDF)
- Filtrage de pertinence : exclusion des agendas non culturels (administratif,
  social, urbanisme...)
- Extraction du texte en langue française en priorité (champs multilingues)
- Suppression du HTML résiduel dans les descriptions
- Rejet des événements sans titre ou sans aucune description exploitable
- Normalisation des dates de début/fin (premier et dernier "timing")
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "events_raw.json"
DATA_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_JSON = DATA_PROCESSED_DIR / "events_clean.json"
OUTPUT_CSV = DATA_PROCESSED_DIR / "events_clean.csv"

# Départements d'Île-de-France (codes INSEE / codes postaux à 2 chiffres)
IDF_DEPARTMENTS = {
    "Paris", "Seine-et-Marne", "Yvelines", "Essonne",
    "Hauts-de-Seine", "Seine-Saint-Denis", "Val-de-Marne", "Val-d'Oise",
}
IDF_POSTAL_PREFIXES = {"75", "77", "78", "91", "92", "93", "94", "95"}

# Agendas non pertinents pour un chatbot d'événements CULTURELS, identifiés lors
# de l'inspection manuelle des résultats de l'étape 2 (annonces administratives,
# aides sociales, urbanisme, jeunesse... plutôt que des événements culturels).
# Filtrage par sous-chaîne insensible à la casse sur le titre de l'agenda source.
EXCLUDED_AGENDA_KEYWORDS = [
    "diocèse", "info jeunes", "aide vacances", "colos apprenantes",
    "cci ", "caue", "catalogue départemental", "pepse",
    "conseil départemental", "haute-garonne", "haute-marne",
]

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def is_relevant_agenda(source_agenda_title: str | None) -> bool:
    """Exclut les agendas identifiés comme non pertinents pour un usage
    culturel (administratif, social, urbanisme...)."""
    if not source_agenda_title:
        return True
    title_lower = source_agenda_title.lower()
    return not any(keyword in title_lower for keyword in EXCLUDED_AGENDA_KEYWORDS)


def strip_html(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def extract_multilingual(field: Any, preferred_lang: str = "fr") -> str:
    """Les champs OpenAgenda (title, description, longDescription) sont
    des dicts {lang: texte}. On privilégie le français, sinon la première
    langue disponible."""
    if field is None:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, dict):
        if preferred_lang in field and field[preferred_lang]:
            return field[preferred_lang]
        for value in field.values():
            if value:
                return value
    return ""


def is_in_ile_de_france(location: dict[str, Any] | None) -> bool:
    if not location:
        return False
    region = (location.get("region") or "").strip()
    department = (location.get("department") or "").strip()
    postal_code = str(location.get("postalCode") or "")

    if region == "Île-de-France":
        return True
    if department in IDF_DEPARTMENTS:
        return True
    if postal_code[:2] in IDF_POSTAL_PREFIXES:
        return True
    return False


def extract_dates(timings: list[dict[str, Any]] | None) -> tuple[str | None, str | None]:
    if not timings:
        return None, None
    begins = [t.get("begin") for t in timings if t.get("begin")]
    ends = [t.get("end") for t in timings if t.get("end")]
    date_start = min(begins) if begins else None
    date_end = max(ends) if ends else (max(begins) if begins else None)
    return date_start, date_end


def clean_event(raw: dict[str, Any]) -> dict[str, Any] | None:
    uid = raw.get("uid")
    if uid is None:
        return None

    title = extract_multilingual(raw.get("title"))
    description = extract_multilingual(raw.get("description"))
    long_description = extract_multilingual(raw.get("longDescription"))

    if not title:
        return None  # un événement sans titre n'est pas exploitable

    full_text = strip_html(long_description) or strip_html(description)
    if not full_text:
        return None  # rien à vectoriser

    location = raw.get("location") or {}
    if not is_in_ile_de_france(location):
        return None

    if not is_relevant_agenda(raw.get("_source_agenda_title")):
        return None

    date_start, date_end = extract_dates(raw.get("timings"))

    return {
        "uid": uid,
        "title": title.strip(),
        "description_short": strip_html(description)[:500],
        "description_full": full_text,
        "city": location.get("city"),
        "address": location.get("address"),
        "postal_code": location.get("postalCode"),
        "department": location.get("department"),
        "region": location.get("region"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "date_start": date_start,
        "date_end": date_end,
        "keywords": raw.get("keywords", {}).get("fr", []) if isinstance(raw.get("keywords"), dict) else [],
        "url": f"https://openagenda.com/{raw.get('_source_agenda_uid', '')}/events/{uid}",
        "source_agenda_uid": raw.get("_source_agenda_uid"),
        "source_agenda_title": raw.get("_source_agenda_title"),
    }


def clean_events(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_uids: set[int] = set()
    cleaned: list[dict[str, Any]] = []
    for raw in raw_events:
        event = clean_event(raw)
        if event is None:
            continue
        if event["uid"] in seen_uids:
            continue
        seen_uids.add(event["uid"])
        cleaned.append(event)
    return cleaned


def save_json(events: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


def save_csv(events: list[dict[str, Any]], path: Path) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(events[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["keywords"] = "|".join(row.get("keywords") or [])
            writer.writerow(row)


def main() -> None:
    if not DATA_RAW.exists():
        raise SystemExit(
            f"Fichier introuvable : {DATA_RAW}\n"
            "Lancez d'abord : python scripts/fetch_open_agenda.py"
        )

    with open(DATA_RAW, encoding="utf-8") as f:
        raw_events = json.load(f)

    print(f"Événements bruts chargés : {len(raw_events)}")
    cleaned = clean_events(raw_events)
    print(f"Événements après nettoyage/filtrage Île-de-France : {len(cleaned)}")

    save_json(cleaned, OUTPUT_JSON)
    save_csv(cleaned, OUTPUT_CSV)
    print(f"Sauvegardé : {OUTPUT_JSON}")
    print(f"Sauvegardé : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()