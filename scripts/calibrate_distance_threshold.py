"""
Aide à calibrer MAX_RETRIEVAL_DISTANCE en affichant, pour plusieurs
requêtes représentatives, la distance de chaque résultat récupéré — pour
repérer visuellement où se situe la frontière entre "pertinent" et "bruit".

Usage :
    python scripts/calibrate_distance_threshold.py

Méthode : regarde où les distances "sautent" nettement (ex. les 3 premiers
résultats sont à 0.28-0.35, puis un saut à 0.61) — le seuil se place juste
après le dernier résultat visiblement pertinent. Teste plusieurs requêtes
avant de choisir une valeur : elle doit bien fonctionner pour un éventail
de questions, pas une seule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

sys.path.append(str(Path(__file__).resolve().parent))
from embeddings_factory import get_embeddings  # noqa: E402

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index" / "faiss_events"
INDEX_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index_metadata.json"

SAMPLE_QUERIES = [
    "concert de jazz à Paris",
    "exposition d'art contemporain",
    "festival en Seine-Saint-Denis",
    "spectacle de danse",
    "activité gratuite pour les enfants",
]


def main() -> None:
    load_dotenv()

    provider = "mistral"
    if INDEX_METADATA_PATH.exists():
        with open(INDEX_METADATA_PATH, encoding="utf-8") as f:
            provider = json.load(f).get("provider", "mistral")

    embeddings = get_embeddings(provider)
    vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

    all_scores: list[float] = []

    for query in SAMPLE_QUERIES:
        print(f"\n{'=' * 70}\nRequête : {query!r}\n{'=' * 70}")
        results = vectorstore.similarity_search_with_score(query, k=10)
        for i, (doc, score) in enumerate(results, start=1):
            all_scores.append(score)
            print(f"  [{i:2d}] distance={score:.4f}  {doc.metadata.get('title', '')[:60]}")

    all_scores.sort()
    print(f"\n{'=' * 70}\nRésumé sur {len(all_scores)} résultats (toutes requêtes confondues)")
    print(f"{'=' * 70}")
    print(f"  min={all_scores[0]:.4f}  max={all_scores[-1]:.4f}")
    print(f"  médiane={all_scores[len(all_scores)//2]:.4f}")
    print(
        "\nChoisis un seuil juste au-dessus des distances des résultats que tu juges "
        "pertinents ci-dessus, puis ajoute dans .env :\n"
        "  MAX_RETRIEVAL_DISTANCE=<ta_valeur>"
    )


if __name__ == "__main__":
    main()
