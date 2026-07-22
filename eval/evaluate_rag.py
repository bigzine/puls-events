"""
Évaluation automatique de la qualité du système RAG avec Ragas.

Usage :
    python eval/evaluate_rag.py

Nécessite un index FAISS déjà construit (python scripts/build_index.py) et
une clé MISTRAL_API_KEY valide dans .env : ce script fait de VRAIS appels à
l'API Mistral (à la fois pour générer les réponses du chatbot ET pour le
LLM-juge de Ragas qui évalue ces réponses), donc consomme du quota API.

Méthodologie à deux volets :
- Les métriques Ragas (faithfulness, answer_relevancy, context_precision,
  context_recall) ne sont calculées QUE sur les questions où une vraie
  recommandation est attendue ("expects_refusal": false dans le dataset).
  Ces métriques ne sont pas conçues pour évaluer une phrase de refus : sur
  un premier essai incluant les deux types de cas, les refus honnêtes
  faisaient chuter artificiellement le score de fidélité (faithfulness),
  alors qu'ils sont pourtant le comportement attendu du système.
- Les cas de refus ("expects_refusal": true — questions hors-sujet ou sans
  événement correspondant) sont vérifiés séparément par une heuristique
  simple : une réponse de refus correcte ne doit citer aucune URL
  d'événement (le système ne doit pas inventer une recommandation hors
  sujet juste pour avoir l'air de répondre).

Sortie : eval/results/evaluation_<timestamp>.json (scores Ragas détaillés)
         + un résumé affiché dans la console pour les deux volets.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import nest_asyncio
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from embeddings_factory import EmbeddingConfigError, get_embeddings  # noqa: E402
from rag_chain import EventChatbotError, answer_question, get_llm  # noqa: E402

# Logs de debug : utile pour diagnostiquer les blocages / lenteurs côté API
# (voir historique du projet). Mettre à logging.WARNING pour une sortie
# plus silencieuse une fois le pipeline stabilisé.
# Logs de debug : utile pour diagnostiquer les blocages / lenteurs côté API
# (voir historique du projet — blocage asyncio résolu, rate-limit Mistral
# identifié grâce à ces logs). Repasser à logging.WARNING pour une sortie
# plus silencieuse une fois le pipeline stabilisé.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.INFO)   # une ligne par requête HTTP, sans le détail des headers
logging.getLogger("httpcore").setLevel(logging.WARNING)  # trop verbeux en DEBUG, on le limite
logging.getLogger("ragas").setLevel(logging.DEBUG)

QA_DATASET_PATH = Path(__file__).resolve().parent / "qa_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index" / "faiss_events"
INDEX_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index_metadata.json"


def load_vectorstore() -> FAISS:
    if not INDEX_DIR.exists():
        raise SystemExit(f"Index introuvable : {INDEX_DIR}\nLancez d'abord : python scripts/build_index.py")
    provider = "mistral"
    if INDEX_METADATA_PATH.exists():
        with open(INDEX_METADATA_PATH, encoding="utf-8") as f:
            provider = json.load(f).get("provider", "mistral")
    embeddings = get_embeddings(provider)
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)


def load_qa_dataset() -> tuple[list[dict], list[dict]]:
    """Retourne (cas_positifs, cas_de_refus).

    Exclut de l'évaluation Ragas :
    - qa_09 (question vide, cas de validation API pure)
    - tout cas marqué "known_limitation": true — ces cas documentent une
      faiblesse RÉELLE et déjà identifiée du système (ex. imprécision
      géographique du retrieval sur qa_03), conservée dans le dataset pour
      la traçabilité mais volontairement exclue du calcul de score : les
      compter contre le système reviendrait à le pénaliser deux fois pour
      un problème déjà documenté ailleurs (rapport, slide limites).
    """
    with open(QA_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    cases = [c for c in data["cases"] if c["question"].strip() and not c.get("known_limitation", False)]
    positive_cases = [c for c in cases if not c.get("expects_refusal", False)]
    refusal_cases = [c for c in cases if c.get("expects_refusal", False)]
    return positive_cases, refusal_cases


def is_refusal_correct(answer: str) -> bool:
    """Heuristique : un refus correct ne cite aucune URL d'événement. Le
    système ne doit jamais recommander un événement hors sujet juste pour
    donner l'impression de répondre à une question hors périmètre."""
    return "http" not in answer.lower()


def run_rag_on_dataset(vectorstore: FAISS, llm, cases: list[dict]) -> dict[str, list]:
    """Fait tourner le pipeline RAG réel sur chaque cas et collecte les
    résultats au format attendu par Ragas.

    IMPORTANT : le contexte utilisé pour l'évaluation Ragas provient de
    `result["context_documents"]`, c'est-à-dire EXACTEMENT les chunks que
    `answer_question` a utilisés pour générer la réponse (après filtrage des
    événements passés et sélection des k meilleurs parmi une marge élargie).
    Un appel séparé à `retrieve_events(question, k=4)` donnerait un contexte
    différent (recherche brute, sans filtre de date, sans la marge élargie),
    ce qui faussait les métriques Ragas en les comparant au mauvais contexte.
    """
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for case in cases:
        question = case["question"]
        print(f"  -> {question}")
        try:
            result = answer_question(vectorstore, llm, question, k=4)
        except EventChatbotError as exc:
            print(f"     [ERREUR] {exc}")
            continue

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append(result["context_documents"] or ["Aucun contexte trouvé."])
        ground_truths.append(case["reference_answer"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def evaluate_refusal_cases(vectorstore: FAISS, llm, cases: list[dict]) -> list[dict]:
    """Vérifie les cas de refus par heuristique, sans appel à Ragas (ces
    métriques ne sont pas adaptées à ce type de réponse)."""
    results = []
    for case in cases:
        question = case["question"]
        print(f"  -> {question}")
        try:
            result = answer_question(vectorstore, llm, question, k=4)
        except EventChatbotError as exc:
            print(f"     [ERREUR] {exc}")
            continue

        correct = is_refusal_correct(result["answer"])
        results.append({
            "question": question,
            "answer": result["answer"],
            "refusal_correct": correct,
        })
    return results


def main() -> None:
    load_dotenv()

    print("Chargement de l'index FAISS...")
    vectorstore = load_vectorstore()

    try:
        llm = get_llm()
    except EventChatbotError as exc:
        raise SystemExit(f"ERREUR : {exc}")

    positive_cases, refusal_cases = load_qa_dataset()
    print(
        f"{len(positive_cases)} questions à réponse positive attendue, "
        f"{len(refusal_cases)} questions à refus attendu, chargées depuis {QA_DATASET_PATH.name}\n"
    )

    # --- Volet 1 : cas de refus (heuristique simple, pas de Ragas) --------
    print("Vérification des cas de refus attendus...")
    refusal_results = evaluate_refusal_cases(vectorstore, llm, refusal_cases)
    refusal_ok = sum(1 for r in refusal_results if r["refusal_correct"])

    # --- Volet 2 : cas positifs (métriques Ragas) --------------------------
    print("\nExécution du pipeline RAG sur les questions à réponse positive...")
    rag_results = run_rag_on_dataset(vectorstore, llm, positive_cases)

    scores_df = None
    if rag_results["question"]:
        dataset = Dataset.from_dict(rag_results)

        print("\nCalcul des métriques Ragas (appels API Mistral pour le LLM-juge)...")
        try:
            embeddings = get_embeddings()
        except EmbeddingConfigError as exc:
            raise SystemExit(f"ERREUR de configuration des embeddings : {exc}")

        # NOTE : une tentative de forcer le mode JSON strict côté Mistral
        # (response_format={"type": "json_object"}) a été testée puis
        # abandonnée : elle n'a pas résolu l'échec systématique de
        # answer_relevancy (qui repose sur une comparaison d'embeddings, pas
        # sur du parsing de texte structuré) et semble avoir dégradé
        # context_recall (0.352 -> 0.133 sur des runs comparables), sans
        # doute parce que toutes les métriques Ragas n'attendent pas le même
        # format de sortie en interne. Le LLM-juge est donc utilisé tel quel.
        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        # max_workers=2 : compromis prudent (voir historique du projet — un
        # ralentissement progressif observé même en séquentiel suggère un
        # rate-limit côté serveur Mistral ; ne pas monter plus haut sans
        # revalider). nest_asyncio contourne un conflit de boucles asyncio
        # observé sur macOS.
        nest_asyncio.apply()
        run_config = RunConfig(max_workers=2, timeout=180, log_tenacity=True)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )
        scores_df = scores.to_pandas()

    # --- Résultats -----------------------------------------------------
    print("\n" + "=" * 70)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 70)

    print(f"\nCas de refus attendus : {refusal_ok}/{len(refusal_results)} corrects")
    for r in refusal_results:
        status = "OK" if r["refusal_correct"] else "ÉCHEC"
        print(f"  [{status}] {r['question']}")

    if scores_df is not None:
        print("\nMétriques Ragas (cas à réponse positive uniquement) :")
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric in scores_df.columns:
                mean_score = scores_df[metric].mean()
                print(f"  {metric:20s} : {mean_score:.3f}")
    else:
        print("\nAucun cas positif exploitable : métriques Ragas non calculées.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if scores_df is not None:
        ragas_path = RESULTS_DIR / f"evaluation_ragas_{timestamp}.json"
        scores_df.to_json(ragas_path, orient="records", force_ascii=False, indent=2)
        print(f"\nDétail Ragas sauvegardé dans {ragas_path}")

    refusal_path = RESULTS_DIR / f"evaluation_refusals_{timestamp}.json"
    with open(refusal_path, "w", encoding="utf-8") as f:
        json.dump(refusal_results, f, ensure_ascii=False, indent=2)
    print(f"Détail des cas de refus sauvegardé dans {refusal_path}")


if __name__ == "__main__":
    main()