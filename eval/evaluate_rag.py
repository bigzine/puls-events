"""
Évaluation automatique de la qualité du système RAG avec Ragas.

Usage :
    python eval/evaluate_rag.py

Nécessite un index FAISS déjà construit (python scripts/build_index.py) et
une clé MISTRAL_API_KEY valide dans .env : ce script fait de VRAIS appels à
l'API Mistral (à la fois pour générer les réponses du chatbot ET pour le
LLM-juge de Ragas qui évalue ces réponses), donc consomme du quota API.

Métriques calculées (voir la doc Ragas pour le détail) :
- faithfulness       : la réponse est-elle fidèle au contexte récupéré (pas d'invention) ?
- answer_relevancy   : la réponse est-elle pertinente par rapport à la question posée ?
- context_precision  : le contexte récupéré est-il pertinent (peu de bruit) ?
- context_recall     : le contexte récupéré couvre-t-il ce qu'il faut pour répondre correctement ?

Sortie : eval/results/evaluation_<timestamp>.json (scores détaillés par question)
         + un résumé affiché dans la console.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
from embeddings_factory import EmbeddingConfigError, get_embeddings  # noqa: E402
from rag_chain import EventChatbotError, answer_question, get_llm, retrieve_events  # noqa: E402

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


def load_qa_dataset() -> list[dict]:
    with open(QA_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # qa_09 teste volontairement une question vide, rejetée au niveau API
    # (validation Pydantic) avant même d'atteindre le RAG : on l'exclut donc
    # de l'évaluation Ragas, qui exige des questions non vides.
    return [c for c in data["cases"] if c["question"].strip()]


def run_rag_on_dataset(vectorstore: FAISS, llm, cases: list[dict]) -> dict[str, list]:
    """Fait tourner le pipeline RAG réel sur chaque cas de test et collecte
    les résultats au format attendu par Ragas."""
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for case in cases:
        question = case["question"]
        print(f"  -> {question}")
        try:
            retrieved_docs = retrieve_events(vectorstore, question, k=4)
            result = answer_question(vectorstore, llm, question, k=4)
        except EventChatbotError as exc:
            print(f"     [ERREUR] {exc}")
            continue

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append([doc.page_content for doc in retrieved_docs] or ["Aucun contexte trouvé."])
        ground_truths.append(case["reference_answer"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def main() -> None:
    load_dotenv()

    print("Chargement de l'index FAISS...")
    vectorstore = load_vectorstore()

    try:
        llm = get_llm()
    except EventChatbotError as exc:
        raise SystemExit(f"ERREUR : {exc}")

    cases = load_qa_dataset()
    print(f"{len(cases)} questions annotées chargées depuis {QA_DATASET_PATH.name}\n")

    print("Exécution du pipeline RAG sur chaque question...")
    rag_results = run_rag_on_dataset(vectorstore, llm, cases)

    if not rag_results["question"]:
        raise SystemExit("Aucun résultat exploitable : évaluation annulée.")

    dataset = Dataset.from_dict(rag_results)

    print("\nCalcul des métriques Ragas (appels API Mistral pour le LLM-juge)...")
    try:
        embeddings = get_embeddings()
    except EmbeddingConfigError as exc:
        raise SystemExit(f"ERREUR de configuration des embeddings : {exc}")

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores_df = scores.to_pandas()

    print("\n" + "=" * 70)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 70)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in scores_df.columns:
            mean_score = scores_df[metric].mean()
            print(f"  {metric:20s} : {mean_score:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"evaluation_{timestamp}.json"
    scores_df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"\nDétail par question sauvegardé dans {output_path}")


if __name__ == "__main__":
    main()
