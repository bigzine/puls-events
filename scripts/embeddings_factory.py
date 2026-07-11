"""
Fabrique d'embeddings : permet de choisir le fournisseur de vectorisation
sans changer le reste du pipeline (build_index.py, search_test.py, l'API).

Deux fournisseurs supportés :
- "mistral"     : mistral-embed via l'API Mistral (nécessite MISTRAL_API_KEY).
                  Qualité optimale, mais chaque appel consomme le quota API.
- "huggingface" : modèle local "paraphrase-multilingual-MiniLM-L12-v2"
                  (aucune clé API, aucun appel réseau après le premier
                  téléchargement du modèle).

Sélection via la variable d'environnement EMBEDDING_PROVIDER
(par défaut : "mistral", conformément au brief technique).
"""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings

DEFAULT_HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MISTRAL_EMBED_MODEL = "mistral-embed"


class EmbeddingConfigError(Exception):
    """Erreur de configuration du fournisseur d'embeddings."""


def get_embeddings(provider: str | None = None) -> Embeddings:
    """
    Retourne une instance d'Embeddings LangChain selon le fournisseur choisi.

    provider : "mistral" ou "huggingface". Si None, lit EMBEDDING_PROVIDER
               dans l'environnement (défaut : "mistral").
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "mistral")).lower().strip()

    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EmbeddingConfigError(
                "EMBEDDING_PROVIDER=mistral mais MISTRAL_API_KEY est absente.\n"
                "Ajoutez votre clé dans .env, ou basculez sur "
                "EMBEDDING_PROVIDER=huggingface pour un modèle local gratuit."
            )
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model=MISTRAL_EMBED_MODEL, api_key=api_key)

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=DEFAULT_HF_MODEL)

    raise EmbeddingConfigError(
        f"Fournisseur d'embeddings inconnu : '{provider}'. "
        "Valeurs acceptées : 'mistral', 'huggingface'."
    )


def embedding_dimension(provider: str) -> int:
    """Dimension du vecteur produit par chaque fournisseur (utile pour la config FAISS)."""
    if provider == "mistral":
        return 1024
    if provider == "huggingface":
        return 384
    raise EmbeddingConfigError(f"Fournisseur inconnu : '{provider}'")