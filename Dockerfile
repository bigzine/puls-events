# Image pour l'API RAG Puls-Events.
#
# Choix de conception importants :
# - L'index vectoriel N'EST PAS construit pendant le build de l'image : ça
#   nécessiterait d'injecter des clés API dans le conteneur au moment du
#   build (mauvaise pratique de sécurité) et ferait des appels API payants
#   à chaque reconstruction de l'image. L'index est monté en volume depuis
#   l'hôte (voir README / commande `docker run` ci-dessous), déjà construit
#   au préalable avec `python scripts/build_index.py`.
# - Les clés API (.env) ne sont jamais copiées dans l'image ; elles sont
#   injectées au lancement via `--env-file .env`.

FROM python:3.11-slim

# build-essential : nécessaire à la compilation de certaines dépendances
# natives (ex. tokenizers, faiss selon la plateforme).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie et installation des dépendances en premier (couche mise en cache
# séparément par Docker) : évite de tout réinstaller à chaque changement de
# code source, seulement quand requirements.txt change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif (PAS des données : voir .dockerignore).
COPY api/ ./api/
COPY scripts/ ./scripts/

EXPOSE 8000

# Vérifie que l'API répond correctement (utilisé par `docker ps` et par
# Docker Compose pour attendre que le service soit prêt).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
