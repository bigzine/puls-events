# Assistant intelligent de recommandation d'événements culturels — Puls-Events

Rapport technique — POC système RAG (Retrieval-Augmented Generation)

---

## 1. Objectifs du projet

**Contexte.** Puls-Events développe une plateforme de recommandations culturelles personnalisées. La mission confiée consistait à livrer un POC démontrant la faisabilité d'un chatbot intelligent capable de répondre en langage naturel à des questions sur les événements culturels à venir, en s'appuyant sur les données de l'API OpenAgenda.

**Problématique.** Une recherche classique par mots-clés ne capture ni le sens réel d'une demande ("un concert ce week-end"), ni sa dimension temporelle implicite. Un système RAG répond à ce besoin en combinant une recherche sémantique (qui comprend l'intention derrière la question) et une génération de réponse en langage naturel ancrée sur des données réelles — plutôt que de laisser un LLM répondre uniquement à partir de sa mémoire générale, avec le risque d'hallucination que cela comporte.

**Objectif du POC.** Démontrer trois choses aux équipes produit et marketing :
- **Faisabilité technique** : une chaîne complète et fonctionnelle, de la donnée brute à la réponse générée.
- **Pertinence métier** : des réponses fiables, sourcées, sans invention.
- **Performance mesurable** : un système testé et évalué avec des métriques reconnues (Ragas), pas seulement une impression subjective.

**Périmètre.**
- **Zone géographique** : Île-de-France (Paris + 7 départements).
- **Fenêtre temporelle** : 12 derniers mois + événements à venir.
- **Données** : événements culturels issus de l'API OpenAgenda v2.

---

## 2. Architecture du système

### Schéma global

```
OpenAgenda API
      │
      ▼
Nettoyage & filtrage (géographie + pertinence des agendas)
      │
      ▼
Chunking & embeddings (LangChain + Mistral)
      │
      ▼
Index vectoriel FAISS
      │
      ▼
API FastAPI (/ask, /rebuild, /health)
      │
      ▼
Conteneur Docker
```

Toute la logique métier (`scripts/`) est découplée de la couche API (`api/`) : chaque brique est testable indépendamment et réutilisable (par exemple, `rag_chain.py` est utilisé à la fois par l'API, le chatbot CLI de test, et le script d'évaluation Ragas).

### Technologies utilisées

| Domaine | Technologie | Version |
|---|---|---|
| Orchestration RAG | LangChain | 0.2.16 |
| Intégration Mistral | langchain-mistralai | 0.1.13 |
| Recherche vectorielle | FAISS (faiss-cpu) | ≥1.9.0 |
| Embeddings (secours local) | langchain-huggingface | 0.0.3 |
| API REST | FastAPI + Uvicorn | ≥0.112 |
| SDK Mistral | mistralai | 1.0.3 |
| Tests | pytest, pytest-mock | ≥8.3 |
| Évaluation | Ragas | 0.1.21 |
| Conteneurisation | Docker, Docker Compose | — |
| CI | GitHub Actions | — |

---

## 3. Préparation et vectorisation des données

### Source de données

L'API OpenAgenda v2 est utilisée en deux temps : recherche d'agendas pertinents par mots-clés géographiques (`/v2/agendas`), puis récupération de leurs événements avec un filtre temporel (`/v2/agendas/{uid}/events`, `timings[gte]`). **L'API ne propose pas de filtre géographique natif sur les événements** : la stratégie retenue recherche des agendas par mots-clés (Paris, Île-de-France, Hauts-de-Seine, Seine-Saint-Denis, Val-de-Marne), puis filtre chaque événement individuellement sur sa région/département/code postal réel.

### Nettoyage

Pipeline appliqué (`scripts/clean_data.py`) :
1. Extraction du texte en français (champs multilingues d'OpenAgenda).
2. Suppression du HTML résiduel dans les descriptions.
3. **Filtrage géographique** : région == "Île-de-France", ou département dans la liste des 8 départements franciliens, ou préfixe de code postal (75/77/78/91/92/93/94/95).
4. **Filtrage de pertinence** : exclusion des agendas identifiés comme non culturels lors d'une inspection manuelle (ex. "Diocèse de Paris" publiait des annonces administratives, pas des événements ; "Info Jeunes", "Colos apprenantes", "CAUE", "CCI", "Conseil départemental" mélangeaient contenu culturel et administratif).
5. Déduplication par identifiant d'événement.
6. Normalisation des dates de début/fin (première et dernière occurrence).

**Anomalie corrigée notable** : la recherche par mots-clés d'agendas via "Hauts-de-Seine" a remonté des agendas sans rapport géographique ("Haute-Garonne", "Haute-Marne") à cause d'un matching textuel approximatif côté OpenAgenda — neutralisé par le filtrage géographique en aval, qui vérifie la localisation réelle de chaque événement plutôt que de faire confiance au nom de l'agenda.

**Chiffres du pipeline :**

| Étape | Volume |
|---|---|
| Agendas interrogés | 47 |
| Événements bruts récupérés | 5 681 |
| Événements après filtrage géographique | 3 790 |
| Événements après filtrage de pertinence | **3 419** |
| Villes distinctes couvertes | ~294 |

### Chunking

Découpage via `RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=100`, séparateurs `["\n\n", "\n", ". ", " ", ""]`). Le texte source de chaque chunk combine titre, ville, mots-clés et description complète — pas seulement la description brute — pour que la recherche sémantique matche aussi sur des requêtes du type "concert à Nanterre" ou "exposition gratuite". Résultat : **8 449 chunks** à partir des 3 419 événements.

### Embedding

- **Modèle** : `mistral-embed-2312` (alias `mistral-embed`), dimension **1024**.
- **Fournisseur alternatif** : HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`, dimension 384), disponible via une fabrique d'embeddings interchangeable (`scripts/embeddings_factory.py`), utile pour des itérations rapides sans consommer de quota API. Comparaison empirique : Mistral a montré une bien meilleure discrimination géographique (ex. requête "festival en Seine-Saint-Denis" retournant exclusivement des résultats du bon département, contre un mélange avec HuggingFace).
- **Logique de batch** : lots de 64 chunks, avec sauvegarde incrémentale de l'index après chaque lot et reprise automatique en cas d'interruption (fichier d'état `build_state.json`), et nouvelle tentative avec backoff exponentiel sur erreur transitoire.
- **Limite de débit rencontrée** : `mistral-embed-2312` est limité à 1 requête/seconde sur le palier utilisé — un `pace_seconds=1.1` a été ajouté entre lots pour respecter cette limite sans provoquer d'erreurs 429.

---

## 4. Choix du modèle NLP

**Modèle sélectionné** : `mistral-small-latest` (génération de réponse), `mistral-embed` (vectorisation), via `langchain-mistralai`.

**Pourquoi ce modèle ?** Choix imposé par le brief technique (stack LangChain + Mistral). Avantages constatés : bonne compatibilité LangChain, coût maîtrisé, qualité de génération en français satisfaisante une fois le prompt affiné.

**Prompting.** Le prompt système (`rag_chain.build_system_prompt`) impose des règles strictes, construites et affinées au fil des tests réels :
- Répondre **uniquement** à partir des événements du contexte fourni, ne jamais inventer.
- Mentionner systématiquement titre, ville et date de chaque événement recommandé.
- **Citer tous les événements pertinents du contexte**, pas seulement 1 ou 2 (ajusté après constat que la concision par défaut faisait chuter les métriques de couverture du contexte — voir section 7).
- Injection dynamique de la **date du jour** et des **dates exactes du week-end en cours** (calculées en Python, pas laissées au calcul du LLM — voir limites du modèle ci-dessous).
- Rester honnête et refuser poliment si aucun événement ne correspond, y compris pour des questions hors périmètre (météo, autre pays, tarifs).
- Stateless : aucun historique de conversation (conforme au brief).

**Limites du modèle constatées en test réel :**
- **Hallucination de dates** : sans injection explicite, le LLM invente une date du jour plausible mais fausse.
- **Erreurs de calcul de dates relatives** : même avec la date du jour fournie, le LLM a pu mal calculer "ce week-end" (ex. annoncer un lundi et un mardi comme le week-end). Corrigé en calculant ces dates programmatiquement plutôt qu'en confiant l'arithmétique au LLM.
- **Confusion sur les événements récurrents** : une date de début passée mais une date de fin future (série d'événements sur plusieurs mois) pouvait être interprétée à tort comme "terminé" — corrigé par une annotation explicite dans le contexte transmis au LLM.
- **Format de sortie non structuré fiable** : pour les besoins de l'évaluation Ragas (LLM-juge), certaines métriques nécessitant une sortie structurée échouent au parsing avec Mistral (voir section 7).

---

## 5. Construction de la base vectorielle

**FAISS utilisé** : `langchain_community.vectorstores.FAISS`, index de type `IndexFlatL2` (recherche par distance euclidienne), construit et fusionné par lots via `FAISS.from_documents` + `merge_from`.

**Stratégie de persistance** :
- Format de sauvegarde : `save_local()` / `load_local()` de LangChain (fichiers `index.faiss` + `index.pkl` dans un dossier dédié).
- Emplacement : `data/index/faiss_events/`.
- Fichier de traçabilité associé `data/index/index_metadata.json` : fournisseur d'embeddings utilisé, nombre d'événements, nombre de chunks, taille/chevauchement de chunk, date de construction — permet de savoir avec quels embeddings recharger l'index sans se tromper de fournisseur.
- Reprise sur incident : fichier d'état `build_state.json` supprimé automatiquement une fois l'indexation terminée avec succès.

**Métadonnées associées à chaque vecteur** (stockées à côté du vecteur, jamais vectorisées elles-mêmes) : `uid`, `title`, `city`, `address`, `postal_code`, `date_start`, `date_end`, `url`, `source_agenda_title`, `chunk_index`, `chunk_count`. Ces métadonnées permettent d'afficher des informations fiables (non générées par le LLM) dans les réponses de l'API, et servent de base au filtrage programmatique des événements passés.

---

## 6. API et endpoints exposés

**Framework utilisé** : FastAPI (Uvicorn comme serveur ASGI), pour la documentation Swagger interactive générée automatiquement.

**Endpoints clés :**

| Méthode | Route | Description |
|---|---|---|
| POST | `/ask` | Question utilisateur → réponse augmentée + sources |
| POST | `/rebuild` | Reconstruit l'index en tâche de fond (protégé par clé API `X-API-Key`) |
| GET | `/rebuild/status` | Suit la progression d'une reconstruction |
| GET | `/health` | Vérifie que l'index est chargé et l'API opérationnelle |
| GET | `/docs` | Documentation Swagger interactive |

**Format des requêtes/réponses.**

Requête `POST /ask` :
```json
{
  "question": "un concert de jazz à Paris",
  "k": 4
}
```

Réponse :
```json
{
  "answer": "Voici quelques concerts de jazz à Paris...",
  "sources": [
    {
      "uid": 43808652,
      "title": "Échecs & Jam ! Entrée à prix libre",
      "city": "Paris",
      "date_start": "2026-07-19T17:00:00+02:00",
      "date_end": "2026-07-19T21:00:00+02:00",
      "url": "https://openagenda.com/20272888/events/43808652"
    }
  ],
  "num_events_found": 4
}
```

**Exemple d'appel API (curl) :**
```bash
curl -X POST 'http://localhost:8000/ask' \
  -H 'Content-Type: application/json' \
  -d '{"question": "un concert de jazz à Paris", "k": 4}'
```

**Tests effectués et documentés.** `api_test.py` (18 tests, via `TestClient` de FastAPI, aucun appel réseau réel — LLM et embeddings simulés) couvre : validation des entrées (question vide, question trop longue, `k` invalide), comportement sans index chargé (503), sécurité de `/rebuild` (401 sans clé, 401 avec mauvaise clé, 409 si reconstruction déjà en cours, 202 accepté), et le contenu de `/health`.

**Gestion des erreurs / limitations.**
- Question vide/espaces uniquement → `422` (validation Pydantic).
- Aucun index chargé → `503` avec message explicite.
- `/rebuild` sans `REBUILD_API_KEY` configurée côté serveur → `503` (échec sécurisé, désactivé par défaut plutôt qu'ouvert).
- `/rebuild` avec mauvaise clé → `401`.
- Reconstruction déjà en cours → `409`.
- Erreur interne inattendue lors de la génération → `500` avec message générique côté client, trace complète loguée côté serveur (pas de fuite d'information sensible).

---

## 7. Évaluation du système

### Jeu de test annoté

`eval/qa_dataset.json` — 10 cas de test :
- **1 cas de validation API** (question vide, hors périmètre de l'évaluation Ragas).
- **4 cas de refus attendu** (question météo, question sur un autre pays, concert de rap sans correspondance, tarif d'un lieu) : vérifiés par une heuristique dédiée (absence de citation d'URL d'événement dans la réponse), pas par Ragas.
- **5 cas de recommandation positive**, dont **1 isolé comme limite connue** (voir plus bas) : les réponses de référence ont été construites à partir des chunks **réellement récupérés** par le système (bonne pratique de construction d'un jeu de test Ragas), plutôt que d'une vérité choisie indépendamment — une référence indépendante du retrieval réel fait mécaniquement chuter `context_recall` sans refléter une vraie faiblesse du système.

**Méthode d'annotation** : inspection manuelle des chunks récupérés (`scripts/inspect_retrieval.py`) pour chaque question, puis rédaction de la réponse de référence à partir des faits effectivement présents dans ce contexte.

### Métriques d'évaluation

Framework **Ragas** (`eval/evaluate_rag.py`), avec Mistral comme LLM-juge :
- **faithfulness** : la réponse s'appuie-t-elle exclusivement sur le contexte fourni (pas d'hallucination) ?
- **answer_relevancy** : la réponse est-elle pertinente par rapport à la question ?
- **context_precision** : le contexte récupéré est-il exempt de bruit ?
- **context_recall** : le contexte récupéré couvre-t-il ce qu'il faut pour répondre complètement ?

### Résultats obtenus

**Analyse quantitative — évolution au fil des corrections :**

| Itération | faithfulness | context_precision | context_recall |
|---|---|---|---|
| Run initial (références abstraites) | 0.093 | 0.278 | 0.306 |
| Références factuelles réécrites | 0.186 | 0.426 | 0.352 |
| Refus séparés des recommandations | 0.202 | 0.383 | 0.133 |
| **Bug de contexte corrigé** (Ragas comparait au mauvais contexte) | 0.800 | 0.050 | 0.067 |
| Références alignées sur le retrieval réel | 0.725 | 0.396 | 0.312 |
| **Prompt ajusté (citer tous les événements pertinents)** | **0.956** | **0.625** | **0.667** |

`answer_relevancy` reste non mesurable (`None`/`nan`) : cette métrique demande au LLM-juge de reformuler la question à partir de la réponse dans un format structuré que Mistral ne respecte pas de façon fiable (`"Failed to parse output. Returning None."`). Une tentative de forcer le mode JSON strict, scopée uniquement à cette métrique (sans affecter les 3 autres, qui donnent déjà de bons résultats), a été implémentée mais n'a pas permis de lever ce blocage de façon concluante — limite documentée plutôt que masquée.

**Analyse qualitative — exemples de bugs réels détectés et corrigés en cours de test :**

1. **Dates hallucinées** : le LLM inventait la date du jour dans ses réponses. → Injection dynamique de la date réelle dans le prompt système.
2. **Événements passés recommandés** : le LLM jugeait lui-même si un événement était terminé (peu fiable). → Filtrage programmatique (`filter_future_events`) avant génération, indépendant du jugement du LLM.
3. **Mauvais calcul du week-end** : "ce week-end" annoncé avec des dates tombant un lundi et un mardi. → Calcul explicite des dates du samedi/dimanche en Python, injecté tel quel dans le prompt.
4. **"Aucun résultat" à tort** : les meilleurs résultats sémantiques bruts pouvaient être majoritairement des événements passés, laissant 0 résultat après filtrage même quand des événements futurs pertinents existaient. → Marge de récupération élargie (`fetch_k = max(k×15, 60)`) avant filtrage par date.
5. **Incohérence de contexte dans l'évaluation** : le script d'évaluation interrogeait l'index séparément de la génération de réponse, comparant Ragas à un contexte différent de celui réellement utilisé. → `answer_question` retourne désormais `context_documents`, les chunks exacts utilisés, garantissant que l'évaluation porte sur le bon contexte.
6. **Imprécision géographique du retrieval** (limite connue, non corrigée) : pour la requête "festivals en Seine-Saint-Denis", le système retourne en top résultat le "Festival de Marne" à Vincennes — qui est en réalité dans le Val-de-Marne. La recherche sémantique pure ne respecte pas de contrainte de département stricte. Ce cas est volontairement conservé et documenté dans le jeu de test (`known_limitation: true`) plutôt que masqué.

**Piste explorée et écartée** : un filtre de seuil de distance sur le retrieval (couper les résultats trop éloignés sémantiquement) a été testé via un script de calibration (`scripts/calibrate_distance_threshold.py`) sur des données réelles. Les distances observées sur 5 requêtes types (0.35 à 0.48) ne montrent pas de rupture nette entre "pertinent" et "bruit" — un seuil unique risquait de dégrader plus de résultats valides qu'il n'en aurait filtré d'invalides. Fonctionnalité implémentée et testée (désactivée par défaut, activable via `MAX_RETRIEVAL_DISTANCE` dans `.env`) mais non retenue en l'état faute de signal empirique clair.

---

## 8. Recommandations et perspectives

### Ce qui fonctionne bien

- Chaîne de bout en bout fonctionnelle et testée (données → index → API → conteneur).
- Robustesse de l'indexation : reprise automatique après interruption, retry avec backoff sur erreurs transitoires.
- Fidélité factuelle élevée (`faithfulness` 0.956) et vérification manuelle croisée systématique (dates, villes, titres recoupés avec les données sources).
- Sécurité de base sur l'endpoint sensible (`/rebuild` protégé par clé API, échec sécurisé par défaut).
- Couverture de test large : 114+ tests unitaires/fonctionnels, deux workflows CI (tests automatiques à chaque push, évaluation Ragas déclenchable manuellement pour ne pas consommer de quota API inutilement).

### Limites du POC

- **Périmètre géographique** limité à l'Île-de-France.
- **Filtre de pertinence des agendas** basé sur une liste de mots-clés statique (maintenance manuelle nécessaire si de nouveaux agendas non pertinents apparaissent).
- **Imprécision géographique ponctuelle** du retrieval sémantique pur (voir section 7, point 6).
- **Pas de contrainte géographique stricte** au niveau de la requête FAISS elle-même (uniquement en filtrage a posteriori des dates).
- **Pas d'historique de conversation** (hors périmètre du brief pour ce POC).
- **`answer_relevancy`** non mesurable de façon fiable avec Mistral comme LLM-juge Ragas.
- **Coût/temps d'évaluation** : le palier API utilisé impose des limites de débit qui allongent le temps d'une évaluation Ragas complète (plusieurs minutes), sans impact sur les performances de l'API en usage normal.

### Améliorations possibles

- Recherche hybride : combiner similarité sémantique et filtres stricts (date, ville/département) directement au niveau de la requête FAISS plutôt qu'en post-filtrage.
- Classification automatique de la pertinence des agendas (modèle ou règles enrichies) plutôt qu'une liste statique.
- Élargir la couverture géographique au-delà de l'Île-de-France.
- Ajouter un historique de conversation multi-tours.
- Réévaluer `answer_relevancy` avec un LLM-juge alternatif mieux supporté par Ragas, ou une métrique de substitution.
- Passage en production : ajout d'authentification sur `/ask` si exposé publiquement, monitoring des temps de réponse et du taux d'erreur, tableau de bord de suivi qualité basé sur des évaluations Ragas régulières.

---

## 9. Organisation du dépôt GitHub

```
puls-events/
├── README.md                      # Ce rapport technique
├── requirements.txt                # Dépendances Python figées
├── .env.example                    # Modèle de configuration (jamais de vraies clés)
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
│
├── scripts/                        # Logique métier (indépendante de l'API)
│   ├── openagenda_client.py        # Client API OpenAgenda v2
│   ├── fetch_open_agenda.py        # Récupération des événements bruts
│   ├── clean_data.py               # Nettoyage, filtrage géo + pertinence
│   ├── embeddings_factory.py       # Fabrique d'embeddings (Mistral / HuggingFace)
│   ├── build_index.py              # Chunking + construction de l'index FAISS
│   ├── rag_chain.py                # Chaîne RAG : retrieval + génération
│   ├── chatbot.py                  # Chatbot CLI pour tests manuels
│   ├── inspect_retrieval.py        # Diagnostic : contexte réellement récupéré
│   └── calibrate_distance_threshold.py  # Calibration du seuil de distance
│
├── api/
│   └── main.py                     # API FastAPI (/ask, /rebuild, /health)
├── api_test.py                     # Tests fonctionnels de l'API
│
├── tests/                          # Tests unitaires
│   ├── test_data_pipeline.py       # Étape 2 : récupération/nettoyage
│   ├── test_indexing.py            # Étape 3 : chunking/indexation
│   └── test_rag_chain.py           # Étape 4 : chaîne RAG
│
├── eval/
│   ├── qa_dataset.json             # Jeu de test annoté
│   ├── evaluate_rag.py             # Évaluation automatique Ragas
│   └── results/                    # Résultats d'évaluation horodatés
│
├── .github/workflows/
│   ├── tests.yml                   # CI automatique (tests unitaires + API)
│   └── evaluate.yml                # CI manuelle (évaluation Ragas, coûte du quota)
│
└── data/                           # Données (non versionnées, voir .gitignore)
    ├── raw/                        # Événements bruts OpenAgenda
    ├── processed/                  # Événements nettoyés
    └── index/                      # Index FAISS + métadonnées
```

**Explication rapide de chaque répertoire :**
- `scripts/` : toute la logique métier, testable et réutilisable indépendamment de l'API.
- `api/` : la couche HTTP, volontairement fine — elle ne fait qu'exposer la logique de `scripts/`.
- `tests/` et `api_test.py` : tests automatisés, aucun appel réseau réel (LLM/embeddings simulés).
- `eval/` : évaluation de la qualité réelle du système, avec de vrais appels API (coûte du quota, à lancer volontairement).
- `.github/workflows/` : automatisation CI, séparée entre tests gratuits (automatiques) et évaluation payante (manuelle).
- `data/` : jamais versionné (fichiers volumineux et régénérables via les scripts).

---

## 10. Annexes

### Extrait du jeu de test annoté (`eval/qa_dataset.json`)

```json
{
  "id": "qa_01",
  "question": "Peux-tu me recommander un concert de jazz à Paris ?",
  "reference_answer": "Il existe plusieurs concerts de jazz à Paris : 'Échecs & Jam ! Entrée à prix libre', une jam session jazz hebdomadaire (dimanches 19 et 26 juillet 2026), 'CINÉ-JAZZ : les thèmes légendaires du grand écran' (1er-2 août 2026), et 'Les samedis Brasil e jazz' par Jean-Baptiste Loutte (18-19 juillet 2026).",
  "expects_refusal": false
}
```

### Prompt système utilisé (extrait, `rag_chain.build_system_prompt`)

```
Tu es l'assistant culturel de Puls-Events. Tu aides les utilisateurs à trouver
des événements culturels en Île-de-France (concerts, expositions, spectacles,
festivals...).

Nous sommes aujourd'hui le {date}. [...]

Règles impératives :
- Réponds UNIQUEMENT à partir des événements fournis dans le contexte
  ci-dessous. N'invente jamais d'événement, de date, de lieu ou de détail
  qui n'y figure pas.
- Si aucun événement du contexte ne correspond à la question, dis-le
  clairement et propose à l'utilisateur de reformuler sa recherche.
- Mentionne systématiquement le titre, la ville et la date de chaque
  événement que tu recommandes.
- Si PLUSIEURS événements du contexte correspondent au thème demandé,
  mentionne-les TOUS, pas seulement le premier ou les deux premiers.
```

### Exemple de réponse JSON réelle (`POST /ask`)

```json
{
  "answer": "Voici deux concerts de jazz à Paris ce mois-ci :\n\n1. Échecs & Jam ! Entrée à prix libre — Paris, le 19 juillet 2026 (17h-21h)\n2. Jo & Charly / Encuentro Cubano-Jazz — Paris, le 31 juillet 2026",
  "sources": [
    {
      "uid": 43808652,
      "title": "Échecs & Jam ! Entrée à prix libre",
      "city": "Paris",
      "date_start": "2026-07-19T17:00:00+02:00",
      "date_end": "2026-07-19T21:00:00+02:00",
      "url": "https://openagenda.com/20272888/events/43808652"
    },
    {
      "uid": 3237586,
      "title": "Jo & Charly / Encuentro Cubano-Jazz",
      "city": "Paris",
      "date_start": "2026-07-31T19:30:00+02:00",
      "date_end": "2026-07-31T22:30:00+02:00",
      "url": "https://openagenda.com/20272888/events/3237586"
    }
  ],
  "num_events_found": 2
}
```

---

## Installation et reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# éditer .env : MISTRAL_API_KEY, OPENAGENDA_API_KEY, REBUILD_API_KEY

python check_setup.py                    # vérifie les imports critiques
python scripts/fetch_open_agenda.py       # récupère les événements bruts
python scripts/clean_data.py              # nettoie et filtre
python scripts/build_index.py             # construit l'index FAISS
pytest tests/ api_test.py -v              # 114+ tests

uvicorn api.main:app --reload --port 8000 # lance l'API (voir /docs)

# Ou via Docker :
docker compose up --build
```