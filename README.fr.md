🇬🇧 [English version](README.md)

## 🐳 Déploiement Docker

L'API RAG (FastAPI) peut être servie dans un conteneur, avec **Ollama tournant sur la machine hôte** (pour bénéficier directement du GPU sans configuration NVIDIA Container Toolkit).

### Prérequis

- Docker Desktop (Windows/Mac) ou Docker Engine + Compose v2 (Linux)
- [Ollama](https://ollama.com) installé et lancé sur l'hôte, avec un modèle disponible :
  ```bash
  ollama pull llama3.1:8b
  ```
- L'index ChromaDB déjà construit (hors Docker, dans le venv local) :
  ```bash
  python ingest.py
  ```

### Lancement

```bash
docker compose up --build
```

Attendre la ligne `Application startup complete.`, puis ouvrir
**http://localhost:8000/docs** (Swagger UI) pour tester l'endpoint `/query`.

Test rapide de la connectivité conteneur → Ollama :

```bash
docker compose exec api python -c "import httpx; print(httpx.get('http://host.docker.internal:11434/api/tags').text)"
```

### Choix d'architecture

| Choix | Justification |
|---|---|
| Index `chroma_db/` monté en **volume** (pas copié dans l'image) | Une ré-ingestion du corpus ne nécessite pas de rebuild de l'image |
| Ollama sur l'**hôte**, joint via `host.docker.internal` | Accès GPU natif, pas de conteneurisation du LLM nécessaire |
| `extra_hosts: host.docker.internal:host-gateway` | Rend le compose portable sous Linux (no-op sous Docker Desktop) |
| Image `python:3.11-slim`, scripts d'ingestion/éval exclus via `.dockerignore` | Image de *serving* minimale : seuls `api.py`/`query.py` et leurs dépendances |
| Modèles chargés au **startup** (pas à la première requête) | Coût payé une fois au démarrage ; latence stable ensuite |

### Pièges rencontrés (et résolus)

| Problème | Cause | Solution |
|---|---|---|
| `chromadb.errors.InternalError: attempt to write a readonly database` au démarrage | Volume ChromaDB monté avec `:ro`. Même en lecture applicative pure, SQLite doit écrire ses fichiers WAL (`-wal`, `-shm`) et poser des verrous à l'ouverture | Monter le volume en lecture-écriture (retirer `:ro`) |
| API injoignable depuis l'hôte malgré le mapping `8000:8000` | uvicorn écoutait sur `127.0.0.1`, joignable uniquement depuis l'intérieur du conteneur | `--host 0.0.0.0` dans le `CMD` du Dockerfile |
| Ollama injoignable depuis le conteneur | `localhost` dans un conteneur désigne le conteneur lui-même, pas l'hôte | `host.docker.internal` dans `OLLAMA_URL` + `extra_hosts` pour Linux |

> **Note** : le premier appel à `/query` peut être lent si Ollama a déchargé le
> modèle de la VRAM (déchargement automatique après ~5 min d'inactivité) —
> ce n'est pas un problème Docker, le modèle est rechargé puis les appels
> suivants retrouvent leur latence normale.


