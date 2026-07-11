🇫🇷 [Version française](README.fr.md)

## 🐳 Docker Deployment

The RAG API (FastAPI) can be served in a container, with **Ollama running on the host machine** (direct GPU access, no NVIDIA Container Toolkit setup required).

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine + Compose v2 (Linux)
- [Ollama](https://ollama.com) installed and running on the host, with a model available:
  ```bash
  ollama pull llama3.1:8b
  ```
- The ChromaDB index already built (outside Docker, in the local venv):
  ```bash
  python ingest.py
  ```

### Run

```bash
docker compose up --build
```

Wait for the `Application startup complete.` log line, then open
**http://localhost:8000/docs** (Swagger UI) to test the `/query` endpoint.

Quick container → Ollama connectivity check:

```bash
docker compose exec api python -c "import httpx; print(httpx.get('http://host.docker.internal:11434/api/tags').text)"
```

### Architecture decisions

| Decision | Rationale |
|---|---|
| `chroma_db/` index mounted as a **volume** (not baked into the image) | Re-ingesting the corpus does not require an image rebuild |
| Ollama on the **host**, reached via `host.docker.internal` | Native GPU access, no need to containerize the LLM |
| `extra_hosts: host.docker.internal:host-gateway` | Makes the compose file portable to Linux (no-op on Docker Desktop) |
| `python:3.11-slim` base image, ingestion/eval scripts excluded via `.dockerignore` | Minimal *serving* image: only `api.py`/`query.py` and their dependencies |
| Models loaded at **startup** (not on first request) | Cost paid once at boot; stable latency afterwards |

### Pitfalls encountered (and fixed)

| Issue | Cause | Fix |
|---|---|---|
| `chromadb.errors.InternalError: attempt to write a readonly database` at startup | ChromaDB volume mounted with `:ro`. Even for purely read-only application usage, SQLite must write its WAL files (`-wal`, `-shm`) and acquire locks on open | Mount the volume read-write (remove `:ro`) |
| API unreachable from the host despite the `8000:8000` port mapping | uvicorn was listening on `127.0.0.1`, only reachable from inside the container | `--host 0.0.0.0` in the Dockerfile `CMD` |
| Ollama unreachable from the container | Inside a container, `localhost` refers to the container itself, not the host | `host.docker.internal` in `OLLAMA_URL` + `extra_hosts` for Linux |

> **Note**: the first `/query` call may be slow if Ollama has unloaded the
> model from VRAM (automatic unload after ~5 min of inactivity) — this is
> not a Docker issue; the model is reloaded and subsequent calls return to
> normal latency.

