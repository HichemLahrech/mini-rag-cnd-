"""
API — Expose le pipeline RAG CND comme service HTTP
=======================================================

Ce module ne redéfinit AUCUNE logique métier : il réutilise directement les
fonctions de query.py (retrieve, rerank, build_where_filter, build_prompt).
L'API est une couche fine autour du pipeline déjà construit et validé par
eval.py — pas une réécriture.

Concepts clés :
- LIFESPAN (démarrage/arrêt) : le modèle d'embedding, le cross-encoder et la
  connexion ChromaDB sont chargés UNE SEULE FOIS au démarrage du serveur, pas
  à chaque requête. Recharger un modèle SentenceTransformer à chaque appel
  HTTP ajouterait plusieurs secondes de latence à chaque requête — un piège
  classique quand on transforme un script CLI en service.
- PYDANTIC (validation automatique) : les modèles de requête/réponse ci-dessous
  définissent un contrat d'API explicite. FastAPI valide automatiquement les
  requêtes entrantes contre ce contrat (types, valeurs autorisées) et génère
  une documentation interactive à /docs sans effort supplémentaire.
- PROVIDER LLM interchangeable : le choix entre Ollama (local, gratuit, plus
  lent) et Claude (cloud, payant, plus capable) est un paramètre de requête,
  pas un choix figé au moment du déploiement.

Lancement local (développement) :
    uvicorn api:app --reload --port 8000

Documentation interactive une fois lancé :
    http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

import chromadb
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

from cnd_metadata import METHOD_KEYWORDS
from query import (
    CANDIDATE_K,
    COLLECTION_NAME,
    DB_DIR,
    EMBEDDING_MODEL,
    FINAL_K,
    MAX_DISTANCE,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RERANK_MODEL,
    TOP_K,
    build_prompt,
    build_where_filter,
    rerank as rerank_chunks,
    retrieve,
)

# Protection optionnelle par clé API : si API_KEY est défini dans
# l'environnement, les requêtes doivent inclure le header X-API-Key
# correspondant. Si non défini (cas par défaut en développement local),
# l'API reste ouverte. Volontairement simple (pas d'OAuth/JWT) pour un
# projet de cette taille — à durcir avant un vrai déploiement public.
API_KEY = os.environ.get("API_KEY")

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# État partagé, peuplé une fois au démarrage (voir lifespan ci-dessous).
# Un dict plutôt que des globals séparées : plus facile à étendre et à
# réinitialiser proprement dans les tests.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"📦 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    state["embed_model"] = SentenceTransformer(EMBEDDING_MODEL)

    print(f"📦 Chargement du cross-encoder : {RERANK_MODEL}")
    state["cross_encoder"] = CrossEncoder(RERANK_MODEL)

    print(f"📚 Connexion à ChromaDB : {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    state["collection"] = client.get_collection(COLLECTION_NAME)
    print(f"   {state['collection'].count()} chunks disponibles — API prête")

    yield  # le serveur tourne ici

    state.clear()


app = FastAPI(
    title="mini-RAG CND API",
    description=(
        "API de retrieval-augmented generation sur un corpus scientifique de "
        "contrôle non destructif (CND). Voir /docs pour la documentation interactive."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS permissif pour la démo/le développement : à restreindre à des origines
# précises avant un déploiement public réel (voir commentaire ci-dessous).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ à restreindre en production (liste d'origines explicites)
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Modèles de requête/réponse (contrat d'API, validé automatiquement) ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question en langage naturel (français ou anglais)")
    rerank: bool = Field(True, description="Active le reranking par cross-encoder (plus précis, plus lent)")
    top_k: int = Field(TOP_K, ge=1, le=50, description="Nombre de chunks retenus (si rerank=False)")
    candidates: int = Field(CANDIDATE_K, ge=1, le=100, description="Chunks pré-sélectionnés avant reranking (si rerank=True)")
    final_k: int = Field(FINAL_K, ge=1, le=20, description="Chunks finaux après reranking (si rerank=True)")
    max_distance: float | None = Field(MAX_DISTANCE, description="Seuil de distance cosinus (si rerank=False)")
    year_min: int | None = Field(None, description="Filtre : année de publication minimale")
    year_max: int | None = Field(None, description="Filtre : année de publication maximale")
    methods: list[str] | None = Field(None, description="Filtre méthode(s) CND en OR — voir GET /methods")
    methods_all: list[str] | None = Field(None, description="Filtre méthode(s) CND en AND (technique combinée)")
    provider: Literal["ollama", "claude"] = Field("ollama", description="Moteur de génération")


class RetrievedChunk(BaseModel):
    source: str
    section: str | None = None
    text: str
    score: float
    metric: Literal["distance", "rerank_score"]


class QueryResponse(BaseModel):
    question: str
    answer: str
    provider_used: str
    chunks: list[RetrievedChunk]
    filter_applied: dict | None = None


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int


class MethodsResponse(BaseModel):
    methods: list[str]


# --- Génération LLM : Ollama (local) et Claude (cloud), interchangeables ---

def generate_with_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama injoignable ({e}). Vérifie qu'Ollama tourne et que '{OLLAMA_MODEL}' est installé.",
        )
    return response.json()["response"]


def generate_with_claude(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="provider='claude' demandé mais ANTHROPIC_API_KEY n'est pas définie côté serveur.",
        )
    try:
        from anthropic import Anthropic
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Le package 'anthropic' n'est pas installé (pip install anthropic).",
        )
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def check_api_key(x_api_key: str | None) -> None:
    """Vérifie la clé API si API_KEY est configurée côté serveur (voir en-tête du module)."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API manquante ou invalide (header X-API-Key).")


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health():
    """Vérifie que l'API est prête et que la base vectorielle est accessible."""
    return HealthResponse(status="ok", chunks_indexed=state["collection"].count())


@app.get("/methods", response_model=MethodsResponse)
def list_methods():
    """Liste les tags de méthode CND reconnus, utilisables dans les filtres /query."""
    return MethodsResponse(methods=sorted(METHOD_KEYWORDS))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, x_api_key: str | None = Header(None)):
    check_api_key(x_api_key)

    try:
        where_filter = build_where_filter(req.year_min, req.year_max, req.methods, req.methods_all)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    embed_model = state["embed_model"]
    collection = state["collection"]

    if req.rerank:
        candidates = retrieve(
            req.question, embed_model, collection,
            top_k=req.candidates, max_distance=None, where=where_filter,
        )
        retrieved = rerank_chunks(req.question, candidates, state["cross_encoder"], top_n=req.final_k)
        metric = "rerank_score"
    else:
        retrieved = retrieve(
            req.question, embed_model, collection,
            top_k=req.top_k, max_distance=req.max_distance, where=where_filter,
        )
        metric = "distance"

    prompt = build_prompt(req.question, retrieved)

    if req.provider == "claude":
        answer = generate_with_claude(prompt)
    else:
        answer = generate_with_ollama(prompt)

    chunks = [
        RetrievedChunk(
            source=meta.get("source", "?"),
            section=meta.get("section"),
            text=doc,
            score=float(value),
            metric=metric,
        )
        for doc, meta, value in retrieved
    ]

    return QueryResponse(
        question=req.question,
        answer=answer,
        provider_used=req.provider,
        chunks=chunks,
        filter_applied=where_filter,
    )
