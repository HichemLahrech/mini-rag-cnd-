"""
QUERY — Étape 2 du pipeline RAG (retrieval + génération)
==========================================================

Ce script :
1. Prend ta question
2. La transforme en embedding avec le MÊME modèle que pour l'ingestion
3. Cherche les chunks les plus proches dans ChromaDB (retrieval)
4. Construit un prompt avec ces chunks comme "contexte"
5. Envoie ce prompt à Claude pour générer une réponse ancrée sur tes sources

Concept clé : le LLM ne "sait" rien de ton corpus par défaut. On lui injecte
les passages pertinents directement dans le prompt (in-context learning),
c'est ça le "R" de RAG (Retrieval-Augmented Generation).

Prérequis :
- avoir lancé ingest.py au moins une fois
- variable d'environnement ANTHROPIC_API_KEY définie
  (ou remplace get_llm_answer() par un appel à un LLM local via Ollama,
  voir la note en bas du fichier)
"""

import argparse

import requests
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from cnd_metadata import METHOD_KEYWORDS

DB_DIR = "chroma_db"
COLLECTION_NAME = "ndt_papers"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
TOP_K = 5  # nombre de chunks récupérés par requête (utilisé seulement si --rerank est désactivé)

# Seuil de distance cosinus (0 = identique, 2 = opposé) au-delà duquel on
# écarte un chunk même s'il fait partie du top_k. À calibrer empiriquement :
# lance quelques questions, regarde les distances affichées, et ajuste.
# None = pas de filtrage (comportement d'origine, top_k pur).
# Ignoré si --rerank est actif : c'est le cross-encoder qui fait le tri fin.
MAX_DISTANCE = None

# --- Reranking (bi-encoder large + cross-encoder précis) ---
# Modèle multilingue léger (mMARCO couvre le français) : bon compromis
# qualité/vitesse pour un premier test sur RTX 4070 Ti. Pour plus de
# précision (au prix de la vitesse), voir "BAAI/bge-reranker-v2-m3".
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CANDIDATE_K = 20  # nombre de candidats remontés par le bi-encoder avant reranking
FINAL_K = 5       # nombre de chunks finalement injectés dans le prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # doit correspondre au modèle téléchargé avec `ollama pull`


def build_where_filter(
    year_min: int | None,
    year_max: int | None,
    methods_any: list[str] | None,
    methods_all: list[str] | None = None,
) -> dict | None:
    """
    Construit un filtre de métadonnées Chroma (paramètre `where`) à partir
    des options CLI --year-min/--year-max/--methods/--methods-all.

    Ce filtre est un filtre EXACT appliqué par Chroma AVANT le calcul de
    similarité — il réduit l'espace de recherche à un sous-ensemble de
    chunks, sur lequel la similarité sémantique s'applique ensuite. C'est
    différent de max_distance qui filtre APRÈS coup sur le score.

    - methods_any (--methods)     : OR — chunks traitant de A OU B
    - methods_all (--methods-all) : AND — chunks traitant de A ET B à LA FOIS
      Utile pour les questions de "combinaison" de méthodes (ex: une technique
      hybride courants de Foucault + thermographie) : le bi-encoder capture
      mal la notion de combinaison (il voit juste deux sujets proches), alors
      qu'un filtre exact ET cible directement les chunks tagués pour les deux
      méthodes simultanément — beaucoup plus précis dans ce cas précis.
    - Année + méthodes combinées = AND (dans la période ET sur ces méthodes)
    """
    conditions = []

    if year_min is not None and year_max is not None:
        conditions.append({"$and": [{"year": {"$gte": year_min}}, {"year": {"$lte": year_max}}]})
    elif year_min is not None:
        conditions.append({"year": {"$gte": year_min}})
    elif year_max is not None:
        conditions.append({"year": {"$lte": year_max}})

    def _validate(methods):
        unknown = [m for m in methods if m not in METHOD_KEYWORDS]
        if unknown:
            raise ValueError(
                f"Méthode(s) inconnue(s) : {unknown}. "
                f"Valeurs possibles : {sorted(METHOD_KEYWORDS)}"
            )

    if methods_any:
        _validate(methods_any)
        method_conditions = [{m: True} for m in methods_any]
        conditions.append(method_conditions[0] if len(method_conditions) == 1 else {"$or": method_conditions})

    if methods_all:
        _validate(methods_all)
        method_conditions = [{m: True} for m in methods_all]
        conditions.append(method_conditions[0] if len(method_conditions) == 1 else {"$and": method_conditions})

    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def retrieve(
    question: str,
    embed_model: SentenceTransformer,
    collection,
    top_k: int = TOP_K,
    max_distance: float | None = MAX_DISTANCE,
    where: dict | None = None,
):
    """
    Retourne les chunks les plus proches sémantiquement de la question.

    On récupère d'abord les top_k par similarité (ANN via HNSW) PARMI les
    chunks satisfaisant `where` (filtre exact sur les métadonnées, ex:
    année ou méthode CND), puis on filtre ceux dont la distance dépasse
    max_distance. Ça évite de forcer systématiquement K chunks dans le
    prompt quand certains sont hors-sujet (ex: question très pointue avec
    un seul passage vraiment pertinent).

    Attention : ce filtrage suppose une collection créée en espace cosinus
    (voir ingest.py, metadata={"hnsw:space": "cosine"}). Avec le L2 par
    défaut de Chroma, l'échelle des distances est différente et un seuil
    calibré ici n'aurait plus le même sens.
    """
    # E5 recommande de préfixer les requêtes avec "query: "
    query_embedding = embed_model.encode([f"query: {question}"])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    triples = list(zip(docs, metadatas, distances))

    if max_distance is not None:
        kept = [t for t in triples if t[2] <= max_distance]
        if not kept:
            # Sécurité pédagogique : plutôt que de renvoyer un contexte vide
            # (le LLM halluciner sans le savoir), on garde au moins le
            # meilleur chunk trouvé, même s'il dépasse le seuil.
            kept = triples[:1]
        return kept

    return triples


def rerank(question: str, candidates, cross_encoder: CrossEncoder, top_n: int = FINAL_K):
    """
    Réordonne une liste de candidats avec un cross-encoder.

    Contrairement au bi-encoder (retrieval initial), le cross-encoder reçoit
    la question ET le chunk ensemble, et produit un score de pertinence par
    attention croisée entre les deux textes — beaucoup plus précis, mais trop
    coûteux pour tourner sur tout le corpus (d'où le "candidates" en entrée :
    on ne rerank qu'une petite liste déjà présélectionnée).

    Retourne des triplets (doc, meta, score) triés du plus au moins pertinent.
    Attention : ce score n'a PAS la même échelle que la distance cosinus du
    bi-encoder — c'est un logit brut du cross-encoder (pas de plage fixe
    type [0,2]), et plus grand = plus pertinent (inverse de la distance, où
    plus petit = plus proche).
    """
    docs = [c[0] for c in candidates]
    pairs = [(question, doc) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(doc, meta, float(score)) for (doc, meta, _dist), score in scored[:top_n]]


def build_prompt(question: str, retrieved_chunks) -> str:
    """Construit le prompt final avec le contexte récupéré."""
    context_blocks = []
    for i, (doc, meta, dist) in enumerate(retrieved_chunks):
        context_blocks.append(f"[Source {i+1} — {meta['source']}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Tu es un assistant de recherche spécialisé en contrôle non destructif (CND).
Réponds à la question en te basant UNIQUEMENT sur les extraits fournis ci-dessous.
Cite la source (ex: [Source 1]) pour chaque affirmation importante.

IMPORTANT — Les extraits sont souvent en ANGLAIS alors que la question est en
FRANÇAIS (le corpus scientifique CND est majoritairement publié en anglais).
Avant de conclure qu'une information est absente, vérifie systématiquement
si un terme technique équivalent en anglais apparaît dans les extraits.
Quelques correspondances courantes (liste non exhaustive) :
- courants de Foucault = eddy current
- contrôle non destructif (CND) = non-destructive testing (NDT)
- thermographie = thermography
- ultrasons / contrôle ultrasonore = ultrasonic testing (UT)
- essai / contrôle = testing / inspection
- fissure = crack ; défaut = defect ; profondeur = depth

Une méthode ou technique combinée peut apparaître sous un acronyme spécifique
dans les extraits (ex: ECPT = Eddy Current Pulsed Thermography = une méthode
qui combine courants de Foucault ET thermographie). Cherche ces acronymes
avant de conclure à l'absence de réponse.

Si, malgré cette vérification terminologique, les extraits ne contiennent
vraiment pas la réponse, dis-le clairement au lieu d'inventer.

Extraits disponibles :
{context}

Question : {question}

Réponse :"""
    return prompt


def get_llm_answer(prompt: str) -> str:
    """Appelle le modèle local via Ollama avec le prompt augmenté du contexte."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (
            "❌ Impossible de se connecter à Ollama. Vérifie qu'Ollama est bien lancé "
            "(il doit tourner en arrière-plan après l'installation) et que le modèle "
            f"'{OLLAMA_MODEL}' est téléchargé (`ollama pull {OLLAMA_MODEL}`)."
        )
    return response.json()["response"]


def parse_args():
    parser = argparse.ArgumentParser(description="Interroge le corpus CND via RAG.")
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Nombre de chunks récupérés avant filtrage (défaut: {TOP_K})",
    )
    parser.add_argument(
        "--max-distance", type=float, default=MAX_DISTANCE,
        help=(
            "Seuil de distance cosinus [0-2] au-delà duquel un chunk est "
            "écarté (défaut: pas de seuil). Regarde les distances affichées "
            "sur quelques questions types pour calibrer une bonne valeur "
            "(typiquement quelque part entre 0.3 et 0.6 selon le corpus)."
        ),
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help=(
            "Active le reranking par cross-encoder : récupère --candidates "
            "chunks par similarité brute, puis les réordonne finement pour "
            "n'en garder que --final-k. Plus précis mais plus lent que le "
            "retrieval seul (--top-k / --max-distance, utilisés si --rerank "
            "est absent)."
        ),
    )
    parser.add_argument(
        "--candidates", type=int, default=CANDIDATE_K,
        help=f"Nombre de candidats remontés avant reranking, si --rerank (défaut: {CANDIDATE_K})",
    )
    parser.add_argument(
        "--final-k", type=int, default=FINAL_K,
        help=f"Nombre de chunks finaux injectés dans le prompt, si --rerank (défaut: {FINAL_K})",
    )
    parser.add_argument(
        "--year-min", type=int, default=None,
        help="Ne garder que les articles publiés à partir de cette année",
    )
    parser.add_argument(
        "--year-max", type=int, default=None,
        help="Ne garder que les articles publiés jusqu'à cette année",
    )
    parser.add_argument(
        "--methods", type=str, default=None,
        help=(
            "Filtrer par méthode(s) CND détectée(s), séparées par des virgules "
            f"(OR entre elles : A ou B). Valeurs possibles : {sorted(METHOD_KEYWORDS)}"
        ),
    )
    parser.add_argument(
        "--methods-all", type=str, default=None,
        help=(
            "Comme --methods, mais en ET (A ET B à la fois dans le même chunk). "
            "Utile pour cibler une technique hybride/combinée (ex: eddy_current,"
            "thermographie pour trouver les chunks parlant des deux ensemble)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    methods_list = [m.strip() for m in args.methods.split(",")] if args.methods else None
    methods_all_list = [m.strip() for m in args.methods_all.split(",")] if args.methods_all else None
    try:
        where_filter = build_where_filter(args.year_min, args.year_max, methods_list, methods_all_list)
    except ValueError as e:
        print(f"❌ {e}")
        return

    print(f"📦 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    cross_encoder = None
    if args.rerank:
        print(f"📦 Chargement du cross-encoder de reranking : {RERANK_MODEL}")
        print("   (premier lancement = téléchargement, ensuite mis en cache)")
        cross_encoder = CrossEncoder(RERANK_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"📚 Collection chargée : {collection.count()} chunks disponibles")
    if where_filter is not None:
        print(f"🔎 Filtre métadonnées actif : {where_filter}")
    if args.rerank:
        print(f"⚙️  reranking actif | candidates={args.candidates} | final_k={args.final_k}\n")
    else:
        print(f"⚙️  reranking désactivé | top_k={args.top_k} | max_distance={args.max_distance}\n")

    print("Pose ta question (ou 'quit' pour sortir) :\n")
    while True:
        question = input("❓ > ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        if args.rerank:
            # On sur-récupère largement (pas de filtre par distance ici :
            # on laisse le cross-encoder faire le tri fin plus bas).
            candidates = retrieve(
                question, embed_model, collection,
                top_k=args.candidates, max_distance=None, where=where_filter,
            )
            retrieved = rerank(question, candidates, cross_encoder, top_n=args.final_k)
            metric_label = "score"
        else:
            retrieved = retrieve(
                question, embed_model, collection,
                top_k=args.top_k, max_distance=args.max_distance, where=where_filter,
            )
            metric_label = "distance"

        print(f"\n🔍 {len(retrieved)} chunk(s) retenu(s) (par ordre de pertinence) :")
        for i, (doc, meta, value) in enumerate(retrieved):
            print(f"  [{i+1}] {meta['source']} ({metric_label}={value:.3f}) — {doc[:100]}...")

        prompt = build_prompt(question, retrieved)
        print("\n🤖 Génération de la réponse...\n")
        answer = get_llm_answer(prompt)
        print(f"💬 {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# NOTE — Utiliser l'API Claude au lieu d'Ollama :
#
# Si tu préfères la qualité de Claude plutôt qu'un modèle local, et que
# tu as une clé API Anthropic :
#   1. `pip install anthropic`
#   2. `set ANTHROPIC_API_KEY=ta-clé` (Windows) ou `export ANTHROPIC_API_KEY=ta-clé` (Linux/Mac)
#   3. Remplace get_llm_answer() par :
#
#      from anthropic import Anthropic
#      client_anthropic = Anthropic()
#      def get_llm_answer(prompt: str) -> str:
#          response = client_anthropic.messages.create(
#              model="claude-sonnet-4-6",
#              max_tokens=1000,
#              messages=[{"role": "user", "content": prompt}],
#          )
#          return response.content[0].text
# ------------------------------------------------------------------
