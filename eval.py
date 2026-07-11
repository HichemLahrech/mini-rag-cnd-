"""
EVAL — Évaluation objective du retrieval
==========================================

Ce script mesure la qualité du retrieval sur un petit jeu de questions
annotées manuellement (eval_set.json), plutôt que de juger "à l'œil" comme
on l'a fait jusqu'ici sur des questions ponctuelles.

Concepts clés :

- GROUND TRUTH (vérité terrain) : pour chaque question, on connaît à
  l'avance le(s) document(s) source(s) qui DEVRAIENT être retrouvés parmi
  les résultats. Ici, la vérité terrain est établie AU NIVEAU DOCUMENT
  (pas au niveau chunk précis) — simplification assumée : annoter au chunk
  près serait plus rigoureux mais bien plus coûteux en temps pour vérifier
  manuellement chaque source. Un "hit" signifie donc "au moins un chunk du
  bon document est remonté", pas "le chunk exact qui répond est remonté".

- RECALL@K : sur les K chunks retournés pour une question, a-t-on retrouvé
  AU MOINS UN chunk provenant d'un document attendu ? Mesure si le bon
  document est "quelque part" dans les résultats, sans regarder où.
- PRECISION@K : parmi les K chunks retournés, quelle proportion provient
  d'un document attendu ? Mesure si les résultats sont "propres" (peu de
  bruit d'autres documents) plutôt que juste "le bon doc est là, perdu
  au milieu d'un tas d'autres".
- MRR (Mean Reciprocal Rank) : 1 / rang du PREMIER chunk pertinent, moyenné
  sur toutes les questions. Si le bon document apparaît en position 1 dans
  les résultats → contribution de 1.0 (parfait). En position 3 → 0.33.
  Sanctionne fortement le fait de devoir "chercher loin" dans les résultats,
  même quand le bon document finit par y être (contrairement au recall qui
  ne voit pas la différence entre position 1 et position 5).

Limite assumée, à ne pas perdre de vue : 8 questions sur 5 documents, c'est
un TRÈS petit échantillon — largement insuffisant pour des conclusions
statistiquement solides ("le retrieval est bon à 87%" n'aurait aucun sens
avec si peu de données). Utile pour repérer des RÉGRESSIONS grossières
(un changement de config qui casse visiblement le retrieval sur telle
question) et comparer deux configurations entre elles, pas pour certifier
une performance absolue.

Usage :
    python eval.py                    # retrieval brut, top_k=5
    python eval.py --rerank           # avec reranking (comme query.py --rerank)
    python eval.py --top-k 3 --rerank
"""

import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from query import retrieve, rerank, RERANK_MODEL, EMBEDDING_MODEL, DB_DIR, COLLECTION_NAME

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def evaluate_question(
    question_data: dict,
    embed_model: SentenceTransformer,
    collection,
    cross_encoder: CrossEncoder | None,
    top_k: int,
    use_rerank: bool,
) -> dict:
    question = question_data["question"]
    expected_sources = set(question_data["expected_sources"])

    if use_rerank:
        # Même logique que query.py : sur-récupération large, puis reranking
        # précis pour ne garder que le top_k final.
        candidates = retrieve(question, embed_model, collection, top_k=20, max_distance=None)
        results = rerank(question, candidates, cross_encoder, top_n=top_k)
    else:
        results = retrieve(question, embed_model, collection, top_k=top_k, max_distance=None)

    retrieved_sources = [meta["source"] for _doc, meta, _val in results]

    hit = any(src in expected_sources for src in retrieved_sources)

    n_relevant = sum(1 for src in retrieved_sources if src in expected_sources)
    precision = n_relevant / len(retrieved_sources) if retrieved_sources else 0.0

    reciprocal_rank = 0.0
    for rank_position, src in enumerate(retrieved_sources, start=1):
        if src in expected_sources:
            reciprocal_rank = 1.0 / rank_position
            break

    return {
        "question": question,
        "expected_sources": sorted(expected_sources),
        "retrieved_sources": retrieved_sources,
        "hit": hit,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
    }


def main():
    parser = argparse.ArgumentParser(description="Évalue le retrieval sur eval_set.json.")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre de chunks évalués par question (défaut: 5)")
    parser.add_argument("--rerank", action="store_true", help="Active le reranking (comme dans query.py --rerank)")
    args = parser.parse_args()

    eval_set = load_eval_set()
    print(f"📋 {len(eval_set)} question(s) dans le jeu d'évaluation\n")

    print(f"📦 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    cross_encoder = None
    if args.rerank:
        print(f"📦 Chargement du cross-encoder : {RERANK_MODEL}")
        cross_encoder = CrossEncoder(RERANK_MODEL)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"📚 Collection chargée : {collection.count()} chunks\n")

    results = []
    for question_data in eval_set:
        result = evaluate_question(question_data, embed_model, collection, cross_encoder, args.top_k, args.rerank)
        results.append(result)

        status = "✅" if result["hit"] else "❌"
        print(f"{status} {result['question'][:90]}")
        print(f"    attendu : {result['expected_sources']}")
        print(f"    trouvé  : {result['retrieved_sources']}")
        print(f"    precision@{args.top_k}={result['precision']:.2f} | RR={result['reciprocal_rank']:.2f}\n")

    n = len(results)
    recall_at_k = sum(r["hit"] for r in results) / n
    mean_precision = sum(r["precision"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n

    print("=" * 60)
    print(f"📊 RÉSULTATS AGRÉGÉS (top_k={args.top_k}, rerank={args.rerank})")
    print(f"   Recall@{args.top_k}            : {recall_at_k:.1%}  ({sum(r['hit'] for r in results)}/{n} questions)")
    print(f"   Precision@{args.top_k} (moyenne) : {mean_precision:.1%}")
    print(f"   MRR                      : {mrr:.3f}")


if __name__ == "__main__":
    main()
