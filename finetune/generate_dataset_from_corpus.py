"""
GENERATE_DATASET_FROM_CORPUS — Dataset QLoRA généré depuis ton corpus CND
============================================================================

Contrairement à prepare_dataset.py (dataset "jouet" générique écrit à la
main), ce script PUISE directement dans ta base ChromaDB déjà indexée
(chroma_db/, construite par ingest.py) pour générer un dataset d'entraînement
ancré dans ton domaine CND réel.

Pipeline :
1. Pour chaque méthode CND connue (voir cnd_metadata.py), on récupère
   quelques chunks réellement tagués pour cette méthode.
2. Pour chaque chunk, on demande à ton modèle Ollama LOCAL (déjà utilisé
   dans query.py — pas de nouvelle dépendance) de générer UNE question
   que ce passage permet de répondre, plus une réponse concise basée
   UNIQUEMENT sur ce passage.
3. On formate chaque paire (question, réponse) au même format JSON strict
   que l'exercice précédent — {"reponse": ..., "methode": ..., "certitude":
   ...} — avec le tag de méthode réellement détecté par ton pipeline RAG
   (cnd_metadata.py), pas une donnée bidon : c'est un vrai réemploi de ce
   qu'on a construit ensemble.
4. Écrit data/train.jsonl et data/val.jsonl — EXACTEMENT le même format que
   prepare_dataset.py, donc train_qlora.py n'a besoin d'aucune modification.

⚠️ LIMITE MAJEURE À COMPRENDRE : la génération des questions/réponses est
elle-même faite par un LLM (ton Ollama local, probablement un 8B). La
QUALITÉ du dataset d'entraînement dépend donc directement de la qualité de
CE modèle générateur — s'il comprend mal un passage technique ou invente
un détail, cette erreur se retrouve dans le dataset et sera "apprise" par
le fine-tuning. Pour un usage sérieux (au-delà de l'exercice pédagogique),
il faudrait relire manuellement les paires générées avant l'entraînement.
Ce script imprime chaque paire générée pour te permettre cette relecture.

⚠️ AUTRE LIMITE : les modèles 8B respectent imparfaitement une consigne de
sortie JSON stricte. On filtre et ignore les générations mal formées (voir
compteur "échecs de parsing" en fin d'exécution) plutôt que de planter.

Prérequis :
- avoir lancé ingest.py au moins une fois (chroma_db/ non vide)
- Ollama lancé en arrière-plan avec le modèle déjà utilisé dans query.py
- Ce script réutilise chromadb/requests/transformers déjà installés pour
  le RAG et pour prepare_dataset.py — pas de nouvelle dépendance à ajouter
  à requirements-finetune.txt
"""

import json
import random
import sys
from pathlib import Path

import chromadb
import requests

# On importe cnd_metadata.py depuis le dossier parent (rag_ndt/) : on
# réutilise la LISTE DES MÉTHODES CND telle que définie par le pipeline
# RAG existant, plutôt que de la redupliquer ici — une seule source de
# vérité pour "quelles méthodes CND on reconnaît".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnd_metadata import METHOD_KEYWORDS  # noqa: E402

DB_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "ndt_papers"  # doit rester synchronisé avec ingest.py

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # doit correspondre au modèle utilisé dans query.py

BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # doit correspondre au modèle ciblé par train_qlora.py
SYSTEM_PROMPT = "Tu es un assistant qui répond aux questions de contrôle non destructif (CND)."

DATA_DIR = Path(__file__).parent / "data"
VAL_FRACTION = 0.15
SEED = 42

CHUNKS_PER_METHOD = 6       # nombre de chunks tirés par méthode CND (augmenté pour compenser le chevauchement de tags)
MIN_CHUNK_LENGTH = 250      # ignore les chunks trop courts, peu informatifs pour générer une bonne question
OVERFETCH_FACTOR = 6        # sur-récupération avant filtrage/dédup (voir fetch_chunks_for_method)


GENERATION_PROMPT_TEMPLATE = """Voici un extrait d'un article scientifique sur le contrôle non destructif (CND) :

---
{chunk}
---

Génère UNE question en français à laquelle cet extrait répond précisément,
et une réponse concise (2 à 4 phrases) basée UNIQUEMENT sur cet extrait
(n'invente aucune information absente du texte).

Réponds STRICTEMENT en JSON, sans aucun texte avant ou après, sur ce format :
{{"question": "...", "reponse": "..."}}"""


def fetch_chunks_for_method(collection, method: str, limit: int) -> list[dict]:
    """
    Récupère jusqu'à `limit` chunks tagués True pour cette méthode CND.

    On sur-récupère (limit * OVERFETCH_FACTOR) puis on MÉLANGE avant de
    couper à `limit` : Chroma renvoie ses résultats dans un ordre stable
    (proche de l'ordre d'insertion), donc sans ce mélange, on retomberait
    systématiquement sur les mêmes premiers chunks à chaque run — un souci
    concret ici car plusieurs méthodes se chevauchent souvent sur un même
    chunk (ex: une phrase qui énumère 5-6 méthodes CND d'un coup), et la
    déduplication (voir seen_chunk_texts dans main()) finit par affamer les
    catégories traitées plus tard si on prend toujours les mêmes candidats.
    """
    result = collection.get(
        where={method: True},
        include=["documents", "metadatas"],
        limit=limit * OVERFETCH_FACTOR,
    )
    candidates = [
        {"text": doc, "source": meta.get("source", "?"), "method": method}
        for doc, meta in zip(result["documents"], result["metadatas"])
        if len(doc) >= MIN_CHUNK_LENGTH
    ]
    random.shuffle(candidates)
    return candidates[:limit]


def generate_qa_from_chunk(chunk_text: str) -> dict | None:
    """
    Appelle Ollama pour générer une paire (question, réponse) à partir d'un
    chunk. Retourne None si la génération échoue ou si le JSON est mal formé
    (voir limite documentée en haut de fichier).
    """
    prompt = GENERATION_PROMPT_TEMPLATE.format(chunk=chunk_text)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},  # un peu de variabilité, mais pas trop (on veut du factuel)
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️  Erreur de connexion Ollama : {e}")
        return None

    raw = response.json()["response"].strip()

    # Les modèles 8B enrobent parfois leur JSON dans des balises markdown
    # ```json ... ``` malgré la consigne "STRICTEMENT en JSON" — on nettoie
    # avant de parser plutôt que d'échouer bêtement sur ce cas fréquent.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()

    try:
        parsed = json.loads(raw)
        if "question" not in parsed or "reponse" not in parsed:
            return None
        return parsed
    except json.JSONDecodeError:
        return None


def build_answer_json(reponse: str, methode: str) -> str:
    # certitude fixée à "elevee" : contrairement au dataset trivia générique,
    # ici la réponse est directement ancrée dans un extrait réel du corpus
    # (pas une connaissance générale du modèle générateur) — simplification
    # assumée, pas une vraie mesure de confiance calibrée.
    return json.dumps({"reponse": reponse, "methode": methode, "certitude": "elevee"}, ensure_ascii=False)


def build_messages(question: str, answer_json: str) -> list[dict]:
    """
    Construit la liste de messages (system/user/assistant) au format attendu
    par trl. Pas de rendu du chat template ici — SFTTrainer s'en charge lui-
    même au moment de l'entraînement (voir train_qlora.py), avec masquage
    automatique du prompt via completion_only_loss=True.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_json},
    ]


def main():
    print(f"📚 Connexion à la base ChromaDB : {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"   {collection.count()} chunks disponibles\n")

    print(f"🤖 Génération des questions/réponses via Ollama ({OLLAMA_MODEL})...")
    print(f"   (vérifie qu'Ollama tourne en arrière-plan avant de lancer ce script)\n")

    seen_chunk_texts = set()  # évite de générer deux fois sur le même chunk (chevauchement entre tags)
    examples = []
    n_attempted = 0
    n_failed_parsing = 0

    # On mélange l'ordre de traitement des méthodes (au lieu de l'ordre
    # alphabétique fixe) : sans ça, les premières méthodes traitées
    # "gagnent" systématiquement les chunks partagés entre plusieurs tags,
    # affamant toujours les mêmes catégories en fin de liste.
    random.seed(SEED)
    methods_order = sorted(METHOD_KEYWORDS)
    random.shuffle(methods_order)

    for method in methods_order:
        candidates = fetch_chunks_for_method(collection, method, CHUNKS_PER_METHOD)
        print(f"📎 {method} : {len(candidates)} chunk(s) candidat(s)")

        for candidate in candidates:
            if candidate["text"] in seen_chunk_texts:
                continue
            seen_chunk_texts.add(candidate["text"])

            n_attempted += 1
            qa = generate_qa_from_chunk(candidate["text"])
            if qa is None:
                n_failed_parsing += 1
                print(f"    ❌ échec de génération/parsing sur un chunk de {candidate['source']}")
                continue

            print(f"    ✅ {qa['question'][:80]}")
            answer_json = build_answer_json(qa["reponse"], method)
            messages = build_messages(qa["question"], answer_json)
            examples.append({"messages": messages})

    print(f"\n📊 {len(examples)} exemples générés avec succès sur {n_attempted} tentatives "
          f"({n_failed_parsing} échecs de parsing JSON)")

    if len(examples) < 10:
        print("⚠️  Peu d'exemples générés — le dataset risque d'être insuffisant pour un "
              "fine-tuning utile. Vérifie qu'Ollama répond correctement, ou augmente "
              "CHUNKS_PER_METHOD.")

    random.seed(SEED)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * VAL_FRACTION))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    DATA_DIR.mkdir(exist_ok=True)
    for name, subset in [("train.jsonl", train_examples), ("val.jsonl", val_examples)]:
        path = DATA_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            for ex in subset:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"✅ {path} : {len(subset)} exemples")


if __name__ == "__main__":
    main()
