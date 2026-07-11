"""
INGESTION — Étape 1 du pipeline RAG
====================================

Ce script :
1. Lit les PDF présents dans data/
2. Découpe le texte en "chunks" (morceaux) de taille raisonnable
3. Calcule un embedding (vecteur) pour chaque chunk avec un modèle local
4. Stocke tout dans une base vectorielle Chroma (locale, sur disque)

Concepts clés à retenir :
- CHUNKING : un LLM ne peut pas "lire" tout un article d'un coup dans le
  contexte de retrieval. On découpe en morceaux de ~500-800 tokens,
  avec un léger recouvrement (overlap) pour ne pas couper une idée en deux.
  Deux stratégies possibles, sélectionnables via --chunking :
    - "fixed"    : découpage aveugle en tranches de N caractères. Simple,
      rapide, mais peut couper n'importe où — au milieu d'une phrase, d'une
      formule, entre un titre et son contenu.
    - "semantic" (par défaut) : découpe d'abord le texte en SECTIONS via les
      titres numérotés détectés (ex: "2.3 Influence du lift-off"), puis
      dans chaque section, accumule des PHRASES ENTIÈRES jusqu'à la taille
      cible — jamais de coupure en plein milieu d'une phrase. Chaque chunk
      garde le titre de sa section en métadonnées.
- EMBEDDING : un modèle transforme chaque chunk de texte en un vecteur
  (ex: 384 ou 768 dimensions). Deux textes proches en sens auront des
  vecteurs proches (mesuré par similarité cosinus).
- VECTOR STORE : une base optimisée pour chercher "les vecteurs les plus
  proches" d'une requête donnée, très rapidement, même avec des millions
  d'entrées.

Modèle d'embedding choisi : intfloat/multilingual-e5-base
-> multilingue (français/anglais/arabe partiellement), tourne bien sur GPU
   ou même CPU pour des corpus de taille modeste (quelques centaines de PDF).
"""

import argparse
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from cnd_metadata import detect_methods, extract_year, has_cnd_context

DATA_DIR = Path(__file__).parent / "data"
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "ndt_papers"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 800           # caractères cibles par chunk (approximation simple, pas de tokenizer ici)
CHUNK_OVERLAP = 150        # chevauchement (en caractères) pour le chunking "fixed"
OVERLAP_SENTENCES = 2      # chevauchement (en phrases) pour le chunking "semantic"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrait le texte brut d'un PDF, page par page."""
    reader = PdfReader(str(pdf_path))
    full_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        full_text.append(text)
    return "\n".join(full_text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Découpe un texte long en morceaux avec chevauchement.

    Exemple pédagogique : avec chunk_size=800 et overlap=150,
    le chunk N+1 commence 150 caractères avant la fin du chunk N.
    Ça évite qu'une phrase importante soit coupée exactement à la frontière
    entre deux chunks sans qu'aucun des deux n'ait le contexte complet.
    """
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# --- Chunking sémantique : sections (titres numérotés) + phrases entières ---

# Détecte les titres de section numérotés (ex: "1. Introduction",
# "2.3 Influence du lift-off", "4.2 Objectifs et enjeux..."). Fonctionne
# bien sur les structures académiques classiques (thèses, articles style
# IEEE/MDPI) — mais c'est une HEURISTIQUE REGEX, pas une analyse de mise en
# page (police, taille de caractère). Limites connues :
# - peut rater un titre si l'extraction PDF le fusionne avec le paragraphe
#   suivant sur la même ligne logique
# - peut matcher un faux positif (ex: une légende de figure numérotée
#   "Figure 3.2 Something", ou une équation numérotée en début de ligne)
SECTION_HEADER_PATTERN = re.compile(
    r"^\s{0,3}(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-ZÀ-ÖØ-Þ][^\n]{2,100})$",
    re.MULTILINE,
)

# Découpage en phrases par regex : approximation, pas un vrai tokenizer
# linguistique (spaCy/nltk seraient plus robustes mais ajoutent une
# dépendance lourde pour un projet pédagogique). Limite connue : les
# abréviations ("et al.", "Fig.", "Eq.") et les nombres décimaux ("3.5")
# peuvent être interprétés à tort comme des fins de phrase.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])")


def split_into_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases (approximation par ponctuation, voir limites ci-dessus)."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Découpe le texte en sections à partir des titres numérotés détectés.
    Retourne une liste de (titre_ou_None, contenu_de_la_section).

    Si aucun titre n'est détecté (article sans structure numérotée claire,
    ou extraction PDF qui a cassé la mise en page), on retombe sur UNE seule
    section sans titre couvrant tout le texte — le chunking sémantique se
    comporte alors comme un simple découpage par phrases, sans le bonus
    "contexte de section".
    """
    matches = list(SECTION_HEADER_PATTERN.finditer(text))
    if not matches:
        return [(None, text)]

    sections = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        # Contenu avant le premier titre détecté (résumé, en-tête...) :
        # on le garde plutôt que de le perdre silencieusement.
        sections.append((None, preamble))

    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((title, content))

    return sections


def chunk_section(content: str, chunk_size: int, overlap_sentences: int) -> list[str]:
    """
    Découpe UNE section en chunks en respectant les frontières de phrases :
    on accumule des phrases entières jusqu'à approcher chunk_size, sans
    jamais couper en plein milieu d'une phrase — contrairement au découpage
    par caractères fixes, qui peut couper une formule ou une idée n'importe où.

    L'overlap se fait en PHRASES (pas en caractères) : le chunk suivant
    repart avec les `overlap_sentences` dernières phrases du chunk précédent.
    """
    sentences = split_into_sentences(content)
    if not sentences:
        return []

    chunks = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        # Cas limite : ajouter cette phrase dépasserait chunk_size. On clôt
        # le chunk courant (s'il n'est pas vide) avant de continuer.
        # Note : si UNE SEULE phrase dépasse déjà chunk_size à elle seule
        # (formule mal segmentée par l'extraction PDF), on la garde entière
        # plutôt que de la tronquer — mieux vaut un chunk trop long qu'une
        # formule coupée en deux.
        if current and current_len + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            # On ne propage l'overlap que si le chunk qu'on vient de clore
            # contenait PLUSIEURS phrases. Sinon (une seule phrase déjà plus
            # grande que chunk_size à elle seule, cas des formules mal
            # segmentées), reprendre cette même phrase comme "overlap"
            # la dupliquerait intégralement dans le chunk suivant et
            # ferait gonfler ce dernier inutilement.
            current = current[-overlap_sentences:] if overlap_sentences and len(current) > 1 else []
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def semantic_chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[tuple[str | None, str]]:
    """
    Chunking "sémantique" : structure du document (sections) + phrases
    entières, plutôt qu'une coupure aveugle en caractères.

    Retourne une liste de (titre_section_ou_None, texte_du_chunk) — le titre
    est conservé pour être stocké en métadonnée (voir main()), ce qui permet
    de savoir de quelle section provient un chunk retrouvé par le retrieval.
    """
    result = []
    for title, content in split_into_sections(text):
        for chunk in chunk_section(content, chunk_size, overlap_sentences):
            result.append((title, chunk))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Ingestion du corpus CND dans ChromaDB.")
    parser.add_argument(
        "--chunking", choices=["semantic", "fixed"], default="semantic",
        help=(
            "Stratégie de découpage : 'semantic' (défaut) respecte les phrases "
            "et la structure en sections ; 'fixed' découpe en tranches de "
            "caractères fixes (comportement d'origine, utile pour comparer)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATA_DIR.exists() or not any(DATA_DIR.glob("*.pdf")):
        print(f"⚠️  Aucun PDF trouvé dans {DATA_DIR}")
        print("   Ajoute tes articles CND/CFRP en .pdf dans ce dossier puis relance.")
        return

    print(f"📦 Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
    print("   (premier lancement = téléchargement, ~1.1 Go, ensuite mis en cache)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # On repart de zéro à chaque ingestion complète pour rester simple.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    # On force l'espace cosinus (plutôt que le L2 par défaut de Chroma) :
    # avec des embeddings normalisés (ce que fait sentence-transformers pour E5),
    # la distance cosinus est bornée dans [0, 2] et beaucoup plus facile à
    # interpréter/seuiller que le L2, dont l'échelle dépend de la norme des vecteurs.
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    print(f"📄 {len(pdf_files)} PDF trouvé(s)")
    print(f"✂️  Stratégie de chunking : {args.chunking}\n")

    all_chunks, all_ids, all_metadatas = [], [], []

    for pdf_path in tqdm(pdf_files, desc="Extraction + chunking"):
        text = extract_text_from_pdf(pdf_path)

        # Les deux stratégies renvoient une liste uniforme de
        # (titre_section_ou_None, texte_du_chunk) pour simplifier la suite —
        # le chunking "fixed" n'a simplement jamais de titre associé.
        if args.chunking == "semantic":
            titled_chunks = semantic_chunk_text(text)
        else:
            titled_chunks = [(None, chunk) for chunk in chunk_text(text)]

        chunks = [chunk for _title, chunk in titled_chunks]

        # Le contexte CND ("est-ce bien un article de contrôle non destructif")
        # est calculé UNE FOIS sur le document entier : c'est une propriété
        # globale de l'article, qui peut n'apparaître que dans le titre ou
        # l'intro, loin du chunk qui nous intéresse.
        doc_is_cnd = has_cnd_context(text)
        year = extract_year(pdf_path.name)

        # Les méthodes spécifiques, elles, sont détectées PAR CHUNK : chaque
        # chunk n'est tagué que pour ce qu'il mentionne réellement, pas pour
        # l'ensemble des méthodes évoquées ailleurs dans l'article (utile
        # pour les articles de synthèse qui passent en revue 6-7 méthodes).
        chunk_methods = [detect_methods(chunk, doc_has_cnd_context=doc_is_cnd) for chunk in chunks]

        # Résumé pédagogique : l'union des tags sur tous les chunks, pour
        # avoir une vue d'ensemble de l'article (à ne pas confondre avec les
        # métadonnées réellement stockées, qui sont par chunk).
        union_tags = sorted({tag for m in chunk_methods for tag, present in m.items() if present})
        tags_str = ", ".join(union_tags) if union_tags else "aucune méthode reconnue"
        cnd_str = "CND ✓" if doc_is_cnd else "CND ✗ (contexte non détecté)"
        year_str = str(year) if year is not None else "année inconnue"
        n_sections = len({t for t, _ in titled_chunks if t is not None})
        section_str = f", {n_sections} section(s) détectée(s)" if n_sections else ""
        tqdm.write(
            f"  📎 {pdf_path.name} → {year_str} | {cnd_str} | "
            f"{len(chunks)} chunks{section_str} | méthodes (union): {tags_str}"
        )

        for i, (title, chunk) in enumerate(titled_chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{pdf_path.stem}_chunk{i}")
            metadata = {"source": pdf_path.name, "chunk_index": i, **chunk_methods[i]}
            if year is not None:
                # Chroma n'accepte pas None comme valeur de métadonnée ;
                # on omet simplement la clé si l'année n'a pas été détectée.
                metadata["year"] = year
            if title is not None:
                metadata["section"] = title
            all_metadatas.append(metadata)

    print(f"✂️  {len(all_chunks)} chunks générés au total")

    print("🧮 Calcul des embeddings...")
    # E5 recommande de préfixer les passages avec "passage: " (convention du modèle)
    prefixed_chunks = [f"passage: {c}" for c in all_chunks]
    embeddings = model.encode(prefixed_chunks, show_progress_bar=True, batch_size=32)

    print("💾 Écriture dans ChromaDB...")
    # Chroma limite la taille des batchs d'insertion, on découpe par sécurité
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            ids=all_ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=all_chunks[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size],
        )

    print(f"✅ Terminé. Base vectorielle stockée dans {DB_DIR}")
    print(f"   Collection '{COLLECTION_NAME}' : {collection.count()} chunks indexés")


if __name__ == "__main__":
    main()
