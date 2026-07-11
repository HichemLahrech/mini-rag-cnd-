"""
INSPECT_METADATA — Diagnostic des métadonnées indexées
=========================================================

Utilitaire de débogage : liste, pour chaque document source, l'année et
l'UNION des méthodes CND détectées sur l'ensemble de ses chunks. Utile
quand un filtre --year-min/--year-max/--methods renvoie 0 résultat et
qu'on veut savoir pourquoi (mauvaise année détectée ? mot-clé non trouvé
sur ce PDF ?) sans repasser par tout le pipeline d'ingestion.

Note : depuis le passage au tagging PAR CHUNK (chaque chunk n'est tagué
que pour ce qu'il mentionne réellement, voir ingest.py), un même document
a des métadonnées DIFFÉRENTES d'un chunk à l'autre. Ce script affiche donc
l'union sur tous les chunks pour donner une vue d'ensemble de l'article,
mais garde à l'esprit que --methods filtre au niveau chunk : un chunk
individuel ne portera qu'un sous-ensemble de ce qui est listé ici.

Usage :
    python inspect_metadata.py
"""

import argparse

import chromadb

DB_DIR = "chroma_db"
COLLECTION_NAME = "ndt_papers"


def main():
    parser = argparse.ArgumentParser(description="Inspecte les métadonnées indexées dans ChromaDB.")
    parser.add_argument(
        "--detail", action="store_true",
        help="Affiche aussi le détail chunk par chunk (pas juste l'union par document)",
    )
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    # On récupère TOUTES les métadonnées (pas de recherche sémantique ici,
    # juste un accès direct aux données stockées).
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data["metadatas"]

    # Regroupement par source : on accumule l'union des méthodes vues sur
    # TOUS les chunks de chaque document (et pas juste le premier rencontré),
    # puisque le tagging varie maintenant d'un chunk à l'autre.
    by_source = {}
    for meta in metadatas:
        source = meta.get("source", "?")
        entry = by_source.setdefault(
            source, {"year": meta.get("year"), "tags": set(), "n_chunks": 0, "chunks": []}
        )
        entry["n_chunks"] += 1
        if entry["year"] is None and meta.get("year") is not None:
            entry["year"] = meta["year"]
        chunk_tags = [k for k, v in meta.items() if k not in ("source", "chunk_index", "year") and v is True]
        entry["tags"].update(chunk_tags)
        entry["chunks"].append((meta.get("chunk_index", "?"), chunk_tags))

    print(f"📚 {len(by_source)} document(s) distinct(s), {len(metadatas)} chunks au total\n")

    for source, entry in sorted(by_source.items()):
        year = entry["year"] if entry["year"] is not None else "❓ non détectée"
        tags_str = ", ".join(sorted(entry["tags"])) if entry["tags"] else "aucune méthode reconnue"
        print(f"  📎 {source}  ({entry['n_chunks']} chunks)")
        print(f"      année            : {year}")
        print(f"      méthodes (union) : {tags_str}")

        if args.detail:
            for chunk_index, chunk_tags in sorted(entry["chunks"], key=lambda x: x[0]):
                tags_display = ", ".join(chunk_tags) if chunk_tags else "—"
                print(f"        chunk {chunk_index:>3} : {tags_display}")
        print()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
