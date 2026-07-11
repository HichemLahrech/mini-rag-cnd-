"""
CND_METADATA — Détection de métadonnées enrichies (année, méthode CND)
========================================================================

Module partagé entre ingest.py et query.py, pour éviter que la logique de
détection dérive entre les deux (si un jour tu ajoutes une méthode CND, tu
ne la modifies qu'ici).

Concept clé : les métadonnées enrichies transforment le retrieval "sémantique
pur" en retrieval "sémantique + filtré". Exemple : au lieu de laisser la
similarité vectorielle seule décider, tu peux forcer "cherche uniquement
parmi les articles sur les courants de Foucault publiés après 2018". C'est
un filtre EXACT (booléen), complémentaire à la similarité qui elle est
approximative.

Limite de l'approche par mots-clés : c'est un classifieur naïf, pas un LLM.
Un article peut mentionner "GMR" une seule fois en passant sans que ce soit
son sujet principal, et sera quand même tagué gmr=True. Pour une détection
plus fine, il faudrait classifier via un LLM (mais ça coûte cher pour un
corpus de recherche qu'on relance souvent) — acceptable comme point de départ.

Garde-fou "contexte CND" : certains termes techniques (courants de Foucault,
ultrasons, thermographie...) désignent un PHÉNOMÈNE PHYSIQUE utilisé dans
plein de contextes non-CND (ex: le chauffage par induction utilise aussi les
courants de Foucault, sans que l'article soit un article de contrôle non
destructif). Pour limiter ce faux positif, un tag n'est validé QUE SI le
document mentionne aussi un marqueur générique de CND (voir CND_CONTEXT_MARKERS)
quelque part dans le texte. Ce n'est pas parfait (un article CND pourrait en
théorie n'utiliser aucun de ces marqueurs), mais ça réduit nettement le bruit
sans complexifier la détection par mot-clé individuel.
"""

import re

# Chaque tag est un nom de champ booléen dans les métadonnées Chroma.
# Les mots-clés sont cherchés en minuscules, sans tenir compte des accents
# n'est PAS géré ici (voir _normalize) pour rester simple.
METHOD_KEYWORDS = {
    "eddy_current": [
        "courants de foucault", "courant de foucault", "eddy current", "eddy-current",
    ],
    "gmr": [
        "gmr", "magnetoresistance", "magnétorésistance", "giant magnetoresistance",
    ],
    "ultrasons": [
        "ultrason", "ultrasonic", "ultrasound",
    ],
    "thermographie": [
        "thermographie", "thermography", "infrared thermography", "thermal imaging",
    ],
    "radiographie": [
        "radiographie", "radiography", "rayons x", "x-ray", "rayons-x",
    ],
    "inspection_visuelle": [
        "inspection visuelle", "visual inspection", "uav", "drone inspection",
    ],
    "mfl": [
        "mfl", "magnetic flux leakage", "fuite de flux magnétique",
    ],
    "acfm": [
        "acfm", "alternating current field measurement",
    ],
    "emat": [
        "emat", "electro-magnetic acoustic transducer", "electromagnetic acoustic transducer",
    ],
}

# Marqueurs génériques signalant qu'on est bien dans un contexte de contrôle
# non destructif (et pas, par ex., de chauffage industriel ou d'imagerie
# médicale). \b force une limite de mot pour éviter qu'un acronyme court
# comme "cnd" ne matche à l'intérieur d'un autre mot.
CND_CONTEXT_MARKERS = [
    r"contrôle non destructif", r"contr[oô]le non-destructif",
    r"essai non destructif", r"évaluation non destructive",
    r"\bcnd\b", r"\bndt\b",
    r"non[- ]destructive testing", r"non[- ]destructive evaluation",
    r"inspection non destructive", r"détection de défaut",
]
CND_CONTEXT_PATTERN = re.compile("|".join(CND_CONTEXT_MARKERS))

# Pour être exploitable par un filtre Chroma "$gte"/"$lte", l'année doit être
# un entier. On l'extrait du nom de fichier (convention observée dans le
# corpus : "... 2019.pdf", "... 2023.pdf").
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def _normalize(text: str) -> str:
    """Minuscule, espaces compressés. Pas de gestion des accents (limite connue)."""
    return " ".join(text.lower().split())


def has_cnd_context(text: str) -> bool:
    """
    Indique si un texte (typiquement un DOCUMENT ENTIER, pas un chunk isolé)
    mentionne quelque part un marqueur générique de contrôle non destructif.

    Calculé au niveau document plutôt que chunk car c'est une propriété
    globale de l'article : la phrase "contrôle non destructif" peut très bien
    n'apparaître que dans le titre ou l'introduction, alors que le chunk qui
    nous intéresse (ex: la description technique d'un capteur) n'en parle pas
    explicitement mais fait bien partie d'un article CND.
    """
    return bool(CND_CONTEXT_PATTERN.search(_normalize(text)))


def detect_methods(text: str, doc_has_cnd_context: bool | None = None) -> dict[str, bool]:
    """
    Retourne un dict {tag: bool} pour CHAQUE tag connu (toujours toutes les
    clés présentes, même à False) — important pour que le filtrage Chroma
    soit cohérent sur tous les chunks, y compris ceux d'articles ne
    mentionnant aucune méthode reconnue.

    Un tag n'est True que si (a) un mot-clé de la méthode est trouvé dans
    `text` ET (b) le document est bien un article CND (voir has_cnd_context).

    Deux façons d'utiliser cette fonction :
    - Niveau DOCUMENT (text = article entier, doc_has_cnd_context=None) :
      le contexte CND est calculé sur le même texte que les mots-clés.
      Risque : un article de synthèse qui évoque 7 méthodes en 7 sections
      se retrouve tagué avec les 7, même si chaque section ne parle que
      d'une seule (voir discussion sur 4-NDAO 2016.pdf).
    - Niveau CHUNK (text = un seul chunk, doc_has_cnd_context=bool déjà
      calculé une fois sur l'article entier via has_cnd_context()) :
      chaque chunk n'est tagué que pour LA méthode qu'il mentionne
      vraiment, tout en gardant le garde-fou "c'est bien un article CND"
      calculé au niveau document. C'est le mode recommandé pour l'ingestion,
      car le chunk est l'unité effectivement injectée dans le prompt RAG.
    """
    normalized = _normalize(text)
    context_ok = has_cnd_context(text) if doc_has_cnd_context is None else doc_has_cnd_context

    return {
        tag: context_ok and any(keyword in normalized for keyword in keywords)
        for tag, keywords in METHOD_KEYWORDS.items()
    }


def extract_year(filename: str) -> int | None:
    """
    Extrait une année plausible (1900-2099) depuis le nom de fichier.
    Retourne None si rien trouvé — dans ce cas la clé "year" est omise des
    métadonnées (voir ingest.py), donc un filtre par année exclura
    naturellement ce document plutôt que de planter.
    """
    match = YEAR_PATTERN.search(filename)
    return int(match.group()) if match else None
