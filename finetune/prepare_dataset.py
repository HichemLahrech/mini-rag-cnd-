"""
PREPARE_DATASET — Construction du dataset d'entraînement QLoRA
==================================================================

Objectif pédagogique de cet exercice : apprendre au modèle un FORMAT DE
SORTIE STRICT (JSON avec deux champs fixes) qu'il ne suit pas naturellement,
sur des questions de culture générale variées (volontairement sans lien
avec le CND, pour rester un exercice "jouet" simple à vérifier).

Pourquoi un format plutôt qu'un nouveau savoir ? Parce que c'est un test
NON AMBIGU du fine-tuning : avec seulement ~35 exemples d'entraînement, un
LoRA a largement de quoi apprendre un PATTERN structurel (comment répondre),
mais pas assez pour mémoriser fiablement de nouveaux FAITS. En comparant le
modèle de base et le modèle fine-tuné sur des questions inédites (voir
inference.py), on peut vérifier objectivement si le format est bien
généralisé — pas juste récité sur les exemples vus.

Ce script :
1. Définit ~35 paires question/réponse formatées en JSON strict
2. Les écrit au format "messages" (system/user/assistant) — le format
   attendu nativement par trl (SFTTrainer applique lui-même le chat
   template au moment de l'entraînement, voir train_qlora.py)
3. Sépare en train/validation et écrit deux fichiers JSONL
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
VAL_FRACTION = 0.15  # ~15% des exemples en validation
SEED = 42

# Paires question/réponse de culture générale, variées en thème pour que le
# modèle généralise le FORMAT plutôt que de l'associer à un seul sujet.
# "certitude" est un champ pédagogique inventé pour l'exercice (pas une
# vraie mesure de confiance calibrée) — juste pour avoir un format à deux
# champs, plus intéressant qu'une simple chaîne de caractères.
TRIVIA_DATA = [
    ("Quelle est la capitale de l'Australie ?", "Canberra est la capitale de l'Australie.", "elevee"),
    ("Qui a écrit 'Les Misérables' ?", "Victor Hugo est l'auteur des Misérables, publié en 1862.", "elevee"),
    ("Quel est le plus grand océan du monde ?", "L'océan Pacifique est le plus grand océan du monde.", "elevee"),
    ("En quelle année a eu lieu la Révolution française ?", "La Révolution française a débuté en 1789.", "elevee"),
    ("Quel est l'élément chimique de symbole Fe ?", "Le fer a pour symbole chimique Fe.", "elevee"),
    ("Qui a peint 'La Nuit étoilée' ?", "Vincent van Gogh a peint La Nuit étoilée en 1889.", "elevee"),
    ("Quelle est la plus haute montagne du monde ?", "L'Everest, avec 8 849 mètres, est la plus haute montagne du monde.", "elevee"),
    ("Combien de continents y a-t-il sur Terre ?", "On considère généralement qu'il y a sept continents.", "moyenne"),
    ("Quel est le plus long fleuve du monde ?", "Le Nil et l'Amazone se disputent ce titre selon la méthode de mesure utilisée ; le Nil est traditionnellement cité comme le plus long.", "faible"),
    ("Qui a composé la 9e symphonie ?", "Ludwig van Beethoven a composé sa 9e symphonie, achevée en 1824.", "elevee"),
    ("Quelle est la monnaie officielle du Japon ?", "Le yen est la monnaie officielle du Japon.", "elevee"),
    ("Quel savant a formulé la théorie de la relativité générale ?", "Albert Einstein a formulé la théorie de la relativité générale en 1915.", "elevee"),
    ("Quelle est la vitesse de la lumière dans le vide ?", "La vitesse de la lumière dans le vide est d'environ 299 792 kilomètres par seconde.", "elevee"),
    ("Qui a été le premier homme à marcher sur la Lune ?", "Neil Armstrong a été le premier homme à marcher sur la Lune, en 1969.", "elevee"),
    ("Quel est le plus petit pays du monde ?", "La Cité du Vatican est le plus petit pays du monde en superficie.", "elevee"),
    ("Combien d'os compte le corps humain adulte ?", "Le corps humain adulte compte généralement 206 os.", "moyenne"),
    ("Quelle est la langue la plus parlée au monde ?", "Le mandarin est la langue maternelle la plus parlée au monde, l'anglais dominant en tant que langue seconde.", "moyenne"),
    ("Qui a écrit 'Roméo et Juliette' ?", "William Shakespeare est l'auteur de Roméo et Juliette.", "elevee"),
    ("Quelle planète est la plus proche du Soleil ?", "Mercure est la planète la plus proche du Soleil.", "elevee"),
    ("Quel est le plus grand désert chaud du monde ?", "Le Sahara est le plus grand désert chaud du monde.", "elevee"),
    ("En quelle année a été fondée l'Organisation des Nations Unies ?", "L'ONU a été fondée en 1945.", "elevee"),
    ("Quel est le symbole chimique de l'or ?", "Le symbole chimique de l'or est Au.", "elevee"),
    ("Qui a réalisé le film 'Pulp Fiction' ?", "Quentin Tarantino a réalisé Pulp Fiction, sorti en 1994.", "elevee"),
    ("Quelle est la capitale du Canada ?", "Ottawa est la capitale du Canada.", "elevee"),
    ("Quel animal est le plus grand mammifère du monde ?", "La baleine bleue est le plus grand mammifère du monde.", "elevee"),
    ("Qui a écrit 'Cent ans de solitude' ?", "Gabriel García Márquez est l'auteur de Cent ans de solitude, publié en 1967.", "elevee"),
    ("Quelle est la capitale de l'Égypte ?", "Le Caire est la capitale de l'Égypte.", "elevee"),
    ("Combien de joueurs compose une équipe de football sur le terrain ?", "Une équipe de football compte onze joueurs sur le terrain.", "elevee"),
    ("Quel est le point culminant d'Afrique ?", "Le Kilimandjaro, en Tanzanie, est le point culminant d'Afrique.", "elevee"),
    ("Qui a inventé l'ampoule électrique à filament pratique et durable ?", "Thomas Edison est généralement crédité de l'ampoule à filament pratique, bien que d'autres inventeurs comme Joseph Swan aient contribué en parallèle.", "moyenne"),
    ("Quelle est la plus grande île du monde ?", "Le Groenland est la plus grande île du monde.", "elevee"),
    ("En quelle année le mur de Berlin est-il tombé ?", "Le mur de Berlin est tombé en 1989.", "elevee"),
    ("Quel est le plus grand pays du monde par superficie ?", "La Russie est le plus grand pays du monde par superficie.", "elevee"),
    ("Qui a peint la Joconde ?", "Léonard de Vinci a peint la Joconde, entre 1503 et 1519 environ.", "elevee"),
    ("Quelle est la capitale de l'Argentine ?", "Buenos Aires est la capitale de l'Argentine.", "elevee"),
    ("Combien de temps met la Terre à faire le tour du Soleil ?", "La Terre met environ 365,25 jours pour faire le tour du Soleil.", "elevee"),
    ("Quel est le plus grand pays d'Amérique du Sud ?", "Le Brésil est le plus grand pays d'Amérique du Sud.", "elevee"),
    ("Qui a écrit 'Don Quichotte' ?", "Miguel de Cervantès est l'auteur de Don Quichotte, publié en deux parties en 1605 et 1615.", "elevee"),
]


def build_answer_json(reponse: str, certitude: str) -> str:
    """Construit la réponse cible en JSON compact (une seule ligne)."""
    return json.dumps({"reponse": reponse, "certitude": certitude}, ensure_ascii=False)


# Le modèle doit associer CE system prompt précis au format JSON appris,
# pas un system prompt qui décrirait explicitement le format attendu
# (sinon on ne testerait pas le fine-tuning, juste le suivi d'instruction
# en contexte). Identique à celui utilisé dans inference.py.
SYSTEM_PROMPT = "Tu es un assistant qui répond aux questions de culture générale."


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
    examples = []
    for question, reponse, certitude in TRIVIA_DATA:
        answer_json = build_answer_json(reponse, certitude)
        messages = build_messages(question, answer_json)
        examples.append({"messages": messages})

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

    print("\n📄 Aperçu d'un exemple (train) :")
    print("-" * 60)
    print(json.dumps(train_examples[0], ensure_ascii=False, indent=2))
    print("-" * 60)


if __name__ == "__main__":
    main()
