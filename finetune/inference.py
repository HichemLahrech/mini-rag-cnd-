"""
INFERENCE — Comparaison modèle de base vs modèle fine-tuné QLoRA
====================================================================

Charge le modèle de base ET le modèle avec l'adaptateur LoRA par-dessus, et
génère une réponse pour LES MÊMES questions — volontairement absentes du
dataset d'entraînement (voir TEST_QUESTIONS et prepare_dataset.py) — avec
les deux. C'est le test qui permet de vérifier OBJECTIVEMENT si le
fine-tuning a bien généralisé le format JSON appris à de nouvelles
questions, plutôt que de "sentir" une amélioration à l'œil.

Ce qu'on s'attend à observer :
- Modèle de base : réponse en prose libre, format non prévisible
- Modèle fine-tuné : réponse en JSON {"reponse": ..., "certitude": ...},
  même si le system prompt ne mentionne JAMAIS ce format explicitement —
  la seule source de ce format, c'est ce que l'adaptateur LoRA a appris.

Prérequis : avoir lancé train_qlora.py au moins une fois (le dossier
qlora-adapter-trivia-json/ doit exister).
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
ADAPTER_DIR = "qlora-adapter-cnd"

# Identique au system prompt utilisé par generate_dataset_from_corpus.py —
# ne mentionne PAS le format JSON, pour que la comparaison base/fine-tuné
# soit un test honnête du fine-tuning et pas du simple suivi d'instruction
# en contexte.
SYSTEM_PROMPT = "Tu es un assistant qui répond aux questions de contrôle non destructif (CND)."

# Questions CND réelles, volontairement formulées différemment de ce qui a
# probablement été généré automatiquement par generate_dataset_from_corpus.py
# (autre angle, autre formulation) — pour limiter le risque de chevauchement
# quasi-littéral avec le dataset d'entraînement.
#
# ⚠️ LIMITE ASSUMÉE : contrairement à l'exercice trivia (où on maîtrisait
# exactement le contenu du dataset d'entraînement), ici les questions
# d'entraînement sont générées automatiquement — on ne peut pas garantir à
# 100% qu'aucune de ces questions de test n'y ressemble de près. C'est un
# test de généralisation plus faible que l'exercice trivia, à prendre avec
# ce recul.
TEST_QUESTIONS = [
    "Comment fonctionne la méthode ACFM pour la détection de fissures ?",
    "Quels capteurs sont utilisés pour mesurer la fuite de flux magnétique dans un rail ?",
    "Pourquoi utilise-t-on des ondes guidées en contrôle ultrasonore des rails ?",
]


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


def generate(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,  # génération déterministe (greedy) : comparaison reproductible d'un run à l'autre
        )

    # On ne garde que les tokens NOUVELLEMENT générés (pas le prompt d'entrée répété)
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    print(f"📦 Chargement du modèle de base (4-bit) : {BASE_MODEL}")
    base_model, tokenizer = load_base_model()

    base_answers = {}
    print("\n" + "#" * 70)
    print("# SANS fine-tuning (modèle de base)")
    print("#" * 70)
    for question in TEST_QUESTIONS:
        answer = generate(base_model, tokenizer, question)
        base_answers[question] = answer
        print(f"\n❓ {question}\n💬 {answer}")

    print(f"\n\n🔧 Chargement de l'adaptateur LoRA ({ADAPTER_DIR}) par-dessus le même modèle...")
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("\n" + "#" * 70)
    print("# AVEC fine-tuning QLoRA")
    print("#" * 70)
    for question in TEST_QUESTIONS:
        answer = generate(finetuned_model, tokenizer, question)
        print(f"\n❓ {question}\n💬 {answer}")

    print("\n" + "=" * 70)
    print("📊 Vérifie visuellement : le modèle fine-tuné produit-il un JSON")
    print('   {"reponse": "...", "certitude": "..."} de façon cohérente,')
    print("   alors que le modèle de base répond en prose libre ?")
    print("   Si oui, l'adaptateur a bien généralisé le format à des")
    print("   questions inédites — pas juste mémorisé les exemples vus.")


if __name__ == "__main__":
    main()
