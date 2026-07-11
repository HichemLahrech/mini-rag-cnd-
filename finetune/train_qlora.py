"""
TRAIN_QLORA -- Fine-tuning QLoRA de Llama 3.1 8B
=================================================

Concepts cles a retenir :

- QUANTIFICATION 4-BIT (le "Q" de QLoRA) : le modele de base est charge en
  4-bit via bitsandbytes, ce qui divise environ par 4 la memoire necessaire
  pour stocker ses poids par rapport a une precision 16-bit classique. Sur
  une RTX 4070 Ti (12 Go de VRAM), un modele 8B en 16-bit ne laisserait
  quasiment aucune marge pour l'entrainement -- en 4-bit, si.

- LoRA (Low-Rank Adaptation) : plutot que d'entrainer les ~8 milliards de
  parametres du modele, on le GELE entierement et on insere de petites
  matrices adaptatrices dans les couches d'attention (q_proj, k_proj,
  v_proj, o_proj). Seules ces matrices sont entrainees -- de l'ordre de
  0.1-1% des parametres totaux (voir le print_trainable_parameters()
  ci-dessous pour le ratio exact sur ce modele).

- COMPLETION_ONLY_LOSS (masquage automatique du prompt) : dans un exemple
  "system + question -> reponse", on ne veut entrainer le modele qu'a
  PREDIRE la reponse, pas a reproduire la question qu'il voit deja en
  entree. SFTConfig(completion_only_loss=True) fait ce masquage
  automatiquement a partir d'un dataset au format "messages" (liste de
  role/content) -- plus besoin de faire correspondre manuellement une
  chaine de caracteres marquant le debut de la reponse (fragile), trl
  applique le chat template ET masque le prompt lui-meme.

Prerequis :
    pip install -r requirements-finetune.txt
    python generate_dataset_from_corpus.py   (ou prepare_dataset.py)
    -> genere data/train.jsonl et data/val.jsonl au format
       {"messages": [{"role": "system", ...}, {"role": "user", ...},
                      {"role": "assistant", ...}]}

Note sur l'API trl : ce script cible trl v1.x (SFTConfig/SFTTrainer avec
processing_class= et peft_config= directement sur le Trainer). Les
tutoriels bases sur trl 0.9.x utilisent une API differente
(tokenizer=, max_seq_length=, DataCollatorForCompletionOnlyLM manuel) qui
n'est plus valide avec trl >= 1.0 -- voir requirements-finetune.txt.
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"  # miroir non-gated de Llama 3.1 8B Instruct
OUTPUT_DIR = "qlora-adapter-cnd"

TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/val.jsonl"

MAX_LENGTH = 512  # largement suffisant pour nos questions/reponses courtes


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # format optimise pour des poids ~normalement distribues
        bnb_4bit_use_double_quant=True,          # quantifie aussi les constantes de quantification (gain marginal)
        bnb_4bit_compute_dtype=torch.bfloat16,   # calculs en bf16, natif sur RTX 4070 Ti (Ada Lovelace)
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Pas besoin d'appeler prepare_model_for_kbit_training() ni
    # get_peft_model() nous-memes : SFTTrainer le fait automatiquement des
    # qu'on lui passe un modele quantifie + peft_config (voir main()).

    return model, tokenizer


def main():
    print(f"Chargement du modele en 4-bit : {BASE_MODEL}")
    print("   (premier lancement = telechargement, ~16 Go, ensuite mis en cache)")
    model, tokenizer = load_model_and_tokenizer()

    lora_config = LoraConfig(
        r=16,                 # rang des matrices adaptatrices : capacite vs cout memoire (8-64 typique)
        lora_alpha=32,        # facteur d'echelle, convention courante : 2x le rang
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # couches d'attention seulement
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Chargement du dataset...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    print(f"   {len(dataset['train'])} exemples d'entrainement, {len(dataset['validation'])} de validation")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_LENGTH,
        completion_only_loss=True,       # masque automatiquement le prompt dans la loss (voir docstring)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # batch effectif = 1*8 = 8 (compense le petit batch physique, contrainte VRAM)
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",        # optimiseur memoire-efficace (bitsandbytes) : "pagine" vers la RAM CPU si besoin
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,   # remplace l'ancien argument tokenizer= (retire en trl v0.16+)
        peft_config=lora_config,      # SFTTrainer applique lui-meme prepare_model_for_kbit_training + get_peft_model
    )

    print("Parametres entraines (adaptateur LoRA) :")
    trainer.model.print_trainable_parameters()  # pedagogique : montre concretement le ratio de parametres entraines

    print("Entrainement (3 epochs sur un tout petit dataset -- quelques minutes attendues)...")
    trainer.train()

    print(f"Sauvegarde de l'adaptateur LoRA dans {OUTPUT_DIR}")
    print("   (seuls les poids de l'adaptateur sont sauvegardes, pas le modele de base -- quelques dizaines de Mo)")
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
