# Case study — mini-RAG CND : construire un pipeline RAG et QLoRA en partant des fondamentaux

## Contexte

Enseignant-chercheur en génie électrique spécialisé en contrôle non destructif (CND), je voulais acquérir une compétence pratique et vérifiable en ingénierie LLM — pas seulement savoir assembler LangChain, mais comprendre ce qui se passe à chaque étape d'un pipeline RAG et d'un fine-tuning QLoRA, au point de pouvoir déboguer et justifier chaque décision.

J'ai choisi mon propre domaine d'expertise comme terrain : cinq articles scientifiques sur le contrôle non destructif ferroviaire (courants de Foucault, GMR, ultrasons, EMAT, MFL, ACFM...). Ce choix n'était pas anodin — connaître le sujet en profondeur me permettait de juger la qualité réelle des réponses générées, pas seulement leur plausibilité apparente.

## Approche

Développement **sans framework RAG** (pas de LangChain, pas de LlamaIndex) : chunking, embeddings, retrieval, reranking et prompt engineering codés directement. L'objectif n'était pas la vitesse de livraison mais la compréhension — un choix délibéré pour un projet d'apprentissage, qui a aussi eu un effet secondaire utile : quand quelque chose ne marchait pas, je savais exactement où chercher, sans devoir remonter une couche d'abstraction.

Méthode de travail : itérative, avec **validation objective à chaque étape** plutôt qu'un jugement à l'œil sur une poignée de questions. Ce principe a structuré tout le projet, du chunking à l'évaluation du fine-tuning.

## Défis techniques et résolutions

### 1. Le chunking naïf casse le sens

Le découpage par caractères fixes coupait des formules et des idées n'importe où. Solution : chunking sémantique en deux niveaux — détection de sections via les titres numérotés du document, puis découpage par phrases entières à l'intérieur de chaque section, avec un overlap au niveau phrase plutôt que caractère.

**Piège rencontré et corrigé** : un edge case où une phrase à elle seule dépassait la taille cible du chunk (fréquent avec les formules mal segmentées par l'extraction PDF) faisait dupliquer intégralement ce chunk surdimensionné dans le chunk suivant via le mécanisme d'overlap — détecté en testant sur un extrait réel avant de lancer l'ingestion complète.

### 2. Les métadonnées sur-étiquettent les articles de synthèse

Premier réflexe : détecter la méthode CND (courants de Foucault, ultrasons...) une fois par document entier. Résultat : un chapitre de thèse passant en revue sept méthodes récupérait les sept tags, même pour des chunks n'en mentionnant qu'une seule — rendant le filtrage par métadonnées quasi inutile sur les documents de synthèse.

**Correction** : détection à deux niveaux — le contexte "est-ce un article CND" reste évalué sur le document entier (propriété globale), mais chaque méthode spécifique est désormais détectée **par chunk** (propriété locale). Un chunk n'est tagué que pour ce qu'il mentionne réellement.

### 3. Un faux positif révélateur : la physique n'est pas le CND

Un article sur le chauffage par induction ferroviaire s'est retrouvé tagué "courants de Foucault" — parce que le chauffage par induction *utilise* physiquement ce phénomène, sans être un article de contrôle non destructif. Un simple mot-clé ne suffisait pas à distinguer l'usage physique de l'usage métier.

**Correction** : garde-fou de co-occurrence — un tag méthode n'est validé que si le document mentionne aussi un marqueur générique de CND ("contrôle non destructif", "NDT", "inspection non destructive"...). Ce cas a aussi révélé, une fois corrigé, une limite inverse (sous-détection possible si un article légitime n'utilise jamais ces marqueurs explicites) — documentée plutôt que masquée.

### 4. Diagnostiquer un échec de retrieval au bon étage

Une question portant sur une technique combinant deux méthodes (courants de Foucault + thermographie) échouait à remonter le bon article, malgré un retrieval en apparence fonctionnel. Diagnostic : le bi-encoder capture la proximité sémantique de sujets, pas la notion de *combinaison* — il remontait des articles mentionnant les deux méthodes séparément plutôt que l'article les combinant réellement.

**Résolution en deux temps**, pour bien isoler la cause :
- Un filtre métadonnées en **ET** (au lieu du OR habituel) a résolu le retrieval — preuve que le problème était bien la logique du bi-encoder, pas les données.
- Une fois le retrieval corrigé, la génération échouait *encore* à répondre correctement (le LLM local ne faisait pas le lien entre "courants de Foucault" en français et "eddy current" dans les extraits anglais). Un deuxième problème, à un étage différent du pipeline, résolu séparément par un prompt enrichi d'un glossaire terminologique.

Cette séparation en deux diagnostics distincts (retrieval vs génération) plutôt qu'un correctif global illustre une discipline de débogage utile en production : ne pas corriger au hasard, identifier précisément l'étage responsable.

### 5. Boucler RAG et QLoRA

Plutôt qu'un dataset de fine-tuning artificiel, génération automatique de paires question/réponse **directement depuis les chunks déjà indexés et tagués** par le pipeline RAG, via le LLM local déjà en place. Déséquilibre initial détecté (certaines méthodes CND sur-représentées à cause du chevauchement de tags sur les chunks d'énumération) et corrigé par sur-récupération et mélange aléatoire des candidats.

Résultat vérifié objectivement : sur des questions absentes du dataset d'entraînement, le modèle fine-tuné produit systématiquement le format JSON appris, alors que le modèle de base répond en prose libre — preuve de généralisation d'un pattern structurel, pas de mémorisation.

### 6. La gestion de versions comme compétence à part entière

Une part non négligeable du travail a consisté à diagnostiquer des incompatibilités de versions entre `bitsandbytes`, `transformers` et `trl` sous Windows — dont un bug connu (`AttributeError` sur un `frozenset`) documenté sur GitHub. Plutôt que de contourner par essais-erreurs, recherche de la cause exacte (changement d'API `trl` v0.9 → v1.x) et migration du code vers l'API actuelle plutôt que de figer des versions obsolètes entre elles.

## Résultats

| Indicateur | Valeur |
|---|---|
| Recall@5 (retrieval, reranking activé) | 100% |
| Precision@5 | 80% (contre 77.5% en chunking naïf, mesuré objectivement) |
| MRR | 0.917 |
| Paramètres entraînés (QLoRA) | 0.17% du modèle (13.6M / 8.04 milliards) |
| Généralisation du format fine-tuné | Vérifiée sur questions inédites |

## Ce que ce projet démontre

- Compréhension des internals d'un pipeline RAG (pas juste l'usage d'un framework)
- Discipline de diagnostic : isoler la cause exacte avant de corriger, distinguer les étages d'un pipeline (retrieval vs génération, données vs code)
- Honnêteté méthodologique : mesurer plutôt que juger à l'œil, documenter les limites plutôt que les taire
- Autonomie sur des problèmes d'ingénierie transverses (gestion d'environnements, compatibilité de versions, spécificités Windows)
- Capacité à boucler plusieurs briques IA (RAG → génération de données → fine-tuning) en un pipeline cohérent

## Prochaines étapes

Déploiement du pipeline en API (FastAPI + Docker), intégration d'un LLM cloud en complément du modèle local, et extension du jeu d'évaluation.

---

*Code source complet : voir le [README technique](./README.md)*
