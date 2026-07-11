# Image de base légère : Python 3.11 sans les outils inutiles pour réduire
# la taille finale de l'image (moins pertinent pour un usage local, mais
# c'est la pratique attendue pour un déploiement réel).
FROM python:3.11-slim

WORKDIR /app

# build-essential : nécessaire à la compilation de certaines dépendances
# transitives (ex: tokenizers) qui n'ont pas toujours de wheel précompilée
# pour toutes les architectures.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances AVANT le code source : Docker met
# en cache chaque étape (layer). Si seul le code change (pas requirements.txt),
# cette étape d'installation n'est pas refaite au prochain build — gain de
# temps important en itération.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Seuls les fichiers réellement nécessaires à l'exécution de l'API sont
# copiés (pas ingest.py, eval.py, les notebooks, etc.) — image plus légère,
# et surtout plus claire sur ce qui tourne réellement en production.
COPY api.py query.py cnd_metadata.py ./

EXPOSE 8000

# --host 0.0.0.0 est indispensable : par défaut uvicorn n'écoute que sur
# localhost, qui à l'intérieur d'un conteneur n'est joignable que depuis CE
# conteneur — le port ne serait pas accessible depuis l'extérieur même avec
# le mapping de port Docker en place.
# Pas de --reload ici (contrairement au développement local) : c'est un
# outil de surveillance de fichiers pour le dev, inutile et coûteux en
# production/conteneur.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
