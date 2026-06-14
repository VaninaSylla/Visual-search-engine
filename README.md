# Visual Search Engine

Prototype de moteur de recherche visuel pour vêtements basé sur des embeddings d'images.

## Objectif

Construire un système de requête d'image qui :
- encode des images en vecteurs de caractéristiques,
- stocke ces représentations sur disque,
- répond à des requêtes d'image en retrouvant les images les plus similaires.

Ce projet répond aux objectifs suivants :
- Service d'encodage d'images acceptant une image en entrée et produisant un vecteur.
- Génération des représentations vectorielles pour un dataset de vêtements.
- Sauvegarde des embeddings et de l'index sur disque.
- Recherche par similarité d'image au lieu de recherche textuelle.

## Structure du projet

- `app.py` : application Flask pour l'upload et la recherche d'images.
- `encoder.py` : définition de l'encodeur d'images basé sur ResNet50.
- `generate_embeddings.py` : script pour calculer et sauvegarder les embeddings du dataset.
- `preprocessing.py` : chargement et prétraitement des images.
- `search_engine.py` : moteur de recherche FAISS pour les embeddings pré-calculés.
- `clothing-dataset/` : dataset d'images de vêtements.
- `embeddings/` : dossier de sortie contenant `embeddings.npy` et `index.json`.

## Installation

1. Activez votre environnement virtuel :

```powershell
cd c:\Users\ngono\Moteur_de_recherche_visuelle\Moteur_de_recherche_visuelle
env\Scripts\activate
```

2. Installez les dépendances :

```powershell
pip install -r requirements.txt
```

## Requirements

- Python 3.12+
- `torch`
- `torchvision`
- `flask`
- `flask-cors`
- `tqdm`
- `faiss-cpu`
- `pillow`
- `numpy`

> Si vous préférez installer manuellement, utilisez `pip install torch torchvision flask flask-cors tqdm faiss-cpu pillow numpy`.

## Génération des embeddings

Le script `generate_embeddings.py` parcourt le dataset `clothing-dataset/images`, encode chaque image avec le modèle ResNet50 et sauvegarde :
- `embeddings/embeddings.npy`
- `embeddings/index.json`

Lancer :

```powershell
python generate_embeddings.py
```

## Lancement de l'application

Démarrez le serveur Flask :

```powershell
python app.py
```

Le service écoute par défaut sur `http://0.0.0.0:5000`.

### Endpoints principaux

- `GET /` : page web principale.
- `POST /upload` : upload d'une image et ajout de son embedding à l'index.
- `POST /search` : recherche d'images similaires à une image de requête.
- `GET /stats` : statistiques du nombre d'images indexées.
- `GET /image?path=...` : accès à une image indexée.

## Fonctionnement du moteur

1. `preprocessing.py` charge et normalise l'image selon les standards ImageNet.
2. `encoder.py` extrait un vecteur de 2048 dimensions avec ResNet50 sans la dernière couche de classification.
3. `generate_embeddings.py` stocke les embeddings du dataset sur disque.
4. `app.py` utilise ces embeddings pour répondre à des requêtes d'image en calculant une similarité cosinus.
5. `search_engine.py` illustre aussi l'utilisation d'un index FAISS pour accélérer la recherche.


## Respect des consignes

Ce projet couvre bien les deux parties obligatoires :
- Construction d'un service d'encodage d'image.
- Construction d'un système de requête d'image pour trouver des correspondances visuelles.

Il permet de passer d'une recherche par texte à une recherche par image, en utilisant une représentation vectorielle de l'image comme base de comparaison.

## Remarques

- Assurez-vous que le dataset `clothing-dataset/images` est présent.
- Les images uploadées sont stockées dans `uploaded_images/`.
- Le moteur de recherche utilise la similarité cosinus sur les embeddings normalisés.
