import os
import numpy as np
import json
import torch
from tqdm import tqdm
from preprocessing import load_image, get_all_image_paths
from encoder import ImageEncoder

DATASET_DIR = "clothing-dataset/images"
OUTPUT_DIR  = "embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_all_embeddings():
    encoder = ImageEncoder()
    image_paths = get_all_image_paths(DATASET_DIR)
    print(f"{len(image_paths)} images trouvées.")

    all_embeddings = []
    valid_paths    = []

    for path in tqdm(image_paths, desc="Encodage"):
        try:
            tensor     = load_image(path)
            embedding  = encoder(tensor).numpy()   # [1, 2048]
            all_embeddings.append(embedding[0])
            valid_paths.append(path)
        except Exception as e:
            print(f"Erreur sur {path}: {e}")

    # Sauvegarde des vecteurs
    embeddings_matrix = np.array(all_embeddings)  # [N, 2048]
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings_matrix)

    # Sauvegarde de l'index (chemin ↔ position)
    with open(os.path.join(OUTPUT_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(valid_paths, f, ensure_ascii=False)

    print(f"Embeddings sauvegardés : shape {embeddings_matrix.shape}")
    return embeddings_matrix, valid_paths

if __name__ == "__main__":
    generate_all_embeddings()
