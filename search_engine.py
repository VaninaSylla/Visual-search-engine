import numpy as np
import json
import faiss
from encoder import ImageEncoder
from preprocessing import load_image

class VisualSearchEngine:
    def __init__(self, embeddings_path, index_path):
        # Chargement des embeddings pré-calculés
        self.embeddings = np.load(embeddings_path).astype("float32")
        with open(index_path, "r", encoding="utf-8") as f:
            self.image_paths = json.load(f)

        # Normalisation L2 pour la similarité cosinus
        faiss.normalize_L2(self.embeddings)

        # Construction de l'index FAISS (Inner Product = cosinus après normalisation)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self.encoder = ImageEncoder()
        print(f"Index FAISS construit : {self.index.ntotal} images indexées.")

    def search(self, query_image_path, top_k=5):
        """
        Encode l'image requête et retourne les top_k images similaires.
        Retourne : liste de (chemin, score)
        """
        tensor       = load_image(query_image_path)
        query_vector = self.encoder(tensor).numpy().astype("float32")
        faiss.normalize_L2(query_vector)

        top_k = max(1, min(top_k, self.index.ntotal))
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "path" : self.image_paths[idx],
                "score": float(score)
            })
        return results
