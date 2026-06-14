import matplotlib.pyplot as plt
from PIL import Image
from search_engine import VisualSearchEngine

def show_results(query_path, top_k=5):
    engine = VisualSearchEngine(
        embeddings_path="embeddings/embeddings.npy",
        index_path="embeddings/index.json"
    )
    results = engine.search(query_path, top_k=top_k)

    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 4))

    # Image requête
    axes[0].imshow(Image.open(query_path))
    axes[0].set_title("Requête", fontsize=10)
    axes[0].axis("off")

    # Résultats
    for i, res in enumerate(results):
        axes[i + 1].imshow(Image.open(res["path"]))
        axes[i + 1].set_title(f"Score: {res['score']:.3f}", fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle("Moteur de recherche visuel — Top-K résultats", fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_results("uploaded_images/re_1751711869_real-madrid-2025-2026-home-shirt-youth.jpg", top_k=5)
