import os
import json
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from encoder import ImageEncoder
from preprocessing import load_image

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "index.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploaded_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

encoder = ImageEncoder()

def load_index():
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH):
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            paths = json.load(f)
        return embeddings, paths
    return np.empty((0, 2048), dtype="float32"), []

def save_index(embeddings, paths):
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False)

def project_path(path):
    resolved = os.path.abspath(os.path.join(BASE_DIR, path))
    if os.path.commonpath([BASE_DIR, resolved]) != BASE_DIR:
        return None
    return resolved

embeddings_db, image_paths = load_index()

# ── ROUTE PAGE PRINCIPALE ──
@app.route("/", methods=["GET"])
def home():
    return send_from_directory(os.path.join(BASE_DIR, "templates"), "index.html")

# ── STATS ──
@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({"total_indexed": len(image_paths)})

# ── UPLOAD ──
@app.route("/upload", methods=["POST"])
def upload():
    global embeddings_db, image_paths
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Aucune image reçue"}), 400
    file = request.files["image"]
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"success": False, "error": "Nom de fichier invalide"}), 400

    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    relative_path = os.path.relpath(save_path, BASE_DIR)
    file.save(save_path)
    try:
        tensor    = load_image(save_path)
        embedding = encoder(tensor).numpy()[0]
        embeddings_db = np.vstack([embeddings_db, embedding]) if embeddings_db.shape[0] > 0 else embedding.reshape(1, -1)
        image_paths.append(relative_path)
        save_index(embeddings_db, image_paths)
        return jsonify({"success": True, "path": relative_path, "embedding_dim": int(embedding.shape[0]), "total_indexed": len(image_paths)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ── SEARCH ──
@app.route("/search", methods=["POST"])
def search():
    global embeddings_db, image_paths
    if "image" not in request.files:
        return jsonify({"error": "Aucune image requête"}), 400
    if embeddings_db.shape[0] == 0:
        return jsonify({"error": "L'index est vide."}), 400
    top_k = max(1, min(int(request.form.get("top_k", 5)), len(image_paths)))
    file  = request.files["image"]
    tmp   = os.path.join(UPLOAD_FOLDER, f"_query_{uuid.uuid4().hex}.jpg")
    file.save(tmp)
    try:
        tensor       = load_image(tmp)
        query_vector = encoder(tensor).numpy()[0].astype("float32")
        db_norm  = embeddings_db / (np.linalg.norm(embeddings_db, axis=1, keepdims=True) + 1e-9)
        q_norm   = query_vector  / (np.linalg.norm(query_vector) + 1e-9)
        scores   = db_norm @ q_norm
        top_idx  = np.argsort(scores)[::-1][:top_k]
        results  = [{"path": image_paths[i], "score": float(scores[i])} for i in top_idx]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp): os.remove(tmp)

# ── SERVIR LES IMAGES ──
@app.route("/image", methods=["GET"])
def serve_image():
    path = request.args.get("path", "")
    resolved_path = project_path(path)
    if not resolved_path or not os.path.exists(resolved_path):
        return jsonify({"error": "Image introuvable"}), 404
    return send_file(resolved_path)

if __name__ == "__main__":
    print(f"Index chargé : {len(image_paths)} images")
    app.run(debug=True, host="0.0.0.0", port=5000)
