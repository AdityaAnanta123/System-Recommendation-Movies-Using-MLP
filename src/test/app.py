from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# --- paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE_DIR, "..", "..", "data", "processed", "movies_processed.csv")
LEARNED_EMB = os.path.join(BASE_DIR, "..", "..", "outputs", "exports", "movie_embeddings_learned.npy")
IDS_FILE = os.path.join(BASE_DIR, "..", "..", "outputs", "exports", "ids.json")

# --- load data once ---
df = pd.read_csv(PROCESSED)
emb = np.load(LEARNED_EMB)
ids = json.load(open(IDS_FILE,"r"))["ids"]

def recommend_by_movie_idx(idx, emb_matrix, top_k=10):
    v = emb_matrix[idx:idx+1]
    sims = cosine_similarity(v, emb_matrix)[0]
    sims[idx] = -1  # exclude self
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return top_idx, sims[top_idx]

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_id = request.args.get("movie_id")
    top_k = int(request.args.get("top_k", 10))

    if movie_id not in ids:
        return jsonify({"error": "movie_id not found"}), 404

    idx = ids.index(movie_id)
    top_idx, scores = recommend_by_movie_idx(idx, emb, top_k=top_k)

    recs = []
    for i, score in zip(top_idx, scores):
        row = df.loc[i]
        recs.append({
            "movie_id": int(row['movie_id']),
            "title": row['title'],
            "genres": row['genres'],
            "release_year": int(row['release_year']) if not np.isnan(row['release_year']) else None,
            "popularity": float(row['popularity']),
            "score": float(score)
        })

    return jsonify({"recommendations": recs})

if __name__ == "__main__":
    app.run(debug=True)
