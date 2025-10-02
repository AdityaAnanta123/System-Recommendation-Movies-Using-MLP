# src/embed/embeddings.py
"""
Bangun SBERT embeddings untuk setiap movie (menggunakan kolom `text`).
Simpan embeddings ke outputs/exports/embeddings.npy dan mapping id -> idx di outputs/exports/ids.json
"""
import os
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"  # cepat & cukup bagus
OUT_DIR = "outputs/exports"
EMB_FILE = os.path.join(OUT_DIR, "embeddings.npy")
IDS_FILE = os.path.join(OUT_DIR, "ids.json")
PROCESSED = "data/processed/movies_processed.csv"

def build_embeddings(model_name=MODEL_NAME):
    df = pd.read_csv(PROCESSED)
    texts = df['text'].fillna("").tolist()
    ids = df['movie_id'].astype(str).tolist() if 'movie_id' in df.columns else df.index.astype(str).tolist()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = SentenceTransformer(model_name)

    embeddings = []
    for t in tqdm(texts, desc="Embedding"):
        emb = model.encode(t, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    np.save(EMB_FILE, embeddings)
    with open(IDS_FILE, "w") as f:
        json.dump({"ids": ids}, f)
    print("Saved embeddings:", EMB_FILE)
    print("Saved ids:", IDS_FILE)

if __name__ == "__main__":
    build_embeddings()
