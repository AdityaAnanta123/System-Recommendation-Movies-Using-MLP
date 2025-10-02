# src/train/train.py
"""
Pipeline training:
- load processed data + sbert embeddings
- gabungkan feature: [sbert_emb, numeric cols, genre one-hot]
- split train/val
- train Keras MLP autoencoder, simpan model akhir
- juga export learned embeddings (bottleneck output) ke outputs/exports/movie_embeddings_learned.npy
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.classifier import build_model, save_model, encoder_from_model
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

PROCESSED = "data/processed/movies_processed.csv"
EMB_FILE = "outputs/exports/embeddings.npy"
IDS_FILE = "outputs/exports/ids.json"
OUT_DIR = "outputs/exports"

def load_meta():
    with open("data/processed/meta.json","r") as f:
        return json.load(f)

def make_feature_matrix():
    df = pd.read_csv(PROCESSED)
    emb = np.load(EMB_FILE)
    with open(IDS_FILE,"r") as f:
        ids = json.load(f)["ids"]

    # numeric cols from meta
    meta = load_meta()
    num_cols = meta['num_cols']
    genre_cols = [c for c in df.columns if c.startswith("genre__")]

    # numeric + genre one-hot
    X_meta = df[num_cols + genre_cols].values.astype(np.float32)

    # concat sbert emb + meta
    X = np.hstack([emb.astype(np.float32), X_meta])

    return X, df

def train():
    X, df = make_feature_matrix()

    # Autoencoder → target = input
    y = X.copy()

    # simple split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    input_dim = X.shape[1]
    model = build_model(input_dim=input_dim, bottleneck_dim=128)

    os.makedirs("models/classifier", exist_ok=True)

    # callbacks
    ckpt = ModelCheckpoint("models/classifier/best.keras", save_best_only=True, monitor="val_loss")
    es = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[ckpt, es]
    )

    # save full model
    save_model(model)

    # extract learned embeddings (bottleneck layer)
    encoder = encoder_from_model(model)
    embeddings_learned = encoder.predict(X, batch_size=256)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "movie_embeddings_learned.npy"), embeddings_learned)

    print("[INFO] Saved learned embeddings.")

if __name__ == "__main__":
    train()
