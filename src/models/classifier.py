# src/models/classifier.py
"""
Keras MLP encoder:
- Input: concatenated [sbert_embedding (d), numeric features, genre one-hot]
- Architecture: Dense -> Dense (bottleneck embedding) -> Dense(recon/regression)
- Kita latih untuk memprediksi `vote_average` (regresi) sehingga bottleneck belajar representasi.
- Setelah training, kita dapat mengambil output bottleneck sebagai learned embedding.
"""
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Lokasi penyimpanan model
MODEL_DIR = "models/classifier"
MODEL_FILE = os.path.join(MODEL_DIR, "mlp_encoder.keras")  # pakai .keras (format baru Keras)
META_FILE = "data/processed/meta.json"

def build_model(input_dim, bottleneck_dim=128):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(512, activation="relu")(inp)
    bottleneck = keras.layers.Dense(bottleneck_dim, activation="relu", name="bottleneck")(x)
    x = keras.layers.Dense(512, activation="relu")(bottleneck)
    out = keras.layers.Dense(input_dim, activation="sigmoid")(x)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model

def save_model(model):
    """Simpan model MLP encoder ke direktori classifier."""
    # pastikan folder ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    # simpan model dengan format .keras
    model.save(MODEL_FILE)
    print(f"[INFO] Model berhasil disimpan ke: {MODEL_FILE}")

def load_model():
    """Load model MLP encoder dari file .keras."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file {MODEL_FILE} tidak ditemukan. Jalankan training dulu.")
    model = keras.models.load_model(MODEL_FILE)
    print(f"[INFO] Model berhasil dimuat dari: {MODEL_FILE}")
    return model

# helper to extract bottleneck embeddings
def encoder_from_model(model):
    return keras.Model(inputs=model.input, outputs=model.get_layer("bottleneck").output)
