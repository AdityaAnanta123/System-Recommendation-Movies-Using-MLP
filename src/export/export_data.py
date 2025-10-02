# src/export/export_data.py
"""
Script sederhana untuk mengekspor embeddings, metadata atau CSV kecil untuk integrasi.
"""
import numpy as np
import pandas as pd
import json
import os

def export_csv_sample(n=1000):
    df = pd.read_parquet("data/processed/movies_processed.parquet")
    df_sample = df.head(n)
    df_sample.to_csv("outputs/exports/movies_sample.csv", index=False)
    print("Saved sample CSV")

def export_embeddings_learned():
    emb = np.load("outputs/exports/movie_embeddings_learned.npy")
    np.save("outputs/exports/movie_embeddings_learned.npy", emb)
    print("Re-saved embeddings (no-op)")

if __name__ == "__main__":
    export_csv_sample()
