# src/test/recommender_app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED = "data/processed/movies_processed.csv"
LEARNED_EMB = "outputs/exports/movie_embeddings_learned.npy"
IDS_FILE = "outputs/exports/ids.json"

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED)
    emb = np.load(LEARNED_EMB)
    ids = json.load(open(IDS_FILE,"r"))["ids"]
    return df, emb, ids

def recommend_by_movie_idx(idx, emb_matrix, top_k=10):
    v = emb_matrix[idx:idx+1]
    sims = cosine_similarity(v, emb_matrix)[0]
    sims[idx] = -1  # exclude self
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return top_idx, sims[top_idx]

def main():
    st.title("Movie Recommender — Hybrid (SBERT + MLP)")
    df, emb, ids = load_data()
    st.write(f"Data loaded: {len(df)} movies")

    q = st.text_input("Cari judul (ketik sebagian) atau masukkan movie_id:")
    top_k = st.slider("Jumlah rekomendasi:", min_value=1, max_value=20, value=10)

    if q:
        try:
            # --- numeric input ---
            if q.isdigit():
                movie_id = q
                if movie_id in ids:
                    idx = ids.index(movie_id)
                else:
                    st.error("movie_id tidak ditemukan")
                    return
            else:
                # --- search by title ---
                res = df[df['title'].str.contains(q, case=False, na=False)]
                if res.empty:
                    st.info("Tidak ketemu, coba kata lain")
                    return
                st.write("Hasil pencarian:")
                st.table(res[['movie_id','title','genres']].head(10))
                idx = res.index[0]

            # --- tampilkan film input ---
            st.subheader("Film input:")
            movie = df.loc[idx]
            st.markdown(f"**{movie['title']}** ({movie['release_year']}) — genres: {movie['genres']} — popularity: {movie['popularity']}")

            # --- rekomendasi ---
            top_idx, scores = recommend_by_movie_idx(idx, emb, top_k=top_k)
            st.subheader("Rekomendasi:")
            for i, score in zip(top_idx, scores):
                row = df.loc[i]
                st.markdown(f"**{row['title']}** ({row['release_year']}) — genres: {row['genres']} — score: {score:.4f}")

        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    main()
