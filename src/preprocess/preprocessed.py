"""
Preprocessing dataset FINAL:
- baca CSV dari data/raw/
- hapus duplikat movie_id
- jika judul sama, pilih versi genres lengkap
- isi genre kosong dari judul sama atau "Unknown"
- gabungkan text (title + overview)
- ekstrak release_year
- encode genres -> multi-hot
- simpan ke data/processed/movies_processed.csv
"""
import os
import pandas as pd
import numpy as np
import json

RAW_PATH = "data/raw/movies_large.csv"
OUT_PATH = "data/processed/movies_processed.csv"

def parse_genres(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        if "|" in x:
            return [g.strip() for g in x.split("|") if g.strip()]
        elif "," in x:
            return [g.strip() for g in x.split(",") if g.strip()]
        else:
            return [x.strip()]
    return []

def preprocess():
    df = pd.read_csv(RAW_PATH)

    # --- Hapus baris tanpa movie_id ---
    df = df.dropna(subset=["movie_id"])

    # --- Hapus duplikat movie_id ---
    before = len(df)
    df = df.drop_duplicates(subset=["movie_id"], keep="first")
    after = len(df)
    print(f"Removed {before - after} duplicate rows based on movie_id")

    # --- Gabungkan text ---
    df["text"] = (df["title"].fillna("") + ". " + df["overview"].fillna("")).str.strip()

    # --- Parse genres ---
    df["genres_list"] = df["genres"].apply(parse_genres)

    # --- Buat mapping title -> genres lengkap ---
    title_to_genres = df[df["genres_list"].map(len)>0].groupby("title")["genres_list"].first().to_dict()

    # --- Isi genre kosong dari judul sama atau "Unknown" ---
    df["genres_list"] = df.apply(
        lambda row: title_to_genres.get(row["title"], ["Unknown"]) if len(row["genres_list"])==0 else row["genres_list"],
        axis=1
    )

    # --- Kalau judul sama, pilih versi pertama saja ---
    df = df.drop_duplicates(subset=["title"], keep="first")

    # --- Extract release year ---
    def get_year(d):
        try:
            return int(str(d)[:4])
        except:
            return np.nan
    df["release_year"] = df["release_date"].apply(get_year)
    df["release_year"] = df["release_year"].fillna(df["release_year"].mode()[0]).astype(int)

    # --- Multi-hot encode genres ---
    all_genres = sorted({g for lst in df["genres_list"] for g in lst})
    for g in all_genres:
        df[f"genre__{g}"] = df["genres_list"].apply(lambda lst: 1 if g in lst else 0)

    # --- Simpan processed dataset ---
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # --- Simpan metadata untuk training MLP ---
    meta = {
        "genres": all_genres,
        "num_cols": ["release_year","popularity"]
    }
    with open("data/processed/meta.json", "w") as f:
        json.dump(meta, f)

    print(f"Saved processed data ({len(df)} rows) to {OUT_PATH}")

if __name__ == "__main__":
    preprocess()
