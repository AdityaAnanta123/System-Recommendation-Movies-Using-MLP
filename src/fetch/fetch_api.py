import os
import time
import requests
import pandas as pd
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    genres = r.json().get("genres", [])
    return {g["id"]: g["name"] for g in genres}

def fetch_data_from_tmdb(query_pages: int = 1, save_path: Optional[str] = None, delay: float = 0.3) -> pd.DataFrame:
    if not TMDB_API_KEY:
        raise ValueError("TMDB_API_KEY not found in environment. Please set it in .env file")

    base = "https://api.themoviedb.org/3"
    rows = []
    genre_map = fetch_genres()  # ambil mapping genre id -> nama

    for page in range(1, query_pages + 1):
        url = f"{base}/movie/popular"
        params = {"api_key": TMDB_API_KEY, "page": page, "language": "en-US"}
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()

        for m in data.get("results", []):
            # ubah genre_ids -> list nama genre
            gnames = [genre_map.get(gid, "") for gid in m.get("genre_ids", [])]
            rows.append({
                "movie_id": m.get("id"),
                "title": m.get("title"),
                "overview": m.get("overview") or "",
                "release_date": m.get("release_date"),
                "popularity": m.get("popularity"),
                "genres": ", ".join(gnames),  # gabungkan jadi string
            })

        print(f"✅ Page {page} done, total rows: {len(rows)}")
        time.sleep(delay)

    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df

if __name__ == "__main__":
    # contoh ambil 100 halaman (~2000 film)
    df = fetch_data_from_tmdb(query_pages=100, save_path="data/raw/movies_large.csv")
    print("Final dataset shape:", df.shape)
    print(df.head())
