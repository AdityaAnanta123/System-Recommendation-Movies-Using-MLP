# System Recommendation Movies Using MLP

A movie recommendation system built using a **Multilayer Perceptron (MLP)** model, consuming data via TMDB API, and exposing interfaces using either Flask or Streamlit.

## Table of Contents

- 📁 Project Structure  
- 🚀 Features  
- 🧩 Components / Modules  
- 🔄 Data Flow & Process  
- 🧪 API & Testing (Flask & Streamlit)  
- 🛠 Installation & Usage  
- 🧮 How It Works (from fetch → inference → response)  
- 🤖 Model Training & Storage  
- ✅ Testing & Evaluation  
- 📂 Directory Layout  
- 📄 License & Acknowledgments  

---

## Project Structure (Directory Layout)

Below is the approximate directory layout of the repository:

```

.
├── data/                      # raw / processed datasets
│   ├── processed/
|   └── raw/                   
├── models/                    # saved MLP / other model files
│   └── classifier/          
├── outputs/
│   └── exports/               # exported artifacts, e.g. prediction outputs, logs
├── src/
│   ├── embed/                 # embedding data from preprocesed
│   ├── export/                # export data from embeddings
│   ├── fetch/                 # fetching data from TMDB using TMDB_API_KEY
│   ├── preprocess/            # Preprocessing data raw
│   ├── models/                # Build model using MLP 
│   ├── train/                 # Training model MLP
│   └── tests/                 # Testing into a Flask and Streamlit
├── requirements.txt
├── .gitignore
└── README.md

````

## Features

- Fetch movie metadata via TMDB (The Movie Database) API  
- Preprocessing & feature engineering
- Made a embeddings for model
- Train a Multilayer Perceptron (MLP) model for recommendation  
- Deploy a prediction API using **Flask**  
- Deploy a UI via **Streamlit**  
- Support for testing/integration end-to-end  
- Save & load model artifacts  

---

## Components / Modules & Their Functions

Here’s a high-level breakdown of core modules and functions you might have in the `src/`:

| Module / File | Responsibility |
|----------------|----------------|
| `fetch/fetch_api.py` | Handle HTTP requests to TMDB API: fetching movie details, search, etc. |
| `train/train.py` | Training logic for the MLP model (defining architecture, training loop) |
| `model/classifier.py` | Build trained model, perform inference / recommendation logic |
| `test/app.py` | Flask application: define endpoints (e.g. `/recommend`) |
| `test/recommendation_app.py` | Streamlit app to create an interactive web UI |
| `preprocess/preprocessed.py` | Data cleaning, feature extraction, transformation before training or inference |
| `embed/embeddings.py` | Build embeddings using Sentence Transformers to processed data |
| `export/export_data.py` | Script to export embeddings, CSV, etc |


---

## Data Flow & Process: From Fetch → Inference → Response

Here is the typical end-to-end process:

1. **Fetch / API call**  
   - The client (UI or external client) sends a request to your endpoint (Flask route or Streamlit widget trigger).  
   - The backend uses the TMDB client module (`utils/tmdb_client.py`) to fetch movie metadata, e.g. via TMDB REST API (using API key, movie title or ID).  
   - The fetched JSON / response is parsed and extracted into structured data (e.g. title, genre vectors, embeddings, etc.).

2. **Preprocessing / Feature Transformation**  
   - The raw metadata is passed into preprocessing utilities (`preprocess/preprocessed.py`)  
   - Feature transformations: scaling, encoding categorical features, embedding generation, etc.  
   - Ensure that transformation is consistent with how the model expects input features.

3. **Model Inference / Recommendation Logic**  
   - The preprocessed features feed into the model inference module (`model/classifier.py`).  
   - The MLP model (loaded from saved weights) computes logits or scores for candidate movies.  
   - You might apply ranking, filtering (e.g. discard already seen movies), and pick top-N recommendations.

4. **Return Response to Client**  
   - The backend wraps the recommended results (movie titles, metadata, scores) into JSON (Flask) or directly renders them in Streamlit UI.  
   - The client receives/display the recommendations.

5. **Logging / Export / Output**  
   - Optionally, the system saves logs, outputs, or metrics in the `outputs/` directory for later analysis.

---

## API & Testing Modes: Flask vs Streamlit

Your system supports **two types of deployment / interface / testing**:

### 🧩 Flask (Backend API)

- **Purpose**: to expose a RESTful endpoint(s) for recommendation inference.
- **Workflow**:
  1. Start the Flask server (`app.py`), e.g. `python app.py`  
  2. Client (postman, frontend, or test suite) sends HTTP requests, e.g. POST `/recommend` with payload `{ "movie_id": 123 }`  
  3. The Flask app receives request → calls TMDB client → preprocess → inference → returns JSON response  
- **Testing**: you might assert that status code is 200, and the JSON has expected keys / value types.

### 📱 Streamlit (Interactive UI)

- **Purpose**: to provide a web user interface where a user can input a movie (or genre, preferences) and see recommended movies.
- **Workflow**:
  1. Run the Streamlit app (`streamlit_app.py`) via `streamlit run streamlit_app.py`  
  2. The UI might accept input (e.g. a text box, dropdown)  
  3. On user action, the UI calls (internally) your recommendation logic (this may reuse the same inference modules)  
  4. The UI displays results (movie posters, names, links, etc.)
- **Testing**: less conventional to write automated tests, but you can simulate / mock internal calls or test pieces of the UI logic.

---

## How to Install & Use

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AdityaAnanta123/System-Recommendation-Movies-Using-MLP.git
   cd System-Recommendation-Movies-Using-MLP
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up TMDB API key**

   * You probably have a file or environment variable (e.g. `TMDB_API_KEY`)
   * Ensure it’s configured before running apps.

4. **Train / prepare model (if needed)**

   ```bash
   python src/model/train.py
   ```

5. **Run Flask server**

   ```bash
   python src/test/app.py
   ```

6. **Run Streamlit UI**

   ```bash
   streamlit run src/test/recommended_app.py
   ```

7. **Use / test endpoints or UI**

   * Visit `http://localhost:5000/recommend` (or whatever route)
   * Or open Streamlit local server (default `http://localhost:8501`)
---
