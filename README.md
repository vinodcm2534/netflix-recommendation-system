# Movie Recommendation System

A content-based recommendation engine that understands what makes movies similar — not by ratings or popularity, but by the actual meaning in their descriptions. Built with TF-IDF and Cosine Similarity, wrapped in a clean Streamlit web app.

---

## What It Does

Type in a movie you like. Get back a list of movies that feel like it — same genre DNA, same themes, same vibe. No collaborative filtering, no black-box neural network — just clean NLP on movie metadata that you can actually reason about.

---

## How It Works

The core idea is straightforward:

1. **Vectorize** — Each movie description is converted into a TF-IDF vector. Words that are common across all movies (like "the", "a") get penalized; words that are distinctive to specific movies get amplified.

2. **Compare** — Cosine similarity measures the angle between any two movie vectors. A score of 1 means identical direction (very similar), 0 means completely unrelated.

3. **Recommend** — Given an input movie, we rank all other movies by their cosine similarity score and return the top N.

The TF-IDF vectorizer and similarity matrix are precomputed and saved as `.joblib` files so the app loads instantly without recomputing on every request.

---

## Project Structure

```
├── analysis/
│   └── lets_recommend.ipynb     # EDA, feature engineering, model building walkthrough
├── data/
│   └── mymoviedb.csv            # Movie dataset with titles, descriptions, and metadata
├── model/
│   ├── TF-IDF-Vectorizer.joblib # Serialized TF-IDF vectorizer
│   └── Cosine-Similarity.joblib # Precomputed similarity matrix
└── web-app/
    └── app.py                   # Streamlit frontend
```

---

## Getting Started

**Clone the repo**
```bash
git clone https://github.com/mynkchn/netflix-recommendation-system.git
cd netflix-recommendation-system
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the app**
```bash
cd web-app
streamlit run app.py
```

Open `http://localhost:8501` in your browser — type a movie title and hit recommend.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.x |
| NLP / ML | scikit-learn (TF-IDF, Cosine Similarity) |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Model persistence | joblib |
| Web interface | Streamlit |

---

## Model Details

**TF-IDF (Term Frequency–Inverse Document Frequency)**  
Converts raw movie text into numerical vectors where each dimension represents a word's importance — high if the word is frequent in that movie's description but rare across the whole dataset.

**Cosine Similarity**  
Measures directional similarity between two vectors, making it robust to differences in description length. Two movies with similar vocabulary will have vectors pointing in similar directions regardless of how long their descriptions are.

Both the fitted vectorizer and the full similarity matrix are serialized and loaded at runtime — keeping inference fast and stateless.

---

## Sample Output

```
Input:  Inception
Output: Interstellar, The Matrix, Shutter Island, Tenet, Memento
```

---

## Author

**Vinod Malage**  
B.Tech CSE 
