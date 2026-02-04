import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Movie Recommender System", page_icon="🎬")
st.title("🎬 Movie Recommender System")

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MOVIE_LIST_PATH = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")

# ---------------------------------------------------
# CREATE MODEL FILES IF NOT PRESENT (AUTO-FIX)
# ---------------------------------------------------
def create_dummy_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    movies = pd.DataFrame({
        "title": [
            "Avatar",
            "Titanic",
            "Inception",
            "Interstellar",
            "The Dark Knight",
            "Avengers: Endgame",
            "Iron Man"
        ]
    })

    vectors = np.random.rand(len(movies), 10)
    similarity = cosine_similarity(vectors)

    with open(MOVIE_LIST_PATH, "wb") as f:
        pickle.dump(movies, f)

    with open(SIMILARITY_PATH, "wb") as f:
        pickle.dump(similarity, f)

# ---------------------------------------------------
# LOAD OR CREATE DATA
# ---------------------------------------------------
if not os.path.exists(MOVIE_LIST_PATH) or not os.path.exists(SIMILARITY_PATH):
    st.warning("Model files not found. Creating demo data...")
    create_dummy_model()

@st.cache_data
def load_data():
    with open(MOVIE_LIST_PATH, "rb") as f:
        movies = pickle.load(f)
    with open(SIMILARITY_PATH, "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity

movies, similarity = load_data()

# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------
def recommend(movie_name):
    index = movies[movies["title"] == movie_name].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
selected_movie = st.selectbox("Select a movie", movies["title"].values)

if st.button("Recommend"):
    st.subheader("Recommended Movies")
    for movie in recommend(selected_movie):
        st.write("🎥", movie)
