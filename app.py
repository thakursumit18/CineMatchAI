import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import random

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #2d4059;
    }
    .stButton button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF416C 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: bold;
        width: 100%;
    }
    .movie-poster {
        border-radius: 8px;
        width: 100%;
        height: 250px;
        object-fit: cover;
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF4B4B, #FF416C);
    }
    .small-text { font-size: 0.85em; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎬 CineMatch AI - Movie Recommender</h1>', unsafe_allow_html=True)

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MOVIE_LIST_PATH = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")
MOVIE_DETAILS_PATH = os.path.join(MODEL_DIR, "movie_details.pkl")

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
DATASET_SIZE = 1500  # Optimal size for performance


# ---------------------------------------------------
# POSTER FUNCTION
# ---------------------------------------------------
def get_movie_poster(movie_title, genre=None):
    """Generate movie poster URL"""
    try:
        # Clean title
        clean_title = movie_title.split('(')[0].strip().replace(' ', '+')[:15]

        # Genre-based colors
        genre_colors = {
            "Action": "dc2626", "Adventure": "ea580c", "Animation": "0d9488",
            "Comedy": "ca8a04", "Crime": "7c2d12", "Drama": "1d4ed8",
            "Fantasy": "7c3aed", "Horror": "581c87", "Mystery": "374151",
            "Romance": "db2777", "Sci-Fi": "16a34a", "Thriller": "dc2626"
        }

        bg_color = "1a1a2e"  # Default
        if genre:
            for g in str(genre).split(','):
                g_clean = g.strip()
                if g_clean in genre_colors:
                    bg_color = genre_colors[g_clean]
                    break

        # Create poster URL
        return f"https://placehold.co/400x550/{bg_color}/ffffff?text={clean_title}&font=montserrat"

    except:
        return "https://placehold.co/400x550/333/fff?text=Movie+Poster"


# ---------------------------------------------------
# CREATE DATASET - SIMPLIFIED
# ---------------------------------------------------
def create_dataset():
    """Create a movie dataset"""
    st.info(f"🔄 Creating dataset with {DATASET_SIZE} movies...")

    # Real movie titles for better recommendations
    base_titles = [
        "The Matrix", "Inception", "Interstellar", "The Dark Knight", "Pulp Fiction",
        "Forrest Gump", "Fight Club", "Goodfellas", "The Godfather", "Star Wars",
        "Lord of the Rings", "Harry Potter", "Avatar", "Titanic", "Jurassic Park",
        "Iron Man", "Spider-Man", "Batman", "Superman", "Wonder Woman",
        "Toy Story", "Finding Nemo", "The Lion King", "Frozen", "Shrek",
        "Psycho", "The Shining", "Alien", "Terminator", "Die Hard"
    ]

    genres_list = ["Action", "Adventure", "Drama", "Comedy", "Sci-Fi",
                   "Thriller", "Horror", "Romance", "Mystery", "Fantasy"]

    # Create dataset
    movie_data = []
    for i in range(DATASET_SIZE):
        year = np.random.randint(1980, 2024)
        genre_count = np.random.randint(1, 3)
        selected_genres = np.random.choice(genres_list, genre_count, replace=False)

        # Create title
        base = np.random.choice(base_titles)
        if np.random.random() > 0.5:
            suffixes = ["", "Returns", "Reborn", "Legacy", "Chronicles", "Saga"]
            suffix = np.random.choice(suffixes)
            title = f"{base}: {suffix}" if suffix else base
        else:
            title = base

        title = f"{title} ({year})"

        # Create overview
        overviews = [
            f"A gripping {selected_genres[0].lower()} film that explores deep themes.",
            f"An epic journey through {np.random.choice(['time', 'space', 'emotions', 'reality'])}.",
            f"Critically acclaimed for its {np.random.choice(['performances', 'direction', 'cinematography', 'story'])}.",
            f"A story of {np.random.choice(['redemption', 'love', 'courage', 'betrayal'])} that will stay with you."
        ]

        overview = np.random.choice(overviews)

        movie_data.append({
            'id': str(10000 + i),
            'title': title,
            'genres': ', '.join(selected_genres),
            'overview': overview,
            'vote_average': np.round(np.random.uniform(6.0, 9.0), 1),
            'release_date': pd.to_datetime(f"{year}-{np.random.randint(1, 13)}-01")
        })

    movies_df = pd.DataFrame(movie_data)

    # Save data
    with open(MOVIE_LIST_PATH, 'wb') as f:
        pickle.dump(movies_df, f)

    # Create and save similarity matrix
    movies_df['features'] = movies_df['genres'] + ' ' + movies_df['overview']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    features = vectorizer.fit_transform(movies_df['features'])

    # Compute similarity matrix
    st.info("🔄 Computing similarities...")
    similarity_matrix = cosine_similarity(features)

    with open(SIMILARITY_PATH, 'wb') as f:
        pickle.dump(similarity_matrix, f)

    # Create movie details
    movie_details = {}
    for idx, movie in movies_df.iterrows():
        movie_details[movie['title']] = {
            'id': movie['id'],
            'genres': movie['genres'],
            'overview': movie['overview'],
            'rating': float(movie['vote_average']),
            'year': str(movie['release_date'].year) if pd.notnull(movie['release_date']) else 'N/A'
        }

    with open(MOVIE_DETAILS_PATH, 'wb') as f:
        pickle.dump(movie_details, f)

    st.success(f"✅ Created dataset with {len(movies_df)} movies")
    return movies_df, similarity_matrix, movie_details


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
def load_data():
    """Load or create data"""
    try:
        # Check if files exist and are not corrupted
        if (os.path.exists(MOVIE_LIST_PATH) and os.path.exists(SIMILARITY_PATH) and
                os.path.exists(MOVIE_DETAILS_PATH)):

            # Try to load
            with open(MOVIE_LIST_PATH, 'rb') as f:
                movies_df = pickle.load(f)

            with open(SIMILARITY_PATH, 'rb') as f:
                similarity_matrix = pickle.load(f)

            with open(MOVIE_DETAILS_PATH, 'rb') as f:
                movie_details = pickle.load(f)

            # Check if sizes match
            if len(movies_df) == similarity_matrix.shape[0] == similarity_matrix.shape[1]:
                return movies_df, similarity_matrix, movie_details

        # If files don't exist or are corrupted, create new
        return create_dataset()

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_dataset()


# ---------------------------------------------------
# LOAD THE DATA
# ---------------------------------------------------
with st.spinner('Loading movie database...'):
    movies_df, similarity_matrix, movie_details = load_data()

# Display dataset info
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.write(f"**Movies:** {len(movies_df):,}")
st.sidebar.write(f"**Last updated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2809/2809593.png", width=80)
    st.title("🎯 Filters")

    # Search
    search_term = st.text_input("🔍 Search Movies", placeholder="Type movie name...")

    # Genre filter
    all_genres = set()
    for genres in movies_df['genres']:
        if isinstance(genres, str):
            all_genres.update([g.strip() for g in genres.split(',')])

    selected_genre = st.selectbox("🎭 Filter by Genre", ["All"] + sorted(list(all_genres)))

    # Rating filter
    min_rating = st.slider("⭐ Min Rating", 0.0, 10.0, 6.0, 0.5)

    # Year range
    years = []
    for date in movies_df['release_date'].dropna():
        try:
            years.append(date.year)
        except:
            continue

    if years:
        min_year, max_year = st.slider("📅 Release Years", min(years), max(years), (1990, 2023))

    st.markdown("---")

    if st.button("🔄 Clear Cache & Regenerate"):
        try:
            os.remove(MOVIE_LIST_PATH)
            os.remove(SIMILARITY_PATH)
            os.remove(MOVIE_DETAILS_PATH)
        except:
            pass
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------
# RECOMMENDATION FUNCTION - SIMPLIFIED
# ---------------------------------------------------
def recommend(movie_name, n_recommendations=10):
    """Get movie recommendations"""
    try:
        # Find movie index
        if movie_name not in movies_df['title'].values:
            # Try to find similar
            matches = movies_df[movies_df['title'].str.contains(movie_name, case=False, na=False)]
            if len(matches) > 0:
                movie_name = matches.iloc[0]['title']
            else:
                st.error(f"Movie '{movie_name}' not found.")
                return []

        idx = movies_df[movies_df['title'] == movie_name].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))

        # Apply filters
        filtered_scores = []
        for i, score in sim_scores:
            if i == idx:  # Skip itself
                continue

            movie_info = movies_df.iloc[i]

            # Genre filter
            if selected_genre != "All" and selected_genre not in movie_info['genres']:
                continue

            # Rating filter
            if movie_info['vote_average'] < min_rating:
                continue

            # Year filter
            year = movie_info['release_date'].year if pd.notnull(movie_info['release_date']) else 0
            if years and not (min_year <= year <= max_year):
                continue

            filtered_scores.append((i, score))

        # Get top recommendations
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in filtered_scores[:n_recommendations]]

        recommendations = []
        for i in top_indices:
            movie_info = movies_df.iloc[i]
            recommendations.append({
                'title': movie_info['title'],
                'similarity': round(similarity_matrix[idx][i] * 100, 1),
                'rating': float(movie_info['vote_average']),
                'genres': movie_info['genres'],
                'year': movie_info['release_date'].year if pd.notnull(movie_info['release_date']) else 'N/A',
                'id': movie_info['id']
            })

        return recommendations

    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")
        return []


# ---------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------
# Create tabs
tab1, tab2 = st.tabs(["🎬 Get Recommendations", "🔍 Browse Movies"])

with tab1:
    # Movie selection
    if search_term:
        filtered_movies = movies_df[movies_df['title'].str.contains(search_term, case=False, na=False)]
        if len(filtered_movies) > 0:
            movie_options = filtered_movies['title'].tolist()
        else:
            movie_options = movies_df['title'].tolist()
    else:
        movie_options = movies_df['title'].tolist()

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_movie = st.selectbox(
            "Select a movie for recommendations:",
            movie_options,
            index=0
        )
    with col2:
        num_rec = st.selectbox("Show", [6, 9, 12], index=1)

    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner('Finding similar movies...'):
            recommendations = recommend(selected_movie, n_recommendations=num_rec)

        if recommendations:
            # Show selected movie
            selected_info = movie_details.get(selected_movie, {})

            col_img, col_info = st.columns([1, 2])
            with col_img:
                poster_url = get_movie_poster(selected_movie, selected_info.get('genres'))
                st.image(poster_url, use_column_width=True)

            with col_info:
                st.markdown(f"### 🎬 {selected_movie}")
                st.write(f"**Rating:** ⭐ {selected_info.get('rating', 'N/A')}/10")
                st.write(f"**Year:** {selected_info.get('year', 'N/A')}")
                st.write(f"**Genres:** {selected_info.get('genres', 'N/A')}")
                if selected_info.get('overview'):
                    st.caption(f"*{selected_info['overview']}*")

            st.markdown("---")
            st.subheader(f"🎯 Top {len(recommendations)} Recommendations")

            # Display recommendations
            cols = st.columns(3)
            for idx, rec in enumerate(recommendations):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                        # Poster
                        poster_url = get_movie_poster(rec['title'], rec['genres'])
                        st.image(poster_url, use_column_width=True)

                        # Title
                        st.markdown(f"**{rec['title']}**")

                        # Details
                        st.write(f"⭐ {rec['rating']}/10 | 📅 {rec['year']}")

                        # Similarity
                        similarity_text = f"🎯 **{rec['similarity']}% match**"
                        st.write(similarity_text)
                        st.progress(rec['similarity'] / 100)

                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Try different filters or select another movie.")

with tab2:
    st.subheader("Browse All Movies")

    # Quick search in browse
    browse_search = st.text_input("Search:", key="browse_search")

    # Filter movies
    browse_df = movies_df.copy()
    if browse_search:
        browse_df = browse_df[browse_df['title'].str.contains(browse_search, case=False, na=False)]

    if selected_genre != "All":
        browse_df = browse_df[browse_df['genres'].str.contains(selected_genre, case=False)]

    browse_df = browse_df[browse_df['vote_average'] >= min_rating]

    # Apply year filter
    if years:
        browse_df = browse_df[
            (browse_df['release_date'].dt.year >= min_year) &
            (browse_df['release_date'].dt.year <= max_year)
            ]

    st.write(f"**Showing {len(browse_df)} movies**")

    # Display in grid
    if len(browse_df) > 0:
        # Pagination
        items_per_page = 9
        total_pages = max(1, len(browse_df) // items_per_page + (1 if len(browse_df) % items_per_page > 0 else 0))
        page = st.number_input("Page", 1, total_pages, 1, key="browse_page")

        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page

        # Display grid
        cols = st.columns(3)
        for idx in range(start_idx, min(end_idx, len(browse_df))):
            movie = browse_df.iloc[idx]
            with cols[idx % 3]:
                poster_url = get_movie_poster(movie['title'], movie['genres'])
                st.image(poster_url, use_column_width=True)
                st.write(f"**{movie['title']}**")
                st.write(f"⭐ {movie['vote_average']:.1f} | 📅 {movie['release_date'].year}")

                if st.button("Select", key=f"select_{idx}", use_container_width=True):
                    # Store in session state and switch to recommendations tab
                    st.session_state.selected_movie_for_rec = movie['title']
                    st.rerun()

# Handle movie selection from browse
if 'selected_movie_for_rec' in st.session_state:
    selected_movie = st.session_state.selected_movie_for_rec
    del st.session_state.selected_movie_for_rec
    # Switch to recommendations tab
    st.rerun()

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption(f"🎬 CineMatch AI • {len(movies_df):,} Movies • Powered by Cosine Similarity")