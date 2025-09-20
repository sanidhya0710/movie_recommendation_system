import pickle
import pandas as pd
import streamlit as st
import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
API_KEY = "f9f86df1941da9ca139ef0d14e36a315"

# Netflix-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: #141414 !important;
        color: white !important;
    }

    .main {
        background: #141414 !important;
        color: white !important;
    }

    .block-container {
        background: #141414 !important;
        padding-top: 2rem !important;
    }

    h1 {
        color: #E50914 !important;
        font-family: 'Netflix Sans', 'Roboto', sans-serif !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }

    .stSelectbox > div > div > div {
        background: #333333 !important;
        color: white !important;
        border: 2px solid #E50914 !important;
        border-radius: 8px !important;
    }

    .stSelectbox label {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #B81D24 0%, #E50914 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6) !important;
    }

    .stApp h2, .stApp h3 {
        color: white !important;
        font-family: 'Netflix Sans', 'Roboto', sans-serif !important;
        font-weight: 600 !important;
        margin: 2rem 0 1rem 0 !important;
    }

    [data-testid="column"] {
        background: transparent !important;
    }

    .stApp p {
        color: white !important;
        font-family: 'Netflix Sans', 'Roboto', sans-serif !important;
    }

    .stImage {
        border-radius: 8px !important;
        transition: transform 0.3s ease !important;
    }

    .stImage:hover {
        transform: scale(1.05) !important;
    }

    .stError {
        background: rgba(229, 9, 20, 0.1) !important;
        border: 1px solid #E50914 !important;
        color: #E50914 !important;
    }

    .element-container {
        background: transparent !important;
    }

    .movie-title {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-align: center !important;
        margin: 0.5rem 0 !important;
        font-family: 'Netflix Sans', 'Roboto', sans-serif !important;
    }

    .similarity-score {
        color: #46D369 !important;
        font-weight: 500 !important;
        text-align: center !important;
        font-size: 0.9rem !important;
    }

    .stSpinner > div {
        border-top-color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)


# Get poster via proxy
@st.cache_data
def get_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    proxy_url = f"https://api.allorigins.win/get?url={requests.utils.quote(url)}"

    try:
        response = requests.get(proxy_url, timeout=10)
        if response.status_code == 200:
            data = json.loads(response.json().get("contents", "{}"))
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w300{poster_path}"
    except:
        pass

    return f"https://via.placeholder.com/300x450/333333/E50914?text=NETFLIX"


# Load data (movies only)
@st.cache_data
def load_data():
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)
    return movies


# Create feature vector for similarity calculation
@st.cache_data
def create_feature_matrix(movies):
    """Create feature matrix from movie data for real-time similarity calculation"""

    # Combine relevant features into a single text string
    # Adjust these columns based on what's available in your movie_dict.pkl
    feature_columns = []

    # Check which columns are available and use them
    if 'genres' in movies.columns:
        feature_columns.append('genres')
    if 'keywords' in movies.columns:
        feature_columns.append('keywords')
    if 'cast' in movies.columns:
        feature_columns.append('cast')
    if 'crew' in movies.columns:
        feature_columns.append('crew')
    if 'overview' in movies.columns:
        feature_columns.append('overview')
    if 'director' in movies.columns:
        feature_columns.append('director')
    if 'production_companies' in movies.columns:
        feature_columns.append('production_companies')

    # Enhanced feature combination with weights
    if feature_columns:
        movies['combined_features'] = movies[feature_columns].fillna('').apply(
            lambda x: ' '.join(x.astype(str)), axis=1
        )

        # Add title words for better matching
        movies['combined_features'] = movies['combined_features'] + ' ' + movies['title'].fillna('')

        # Add repeated important features for higher weight
        if 'genres' in movies.columns:
            movies['combined_features'] = movies['combined_features'] + ' ' + movies['genres'].fillna('') * 3
    else:
        # Fallback: use title with some processing
        movies['combined_features'] = movies['title'].fillna('').str.lower()

    # Create TF-IDF matrix with optimized parameters
    tfidf = TfidfVectorizer(
        max_features=8000,  # Increased features
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=1,  # Include rare terms
        max_df=0.95,  # Exclude very common terms
        sublinear_tf=True  # Apply sublinear tf scaling
    )

    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

    return tfidf_matrix


# Get recommendations using real-time similarity calculation
@st.cache_data
def recommend(movie_title, movies, _tfidf_matrix):
    """Calculate recommendations in real-time with improved similarity"""
    try:
        # Find the index of the selected movie
        idx = movies[movies['title'] == movie_title].index[0]

        # Calculate cosine similarity for the selected movie with all others
        movie_vector = _tfidf_matrix[idx:idx + 1]
        cosine_similarities = cosine_similarity(movie_vector, _tfidf_matrix).flatten()

        # Get indices of movies sorted by similarity (excluding the movie itself)
        similar_indices = cosine_similarities.argsort()[::-1]

        recommendations = []
        added_movies = set()

        # Get top recommendations, ensuring we don't include the same movie
        for movie_idx in similar_indices:
            if len(recommendations) >= 5:
                break

            if movie_idx == idx:  # Skip the selected movie itself
                continue

            movie_data = movies.iloc[movie_idx]
            movie_title_clean = movie_data['title']

            # Avoid duplicate movies (sometimes there are slight variations)
            if movie_title_clean not in added_movies:
                similarity_score = cosine_similarities[movie_idx]

                # Apply minimum similarity boost to avoid 0 scores
                if similarity_score < 0.01:
                    # Use title similarity as fallback
                    title_similarity = calculate_title_similarity(movie_title, movie_title_clean)
                    similarity_score = max(similarity_score, title_similarity * 0.1)

                recommendations.append({
                    'title': movie_title_clean,
                    'id': movie_data['id'],
                    'similarity': similarity_score
                })
                added_movies.add(movie_title_clean)

        # If still very low similarities, add some popular/random movies as fallback
        if len(recommendations) < 5 or all(r['similarity'] < 0.05 for r in recommendations):
            fallback_movies = get_fallback_recommendations(movies, idx)
            for fallback in fallback_movies:
                if len(recommendations) >= 5:
                    break
                if fallback['title'] not in added_movies:
                    recommendations.append(fallback)
                    added_movies.add(fallback['title'])

        return recommendations[:5]  # Return top 5

    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []


def calculate_title_similarity(title1, title2):
    """Calculate basic title similarity as fallback"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()


def get_fallback_recommendations(movies, selected_idx):
    """Get fallback recommendations when similarity is too low"""
    fallback_movies = []

    # Get some popular movies or random movies as fallbacks
    popular_indices = movies.sample(n=min(10, len(movies))).index.tolist()

    for idx in popular_indices:
        if idx != selected_idx:
            movie_data = movies.iloc[idx]
            fallback_movies.append({
                'title': movie_data['title'],
                'id': movie_data['id'],
                'similarity': 0.15  # Give a reasonable fallback similarity
            })

    return fallback_movies


# Set page config
st.set_page_config(
    page_title="Ssnidhya Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main app
st.title("🎬 Sanidhya Movie Recommender")

# Load data and create feature matrix
with st.spinner("Loading movie database..."):
    movies = load_data()
    tfidf_matrix = create_feature_matrix(movies)

# Create centered layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_movie = st.selectbox(
        "Choose a movie:",
        movies["title"].values,
        help="Select any movie to get AI-powered recommendations"
    )

    if st.button("Get Recommendations", use_container_width=True):
        with st.spinner("Finding perfect matches..."):
            recommendations = recommend(selected_movie, movies, tfidf_matrix)

        if recommendations:
            st.subheader(f"Movies similar to '{selected_movie}':")
            st.markdown("<br>", unsafe_allow_html=True)

            cols = st.columns(5, gap="medium")
            for i, rec in enumerate(recommendations):
                with cols[i]:
                    poster_url = get_poster(rec['id'])
                    st.image(poster_url, use_container_width=True)

                    st.markdown(f'<div class="movie-title">{rec["title"]}</div>', unsafe_allow_html=True)
        else:
            st.error("No recommendations found! Please try a different movie.")

# Quick movie suggestions
if 'recommendations' not in locals() or not recommendations:
    st.markdown("---")
    st.markdown("### 🔥 Try These Popular Movies")

    popular_movies = [
        "The Dark Knight", "Avatar", "Inception", "Titanic",
        "The Matrix", "Interstellar", "Pulp Fiction", "The Godfather"
    ]

    # Filter to available movies
    available_popular = [movie for movie in popular_movies if movie in movies["title"].values]

    if available_popular:
        cols = st.columns(min(4, len(available_popular)))
        for i, movie in enumerate(available_popular[:4]):
            with cols[i]:
                if st.button(f"🎬 {movie}", key=f"pop_{i}", use_container_width=True):
                    st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #808080; font-size: 0.9rem; margin-top: 3rem;">
    Powered by TMDB API • Real-time AI Recommendations
</div>
""", unsafe_allow_html=True)