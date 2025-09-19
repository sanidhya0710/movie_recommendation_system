import pickle
import pandas as pd
import streamlit as st
import requests
import json

# Configuration
API_KEY = "f9f86df1941da9ca139ef0d14e36a315"

# Netflix-style CSS
st.markdown("""
<style>
    /* Import Netflix font */
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;600;700&display=swap');

    /* Netflix dark theme */
    .stApp {
        background: #141414 !important;
        color: white !important;
    }

    /* Main container */
    .main {
        background: #141414 !important;
        color: white !important;
    }

    .block-container {
        background: #141414 !important;
        padding-top: 2rem !important;
    }

    /* Title styling */
    h1 {
        color: #E50914 !important;
        font-family: 'Netflix Sans', 'Roboto', sans-serif !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }

    /* Selectbox styling */
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

    /* Button styling */
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

    /* Subheader styling */
    .stApp h2, .stApp h3 {
        color: white !important;
        font-family: 'Netflix Sans', 'Roboto', sans-serif !important;
        font-weight: 600 !important;
        margin: 2rem 0 1rem 0 !important;
    }

    /* Movie cards */
    [data-testid="column"] {
        background: transparent !important;
    }

    /* Movie text styling */
    .stApp p {
        color: white !important;
        font-family: 'Netflix Sans', 'Roboto', sans-serif !important;
    }

    /* Image containers */
    .stImage {
        border-radius: 8px !important;
        transition: transform 0.3s ease !important;
    }

    .stImage:hover {
        transform: scale(1.05) !important;
    }

    /* Error message styling */
    .stError {
        background: rgba(229, 9, 20, 0.1) !important;
        border: 1px solid #E50914 !important;
        color: #E50914 !important;
    }

    /* Remove white backgrounds */
    .element-container {
        background: transparent !important;
    }

    /* Sidebar styling if visible */
    .sidebar .sidebar-content {
        background: #222222 !important;
    }

    /* Text input and other elements */
    .stTextInput > div > div > input {
        background: #333333 !important;
        color: white !important;
        border: 2px solid #E50914 !important;
    }

    /* Netflix-style movie title formatting */
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

    # Netflix-style fallback placeholder
    return f"https://via.placeholder.com/300x450/333333/E50914?text=NETFLIX"


# Load data
@st.cache_data
def load_data():
    movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity


# Get recommendations
def recommend(movie_title, movies, similarity):
    try:
        idx = movies[movies["title"] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

        recommendations = []
        for movie_idx, score in distances[1:6]:
            movie_data = movies.iloc[movie_idx]
            recommendations.append({
                'title': movie_data['title'],
                'id': movie_data['id'],
                'similarity': score
            })
        return recommendations
    except:
        return []


# Set page config
st.set_page_config(
    page_title="Sanidhya Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main app
st.title("🎬 Sanidhya Movie Recommender")

movies, similarity = load_data()

# Create centered layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_movie = st.selectbox("Choose a movie:", movies["title"].values)

    if st.button("Get Recommendations", use_container_width=True):
        recommendations = recommend(selected_movie, movies, similarity)

        if recommendations:
            st.subheader(f"Movies similar to '{selected_movie}':")

            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)

            cols = st.columns(5, gap="medium")
            for i, rec in enumerate(recommendations):
                with cols[i]:
                    poster_url = get_poster(rec['id'])
                    st.image(poster_url, use_container_width=True)

                    # Custom HTML for better text styling
                    st.markdown(f'<div class="movie-title">{rec["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="similarity-score">Match: {rec["similarity"]:.1%}</div>',
                                unsafe_allow_html=True)
        else:
            st.error("No recommendations found!")

# Netflix-style footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #808080; font-size: 0.9rem; margin-top: 3rem;">
    Powered by TMDB API • Netflix-Style Interface
</div>
""", unsafe_allow_html=True)