# streamlit/pages/3_📌_recommendation.py

import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# Fix import typos and add error handling
try:
    from models.collaborative_filtering.book_item_cf import recommend_books as book_cf
except ImportError as e:
    st.error(f"❌ Book CF import error: {e}")
    book_cf = None

try:
    from models.collaborative_filtering.movie_item_cf import recommend_movies as movie_cf
except ImportError as e:
    st.error(f"❌ Movie CF import error: {e}")
    movie_cf = None

try:
    from models.collaborative_filtering.music_cf import recommend_artists as music_cf
except ImportError as e:
    st.error(f"❌ Music CF import error: {e}")
    music_cf = None

try:
    from models.matrix_factorization.book_svd import recommend_books as book_svd
except ImportError as e:
    st.error(f"❌ Book SVD import error: {e}")
    book_svd = None

try:
    from models.matrix_factorization.movie_svd import recommend_movies as movie_svd
except ImportError as e:
    st.error(f"❌ Movie SVD import error: {e}")
    movie_svd = None

try:
    from models.matrix_factorization.music_svd import recommend_artists as music_svd
except ImportError as e:
    st.error(f"❌ Music SVD import error: {e}")
    music_svd = None

# Popularity based imports with error handling
try:
    from models.popularity_based.book_popularity import get_top_books
except ImportError as e:
    st.error(f"❌ Book popularity import error: {e}")
    get_top_books = None

try:
    from models.popularity_based.movie_popularity import get_top_movies
except ImportError as e:
    st.error(f"❌ Movie popularity import error: {e}")
    get_top_movies = None

try:
    from models.popularity_based.music_popularity import get_top_artists
except ImportError as e:
    st.error(f"❌ Music popularity import error: {e}")
    get_top_artists = None

st.set_page_config(page_title="Recommendations", page_icon="📌", layout="wide")
st.title("📌 Multi-Domain Recommendations")

st.markdown("""
Select a recommendation algorithm and provide a user ID to get suggestions for:
- 📚 Books
- 🎬 Movies
- 🎵 Music

This will return recommendations from all domains based on your selected algorithm.
""")

@st.cache_data
def load_book_data():
    try:
        filtered_df = pd.read_csv("../data/processed/book/book_filtered_rating.csv")
        book_info_df = pd.read_csv("../data/processed/book/book_info.csv")

        user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='book_id', values='Rating')

        # Always compute fresh similarity matrix to ensure alignment
        item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
        similarity_df = pd.DataFrame(item_similarity,
                                     index=user_item_matrix.columns,
                                     columns=user_item_matrix.columns)

        return user_item_matrix, similarity_df, book_info_df
    except Exception as e:
        st.error(f"❌ Error loading book data: {e}")
        return None, None, None


@st.cache_data
def load_movie_data():
    try:
        # Check if files exist
        movie_rating_path = "../data/processed/movie/movie_rating.csv"
        movie_info_path = "../data/processed/movie/movie_info.csv"
        
        if not os.path.exists(movie_rating_path):
            st.error(f"❌ Movie rating file not found: {movie_rating_path}")
            return None, None, None
            
        if not os.path.exists(movie_info_path):
            st.error(f"❌ Movie info file not found: {movie_info_path}")
            return None, None, None
        
        filtered_df = pd.read_csv(movie_rating_path)
        movie_info_df = pd.read_csv(movie_info_path)

        # Debug: Show column names
        st.write("🔍 Movie rating columns:", filtered_df.columns.tolist())
        st.write("🔍 Movie info columns:", movie_info_df.columns.tolist())

        user_item_matrix = filtered_df.pivot_table(index='user_id', columns='item_id', values='rating')

        # Always compute fresh similarity matrix to ensure alignment
        item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
        similarity_df = pd.DataFrame(item_similarity,
                                     index=user_item_matrix.columns,
                                     columns=user_item_matrix.columns)

        return user_item_matrix, similarity_df, movie_info_df
    except Exception as e:
        st.error(f"❌ Error loading movie data: {e}")
        st.exception(e)
        return None, None, None


@st.cache_data
def load_music_data():
    try:
        filtered_df = pd.read_csv("../data/processed/music/music_rating.csv")
        music_info_df = pd.read_csv("../data/processed/music/music_info.csv")

        user_item_matrix = filtered_df.pivot_table(index='user_id', columns='artist_id', values='weight')

        # Always compute fresh similarity matrix to ensure alignment
        item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
        similarity_df = pd.DataFrame(item_similarity,
                                     index=user_item_matrix.columns,
                                     columns=user_item_matrix.columns)

        return user_item_matrix, similarity_df, music_info_df
    except Exception as e:
        st.error(f"❌ Error loading music data: {e}")
        return None, None, None


st.divider()

# Dropdown for algorithm type
algorithm = st.selectbox("🤖 Select Recommendation Algorithm", [
    "Collaborative Filtering",
    "Matrix Factorization (SVD)",
    "Popularity Based"
])

user_id = st.number_input("🔢 Enter User ID", min_value=1, step=1)

if st.button("🎯 Get Recommendations"):
    if algorithm == "Collaborative Filtering":
        st.subheader("📚 Book Recommendations (Collaborative Filtering)")
        try:
            if book_cf is None:
                st.error("❌ Book CF module not available")
            else:
                book_matrix, book_sim, book_info = load_book_data()
                if book_matrix is not None:
                    recommendations = book_cf(user_id, book_matrix, book_sim, book_info)
                    st.dataframe(recommendations, use_container_width=True)
                else:
                    st.error("❌ Failed to load book data")
        except Exception as e:
            st.error(f"Book CF error: {e}")
            st.exception(e)

        st.subheader("🎬 Movie Recommendations (Collaborative Filtering)")
        try:
            if movie_cf is None:
                st.error("❌ Movie CF module not available")
            else:
                movie_matrix, movie_sim, movie_info = load_movie_data()
                
                # Debug information
                if movie_matrix is not None:
                    st.write(f"🔍 User {user_id} exists in matrix: {user_id in movie_matrix.index}")
                    st.write(f"🔍 Matrix shape: {movie_matrix.shape}")
                    st.write(f"🔍 Available users (first 10): {movie_matrix.index.tolist()[:10]}")
                    
                    if user_id in movie_matrix.index:
                        user_ratings = movie_matrix.loc[user_id].dropna()
                        st.write(f"🔍 User {user_id} has {len(user_ratings)} ratings")
                        st.write(f"🔍 User ratings: {user_ratings.head().to_dict()}")
                    
                    recommendations = movie_cf(user_id, movie_matrix, movie_sim, movie_info)
                    st.dataframe(recommendations, use_container_width=True)
                else:
                    st.error("❌ Failed to load movie data")
        except Exception as e:
            st.error(f"Movie CF error: {e}")
            st.exception(e)

        st.subheader("🎵 Music Recommendations (Collaborative Filtering)")
        try:
            if music_cf is None:
                st.error("❌ Music CF module not available")
            else:
                music_matrix, music_sim, music_info = load_music_data()
                if music_matrix is not None:
                    recommendations = music_cf(user_id, music_matrix, music_sim, music_info)
                    st.dataframe(recommendations, use_container_width=True)
                else:
                    st.error("❌ Failed to load music data")
        except Exception as e:
            st.error(f"Music CF error: {e}")
            st.exception(e)

    elif algorithm == "Matrix Factorization (SVD)":
        st.subheader("📚 Book Recommendations (SVD)")
        try:
            if book_svd is None:
                st.error("❌ Book SVD module not available")
            else:
                # Load data for SVD
                book_matrix, _, book_info = load_book_data()
                if book_matrix is not None:
                    # You'll need to implement SVD matrix loading
                    st.warning("⚠️ SVD matrix loading not implemented yet")
                else:
                    st.error("❌ Failed to load book data")
        except Exception as e:
            st.error(f"Book SVD error: {e}")

        st.subheader("🎬 Movie Recommendations (SVD)")
        try:
            if movie_svd is None:
                st.error("❌ Movie SVD module not available")
            else:
                # Load data for SVD
                movie_matrix, _, movie_info = load_movie_data()
                if movie_matrix is not None:
                    # You'll need to implement SVD matrix loading
                    st.warning("⚠️ SVD matrix loading not implemented yet")
                else:
                    st.error("❌ Failed to load movie data")
        except Exception as e:
            st.error(f"Movie SVD error: {e}")

        st.subheader("🎵 Music Recommendations (SVD)")
        try:
            if music_svd is None:
                st.error("❌ Music SVD module not available")
            else:
                # Load data for SVD
                music_matrix, _, music_info = load_music_data()
                if music_matrix is not None:
                    # You'll need to implement SVD matrix loading
                    st.warning("⚠️ SVD matrix loading not implemented yet")
                else:
                    st.error("❌ Failed to load music data")
        except Exception as e:
            st.error(f"Music SVD error: {e}")

    elif algorithm == "Popularity Based":
        st.subheader("📚 Top Books (Popularity)")
        try:
            if get_top_books is None:
                st.error("❌ Book popularity module not available")
            else:
                # Load data for popularity
                book_info = pd.read_csv("../data/processed/book/book_info.csv")
                book_ratings = pd.read_csv("../data/processed/book/book_rating.csv")
                recommendations = get_top_books(book_ratings, book_info)
                st.dataframe(recommendations, use_container_width=True)
        except Exception as e:
            st.error(f"Popular Book error: {e}")

        st.subheader("🎬 Top Movies (Popularity)")
        try:
            if get_top_movies is None:
                st.error("❌ Movie popularity module not available")
            else:
                # Load data for popularity
                movie_info = pd.read_csv("../data/processed/movie/movie_info.csv")
                movie_ratings = pd.read_csv("../data/processed/movie/movie_rating.csv")
                recommendations = get_top_movies(movie_ratings, movie_info)
                st.dataframe(recommendations, use_container_width=True)
        except Exception as e:
            st.error(f"Popular Movie error: {e}")

        st.subheader("🎵 Top Music (Popularity)")
        try:
            if get_top_artists is None:
                st.error("❌ Music popularity module not available")
            else:
                # Load data for popularity
                music_info = pd.read_csv("../data/processed/music/music_info.csv")
                music_ratings = pd.read_csv("../data/processed/music/music_rating.csv")
                recommendations = get_top_artists(music_ratings, music_info)
                st.dataframe(recommendations, use_container_width=True)
        except Exception as e:
            st.error(f"Popular Music error: {e}") 