# streamlit/pages/3_📌_recommendation.py

import streamlit as st
import pandas as pd

from models.colaborative_filtering.book_item_cf import (
    recommend_books as book_cf,
    # load_book_data
)
from models.colaborative_filtering.movie_cf import (
    recommend_movies as movie_cf,
    # load_movie_data
)
from models.colaborative_filtering.music_cf import (
    recommend_artists as music_cf,
    # load_music_data
)

from models.matrix_factorization.book_svd import recommend_books_svd as book_svd
from models.matrix_factorization.movies_svd import recommend_movies_svd as movie_svd
from models.matrix_factorization.music_svd import recommend_music_svd as music_svd

from models.popularity_based.book_popularity import get_popular_books
from models.popularity_based.movie_popularity import get_popular_movies
from models.popularity_based.music_popularity import get_popular_music


st.set_page_config(page_title="Recommendations", page_icon="📌", layout="wide")
st.title("📌 Multi-Domain Recommendations")

st.markdown("""
Select a recommendation algorithm and provide a user ID to get suggestions for:
- 📚 Books
- 🎬 Movies
- 🎵 Music

This will return recommendations from all domains based on your selected algorithm.
""")



import streamlit as st
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_book_data():
    filtered_df = pd.read_csv("data/processed/book/book_filtered_rating.csv")
    book_info_df = pd.read_csv("data/processed/book/book_info.csv")

    user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='book_id', values='Rating')

    similarity_path = "models/colaborative_filtering/book_item_similarity_matrix.csv"
    if os.path.exists(similarity_path):
        similarity_df = pd.read_csv(similarity_path, index_col=0)
        similarity_df.columns = similarity_df.columns.astype(int)
        similarity_df.index = similarity_df.index.astype(int)
    else:
        item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
        similarity_df = pd.DataFrame(item_similarity,
                                     index=user_item_matrix.columns,
                                     columns=user_item_matrix.columns)
        similarity_df.to_csv(similarity_path)

    return user_item_matrix, similarity_df, book_info_df


@st.cache_data
def load_movie_data():
    filtered_df = pd.read_csv("data/processed/movie/movie_rating.csv")
    movie_info_df = pd.read_csv("data/processed/movie/movie_info.csv")

    user_item_matrix = filtered_df.pivot_table(index='user_id', columns='item_id', values='rating')

    similarity_path = "models/colaborative_filtering/movie_item_similarity_matrix.csv"
    if os.path.exists(similarity_path):
        similarity_df = pd.read_csv(similarity_path, index_col=0)
        similarity_df.columns = similarity_df.columns.astype(int)
        similarity_df.index = similarity_df.index.astype(int)
    else:
        item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
        similarity_df = pd.DataFrame(item_similarity,
                                     index=user_item_matrix.columns,
                                     columns=user_item_matrix.columns)
        similarity_df.to_csv(similarity_path)

    return user_item_matrix, similarity_df, movie_info_df


@st.cache_data
def load_music_data():
    filtered_df = pd.read_csv("data/processed/music/music_rating.csv")
    music_info_df = pd.read_csv("data/processed/music/music_info.csv")

    user_item_matrix = filtered_df.pivot_table(index='user_id', columns='artist_id', values='weight')

    similarity_path = "models/colaborative_filtering/music_item_similarity_matrix.csv"
    if os.path.exists(similarity_path):
        similarity_df = pd.read_csv(similarity_path, index_col=0)
        similarity_df.columns = similarity_df.columns.astype(int)
        similarity_df.index = similarity_df.index.astype(int)
    else:
        item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
        similarity_df = pd.DataFrame(item_similarity,
                                     index=user_item_matrix.columns,
                                     columns=user_item_matrix.columns)
        similarity_df.to_csv(similarity_path)

    return user_item_matrix, similarity_df, music_info_df


# --- USERNAME SELECTION LOGIC ---
# Load user files with usernames for each domain
book_user_df = pd.read_csv(r'data\processed\book\book_user_with_names.csv')
movie_user_df = pd.read_csv(r'data\processed\movie\movie_user_with_names.csv')
music_user_df = pd.read_csv(r'data\processed\movie\movie_user_with_names.csv')

# Get the set of all usernames (assuming usernames are unique across all domains)
all_usernames = set(book_user_df['username']).union(movie_user_df['username']).union(music_user_df['username'])

username = st.selectbox("👤 Select Username", sorted(all_usernames))

# Find user_id for each domain (if username exists in that domain)
book_user_id = book_user_df[book_user_df['username'] == username]['user_id'].iloc[0] if username in book_user_df['username'].values else None
movie_user_id = movie_user_df[movie_user_df['username'] == username]['user_id'].iloc[0] if username in movie_user_df['username'].values else None
music_user_id = music_user_df[music_user_df['username'] == username]['user_id'].iloc[0] if username in music_user_df['username'].values else None

# --- OLD USER ID INPUT (commented out) ---
# user_id = st.number_input("🔢 Enter User ID", min_value=1, step=1)


st.divider()

# Dropdown for algorithm type
algorithm = st.selectbox("🤖 Select Recommendation Algorithm", [
    "Collaborative Filtering",
    "Matrix Factorization (SVD)",
    "Popularity Based"
])


if st.button("🎯 Get Recommendations"):
    if algorithm == "Collaborative Filtering":
        st.subheader("📚 Book Recommendations (Collaborative Filtering)")
        try:
            book_matrix, book_sim, book_info = load_book_data()
            if book_user_id is not None:
                st.dataframe(book_cf(book_user_id, book_matrix, book_sim, book_info), use_container_width=True)
            else:
                st.warning("No Book user ID found for this username.")
        except Exception as e:
            st.error(f"Book CF error: {e}")

        st.subheader("🎬 Movie Recommendations (Collaborative Filtering)")
        movie_matrix, movie_sim, movie_info = load_movie_data()
        if movie_user_id is not None:
            st.dataframe(movie_cf(movie_user_id, movie_matrix, movie_sim, movie_info), use_container_width=True)
        else:
            st.warning("No Movie user ID found for this username.")

        st.subheader("🎵 Music Recommendations (Collaborative Filtering)")
        music_matrix, music_sim, music_info = load_music_data()
        if music_user_id is not None:
            st.dataframe(music_cf(music_user_id, music_matrix, music_sim, music_info), use_container_width=True)
        else:
            st.warning("No Music user ID found for this username.")

    elif algorithm == "Matrix Factorization (SVD)":
        st.subheader("📚 Book Recommendations (SVD)")
        if book_user_id is not None:
            st.dataframe(book_svd(book_user_id), use_container_width=True)
        else:
            st.warning("No Book user ID found for this username.")

        st.subheader("🎬 Movie Recommendations (SVD)")
        if movie_user_id is not None:
            st.dataframe(movie_svd(movie_user_id), use_container_width=True)
        else:
            st.warning("No Movie user ID found for this username.")

        st.subheader("🎵 Music Recommendations (SVD)")
        if music_user_id is not None:
            try:
                st.dataframe(music_svd(music_user_id), use_container_width=True)
            except Exception as e:
                st.error(f"Music SVD error: {e}")
        else:
            st.warning("No Music user ID found for this username.")

    elif algorithm == "Popularity Based":
        st.subheader("📚 Top Books (Popularity)")
        try:
            st.dataframe(get_popular_books(), use_container_width=True)
        except Exception as e:
            st.error(f"Popular Book error: {e}")

        st.subheader("🎬 Top Movies (Popularity)")
        try:
            st.dataframe(get_popular_movies(), use_container_width=True)
        except Exception as e:
            st.error(f"Popular Movie error: {e}")

        st.subheader("🎵 Top Music (Popularity)")
        try:
            st.dataframe(get_popular_music(), use_container_width=True)
        except Exception as e:
            st.error(f"Popular Music error: {e}")
