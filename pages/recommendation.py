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

from models.matrix_factorization.book_svd import recommend_books as book_svd
from models.matrix_factorization.movies_svd import recommend_movies as movie_svd
from models.matrix_factorization.music_svd import recommend_artists as music_svd

from models.popularity_based.book_popularity import get_top_books
from models.popularity_based.movie_popularity import get_top_movies
from models.popularity_based.music_popularity import get_top_artists

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
            book_matrix, book_sim, book_info = load_book_data()
            st.dataframe(book_cf(user_id, book_matrix, book_sim, book_info), use_container_width=True)
        except Exception as e:
            st.error(f"Book CF error: {e}")

        st.subheader("🎬 Movie Recommendations (Collaborative Filtering)")


        movie_matrix, movie_sim, movie_info = load_movie_data()
        # print("a")
        # print(movie_matrix)
        # print()
        # print('b')
        # print(movie_sim)

        # print()
        # print('c')
        # print(movie_info)
        
        st.dataframe(movie_cf(user_id, movie_matrix, movie_sim, movie_info), use_container_width=True)
        # try:
        #     movie_matrix, movie_sim, movie_info = load_movie_data()
        #     print("a")
        #     print(movie_matrix)
        #     print()
        #     print('b')
        #     print(movie_sim)

        #     print()
        #     print('c')
        #     print(movie_info)
        #     st.dataframe(movie_cf(user_id, movie_matrix, movie_sim, movie_info), use_container_width=True)

        # except Exception as e:
        #     st.error(f"Movie CF error: {e}")

        st.subheader("🎵 Music Recommendations (Collaborative Filtering)")
        music_matrix, music_sim, music_info = load_music_data()
        st.dataframe(music_cf(user_id, music_matrix, music_sim, music_info), use_container_width=True)


    elif algorithm == "Matrix Factorization (SVD)":
        st.subheader("📚 Book Recommendations (SVD)")
        try:
            st.dataframe(book_svd(user_id), use_container_width=True)
        except Exception as e:
            st.error(f"Book SVD error: {e}")

        st.subheader("🎬 Movie Recommendations (SVD)")
        try:
            st.dataframe(movie_svd(user_id), use_container_width=True)
        except Exception as e:
            st.error(f"Movie SVD error: {e}")

        st.subheader("🎵 Music Recommendations (SVD)")
        try:
            st.dataframe(music_svd(user_id), use_container_width=True)
        except Exception as e:
            st.error(f"Music SVD error: {e}")

    elif algorithm == "Popularity Based":
        st.subheader("📚 Top Books (Popularity)")
        try:
            st.dataframe(get_top_books(), use_container_width=True)
        except Exception as e:
            st.error(f"Popular Book error: {e}")

        st.subheader("🎬 Top Movies (Popularity)")
        try:
            st.dataframe(get_top_movies(), use_container_width=True)
        except Exception as e:
            st.error(f"Popular Movie error: {e}")

        st.subheader("🎵 Top Music (Popularity)")
        try:
            st.dataframe(get_top_artists(), use_container_width=True)
        except Exception as e:
            st.error(f"Popular Music error: {e}")
