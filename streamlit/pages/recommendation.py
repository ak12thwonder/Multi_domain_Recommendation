# streamlit/pages/3_📌_recommendation.py
import os
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

import psycopg2
from dotenv import load_dotenv


st.set_page_config(page_title="Recommendations", page_icon="📌", layout="wide")
st.title("📌 Multi-Domain Recommendations")

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# Show main area success message if just registered (must be at the very top)
if st.session_state.get('user_just_registered'):
    st.success(f"User '{st.session_state['user_just_registered']}' has been registered successfully!")
    st.session_state['user_just_registered'] = None

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
music_user_df = pd.read_csv(r'data\processed\music\music_rating_with_names.csv')

# Get the set of all usernames (assuming usernames are unique across all domains)
all_usernames = set(book_user_df['username']).union(movie_user_df['username']).union(music_user_df['username'])

# --- New User Registration UI at the top ---
# st.sidebar.header('🆕 RFas

if 'new_user_registered' not in st.session_state:
    st.session_state['new_user_registered'] = False
# Registration button
st.sidebar.header('🆕 Register New User')
with st.sidebar.form("new_user_form", clear_on_submit=True):
    st.subheader("👤 Enter Your Details")
    username = st.text_input("Username")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["M", "F", "Other"])
    occupation = st.text_input("Occupation", value="Other")
    submitted = st.form_submit_button("Submit")

    if submitted:
        new_users_path = 'data/processed/new_users.csv'
        # Load or initialize users file
        if os.path.exists(new_users_path):
            existing_users_df = pd.read_csv(new_users_path)
            all_usernames = set(existing_users_df['username'])
        else:
            existing_users_df = pd.DataFrame(columns=['user_id', 'username', 'age', 'gender', 'occupation'])
            all_usernames = set()
        # Validate uniqueness
        if username in all_usernames:
            st.warning(f"Username '{username}' already exists. Please choose a different one.")
        else:
            # Generate numeric user_id starting from 1000233
            if not existing_users_df.empty:
                # Only consider numeric user_ids
                numeric_ids = pd.to_numeric(existing_users_df['user_id'], errors='coerce')
                numeric_ids = numeric_ids.dropna().astype(int)
                next_id = numeric_ids.max() + 1 if not numeric_ids.empty else 1000233
            else:
                next_id = 1000231
            new_user_id = next_id  # Always an integer
            new_user_row = {
                'user_id': new_user_id,
                'username': username,
                'age': age,
                'gender': gender,
                'occupation': occupation
            }
            new_user_df = pd.DataFrame([new_user_row])[['user_id', 'username', 'age', 'gender', 'occupation']]
            new_user_df.to_csv(new_users_path, mode='a', header=not os.path.exists(new_users_path), index=False)
            st.success(f"🎉 User '{username}' registered successfully!")
            # Show popularity-based suggestions
            st.subheader('📚 Top Books (Popularity)')
            book_pop = get_popular_books()
            if not book_pop.empty and 'title' in book_pop.columns:
                display_df = book_pop[['title']].reset_index(drop=True)
                display_df.index = display_df.index + 1
                st.dataframe(display_df, use_container_width=True)
            # else:
                # st.info('No popular books available.')
            st.subheader('🎬 Top Movies (Popularity)')
            movie_pop = get_popular_movies()
            if not movie_pop.empty and 'title' in movie_pop.columns:
                display_df = movie_pop[['title']].reset_index(drop=True)
                display_df.index = display_df.index + 1
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info('No popular movies available.')
            st.subheader('🎵 Top Music (Popularity)')
            music_pop = get_popular_music()
            if not music_pop.empty and 'name' in music_pop.columns:
                display_df = music_pop[['name']].reset_index(drop=True)
                display_df.index = display_df.index + 1
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info('No popular music available.')
# Show main area success message if just registered
if st.session_state.get('user_just_registered'):
    st.success(f"User '{st.session_state['user_just_registered']}' has been registered successfully!")
    # Optionally clear the flag after showing the message
    st.session_state['user_just_registered'] = None

# --- Username selection dropdown (only after registration UI) ---
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
                book_recs = book_cf(book_user_id, book_matrix, book_sim, book_info)
                if book_recs.empty or 'title' not in book_recs.columns:
                    st.info('No book recommendations available.')
                else:
                    book_recs = book_recs[['title']].reset_index(drop=True)
                    book_recs.index += 1
                    st.dataframe(book_recs)
            else:
                st.warning("No Book user ID found for this username.")
        except Exception as e:
            st.error(f"Book CF error: {e}")

        st.subheader("🎬 Movie Recommendations (Collaborative Filtering)")
        movie_matrix, movie_sim, movie_info = load_movie_data()
        if movie_user_id is not None:
            movie_recs = movie_cf(movie_user_id, movie_matrix, movie_sim, movie_info)
            if movie_recs.empty or 'title' not in movie_recs.columns:
                st.info('No movie recommendations available.')
            else:
                movie_recs = movie_recs[['title']].reset_index(drop=True)
                movie_recs.index += 1
                st.dataframe(movie_recs)
        else:
            st.warning("No Movie user ID found for this username.")

        st.subheader("🎵 Music Recommendations (Collaborative Filtering)")
        music_matrix, music_sim, music_info = load_music_data()
        if music_user_id is not None:
            music_recs = music_cf(music_user_id, music_matrix, music_sim, music_info)
            if music_recs.empty or 'name' not in music_recs.columns:
                st.info('No music recommendations available.')
            else:
                music_recs = music_recs[['name']].reset_index(drop=True)
                music_recs.index += 1
                st.dataframe(music_recs)
        else:
            st.warning("No Music user ID found for this username.")

    elif algorithm == "Matrix Factorization (SVD)":
        st.subheader("📚 Book Recommendations (SVD)")
        if book_user_id is not None:
            book_svd_recs = book_svd(book_user_id)
            required_cols = ['title', 'rating_count', 'average_rating']
            if book_svd_recs.empty or not all(col in book_svd_recs.columns for col in required_cols):
                st.info('No book recommendations available.')
            else:
                display_df = book_svd_recs[required_cols].reset_index(drop=True)
                display_df.index += 1
                st.dataframe(display_df)
        else:
            st.warning("No Book user ID found for this username.")

        st.subheader("🎬 Movie Recommendations (SVD)")
        if movie_user_id is not None:
            movie_svd_recs = movie_svd(movie_user_id)
            required_cols = ['title', 'rating_count', 'average_rating']
            if movie_svd_recs.empty or not all(col in movie_svd_recs.columns for col in required_cols):
                st.info('No movie recommendations available.')
            else:
                display_df = movie_svd_recs[required_cols].reset_index(drop=True)
                display_df.index += 1
                st.dataframe(display_df)
        else:
            st.warning("No Movie user ID found for this username.")

        st.subheader("🎵 Music Recommendations (SVD)")
        if music_user_id is not None:
            try:
                music_svd_recs = music_svd(music_user_id)
                required_cols = ['name', 'rating_count', 'average_rating']
                if music_svd_recs.empty or not all(col in music_svd_recs.columns for col in required_cols):
                    st.info('No music recommendations available.')
                else:
                    display_df = music_svd_recs[['name', 'rating_count', 'average_rating']].reset_index(drop=True)
                    display_df.index += 1
                    st.dataframe(display_df)
            except Exception as e:
                st.error(f"Music SVD error: {e}")
        else:
            st.warning("No Music user ID found for this username.")

    elif algorithm == "Popularity Based":
        st.subheader("📚 Top Books (Popularity)")
        try:
            book_pop = get_popular_books()
            required_cols = ['title', 'rating_count', 'average_rating']
            if book_pop.empty or not all(col in book_pop.columns for col in required_cols):
                st.info('No popular books available.')
            else:
                display_df = book_pop[required_cols].reset_index(drop=True)
                display_df.index += 1
                st.dataframe(display_df)
        except Exception as e:
            st.error(f"Popular Book error: {e}")

        st.subheader("🎬 Top Movies (Popularity)")
        try:
            movie_pop = get_popular_movies()
            required_cols = ['title', 'rating_count', 'average_rating']
            if movie_pop.empty or not all(col in movie_pop.columns for col in required_cols):
                st.info('No popular movies available.')
            else:
                display_df = movie_pop[required_cols].reset_index(drop=True)
                display_df.index += 1
                st.dataframe(display_df)
        except Exception as e:
            st.error(f"Popular Movie error: {e}")

        st.subheader("🎵 Top Music (Popularity)")
        try:
            music_pop = get_popular_music()
            required_cols = ['name', 'rating_count', 'average_rating']
            if music_pop.empty or not all(col in music_pop.columns for col in required_cols):
                st.info('No popular music available.')
            else:
                display_df = music_pop[['name', 'rating_count', 'average_rating']].reset_index(drop=True)
                display_df.index += 1
                st.dataframe(display_df)
        except Exception as e:
            st.error(f"Popular Music error: {e}")

# --- Additional Button: Save to Database ---
def save_new_users_to_db():
    # from database.new_items import insert_new_users
    from database.new_items import insert_new_user_profile
    import os
    print("Current working directory:", os.getcwd())
    df = pd.read_csv(os.path.join("data", "processed", "new_users.csv"))
    insert_new_user_profile(df)
    st.success('New users saved to the database!')
    # try:
    # except Exception as e:
    #     st.error(f'Error saving to database: {e}')

if st.sidebar.button('Save to Database'):
    save_new_users_to_db()


def soft_delete_user(user_id):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    cur.execute("UPDATE user_profile SET active_user = 0 WHERE user_id = %s", (user_id,))
    conn.commit()
    cur.close()
    conn.close()

# Example: Load users from the database (only active users)
import psycopg2.extras

def get_active_users():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT user_id, username FROM user_profile WHERE active_user = 1")
    users = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(users)

users_df = get_active_users()

import streamlit as st

username_to_delete = st.text_input("Enter Username to delete (soft delete):")

if st.button("Delete User"):
    if username_to_delete:
        try:
            # Look up user_id by username
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM user_profile WHERE username = %s", (username_to_delete,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result:
                user_id = result[0]
                soft_delete_user(user_id)
                st.success(f"User '{username_to_delete}' deleted (soft delete)!")
            else:
                st.warning(f"No user found with username '{username_to_delete}'.")
        except Exception as e:
            st.error(f"Error deleting user: {e}")
    else:
        st.warning("Please enter a User ID.")
