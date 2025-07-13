# streamlit_app/Home.py

import streamlit as st
import pandas as pd
import psycopg2
st.set_page_config(page_title="Multi-Domain Recommendation Dashboard", page_icon="📚", layout="wide")

st.title("📚 Multi-Domain Recommendation System")
st.markdown("Welcome to the unified recommendation platform for **Books**, **Movies**, and **Music**!")

st.markdown("""
This system uses multiple recommendation algorithms:
- ✅ **Collaborative Filtering** (User-Based / Item-Based)
- ✅ **Matrix Factorization** (SVD)
- ✅ **Content Based**
- ✅ **Popularity-Based Filtering**

You can explore visualizations, generate recommendations for users, and evaluate model performance using the sidebar navigation.
""")

st.divider()
st.header("📊 Dataset Overview")

# 🔌 PostgreSQL connection to fetch combined stats
from dotenv import load_dotenv
import os
import psycopg2
import pandas as pd
import streamlit as st

# Load .env variables
load_dotenv()

# # Streamlit layout
# st.set_page_config(page_title="Multi-Domain Recommendation Dashboard", page_icon="📚", layout="wide")
# st.title("📚 Multi-Domain Recommendation System")

# 📦 Fetch DB credentials from .env
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# 🔌 PostgreSQL connection using .env credentials
def fetch_combined_stats():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(DISTINCT user_id),
                COUNT(DISTINCT item_id),
                COUNT(*)
            FROM ratings;
        """)
        users, items, interactions = cur.fetchone()
        cur.close()
        conn.close()
        return users, items, interactions
    except Exception as e:
        st.error(f"❌ Error fetching combined stats: {e}")
        return 0, 0, 0



# Load stats for each domain + combined
def load_stats():
    # Book
    book_rating = pd.read_csv("data/processed/book/book_rating.csv")
    book_users = book_rating['User-ID'].nunique()
    book_items = book_rating['book_id'].nunique()
    book_interactions = len(book_rating)

    # Movie
    movie_rating = pd.read_csv("data/processed/movie/movie_rating.csv")
    movie_users = movie_rating['user_id'].nunique()
    movie_items = movie_rating['item_id'].nunique()
    movie_interactions = len(movie_rating)

    # Music
    music_rating = pd.read_csv("data/processed/music/music_rating.csv")
    music_users = music_rating['user_id'].nunique()
    music_items = music_rating['artist_id'].nunique()
    music_interactions = len(music_rating)

    # Combined (from DB)
    combined_users, combined_items, combined_interactions = fetch_combined_stats()

    return {
        "Books": (book_users, book_items, book_interactions),
        "Movies": (movie_users, movie_items, movie_interactions),
        "Music": (music_users, music_items, music_interactions),
        "Combined": (combined_users, combined_items, combined_interactions)
    }

# 🔢 Get all stats
stats = load_stats()

# 📊 Show metrics in 4 columns
col1, col2, col3, col4 = st.columns(4)
for col, domain in zip([col1, col2, col3, col4], stats.keys()):
    users, items, interactions = stats[domain]
    with col:
        st.subheader(f"📁 {domain}")
        st.metric("Users", f"{users:,}")
        st.metric("Items", f"{items:,}")
        st.metric("Interactions", f"{interactions:,}")

st.divider()
