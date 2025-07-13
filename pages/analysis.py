# streamlit_app/pages/1_📊_Analysis.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import os

st.set_page_config(page_title="Data Analysis", layout="wide")

st.title("📊 Data Analysis")

st.markdown("""
Explore **structured previews** and visualizations of your datasets:
- 📚 Books: info, users, ratings
- 🎮 Movies
- 🎵 Music
- 🔁 Combined interactions
""")

st.divider()

# Paths
DATA_PATH = "../data/processed"
NOTEBOOK_PATH = "../notebooks"

# Select domain
domain = st.selectbox("📂 Select Domain", ["Books", "Movies", "Music", "Combined"])

if domain == "Books":
    st.subheader("📚 Book Dataset Components")
    col1, col2, col3 = st.columns(3)

    file_map = {
        "Book_info": os.path.join(DATA_PATH, "book/book_info.csv"),
        "Book_Users": os.path.join(DATA_PATH, "book/book_user.csv"),
        "Book_Ratings": os.path.join(DATA_PATH, "book/book_rating.csv")
    }

    dfs = {}
    for col, name in zip([col1, col2, col3], file_map.keys()):
        with col:
            try:
                df = pd.read_csv(file_map[name])
                dfs[name] = df
                st.markdown(f"**{name} (Top 5 rows)**")
                st.dataframe(df.head(), use_container_width=True)
                st.markdown(f"**{name} Description**")
                st.dataframe(df.describe(include='all'), use_container_width=True)
            except Exception as e:
                st.error(f"{name} file not found or error reading: {e}")

    try:
        rating_df = dfs.get("Book_Ratings", pd.read_csv(file_map["Book_Ratings"]))
        rating_df = rating_df.drop(columns=["ISBN"], errors="ignore")

        st.subheader("📊 Book Dataset Visual Summaries")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Rating Distribution**")
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            sns.histplot(rating_df["Rating"], bins=20, kde=True, ax=ax1)
            ax1.set_title("Book Ratings Histogram")
            st.pyplot(fig1)

        with col2:
            st.markdown("**Top Rated Books**")
            top_books = rating_df["book_id"].value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.barplot(x=top_books.values, y=top_books.index, ax=ax2)
            ax2.set_xlabel("Rating Count")
            ax2.set_ylabel("Book ID")
            ax2.set_title("Top Rated Books")
            st.pyplot(fig2)

        with col3:
            st.markdown("**Most Active Users**")
            top_users = rating_df["User-ID"].value_counts().head(10)
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            sns.barplot(x=top_users.values, y=top_users.index, ax=ax3)
            ax3.set_xlabel("Ratings Given")
            ax3.set_ylabel("User ID")
            ax3.set_title("Most Active Users")
            st.pyplot(fig3)

    except Exception as e:
        st.warning(f"📉 Rating plots could not be generated: {e}")

    notebook_file = os.path.join(NOTEBOOK_PATH, "book_cleaning.ipynb")


elif domain == "Movies":
    st.subheader("🎮 Movie Ratings Dataset")
    file_path = os.path.join(DATA_PATH, "movie/movie_rating.csv")
    try:
        df = pd.read_csv(file_path)
        st.markdown("**Top 5 rows**")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("**Description**")
        st.dataframe(df.describe(include='all'), use_container_width=True)

        st.subheader("📊 Movie Dataset Visual Summaries")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Rating Distribution**")
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            sns.histplot(df["rating"], bins=15, kde=True, ax=ax1)
            ax1.set_title("Rating Histogram")
            st.pyplot(fig1)

        with col2:
            st.markdown("**Top 10 Rated Movies**")
            top_movies = df["item_id"].value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.barplot(x=top_movies.values, y=top_movies.index, ax=ax2)
            ax2.set_xlabel("Rating Count")
            ax2.set_ylabel("Movie ID")
            ax2.set_title("Top Rated Movies")
            st.pyplot(fig2)

        with col3:
            st.markdown("**User Activity**")
            top_users = df["user_id"].value_counts().head(10)
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            sns.barplot(x=top_users.values, y=top_users.index, ax=ax3)
            ax3.set_xlabel("Ratings Given")
            ax3.set_ylabel("User ID")
            ax3.set_title("Most Active Users")
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error loading movie dataset: {e}")

    notebook_file = os.path.join(NOTEBOOK_PATH, "movie_cleaning.ipynb")

elif domain == "Music":
    st.subheader("🎵 Music Ratings Dataset")
    file_path = os.path.join(DATA_PATH, "music/music_rating.csv")
    try:
        df = pd.read_csv(file_path)
        st.markdown("**Top 5 rows**")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("**Description**")
        st.dataframe(df.describe(include='all'), use_container_width=True)

        st.subheader("📊 Music Dataset Visual Summaries")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Weight Distribution**")
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            sns.histplot(df["weight"], bins=20, kde=True, ax=ax1)
            ax1.set_title("Listening Weight Distribution")
            st.pyplot(fig1)

        with col2:
            st.markdown("**Top Artists (by interactions)**")
            top_artists = df["artist_id"].value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.barplot(x=top_artists.values, y=top_artists.index, ax=ax2)
            ax2.set_xlabel("Total Interactions")
            ax2.set_ylabel("Artist ID")
            ax2.set_title("Top Artists")
            st.pyplot(fig2)

        with col3:
            st.markdown("**Most Active Users**")
            top_users = df["user_id"].value_counts().head(10)
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            sns.barplot(x=top_users.values, y=top_users.index, ax=ax3)
            ax3.set_xlabel("Total Interactions")
            ax3.set_ylabel("User ID")
            ax3.set_title("Top Users")
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"Error loading music dataset: {e}")

    notebook_file = os.path.join(NOTEBOOK_PATH, "music_cleaning.ipynb")


elif domain == "Combined":
    st.subheader("🔁 Combined Interactions Dataset")

    from dotenv import load_dotenv
    import psycopg2
    import os

    load_dotenv()
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT", 5432)

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )

        # 1. Load full ratings table
        df_ratings = pd.read_sql("SELECT * FROM ratings", conn)

        # 2. Get users who are present in all three domains
        query = """
        SELECT user_id FROM users
        WHERE from_books = 1 AND from_movies = 1 AND from_music = 1;
        """
        df_user = pd.read_sql(query, conn)
        conn.close()

        # 3. Filter ratings to only those from users in all three domains
        common_user_ids = df_user['user_id'].tolist()
        df_common = df_ratings[df_ratings['user_id'].isin(common_user_ids)]

        # ✅ Show top rows
        st.markdown("**Top 5 rows (Common Users Only)**")
        st.dataframe(df_common.head(), use_container_width=True)

        # ✅ Show summary stats
        st.markdown("**Description**")
        st.dataframe(df_common.describe(include='all'), use_container_width=True)

        # ✅ Rating distribution plot
        st.subheader("📊 Rating Distribution (Users in All Domains)")
        plt.figure(figsize=(8, 4))
        sns.histplot(df_common["rating"], bins=20, kde=True)
        plt.title("Distribution of Ratings (Users in All Domains)")
        st.pyplot()

        # ✅ Domain distribution
        st.subheader("🔍 Interaction Count per Domain (Common Users Only)")
        st.bar_chart(df_common['source'].value_counts())

    except Exception as e:
        st.error(f"❌ Error fetching or processing combined data: {e}")

    notebook_file = os.path.join(NOTEBOOK_PATH, "music_cleaning.ipynb")



# Notebook download
st.divider()
st.subheader("📘 Full Analysis Notebook")

try:
    with open(notebook_file, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        notebook_name = os.path.basename(notebook_file)
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{notebook_name}">📅 Download {notebook_name}</a>'
        st.markdown(href, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ Notebook not found. Please ensure it exists in the `notebooks/` folder.")
