import streamlit as st
import pandas as pd

# Load info for all domains
book_info = pd.read_csv("data/processed/book/book_info.csv")
movie_info = pd.read_csv("data/processed/movie/movie_info.csv")
music_info = pd.read_csv("data/processed/music/music_info.csv")

st.title("Domain-Specific Quickstart Recommendations")

# 1. Select domain
domain = st.selectbox("Which type of recommendation do you want?", ["Books", "Movies", "Music"])

# 2. Let user rate items in that domain
if domain == "Books":
    st.subheader("Rate Books You've Read")
    choices = book_info['Title'].unique()
elif domain == "Movies":
    st.subheader("Rate Movies You've Watched")
    choices = movie_info['title'].unique()
else:
    st.subheader("Rate Artists You've Listened To")
    choices = music_info['name'].unique()

user_items = []
user_ratings = []

for i in range(3):
    col1, col2 = st.columns([3, 1])
    with col1:
        item = st.selectbox(f"{domain[:-1]} {i+1}", options=[""] + list(choices), key=f"item_{i}")
    with col2:
        rating = st.slider(f"Your Rating {i+1}", 1, 5, 3, key=f"rating_{i}")
    if item:
        user_items.append(item)
        user_ratings.append(rating)

# 3. Recommend items from the same domain
if st.button("Get My Recommendations") and user_items:
    temp_user_id = -1
    if domain == "Books":
        # Map book titles to book_id
        title_to_id = dict(zip(book_info['Title'], book_info['book_id']))
        temp_ratings = [{"User-ID": temp_user_id, "book_id": title_to_id[item], "Rating": rating}
                        for item, rating in zip(user_items, user_ratings) if item in title_to_id]
        temp_ratings_df = pd.DataFrame(temp_ratings)
        # Load collaborative filtering data/functions
        from models.colaborative_filtering.book_item_cf import load_book_data, recommend_books
        user_item_matrix, similarity_df, book_info_df = load_book_data()
        # Add temp user to matrix
        user_item_matrix.loc[temp_user_id] = 0
        for row in temp_ratings:
            user_item_matrix.at[temp_user_id, row["book_id"]] = row["Rating"]
        recs = recommend_books(temp_user_id, user_item_matrix, similarity_df, book_info_df)
        st.subheader("📚 Book Recommendations")
        if recs.empty or 'Title' not in recs.columns:
            # st.info("No personalized recommendations found. Showing popular books instead.")
            from models.popularity_based.book_popularity import get_popular_books
            recs = get_popular_books()
            display_df = recs[['Title']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df)
        else:
            display_df = recs[['Title']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df)

    elif domain == "Movies":
        # Map movie titles to movie_id
        title_to_id = dict(zip(movie_info['title'], movie_info['movie_id']))
        temp_ratings = [{"user_id": temp_user_id, "movie_id": title_to_id[item], "rating": rating}
                        for item, rating in zip(user_items, user_ratings) if item in title_to_id]
        temp_ratings_df = pd.DataFrame(temp_ratings)
        from models.colaborative_filtering.movie_cf import load_movie_data, recommend_movies
        user_item_matrix, similarity_df, movie_info_df = load_movie_data()
        user_item_matrix.loc[temp_user_id] = 0
        for row in temp_ratings:
            user_item_matrix.at[temp_user_id, row["movie_id"]] = row["rating"]
        recs = recommend_movies(temp_user_id, user_item_matrix, similarity_df, movie_info_df)
        st.subheader("🎬 Movie Recommendations")
        if recs.empty or 'title' not in recs.columns:
            # st.info("No personalized recommendations found. Showing popular movies instead.")
            from models.popularity_based.movie_popularity import get_popular_movies
            recs = get_popular_movies()
            if not recs.empty and 'title' in recs.columns:
                display_df = recs[['title']].reset_index(drop=True)
                display_df.index = display_df.index + 1
                st.dataframe(display_df)
            else:
                st.info("No popular movies available.")
        else:
            display_df = recs[['title']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df)

    else:  # Music
        # Map artist names to id
        name_to_id = dict(zip(music_info['name'], music_info['id']))
        temp_ratings = [{"user_id": temp_user_id, "id": name_to_id[item], "weight": rating}
                        for item, rating in zip(user_items, user_ratings) if item in name_to_id]
        temp_ratings_df = pd.DataFrame(temp_ratings)
        from models.colaborative_filtering.music_cf import load_music_data, recommend_artists
        user_item_matrix, similarity_df, music_info_df = load_music_data()
        user_item_matrix.loc[temp_user_id] = 0
        for row in temp_ratings:
            user_item_matrix.at[temp_user_id, row["id"]] = row["weight"]
        recs = recommend_artists(temp_user_id, user_item_matrix, similarity_df, music_info_df)
        st.subheader("🎵 Music Recommendations")
        if recs.empty or 'name' not in recs.columns:
            # st.info("Showing popular music instead.")
            from models.popularity_based.music_popularity import get_popular_music
            recs = get_popular_music()
            display_df = recs[['name']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df)
        else:
            display_df = recs[['name']].reset_index(drop=True)
            display_df.index = display_df.index + 1
            st.dataframe(display_df)
else:
    st.info("Select at least one item and rate it to get recommendations.")