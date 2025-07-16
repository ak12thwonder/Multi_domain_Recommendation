import streamlit as st
import pandas as pd
import os

def get_recommendations_by_genre_or_decade(genre_or_decade):
    movie_info_path = os.path.join('data', 'processed', 'movie', 'movie_info.csv')
    df = pd.read_csv(movie_info_path)
    genre_map = {
        'Adventure': 'Adventure',
        'Classic': 'Classic',  # Classic is not a column, so filter by year < 1970
        'Romantic': 'Romance',
        'Fight': 'Action',
        '90s': '90s',
        # '20s': '20s'
    }
    if genre_or_decade in ['Adventure', 'Romantic', 'Fight']:
        genre_col = genre_map[genre_or_decade]
        filtered = df[df[genre_col] == 1]
        filtered = filtered[['title', 'release_date']]
        filtered['category'] = genre_or_decade
    elif genre_or_decade == 'Classic':
        # Classic: movies before 1970
        filtered = df[df['release_date'].str[-4:].astype(int) < 1970][['title', 'release_date']]
        filtered['category'] = 'Classic'
    elif genre_or_decade == '90s':
        filtered = df[df['release_date'].str[-4:].astype(int).between(1990, 1999)][['title', 'release_date']]
        filtered['category'] = '90s'
    # elif genre_or_decade == '20s':
    #     filtered = df[df['release_date'].str[-4:].astype(int).between(1920, 1929)][['title', 'release_date']]
    #     filtered['category'] = '20s'
    else:
        filtered = pd.DataFrame(columns=['title', 'release_date', 'category'])
    filtered = filtered.reset_index(drop=True)
    filtered.index += 1
    return filtered

st.set_page_config(page_title="Genre/Decade Recommendations", page_icon="🎬", layout="wide")
st.title("🎬 Genre & Decade Recommendations")

st.markdown("""
Click a block below to get recommendations for that genre or decade.
""")

blocks = [
    ("Adventure", "🧭"),
    ("Classic", "🎩"),
    ("Romantic", "💖"),
    ("Fight", "🥊"),
    ("90s", "📼"),
    # ("20s", "🎷")
]

cols = st.columns(len(blocks))
clicked = None
for i, (label, emoji) in enumerate(blocks):
    if cols[i].button(f"{emoji} {label}"):
        clicked = label

if clicked:
    st.subheader(f"Recommendations for {clicked}")
    recs = get_recommendations_by_genre_or_decade(clicked)
    if recs.empty:
        st.info("No recommendations available for this category.")
    else:
        st.dataframe(recs) 

# Example user data
user_df = pd.read_csv("data/processed/book/book_user_with_names.csv")  # or movie/music user file

age_groups = [
    ("16-20", (16, 20)),
    ("21-30", (21, 30)),
    ("31-40", (31, 40)),
    ("41-50", (41, 50)),
    ("50+", (51, 200)),
]

st.subheader("Browse by Age Group")
age_group_labels = [label for label, _ in age_groups]
cols = st.columns(len(age_group_labels))

selected_age_group = None
for i, (label, _) in enumerate(age_groups):
    if cols[i].button(label):
        selected_age_group = label

if selected_age_group:
    min_age, max_age = dict(age_groups)[selected_age_group]
    filtered_users = user_df[(user_df['age'] >= min_age) & (user_df['age'] <= max_age)]
    user_ids = filtered_users['user_id'].unique()

    # Load movie ratings and info
    movie_ratings = pd.read_csv("data/processed/movie/movie_rating.csv")
    movie_info = pd.read_csv("data/processed/movie/movie_info.csv")

    # Filter ratings for users in this age group
    group_ratings = movie_ratings[movie_ratings['user_id'].isin(user_ids)]

    # Get top 5 most rated movies (or you can use average rating, etc.)
    top_movies = (
        group_ratings.groupby('item_id')
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Get movie titles (use the correct ID column)
    top_movie_titles = movie_info[movie_info['movie_id'].isin(top_movies)][['title']].drop_duplicates().reset_index(drop=True)
    top_movie_titles.index = top_movie_titles.index + 1  # Start index from 1

    st.subheader(f"Top 5 Movies for Age Group {selected_age_group}")
    st.dataframe(top_movie_titles)

st.subheader("Browse by Gender")
gender_cols = st.columns(2)
selected_gender = None
if gender_cols[0].button("Male"):
    selected_gender = "M"
if gender_cols[1].button("Female"):
    selected_gender = "F"

if selected_gender:
    # Filter users by gender
    filtered_users = user_df[user_df['gender'] == selected_gender]
    user_ids = filtered_users['user_id'].unique()

    # Load movie ratings and info
    movie_ratings = pd.read_csv("data/processed/movie/movie_rating.csv")
    movie_info = pd.read_csv("data/processed/movie/movie_info.csv")

    # Filter ratings for users of this gender
    group_ratings = movie_ratings[movie_ratings['user_id'].isin(user_ids)]

    # Get top 5 most rated movies
    top_movies = (
        group_ratings.groupby('item_id')
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Get movie titles
    top_movie_titles = movie_info[movie_info['movie_id'].isin(top_movies)][['title']].drop_duplicates().reset_index(drop=True)
    top_movie_titles.index = top_movie_titles.index + 1  # Start index from 1

    st.subheader(f"Top 5 Movies for {'Male' if selected_gender == 'M' else 'Female'} Users")
    st.dataframe(top_movie_titles)