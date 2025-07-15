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