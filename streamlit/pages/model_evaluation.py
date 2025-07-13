import streamlit as st
from evaluation.collaborative.book_cf_eval import evaluate_book_cf
from evaluation.collaborative.movie_cf_eval import evaluate_movie_cf
from evaluation.collaborative.music_cf_eval import evaluate_music_cf

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
st.title("📊 Collaborative Filtering Evaluation")

# # Book CF Evaluation
# st.subheader("📚 Book Recommendation - Collaborative Filtering")
# book_eval_df = evaluate_book_cf()
# st.dataframe(book_eval_df, use_container_width=True)

# Movie CF Evaluation
st.subheader("🎬 Movie Recommendation - Collaborative Filtering")
movie_eval_df = evaluate_movie_cf()
st.dataframe(movie_eval_df, use_container_width=True)

# Music CF Evaluation
st.subheader("🎵 Music Recommendation - Collaborative Filtering")
music_eval_df = evaluate_music_cf()
st.dataframe(music_eval_df, use_container_width=True)
