import streamlit as st

# from evaluation.collaborative.book_cf_eval import evaluate_book_cf
# from evaluation.collaborative.movie_cf_eval import evaluate_movie_cf
# from evaluation.collaborative.music_cf_eval import evaluate_music_cf

from model_evaluation.colaborative_filter.book_cf_eval import evaluate_book_cf
from model_evaluation.colaborative_filter.movie_cf_eval import evaluate_movie_cf
from model_evaluation.colaborative_filter.music_cf_eval import evaluate_music_cf
from model_evaluation.matrix_factorization.book_mf_eval import evaluate_book_mf
from model_evaluation.matrix_factorization.movie_mf_eval import evaluate_movie_mf
from model_evaluation.matrix_factorization.music_mf_eval import evaluate_music_mf

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
st.title("📊 Collaborative Filtering Evaluation")

# Book CF Evaluation
st.subheader("📚 Book Recommendation - Collaborative Filtering")
book_eval_df = evaluate_book_cf()
st.dataframe(book_eval_df, use_container_width=True)

# Movie CF Evaluation
st.subheader("🎬 Movie Recommendation - Collaborative Filtering")
movie_eval_df = evaluate_movie_cf()
st.dataframe(movie_eval_df, use_container_width=True)

# Music CF Evaluation
st.subheader("🎵 Music Recommendation - Collaborative Filtering")
music_eval_df = evaluate_music_cf()
st.dataframe(music_eval_df, use_container_width=True)

# --- Matrix Factorization (SVD) Evaluation ---
st.title("📊 Matrix Factorization (SVD) Evaluation")

# Book MF Evaluation
st.subheader("📚 Book Recommendation - Matrix Factorization (SVD)")
book_mf_eval_df = evaluate_book_mf()
st.dataframe(book_mf_eval_df, use_container_width=True)

# Movie MF Evaluation
st.subheader("🎬 Movie Recommendation - Matrix Factorization (SVD)")
movie_mf_eval_df = evaluate_movie_mf()
st.dataframe(movie_mf_eval_df, use_container_width=True)

# Music MF Evaluation
st.subheader("🎵 Music Recommendation - Matrix Factorization (SVD)")
music_mf_eval_df = evaluate_music_mf()
st.dataframe(music_mf_eval_df, use_container_width=True)