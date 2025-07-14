# evaluation/model_evaluation/movie_cf_eval.py

import pandas as pd
from models.colaborative_filtering.movie_cf import (
    build_user_item_matrix,
    compute_item_similarity,
    recommend_movies
)

def evaluate_movie_cf():
    """
    Evaluate item-based collaborative filtering on movie domain using Precision@5
    Returns: DataFrame with evaluation summary
    """
    ratings_df = pd.read_csv("data/processed/movie/movie_rating.csv")
    movie_info_df = pd.read_csv("data/processed/movie/movie_info.csv")

    ratings_df = ratings_df.sort_values(['user_id', 'timestamp'])
    test = ratings_df.groupby('user_id').tail(1)
    train = ratings_df.drop(labels=list(test.index))

    user_item_matrix = build_user_item_matrix(train)
    similarity_df = compute_item_similarity(user_item_matrix)

    hits = 0
    total_users = 0
    for _, row in test.iterrows():
        user_id = row['user_id']
        actual_movie = row['item_id']
        if user_id not in user_item_matrix.index:
            continue
        recommended = recommend_movies(user_id, user_item_matrix, similarity_df, movie_info_df, top_n=5)
        rec_ids = recommended['movie_id'].tolist()
        if actual_movie in rec_ids:
            hits += 1
        total_users += 1

    precision_at_5 = hits / total_users if total_users > 0 else 0

    return pd.DataFrame({
        "Model": ["Movie Collaborative Filtering"],
        "Accuracy (Precision@5)": [round(precision_at_5, 4)],
        "Users Evaluated": [total_users]
    }) 