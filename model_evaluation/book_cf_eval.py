# evaluation/model_evaluation/book_cf_eval.py

import pandas as pd
from models.colaborative_filtering.book_item_cf import (
    build_user_item_matrix,
    compute_item_similarity,
    recommend_books
)

def evaluate_book_cf():
    """
    Evaluate item-based collaborative filtering on book domain using Precision@5
    Returns: DataFrame with evaluation summary
    """
    ratings_df = pd.read_csv("data/processed/book/book_rating.csv")
    book_info_df = pd.read_csv("data/processed/book/book_info.csv")

    ratings_df = ratings_df.sort_values(['user_id', 'timestamp'])
    test = ratings_df.groupby('user_id').tail(1)
    train = ratings_df.drop(labels=list(test.index))

    user_item_matrix = build_user_item_matrix(train)
    similarity_df = compute_item_similarity(user_item_matrix)

    hits = 0
    total_users = 0
    for _, row in test.iterrows():
        user_id = row['user_id']
        actual_book = row['item_id']
        if user_id not in user_item_matrix.index:
            continue
        recommended = recommend_books(user_id, user_item_matrix, similarity_df, book_info_df, top_n=5)
        rec_ids = recommended['book_id'].tolist()
        if actual_book in rec_ids:
            hits += 1
        total_users += 1

    precision_at_5 = hits / total_users if total_users > 0 else 0

    return pd.DataFrame({
        "Model": ["Book Collaborative Filtering"],
        "Accuracy (Precision@5)": [round(precision_at_5, 4)],
        "Users Evaluated": [total_users]
    }) 