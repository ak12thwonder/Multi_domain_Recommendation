import pandas as pd
from models.colaborative_filtering.music_cf import (
    build_user_item_matrix,
    compute_item_similarity,
    recommend_artists
)

def evaluate_music_cf():
    """
    Evaluate item-based collaborative filtering on music domain using Precision@5.
    Returns: DataFrame with evaluation summary
    """
    ratings_df = pd.read_csv("data/processed/music/music_rating.csv")
    music_info_df = pd.read_csv("data/processed/music/music_info.csv")

    # Split data: last interaction per user as test, rest as train
    train = ratings_df.groupby("user_id").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    test = ratings_df.groupby("user_id").apply(lambda x: x.iloc[-1:]).reset_index(drop=True)

    user_item_matrix = build_user_item_matrix(train)
    similarity_df = compute_item_similarity(user_item_matrix)

    hits = 0
    total_users = 0

    for _, row in test.iterrows():
        user_id = row['user_id']
        actual_artist = row['artist_id']

        if user_id not in user_item_matrix.index:
            continue

        recommended = recommend_artists(user_id, user_item_matrix, similarity_df, music_info_df, top_n=5)
        rec_ids = recommended['artist_id'].tolist()

        if actual_artist in rec_ids:
            hits += 1
        total_users += 1

    precision_at_5 = hits / total_users if total_users > 0 else 0

    return pd.DataFrame({
        "Model": ["Music Collaborative Filtering"],
        "Accuracy (Precision@5)": [round(precision_at_5, 4)],
        "Users Evaluated": [total_users]
    })
