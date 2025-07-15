# evaluation/model_evaluation/music_cf_eval.py

import os
import pandas as pd
import json
from models.colaborative_filtering.music_cf import (
    build_user_item_matrix,
    compute_item_similarity,
    recommend_artists
)
import psycopg2
from dotenv import load_dotenv

def save_accuracy_to_db(model_name, accuracy, users_evaluated, recall, f1, mae):
    load_dotenv()
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_PORT = os.getenv("DB_PORT", 5432)
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_evaluation (
            model_name TEXT PRIMARY KEY,
            accuracy FLOAT,
            users_evaluated INT,
            recall FLOAT,
            f1 FLOAT,
            mae FLOAT
        );
    ''')
    cur.execute('''
        INSERT INTO model_evaluation (model_name, accuracy, users_evaluated, recall, f1, mae)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name) DO UPDATE SET accuracy = EXCLUDED.accuracy, users_evaluated = EXCLUDED.users_evaluated, recall = EXCLUDED.recall, f1 = EXCLUDED.f1, mae = EXCLUDED.mae;
    ''', (model_name, accuracy, users_evaluated, recall, f1, mae))
    conn.commit()
    cur.close()
    conn.close()

def evaluate_music_cf():
    """
    Evaluate item-based collaborative filtering on music domain using Precision@5
    Returns: DataFrame with evaluation summary
    """
    ratings_df = pd.read_csv("data/processed/music/music_rating.csv")
    music_info_df = pd.read_csv("data/processed/music/music_info.csv")

    ratings_df = ratings_df.sort_values(['user_id'])
    test = ratings_df.groupby('user_id').tail(1)
    train = ratings_df.drop(labels=list(test.index))
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    uim_path = os.path.join(cache_dir, "user_item_matrix.pkl")
    sim_path = os.path.join(cache_dir, "music_item_similarity_matrix.csv")
    if os.path.exists(uim_path):
        user_item_matrix = pd.read_pickle(uim_path)
    else:
        user_item_matrix = build_user_item_matrix(train)
        user_item_matrix.to_pickle(uim_path)
    if os.path.exists(sim_path):
        similarity_df = pd.read_csv(sim_path, index_col=0)
    else:
        similarity_df = compute_item_similarity(user_item_matrix)
        similarity_df.to_csv(sim_path)
    hits = 0
    total_users = 0
    abs_errors = []
    for _, row in test.iterrows():
        user_id = row['user_id']
        actual_music = row['artist_id']
        if user_id not in user_item_matrix.index:
            continue
        recommended = recommend_artists(user_id, user_item_matrix, similarity_df, music_info_df, top_n=5)
        rec_ids = recommended['music_id'].tolist() if 'music_id' in recommended.columns else []
        hit = int(actual_music in rec_ids)
        hits += hit
        total_users += 1
        abs_errors.append(abs(1 - hit))
    precision_at_5 = hits / total_users if total_users > 0 else 0
    recall_at_5 = precision_at_5
    f1_at_5 = (2 * precision_at_5 * recall_at_5) / (precision_at_5 + recall_at_5) if (precision_at_5 + recall_at_5) > 0 else 0
    mae = sum(abs_errors) / total_users if total_users > 0 else 0
    save_accuracy_to_db('music_cf', round(precision_at_5, 4), total_users, round(recall_at_5, 4), round(f1_at_5, 4), round(mae, 4))
    return pd.DataFrame({
        "Model": ["Music Collaborative Filtering"],
        "Accuracy (Precision@5)": [round(precision_at_5, 4)],
        "Recall@5": [round(recall_at_5, 4)],
        "F1@5": [round(f1_at_5, 4)],
        "MAE": [round(mae, 4)],
        "Users Evaluated": [total_users]
    }) 