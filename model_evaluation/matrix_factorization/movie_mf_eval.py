import os
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
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

def evaluate_movie_mf():
    ratings_df = pd.read_csv("data/processed/movie/movie_rating.csv")
    ratings_df = ratings_df.sort_values(['user_id', 'timestamp'])
    test = ratings_df.groupby('user_id').tail(1)
    train = ratings_df.drop(labels=list(test.index))
    user_item_matrix = train.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd_matrix = svd.fit_transform(user_item_matrix)
    user_index = {u: i for i, u in enumerate(user_item_matrix.index)}
    item_index = {i: j for j, i in enumerate(user_item_matrix.columns)}
    hits = 0
    total_users = 0
    abs_errors = []
    for _, row in test.iterrows():
        u, i = row['user_id'], row['item_id']
        if u in user_index and i in item_index:
            scores = np.dot(svd_matrix[user_index[u], :], svd.components_)
            top_n_idx = np.argsort(scores)[::-1][:5]
            top_n_items = [user_item_matrix.columns[idx] for idx in top_n_idx]
            hit = int(i in top_n_items)
            hits += hit
            abs_errors.append(abs(1 - hit))
            total_users += 1
    precision_at_5 = hits / total_users if total_users > 0 else 0
    recall_at_5 = precision_at_5
    f1_at_5 = (2 * precision_at_5 * recall_at_5) / (precision_at_5 + recall_at_5) if (precision_at_5 + recall_at_5) > 0 else 0
    mae = sum(abs_errors) / total_users if total_users > 0 else 0
    save_accuracy_to_db('movie_mf', round(precision_at_5, 4), total_users, round(recall_at_5, 4), round(f1_at_5, 4), round(mae, 4))
    return pd.DataFrame({
        "Model": ["Movie Matrix Factorization"],
        "Accuracy (Precision@5)": [round(precision_at_5, 4)],
        "Recall@5": [round(recall_at_5, 4)],
        "F1@5": [round(f1_at_5, 4)],
        "MAE": [round(mae, 4)],
        "Users Evaluated": [total_users]
    }) 