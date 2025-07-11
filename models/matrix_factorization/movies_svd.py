import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data():
    ratings = pd.read_csv("../../data/processed/movie/movie_rating.csv")
    movies = pd.read_csv("../../data/processed/movie/movie_info.csv")
    return ratings, movies

def build_user_item_matrix(ratings_df):
    print("🔧 Building user-item matrix...")
    user_item_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating')
    return user_item_matrix.fillna(0)

def apply_svd(user_item_matrix, n_components=50):
    print(f"🔍 Applying Truncated SVD with {n_components} latent features...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_reduced = svd.fit_transform(user_item_matrix)
    print("✅ SVD applied.")
    return matrix_reduced, svd

def recommend_movies(user_id, user_item_matrix, movies_df, svd_matrix, top_n=5):
    if user_id not in user_item_matrix.index:
        print(f"🚫 User {user_id} not found.")
        return pd.DataFrame()

    print(f"🎬 Generating recommendations for user {user_id}...")
    user_idx = list(user_item_matrix.index).index(user_id)
    user_vector = svd_matrix[user_idx]

    similarities = cosine_similarity([user_vector], svd_matrix)[0]
    similar_user_indices = similarities.argsort()[::-1][1:6]
    similar_users = [user_item_matrix.index[i] for i in similar_user_indices]

    recommendations = pd.DataFrame(columns=['movie_id', 'rating'])

    for sim_user in similar_users:
        user_ratings = user_item_matrix.loc[sim_user]
        top_movies = user_ratings[user_ratings > 0].sort_values(ascending=False).head(10)
        top_movies_df = top_movies.reset_index()
        top_movies_df.columns = ['movie_id', 'rating']
        recommendations = pd.concat([recommendations, top_movies_df])

    rated_movies = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    recommendations = recommendations[~recommendations['movie_id'].isin(rated_movies)]

    if recommendations.empty:
        print("⚠️ No recommendations available after filtering.")
        return pd.DataFrame()

    top_recommendations = (
        recommendations.groupby('movie_id')
        .mean()
        .sort_values(by='rating', ascending=False)
        .head(top_n)
    )

    return movies_df[movies_df['movie_id'].isin(top_recommendations.index)][['movie_id', 'title']].drop_duplicates()

def main():
    ratings_df, movies_df = load_data()
    user_item_matrix = build_user_item_matrix(ratings_df)
    svd_matrix, svd_model = apply_svd(user_item_matrix, n_components=50)

    user_id = 8  # Change to test other users
    recommendations = recommend_movies(user_id, user_item_matrix, movies_df, svd_matrix, top_n=5)

    print("\n🎉 Recommended Movies:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            print(f"🎥 {row['title']} (movie_id: {row['movie_id']})")

if __name__ == "__main__":
    main()
