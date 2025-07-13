import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    ratings_df = pd.read_csv("../../data/processed/movie/movie_rating.csv")
    movie_info_df = pd.read_csv("../../data/processed/movie/movie_info.csv")
    return ratings_df, movie_info_df

def prepare_genre_matrix(movie_info_df):
    print("🎬 Preparing genre feature matrix...")
    genre_cols = movie_info_df.columns[5:]  # Assuming first 5 are meta, rest are genres
    genre_matrix = movie_info_df[genre_cols].astype(float)
    print(f"✅ Genre matrix shape: {genre_matrix.shape}")
    return genre_matrix, genre_cols

def compute_similarity_matrix(genre_matrix):
    print("🔍 Computing cosine similarity between movies...")
    similarity_matrix = cosine_similarity(genre_matrix)
    print("✅ Similarity matrix computed.")
    return similarity_matrix

def recommend_movies(user_id, ratings_df, movie_info_df, similarity_matrix, top_n=5):
    print(f"🎯 Generating recommendations for user {user_id}...")

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    if user_ratings.empty:
        print("🚫 No ratings found for this user.")
        return pd.DataFrame()

    liked_movies = user_ratings[user_ratings['rating'] >= 4]['item_id'].tolist()
    if not liked_movies:
        print("🚫 No high-rated movies by user to base recommendations on.")
        return pd.DataFrame()

    # Get indices of liked movies
    movie_indices = movie_info_df[movie_info_df['movie_id'].isin(liked_movies)].index.tolist()

    # Sum similarities across liked movies
    sim_scores = similarity_matrix[movie_indices].sum(axis=0)
    sim_scores_series = pd.Series(sim_scores, index=movie_info_df.index)

    # Exclude already watched movies
    watched_indices = movie_info_df[movie_info_df['movie_id'].isin(liked_movies)].index
    sim_scores_series = sim_scores_series.drop(index=watched_indices)

    # Top-N recommendations
    top_indices = sim_scores_series.sort_values(ascending=False).head(top_n).index
    recommendations = movie_info_df.loc[top_indices][['movie_id', 'title']]

    return recommendations

def main():
    ratings_df, movie_info_df = load_data()
    genre_matrix, genre_cols = prepare_genre_matrix(movie_info_df)
    similarity_matrix = compute_similarity_matrix(genre_matrix)

    user_id = 8  # Change this to test other users
    recommendations = recommend_movies(user_id, ratings_df, movie_info_df, similarity_matrix, top_n=5)

    print("\n🎥 Recommended Movies:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            print(f"🎬 {row['title']} (movie_id: {row['movie_id']})")

if __name__ == "__main__":
    main()
