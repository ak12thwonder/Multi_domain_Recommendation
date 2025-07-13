import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    print("📥 Loading movie ratings...")
    ratings_df = pd.read_csv("../../data/processed/movie/movie_rating.csv")
    movie_info_df = pd.read_csv("../../data/processed/movie/movie_info.csv")  # Contains movie_id ↔ title
    return ratings_df, movie_info_df


def filter_popular_movies(ratings_df, min_count=70):
    print("🔎 Filtering popular movies...")
    popular_movie_ids = ratings_df['item_id'].value_counts()
    popular_movie_ids = popular_movie_ids[popular_movie_ids > min_count].index
    return ratings_df[ratings_df['item_id'].isin(popular_movie_ids)]


def build_user_item_matrix(filtered_df):
    print("🔧 Building user-item matrix...")
    user_item_matrix = filtered_df.pivot_table(index='user_id', columns='item_id', values='rating')
    return user_item_matrix


def compute_item_similarity(user_item_matrix):
    print("🧠 Calculating item-to-item similarity...")
    similarity_matrix = cosine_similarity(user_item_matrix.T.fillna(0))
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return similarity_df


def save_similarity_matrix(similarity_df, filename="movie_item_similarity_matrix.csv"):
    print(f"💾 Saving similarity matrix to {filename}...")
    similarity_df.to_csv(filename)
    print("✅ Similarity matrix saved.")


def recommend_movies(user_id, user_item_matrix, similarity_df, movie_info_df, top_n=5):
    if user_id not in user_item_matrix.index:
        print(f"🚫 User {user_id} not found.")
        return pd.DataFrame()

    print(f"🎬 Recommending movies for user {user_id}...")
    user_ratings = user_item_matrix.loc[user_id].dropna()
    
    scores = {}
    print(user_ratings)
    print(len(user_ratings))
    print(type(user_ratings))
    for item, rating in user_ratings.items():
        # Check if item exists in similarity matrix
        if item not in similarity_df.columns:
            print(f"⚠️ Item {item} not found in similarity matrix, skipping...")
            continue
            
        print(f"Processing item {item} with rating {rating}")
        
        # Get similar items, but only drop labels that exist in the similarity matrix
        similar_items = similarity_df[item]
        
        # Filter out items that the user has already rated (only if they exist in similarity matrix)
        user_rated_items = set(user_ratings.index) & set(similar_items.index)
        if user_rated_items:
            similar_items = similar_items.drop(labels=list(user_rated_items))
        
        for similar_item, similarity in similar_items.items():
            scores[similar_item] = scores.get(similar_item, 0) + similarity * rating

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_ids = [item_id for item_id, _ in sorted_items[:top_n]]

    # Join with movie_info_df to get titles
    recommended_movies = movie_info_df[movie_info_df['movie_id'].isin(recommended_ids)][['movie_id', 'title']].drop_duplicates()
    return recommended_movies


def main():
    ratings_df, movie_info_df = load_data()
    filtered_df = filter_popular_movies(ratings_df, min_count=70)
    user_item_matrix = build_user_item_matrix(filtered_df)
    similarity_df = compute_item_similarity(user_item_matrix)

    save_similarity_matrix(similarity_df)
    
    user_id = 20 # change as needed
    recommendations = recommend_movies(user_id, user_item_matrix, similarity_df, movie_info_df, top_n=5)

    print("\n✅ Final Recommendations:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            print(f"🎥 {row['title']} (Movie ID: {row['movie_id']})")


if __name__ == "__main__":
    main()
