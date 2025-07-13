import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    print("📥 Loading music ratings and artist info...")
    ratings_df = pd.read_csv("../../data/processed/music/music_rating.csv")  # has user_id, artist_id, weight
    artist_info_df = pd.read_csv("../../data/processed/music/music_info.csv")  # should contain artist_id, artist_name
    return ratings_df, artist_info_df


def filter_popular_artists(ratings_df, min_count=70):
    print("🔎 Filtering popular artists...")
    popular_artists = ratings_df['artist_id'].value_counts()
    popular_artists = popular_artists[popular_artists > min_count].index
    return ratings_df[ratings_df['artist_id'].isin(popular_artists)]


def build_user_item_matrix(filtered_df):
    print("🔧 Building user-item matrix...")
    user_item_matrix = filtered_df.pivot_table(index='user_id', columns='artist_id', values='weight')
    return user_item_matrix


def compute_item_similarity(user_item_matrix):
    print("🧠 Calculating item-to-item similarity...")
    similarity_matrix = cosine_similarity(user_item_matrix.T.fillna(0))
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return similarity_df


def save_similarity_matrix(similarity_df, filename="music_item_similarity_matrix.csv"):
    print(f"💾 Saving similarity matrix to {filename}...")
    similarity_df.to_csv(filename)
    print("✅ Similarity matrix saved.")


def recommend_artists(user_id, user_item_matrix, similarity_df, artist_info_df, top_n=5):
    if user_id not in user_item_matrix.index:
        print(f"🚫 User {user_id} not found.")
        return pd.DataFrame()

    print(f"🎧 Recommending artists for user {user_id}...")
    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = {}
    
    for item, rating in user_ratings.items():
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
        similar_items = similarity_df[item].drop(labels=user_ratings.index, errors='ignore')
        for similar_item, similarity in similar_items.items():
            scores[similar_item] = scores.get(similar_item, 0) + similarity * rating

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_ids = [artist_id for artist_id, _ in sorted_items[:top_n]]

    # Join with artist info
    recommended_artists = artist_info_df[artist_info_df['id'].isin(recommended_ids)][['id', 'name']].drop_duplicates()
    return recommended_artists


def main():
    ratings_df, artist_info_df = load_data()
    filtered_df = filter_popular_artists(ratings_df, min_count=70)
    user_item_matrix = build_user_item_matrix(filtered_df)
    similarity_df = compute_item_similarity(user_item_matrix)

    save_similarity_matrix(similarity_df)

    user_id = 20  # Change this to test different users
    recommendations = recommend_artists(user_id, user_item_matrix, similarity_df, artist_info_df, top_n=5)

    print("\n✅ Final Recommendations:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            print(f"🎤 {row['name']} (Artist ID: {row['id']})")


if __name__ == "__main__":
    main()
