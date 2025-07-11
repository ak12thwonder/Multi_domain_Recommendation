import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    ratings = pd.read_csv("../../data/processed/music/music_rating.csv")
    music_info = pd.read_csv("../../data/processed/music/music_info.csv")
    return ratings, music_info

def build_user_item_matrix(ratings_df):
    print("🔧 Building user-item matrix...")
    print(ratings_df.head())
    user_item_matrix = ratings_df.pivot_table(index='user_id', columns='artist_id', values='weight')
    user_item_matrix = user_item_matrix.fillna(0)

    print(f"✅ Matrix shape: {user_item_matrix.shape}")
    if user_item_matrix.empty:
        raise ValueError("🚫 User-item matrix is empty. Check input data.")
    
    return user_item_matrix

def apply_svd(user_item_matrix, n_components=50):
    print(f"🔍 Applying Truncated SVD with {n_components} latent features...")

    if user_item_matrix.shape[1] < 2:
        raise ValueError("❗ SVD requires at least 2 items (columns). Not enough artist ratings.")

    svd = TruncatedSVD(n_components=min(n_components, user_item_matrix.shape[1]-1), random_state=42)
    matrix_reduced = svd.fit_transform(user_item_matrix)
    print("✅ SVD applied.")
    return matrix_reduced, svd

def recommend_artists(user_id, user_item_matrix, music_info_df, svd_matrix, top_n=5):
    if user_id not in user_item_matrix.index:
        print(f"🚫 User {user_id} not found.")
        return pd.DataFrame()

    print(f"🎶 Generating recommendations for user {user_id}...")
    user_idx = list(user_item_matrix.index).index(user_id)
    user_vector = svd_matrix[user_idx]

    similarities = cosine_similarity([user_vector], svd_matrix)[0]
    similar_user_indices = similarities.argsort()[::-1][1:6]
    similar_users = [user_item_matrix.index[i] for i in similar_user_indices]

    recommendations = pd.DataFrame(columns=['artist_id', 'weight'])

    for sim_user in similar_users:
        user_ratings = user_item_matrix.loc[sim_user]
        top_artists = user_ratings[user_ratings > 0].sort_values(ascending=False).head(10)
        top_artists_df = top_artists.reset_index()
        top_artists_df.columns = ['artist_id', 'weight']
        recommendations = pd.concat([recommendations, top_artists_df])

    rated_artists = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    recommendations = recommendations[~recommendations['artist_id'].isin(rated_artists)]

    if recommendations.empty:
        print("⚠️ No recommendations available after filtering.")
        return pd.DataFrame()

    top_recommendations = (
        recommendations.groupby('artist_id')
        .mean()
        .sort_values(by='weight', ascending=False)
        .head(top_n)
    )

    return music_info_df[music_info_df['id'].isin(top_recommendations.index)][['id', 'name']].drop_duplicates()

def main():
    ratings_df, music_info_df = load_data()
    user_item_matrix = build_user_item_matrix(ratings_df)

    try:
        svd_matrix, svd_model = apply_svd(user_item_matrix, n_components=50)
    except ValueError as e:
        print(e)
        return

    user_id = 2  # Change this to test other users
    recommendations = recommend_artists(user_id, user_item_matrix, music_info_df, svd_matrix, top_n=5)

    print("\n🎉 Recommended Artists:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            print(f"🎤 {row['name']} (artist_id: {row['id']})")

if __name__ == "__main__":
    main()
