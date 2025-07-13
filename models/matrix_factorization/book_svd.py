# import pandas as pd
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity

# def load_data():
#     ratings = pd.read_csv("../../data/processed/book/book_filtered_rating.csv")
#     books = pd.read_csv("../../data/processed/book/book_info.csv")
#     return ratings, books

# def build_user_item_matrix(ratings_df):
#     print("🔧 Building user-item matrix...")
#     user_item_matrix = ratings_df.pivot_table(index='User-ID', columns='book_id', values='Rating')
#     return user_item_matrix.fillna(0)

# def apply_svd(user_item_matrix, n_components=50):
#     print(f"🔍 Applying Truncated SVD with {n_components} latent features...")
#     svd = TruncatedSVD(n_components=n_components, random_state=42)
#     matrix_reduced = svd.fit_transform(user_item_matrix)
#     print("✅ SVD applied.")
#     return matrix_reduced, svd

# def recommend_books(user_id, user_item_matrix, books_df, svd_matrix, top_n=5):
#     if user_id not in user_item_matrix.index:
#         print(f"🚫 User {user_id} not found.")
#         return pd.DataFrame()

#     print(f"📚 Generating recommendations for user {user_id}...")
#     user_idx = list(user_item_matrix.index).index(user_id)
#     user_vector = svd_matrix[user_idx]
    
#     # Cosine similarity between this user and all users
#     similarities = cosine_similarity([user_vector], svd_matrix)[0]
    
#     # Most similar users
#     similar_user_indices = similarities.argsort()[::-1][1:6]  # Top 5 similar (excluding self)
#     similar_users = [user_item_matrix.index[i] for i in similar_user_indices]

#     # Get books rated by similar users
#     recommendations = pd.DataFrame()
#     for sim_user in similar_users:
#         user_ratings = user_item_matrix.loc[sim_user]
#         top_books = user_ratings[user_ratings > 0].sort_values(ascending=False).head(10)
#         recommendations = pd.concat([recommendations, top_books])

#     # Filter books user already rated
#     rated_books = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
#     recommendations = recommendations[~recommendations.index.isin(rated_books)]

#     # Final top-N books
#     top_recommendations = recommendations.groupby(recommendations.index).mean().sort_values(by='Rating',ascending=False).head(top_n)
    
#     return books_df[books_df['book_id'].isin(top_recommendations.index)][['book_id', 'Title']].drop_duplicates()

# def main():
#     ratings_df, books_df = load_data()
#     user_item_matrix = build_user_item_matrix(ratings_df)

#     svd_matrix, svd_model = apply_svd(user_item_matrix, n_components=50)

#     user_id = 8  # 🔁 Change this to test other users
#     recommendations = recommend_books(user_id, user_item_matrix, books_df, svd_matrix, top_n=5)

#     print("\n🎉 Recommended Books:")
#     if recommendations.empty:
#         print("No recommendations found.")
#     else:
#         for _, row in recommendations.iterrows():
#             print(f"📖 {row['Title']} (book_id: {row['book_id']})")

# if __name__ == "__main__":
#     main()




import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    ratings = pd.read_csv("data/processed/book/book_filtered_rating.csv")
    books = pd.read_csv("data/processed/book/book_info.csv")
    return ratings, books

def build_user_item_matrix(ratings_df):
    print("🔧 Building user-item matrix...")
    user_item_matrix = ratings_df.pivot_table(index='User-ID', columns='book_id', values='Rating')
    return user_item_matrix.fillna(0)

def apply_svd(user_item_matrix, n_components=50):
    print(f"🔍 Applying Truncated SVD with {n_components} latent features...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    matrix_reduced = svd.fit_transform(user_item_matrix)
    print("✅ SVD applied.")
    return matrix_reduced, svd

def recommend_books(user_id, user_item_matrix, books_df, svd_matrix, top_n=5):
    if user_id not in user_item_matrix.index:
        print(f"🚫 User {user_id} not found.")
        return pd.DataFrame()

    print(f"📚 Generating recommendations for user {user_id}...")
    user_idx = list(user_item_matrix.index).index(user_id)
    user_vector = svd_matrix[user_idx]
    
    # Cosine similarity between this user and all users
    similarities = cosine_similarity([user_vector], svd_matrix)[0]
    
    # Most similar users
    similar_user_indices = similarities.argsort()[::-1][1:6]  # Top 5 similar (excluding self)
    similar_users = [user_item_matrix.index[i] for i in similar_user_indices]

    # Get books rated by similar users
    recommendations = pd.DataFrame(columns=['book_id', 'Rating'])

    for sim_user in similar_users:
        user_ratings = user_item_matrix.loc[sim_user]
        top_books = user_ratings[user_ratings > 0].sort_values(ascending=False).head(10)

        # ✅ Convert Series to DataFrame and name columns
        top_books_df = top_books.reset_index()
        top_books_df.columns = ['book_id', 'Rating']
        recommendations = pd.concat([recommendations, top_books_df])

    # Filter books user already rated
    rated_books = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    recommendations = recommendations[~recommendations['book_id'].isin(rated_books)]

    # Final top-N books
    if recommendations.empty:
        print("⚠️ No recommendations available after filtering.")
        return pd.DataFrame()

    top_recommendations = (
        recommendations.groupby('book_id')
        .mean()
        .sort_values(by='Rating', ascending=False)
        .head(top_n)
    )

    return books_df[books_df['book_id'].isin(top_recommendations.index)][['book_id', 'Title']].drop_duplicates()

def recommend_books_svd(user_id, top_n=5):
    ratings_df, books_df = load_data()
    user_item_matrix = build_user_item_matrix(ratings_df)
    svd_matrix, svd_model = apply_svd(user_item_matrix, n_components=50)
    return recommend_books(user_id, user_item_matrix, books_df, svd_matrix, top_n)


def main():
    ratings_df, books_df = load_data()
    user_item_matrix = build_user_item_matrix(ratings_df)
    svd_matrix, svd_model = apply_svd(user_item_matrix, n_components=50)

    user_id = 8  # 🔁 Change this to test other users
    recommendations = recommend_books(user_id, user_item_matrix, books_df, svd_matrix, top_n=5)

    print("\n🎉 Recommended Books:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            print(f"📖 {row['Title']} (book_id: {row['book_id']})")

if __name__ == "__main__":
    main()

