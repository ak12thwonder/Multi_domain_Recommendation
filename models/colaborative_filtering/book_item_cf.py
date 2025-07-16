import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    print("📥 Loading rating and book info data...")
    ratings_df = pd.read_csv("../../data/processed/book/book_rating.csv")
    book_info_df = pd.read_csv("../../data/processed/book/book_info.csv")
    return ratings_df, book_info_df


def filter_popular_books(ratings_df, min_count=70):
    print(f"🔍 Filtering books with more than {min_count} ratings...")
    popular_books = ratings_df['book_id'].value_counts()
    popular_books = popular_books[popular_books > min_count].index
    filtered_df = ratings_df[ratings_df['book_id'].isin(popular_books)]
    print(f"✅ Filtered to {filtered_df['book_id'].nunique()} popular books.")
    return filtered_df


def build_user_item_matrix(filtered_df):
    print("🔧 Building user-item matrix...")
    user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='book_id', values='Rating')
    return user_item_matrix


def compute_item_similarity(user_item_matrix):
    print("🧠 Calculating item-to-item similarity...")
    item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
    similarity_df = pd.DataFrame(item_similarity,
                                 index=user_item_matrix.columns,
                                 columns=user_item_matrix.columns)
    return similarity_df


def save_similarity_matrix(similarity_df, filename="book_item_similarity_matrix.csv"):
    print(f"💾 Saving similarity matrix to {filename}...")
    similarity_df.to_csv(filename)
    print("✅ Similarity matrix saved.")


def recommend_books(user_id, user_item_matrix, similarity_df, book_info_df, top_n=5):
    if user_id not in user_item_matrix.index:
        print(f"🚫 User {user_id} not found.")
        return pd.DataFrame()

    print(f"📚 Recommending books for user {user_id}...")
    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = {}

    for item, rating in user_ratings.items():
        if item not in similarity_df.columns:
            continue
        similar_items = similarity_df[item].drop(labels=user_ratings.index, errors='ignore')
        for similar_item, similarity in similar_items.items():
            scores[similar_item] = scores.get(similar_item, 0) + similarity * rating

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_ids = [book_id for book_id, _ in sorted_items[:top_n]]

    recommended_books = book_info_df[
        book_info_df['book_id'].isin(recommended_ids)
    ][['book_id', 'Title']].drop_duplicates()

    print("User ratings:", user_ratings)
    print("Scores:", scores)
    print("Recommended IDs:", recommended_ids)
    print("Book info book_ids:", book_info_df['book_id'].tolist()[:10])
    print("Book info columns:", book_info_df.columns)

    return recommended_books


def load_book_data():
    """
    Loads the filtered book ratings, book info, user-item matrix, and item similarity matrix.
    Returns:
        user_item_matrix (pd.DataFrame)
        similarity_df (pd.DataFrame)
        book_info_df (pd.DataFrame)
    """
    filtered_df = pd.read_csv("data/processed/book/book_filtered_rating.csv")
    book_info_df = pd.read_csv("data/processed/book/book_info.csv")
    user_item_matrix = filtered_df.pivot_table(index='User-ID', columns='book_id', values='Rating')

    similarity_path = "models/colaborative_filtering/book_item_similarity_matrix.csv"
    if os.path.exists(similarity_path):
        similarity_df = pd.read_csv(similarity_path, index_col=0)
        similarity_df.columns = similarity_df.columns.astype(int)
        similarity_df.index = similarity_df.index.astype(int)
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        item_similarity = cosine_similarity(user_item_matrix.T.fillna(0))
        similarity_df = pd.DataFrame(item_similarity,
                                     index=user_item_matrix.columns,
                                     columns=user_item_matrix.columns)
        similarity_df.to_csv(similarity_path)
    return user_item_matrix, similarity_df, book_info_df


def main():
    ratings_df, book_info_df = load_data()
    # filtered_df = filter_popular_books(ratings_df, min_count=70)
    filtered_df = pd.read_csv("../../data/processed/book/book_filtered_rating.csv")

    user_item_matrix = build_user_item_matrix(filtered_df)
    print("👀 Available user IDs (sample):", user_item_matrix.index.tolist()[:10])
    print("✅ User 2 exists?" , 2 in user_item_matrix.index)


    similarity_path = "book_item_similarity_matrix.csv"

    if os.path.exists(similarity_path):
        print("📂 Loading precomputed similarity matrix...")
        similarity_df = pd.read_csv(similarity_path, index_col=0)
        print(similarity_df.head())
        similarity_df.columns = similarity_df.columns.astype(int)
        similarity_df.index = similarity_df.index.astype(int)
    else:
        similarity_df = compute_item_similarity(user_item_matrix)
        save_similarity_matrix(similarity_df)

    user_id = 8 # Change this to test other users
    recommendations = recommend_books(user_id, user_item_matrix, similarity_df, book_info_df, top_n=5)
    # print("🔍 Ratings for user 2:")
    # print(user_item_matrix.loc[2].dropna())

    print("\n✅ Final Recommendations:")
    if recommendations.empty:
        print("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            print(f"📖 {row['Title']} (book_id: {row['book_id']})")


if __name__ == "__main__":
    main()
