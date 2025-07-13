import pandas as pd

def load_data():
    print("📥 Loading book rating and info data...")
    ratings_df = pd.read_csv("../../data/processed/book/book_filtered_rating.csv")
    book_info_df = pd.read_csv("../../data/processed/book/book_info.csv")
    return ratings_df, book_info_df

def compute_popularity(ratings_df):
    print("📊 Calculating popularity metrics...")

    popularity_df = ratings_df.groupby('book_id').agg(
        rating_count=('Rating', 'count'),
        average_rating=('Rating', 'mean')
    ).reset_index()

    # IMDb-style weighted rating score
    C = popularity_df['average_rating'].mean()
    m = popularity_df['rating_count'].quantile(0.50)  # median threshold

    popularity_df = popularity_df[popularity_df['rating_count'] >= m].copy()

    popularity_df['weighted_score'] = (
        (popularity_df['rating_count'] / (popularity_df['rating_count'] + m)) * popularity_df['average_rating'] +
        (m / (popularity_df['rating_count'] + m)) * C
    )

    return popularity_df.sort_values(by='weighted_score', ascending=False)

def get_top_books(popularity_df, book_info_df, top_n=10):
    print(f"📚 Fetching top {top_n} popular books...")
    top_books = popularity_df.head(top_n)
    merged = pd.merge(top_books, book_info_df, on='book_id')
    return merged[['book_id', 'Title', 'rating_count', 'average_rating', 'weighted_score']]

def main():
    ratings_df, book_info_df = load_data()
    popularity_df = compute_popularity(ratings_df)
    top_books = get_top_books(popularity_df, book_info_df, top_n=5)

    print("\n📖 Top 10 Popular Books:")
    for _, row in top_books.iterrows():
        print(f"⭐ {row['Title']} (book_id: {row['book_id']}")

if __name__ == "__main__":
    main()
