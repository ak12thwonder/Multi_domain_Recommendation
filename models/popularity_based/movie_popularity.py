import pandas as pd

def load_data():
    print("📥 Loading movie rating and info data...")
    ratings_df = pd.read_csv("../../data/processed/movie/movie_rating.csv")
    movie_info_df = pd.read_csv("../../data/processed/movie/movie_info.csv")
    return ratings_df, movie_info_df

def compute_popularity(ratings_df):
    print("📊 Calculating popularity metrics...")

    # Group by item_id (which is movie_id) and calculate:
    # - number of ratings
    # - average rating
    popularity_df = ratings_df.groupby('item_id').agg(
        rating_count=('rating', 'count'),
        average_rating=('rating', 'mean')
    ).reset_index()

    # Compute weighted score like IMDb (optional but better)
    C = popularity_df['average_rating'].mean()
    m = popularity_df['rating_count'].quantile(0.50)  # 50th percentile

    # Only consider movies with rating count above threshold
    popularity_df = popularity_df[popularity_df['rating_count'] >= m].copy()

    # IMDb-style weighted rating
    popularity_df['weighted_score'] = (
        (popularity_df['rating_count'] / (popularity_df['rating_count'] + m)) * popularity_df['average_rating'] +
        (m / (popularity_df['rating_count'] + m)) * C
    )

    # Sort by popularity score
    return popularity_df.sort_values(by='weighted_score', ascending=False)

def get_top_movies(popularity_df, movie_info_df, top_n=10):
    print(f"🎯 Fetching top {top_n} popular movies...")
    top_movies = popularity_df.head(top_n)
    merged = pd.merge(top_movies, movie_info_df, left_on='item_id', right_on='movie_id')
    return merged[['movie_id', 'title', 'rating_count', 'average_rating', 'weighted_score']]

def main():
    ratings_df, movie_info_df = load_data()
    popularity_df = compute_popularity(ratings_df)
    top_movies = get_top_movies(popularity_df, movie_info_df, top_n=5)

    print("\n🎥 Top 10 Popular Movies:")
    for _, row in top_movies.iterrows():
        print(f"⭐ {row['title']} (movie_id: {row['movie_id']}") 

if __name__ == "__main__":
    main()
