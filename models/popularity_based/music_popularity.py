import pandas as pd

def load_data():
    print("📥 Loading music rating and artist info data...")
    ratings_df = pd.read_csv("../../data/processed/music/music_rating.csv")
    artist_info_df = pd.read_csv("../../data/processed/music/music_info.csv")
    return ratings_df, artist_info_df

def compute_popularity(ratings_df):
    print("📊 Calculating popularity metrics...")

    popularity_df = ratings_df.groupby('artist_id').agg(
        rating_count=('weight', 'count'),
        average_rating=('weight', 'mean')
    ).reset_index()

    # IMDb-style weighted score
    C = popularity_df['average_rating'].mean()
    m = popularity_df['rating_count'].quantile(0.50)  # consider top 50%

    popularity_df = popularity_df[popularity_df['rating_count'] >= m].copy()

    popularity_df['weighted_score'] = (
        (popularity_df['rating_count'] / (popularity_df['rating_count'] + m)) * popularity_df['average_rating'] +
        (m / (popularity_df['rating_count'] + m)) * C
    )

    return popularity_df.sort_values(by='weighted_score', ascending=False)

def get_top_artists(popularity_df, artist_info_df, top_n=10):
    print(f"🎯 Fetching top {top_n} popular artists...")
    top_artists = popularity_df.head(top_n)
    merged = pd.merge(top_artists, artist_info_df, left_on='artist_id', right_on='id')
    return merged[['id', 'name', 'rating_count', 'average_rating', 'weighted_score']]

def main():
    ratings_df, artist_info_df = load_data()
    popularity_df = compute_popularity(ratings_df)
    top_artists = get_top_artists(popularity_df, artist_info_df, top_n=5)

    print("\n🎵 Top 10 Popular Artists:")
    for _, row in top_artists.iterrows():
        print(f"⭐ {row['name']} (artist_id: {row['id']}") 

if __name__ == "__main__":
    main()
