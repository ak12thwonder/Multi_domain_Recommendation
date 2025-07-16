import pandas as pd
import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_user_file(path):
    cols = ['user_id', 'username', 'age', 'gender', 'occupation']
    df = pd.read_csv(path)
    for col in cols:
        if col not in df.columns:
            df[col] = None
    return df[cols]

# Load all user mapping files
book_users = load_user_file("../data/processed/book/book_user_with_names.csv")
movie_users = load_user_file("../data/processed/movie/movie_user_with_names.csv")
music_users = load_user_file("../data/processed/music/music_rating_with_names.csv")
new_users = load_user_file("../data/processed/new_users.csv")

# Add source flags
book_users['from_book'] = 1
book_users['from_movies'] = 0
book_users['from_music'] = 0

movie_users['from_book'] = 0
movie_users['from_movies'] = 1
movie_users['from_music'] = 0

music_users['from_book'] = 0
music_users['from_movies'] = 0
music_users['from_music'] = 1

# For new_users, set all source flags to 0 (or you can set to 1 if you want to track them as a separate source)
new_users['from_book'] = 0
new_users['from_movies'] = 0
new_users['from_music'] = 0

# Combine all users
all_users = pd.concat([book_users, movie_users, music_users, new_users], ignore_index=True)

# Group by user_id, aggregate: take first non-null for each field, and max for source flags
def agg_func(series):
    return series.dropna().iloc[0] if not series.dropna().empty else None

user_profile = all_users.groupby('user_id').agg({
    'username': agg_func,
    'age': agg_func,
    'gender': agg_func,
    'occupation': agg_func,
    'from_book': 'max',
    'from_movies': 'max',
    'from_music': 'max'
}).reset_index()

# Add active_user column (default 1)
user_profile['active_user'] = 1

# Connect to your database
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()

# Create the user_profile table if it doesn't exist
cur.execute("""
CREATE TABLE IF NOT EXISTS user_profile (
    user_id BIGINT PRIMARY KEY,
    username VARCHAR(255),
    age INTEGER,
    gender VARCHAR(10),
    occupation VARCHAR(255),
    from_movies INTEGER DEFAULT 0,
    from_book INTEGER DEFAULT 0,
    from_music INTEGER DEFAULT 0,
    active_user INTEGER DEFAULT 1
);
""")
conn.commit()

# Insert or upsert into user_profile table
for _, row in user_profile.iterrows():
    cur.execute("""
        INSERT INTO user_profile (user_id, username, age, gender, occupation, from_movies, from_book, from_music, active_user)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            username = EXCLUDED.username,
            age = EXCLUDED.age,
            gender = EXCLUDED.gender,
            occupation = EXCLUDED.occupation,
            from_movies = EXCLUDED.from_movies,
            from_book = EXCLUDED.from_book,
            from_music = EXCLUDED.from_music,
            active_user = EXCLUDED.active_user;
    """, (
        int(row['user_id']),
        row['username'],
        int(row['age']) if pd.notnull(row['age']) else None,
        row['gender'],
        row['occupation'],
        int(row['from_movies']),
        int(row['from_book']),
        int(row['from_music']),
        int(row['active_user'])
    ))

conn.commit()
cur.close()
conn.close()
print("user_profile table populated/updated.")
