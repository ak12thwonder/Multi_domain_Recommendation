import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()


# Step 1: Create ratings table
def create_ratings_table():
    print("🛠️ Creating 'ratings' table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            rating_id SERIAL PRIMARY KEY,
            user_id INT,
            item_id TEXT,
            rating FLOAT,
            source TEXT,
            UNIQUE(user_id, item_id)
        );
    """)
    conn.commit()
    print("✅ 'ratings' table created.")


# Step 2: Insert movie ratings
def insert_movie_ratings():
    print("🎬 Inserting movie ratings...")
    df = pd.read_csv("../data/processed/movie/movie_rating.csv")

    for _, row in df.iterrows():
        try:
            user_id = int(row["user_id"])
            item_id = f"movies_{int(row['item_id'])}"
            rating = float(row["rating"])

            cur.execute("""
                INSERT INTO ratings (user_id, item_id, rating, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, item_id) DO NOTHING;
            """, (user_id, item_id, rating, "movies"))

        except Exception as e:
            conn.rollback()
            print(f"❌ Skipped movie row due to: {e}")

    conn.commit()
    print("✅ Movie ratings inserted.")


# Step 3: Insert book ratings
def insert_book_ratings():
    print("📚 Inserting book ratings...")
    df = pd.read_csv("../data/processed/book/book_rating.csv")

    for _, row in df.iterrows():
        try:
            user_id = int(row["User-ID"])
            isbn_raw = str(row["ISBN"]).strip()
            if not isbn_raw:
                raise ValueError("Missing ISBN")
            item_id = f"books_{isbn_raw}"
            rating = float(row["Rating"])

            cur.execute("""
                INSERT INTO ratings (user_id, item_id, rating, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, item_id) DO NOTHING;
            """, (user_id, item_id, rating, "books"))

        except Exception as e:
            conn.rollback()
            print(f"❌ Skipped book row due to: {e}")

    conn.commit()
    print("✅ Book ratings inserted.")


# Step 4: Insert music ratings
def insert_music_ratings():
    print("🎵 Inserting music ratings...")
    df = pd.read_csv("../data/processed/music/music_rating.csv")

    for _, row in df.iterrows():
        try:
            user_id = int(row["user_id"])
            item_id = f"music_{int(row['artist_id'])}"
            rating = float(row["weight"])

            cur.execute("""
                INSERT INTO ratings (user_id, item_id, rating, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, item_id) DO NOTHING;
            """, (user_id, item_id, rating, "music"))

        except Exception as e:
            conn.rollback()
            print(f"❌ Skipped music row due to: {e}")

    conn.commit()
    print("✅ Music ratings inserted.")


# Main execution block
if __name__ == "__main__":
    create_ratings_table()
    insert_movie_ratings()
    insert_book_ratings()
    insert_music_ratings()

    cur.close()
    conn.close()
    print("🚀 All ratings loaded into the database.")
