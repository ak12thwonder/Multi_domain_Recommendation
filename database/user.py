"""
user.py

This script unifies user data from MovieLens, Book-Crossing, and Last.fm datasets
into a single PostgreSQL 'users' table. It loads environment variables for database
connection, creates the unified table if it does not exist, and inserts/updates user
records from each source.

-We have collect the user id and the demographic behaviour from the different dataset.
-only movies dataset contain this cols so we just merged this data with the other data set.
-Finally we are combining data same data than table will getting rich in the vertically if the user in the 
table than it getting rich horizontly.

Usage:
    python user.py

Requirements:
    - psycopg2
    - pandas
    - python-dotenv
    - Environment variables set in a .env file

Author: [Aditya Kaushal]
Date: [10/07/2025] Thrusday
"""
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv


# === Load environment variables from .env ===
load_dotenv()

DB_PARAMS = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT"),
}

# === PostgreSQL Connection ===
conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

# === Step 1: Create Unified Users Table ===
def create_users_table():
    print("🛠️ Creating 'users' table with integer flags...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            occupation TEXT,
            from_movies INTEGER DEFAULT 0,
            from_books INTEGER DEFAULT 0,
            from_music INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    print("✅ 'users' table ready.")


# === Step 2: Insert Users from Processed Movies CSV ===
def insert_movie_users():
    print("🎬 Inserting movie users...")
    df = pd.read_csv("../data/processed/movie/movie_user.csv")

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO users (user_id, age, gender, occupation, from_movies)
            VALUES (%s, %s, %s, %s, 1)
            ON CONFLICT (user_id) DO UPDATE
            SET
                age = COALESCE(users.age, EXCLUDED.age),
                gender = COALESCE(users.gender, EXCLUDED.gender),
                occupation = COALESCE(users.occupation, EXCLUDED.occupation),
                from_movies = 1;
        """, (
            int(row["user_id"]),
            int(row["age"]) if pd.notnull(row["age"]) else None,
            row["gender"] if pd.notnull(row["gender"]) else None,
            row["occupation"] if pd.notnull(row["occupation"]) else None
        ))
    print("✅ Movie users inserted.")


# === Step 3: Insert Users from Processed Books CSV ===
def insert_book_users():
    print("📚 Inserting book users...")
    df = pd.read_csv("../data/processed/book/book_user.csv")

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO users (user_id, from_books)
                VALUES (%s, 1)
                ON CONFLICT (user_id) DO UPDATE
                SET
                    age = COALESCE(users.age, EXCLUDED.age),
                    from_books = 1;
            """, (
                int(row["User-ID"]),
                # int(row["Age"]) if pd.notnull(row["Age"]) else None,
                # row["location"] if pd.notnull(row["location"]) else None
            ))
        except Exception as e:
            print(f"❌ Skipped row due to error: {e}")
    print("✅ Book users inserted.")


# === Step 4: Insert Users from Processed Music CSV ===
def insert_music_users():
    print("🎵 Inserting music users...")
    df = pd.read_csv("../data/processed/music/music_rating.csv")

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO users (user_id, from_music)
            VALUES (%s, 1)
            ON CONFLICT (user_id) DO UPDATE
            SET from_music = 1;
        """, (int(row["user_id"]),))
    print("✅ Music users inserted.")


# === Run All ===
if __name__ == "__main__":
    create_users_table()
    insert_movie_users()
    insert_book_users()
    insert_music_users()

    conn.commit()
    cur.close()
    conn.close()
    print("🚀 All user data loaded successfully into 'users' table.")
