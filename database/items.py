import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

# 📦 Load environment variables from .env
load_dotenv()

DB_NAME     = os.getenv("DB_NAME")
DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")

# 🔌 Connect to PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# 🛠️ Create the 'items' table
cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY,
        title TEXT,
        year INT,
        source TEXT
    );
""")
conn.commit()
print("✅ 'items' table created.")

# === Insert Movies ===
def insert_movies():
    print("🎬 Inserting movie items...")
    df = pd.read_csv("../data/processed/movie/movie_info.csv")

    # Extract year from release_date safely
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce', dayfirst=True).dt.year

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO items (item_id, title, year, source)
            VALUES (%s, %s, %s, 'movies')
            ON CONFLICT (item_id) DO NOTHING;
        """, (
            f"movies_{int(row['movie_id'])}",
            row['title'],
            int(row['year']) if not pd.isna(row['year']) else None
        ))
    print("✅ Movies inserted.")

# === Insert Books ===
def insert_books():
    print("📚 Inserting book items...")
    df = pd.read_csv("../data/processed/book/book_info.csv")

    for _, row in df.iterrows():
        isbn = str(row['ISBN']).strip()
        title = row.get('Title', '').strip()
        year = row.get('Year', None)

        # Skip rows with missing or bad ISBN/title
        if not isbn or isbn.lower() == 'nan' or not title:
            continue

        cur.execute("""
            INSERT INTO items (item_id, title, year, source)
            VALUES (%s, %s, %s, 'books')
            ON CONFLICT (item_id) DO NOTHING;
        """, (
            f"books_{isbn}",
            title,
            year
        ))
    print("✅ Books inserted.")


# === Insert Music ===
def insert_music():
    print("🎵 Inserting music items...")
    df = pd.read_csv("../data/processed/music/music_info.csv")

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO items (item_id, title, year, source)
            VALUES (%s, %s, NULL, 'music')
            ON CONFLICT (item_id) DO NOTHING;
        """, (
            f"music_{int(row['id'])}",
            row['name']
        ))
    print("✅ Music inserted.")

if __name__ == "__main__":
    # 🚀 Run all inserts
    insert_movies()
    insert_books()
    insert_music()

    conn.commit()
    cur.close()
    conn.close()
    print("🚀 All items inserted successfully.")
