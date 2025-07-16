import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

# Load environment variables from .env file
load_dotenv()
DB_PARAMS = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT"),
}

def create_modification_log_table():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_modification_log (
            id SERIAL PRIMARY KEY,
            user_id BIGINT,
            username VARCHAR(255),
            modified_on TIMESTAMP,
            modified_by VARCHAR(255)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def log_user_modification(user_id, username, modified_by="admin"):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    modified_on = datetime.now()
    cur.execute("""
        INSERT INTO user_modification_log (user_id, username, modified_on, modified_by)
        VALUES (%s, %s, %s, %s)
    """, (user_id, username, modified_on, modified_by))
    conn.commit()
    cur.close()
    conn.close()

def log_all_new_users_from_csv(csv_path="../data/processed/new_users.csv"):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        user_id = int(row["user_id"])
        username = row.get("username")
        log_user_modification(user_id, username)
    print("All new users from CSV logged in modification table.")

if __name__ == "__main__":
    create_modification_log_table()
    log_all_new_users_from_csv()  # This will log all users in new_users.csv
    print("All new users logged.") 