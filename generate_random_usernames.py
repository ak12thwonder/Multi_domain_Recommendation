import pandas as pd
import random
import string

try:
    import names
    get_random_name = lambda: names.get_full_name()
except ImportError:
    get_random_name = lambda: ''.join(random.choices(string.ascii_letters, k=8))

user_files = [
    ('data/processed/book/book_user.csv', 'user_id', 'book'),
    ('data/processed/movie/movie_user.csv', 'user_id', 'movie'),
    ('data/processed/music/music_rating.csv', 'user_id', 'music'),
]

# Collect user IDs per domain
user_ids_by_domain = {}
all_user_ids = set()
for path, col, domain in user_files:
    df = pd.read_csv(path)
    user_ids = set(df[col].unique())
    user_ids_by_domain[domain] = user_ids
    all_user_ids.update(user_ids)

from collections import Counter
user_id_counts = Counter()
for domain, ids in user_ids_by_domain.items():
    user_id_counts.update(ids)

# Map user_id to username, ensuring uniqueness
assigned_names = set()
def unique_random_name():
    while True:
        name = get_random_name()
        if name not in assigned_names:
            assigned_names.add(name)
            return name

user_id_to_name = {user_id: unique_random_name() for user_id in all_user_ids}

# Now assign usernames to each file
for path, col, domain in user_files:
    df = pd.read_csv(path)
    df['username'] = df[col].map(lambda uid: user_id_to_name[uid])
    output_path = path.replace('.csv', '_with_names.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved with usernames: {output_path}") 