# app.py
# Music Recommendation System
# Data structures used:
#   - dict (hash table)  -> O(1) song/user lookup
#   - dict of dicts      -> adjacency list for similarity graph
#   - heapq (min-heap)   -> efficient top-N ranking

import math
import heapq
import pandas as pd
from collections import defaultdict, deque
from flask import Flask, render_template, jsonify

app = Flask(__name__)


# -------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------

def load_songs(filepath="songs_normalize.csv"):
    df = pd.read_csv(filepath)
    df = df[["artist", "song", "energy", "danceability",
             "valence", "tempo", "acousticness",
             "popularity", "genre"]].dropna()
    df = df.drop_duplicates(subset="song")
    return df


# -------------------------------------------------
# 2. DATA STORAGE (hash tables)
# -------------------------------------------------

class SongStore:
    def __init__(self):
        self._data = {}

    def add_song(self, song_id, attributes):
        self._data[song_id] = attributes

    def get_song(self, song_id):
        return self._data.get(song_id, {})

    def all_songs(self):
        return list(self._data.keys())


class UserStore:
    def __init__(self):
        self._data = {}

    def add_user(self, user_id, ratings):
        self._data[user_id] = ratings

    def get_user(self, user_id):
        return self._data.get(user_id, {})

    def all_users(self):
        return list(self._data.keys())


# -------------------------------------------------
# 3. SIMILARITY ENGINE
# -------------------------------------------------

def cosine_similarity(vec_a, vec_b):
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(vec_a[k] * vec_b[k] for k in vec_a if k in vec_b)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# -------------------------------------------------
# 4. SIMILARITY GRAPH
# -------------------------------------------------

class SimilarityGraph:
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self._adj = defaultdict(dict)

    def build(self, user_store):
        users = user_store.all_users()
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u, v = users[i], users[j]
                score = cosine_similarity(
                    user_store.get_user(u),
                    user_store.get_user(v)
                )
                if score >= self.threshold:
                    self._adj[u][v] = score
                    self._adj[v][u] = score

    def neighbors(self, user_id):
        return self._adj.get(user_id, {})


# -------------------------------------------------
# 5. RECOMMENDATION GENERATOR (BFS)
# -------------------------------------------------

def get_candidates(target_user, user_store, graph, hops=1):
    seen_users = {target_user}
    target_songs = set(user_store.get_user(target_user).keys())
    candidates = defaultdict(float)
    queue = deque([(target_user, 1)])

    while queue:
        current_user, depth = queue.popleft()
        for neighbor, similarity in graph.neighbors(current_user).items():
            if neighbor in seen_users:
                continue
            seen_users.add(neighbor)
            for song_id, rating in user_store.get_user(neighbor).items():
                if song_id not in target_songs:
                    candidates[song_id] += similarity * rating
            if depth < hops:
                queue.append((neighbor, depth + 1))

    return dict(candidates)


# -------------------------------------------------
# 6. RANKER (min-heap)
# -------------------------------------------------

def top_n(candidates, n=5):
    return heapq.nlargest(n, candidates.items(), key=lambda x: x[1])


# -------------------------------------------------
# 7. BUILD SYSTEM
# -------------------------------------------------

def build_system():
    df = load_songs()
    song_store = SongStore()
    user_store = UserStore()

    for _, row in df.iterrows():
        song_store.add_song(row["song"], {
            "artist": row["artist"],
            "energy": row["energy"],
            "danceability": row["danceability"],
            "valence": row["valence"],
            "tempo": row["tempo"],
            "acousticness": row["acousticness"],
            "popularity": row["popularity"],
            "genre": row["genre"]
        })

    all_songs = df["song"].tolist()
    pop = df.set_index("song")["popularity"]

    # Simulated users with real names and overlapping song pools
    users_raw = {
        "Alex":   {s: pop[s] / 100 for s in all_songs[0:120]},
        "Jordan": {s: pop[s] / 100 for s in all_songs[60:180]},
        "Maya":   {s: pop[s] / 100 for s in all_songs[120:240]},
        "Tyler":  {s: pop[s] / 100 for s in all_songs[180:300]},
        "Sofia":  {s: pop[s] / 100 for s in all_songs[30:150]},
        "Marcus": {s: pop[s] / 100 for s in all_songs[90:210]},
        "Priya":  {s: pop[s] / 100 for s in all_songs[150:270]},
        "Lena":   {s: pop[s] / 100 for s in all_songs[10:130]},
    }

    for user_id, ratings in users_raw.items():
        user_store.add_user(user_id, ratings)

    graph = SimilarityGraph(threshold=0.3)
    graph.build(user_store)

    return song_store, user_store, graph


song_store, user_store, graph = build_system()


# -------------------------------------------------
# 8. FLASK ROUTES
# -------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/users")
def get_users():
    return jsonify(user_store.all_users())


@app.route("/recommend/<user_id>")
def recommend(user_id):
    candidates = get_candidates(user_id, user_store, graph, hops=1)
    if not candidates:
        return jsonify([])
    top = top_n(candidates, n=5)
    results = []
    for song_id, score in top:
        info = song_store.get_song(song_id)
        results.append({
            "song": song_id,
            "artist": info.get("artist", "Unknown"),
            "genre": info.get("genre", ""),
            "score": round(score, 3),
            "energy": round(info.get("energy", 0), 2),
            "danceability": round(info.get("danceability", 0), 2),
            "popularity": int(info.get("popularity", 0))
        })
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)