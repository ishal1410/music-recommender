# Music Recommendation System

A Python and Flask web app that recommends songs based on what users with similar taste are listening to.

## How it works

The system loads a dataset of real Spotify songs. Each simulated user has a listening history. The app compares users, connects similar ones in a graph, then walks that graph to find songs a user has not heard yet. Results are ranked and the top 5 are returned.

## Data structures

**Hash table (dict)**
Stores users and songs. O(1) lookup by ID. A list would scan the whole dataset on every request.

**Adjacency dict (dict of dicts)**
Stores the similarity graph. Each user points to their neighbors and the similarity score. Skips pairs with zero similarity so memory stays low.

**Cosine similarity**
Compares two users by the angle between their rating vectors, not the size. A user who rated 100 songs is not assumed to be more similar to someone than a user who rated 10.

**Deque for BFS**
Traverses the graph to collect candidate songs. O(1) on both ends. BFS goes to the closest neighbors first, which are the most relevant.

**Min-heap (heapq)**
Returns top 5 results in O(n log k) instead of sorting the full list at O(n log n).

## Dataset

Spotify Top Hits 2000–2019 from Kaggle. Contains artist, song name, energy, danceability, valence, tempo, and genre.

## Setup
pip install flask pandas
py app.py

Then open `http://127.0.0.1:5000`

## Structure
app.py                  recommendation logic and Flask routes
songs_normalize.csv     dataset
templates/index.html    frontend
static/style.css        styling

Save and push:
git add README.md
git commit -m "Update README"
git push
