import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Genre-to-mood mapping
MOOD_GENRE_AFFINITY = {
    "high_val_high_aro":  ["Action", "Adventure", "Comedy", "Animation"],
    "high_val_low_aro":   ["Romance", "Musical", "Fantasy", "Children"],
    "low_val_high_aro":   ["Thriller", "Horror", "Crime", "Mystery"],
    "low_val_low_aro":    ["Drama", "Documentary", "Film-Noir", "War"],
    "neutral":            ["Comedy", "Drama", "Adventure", "Sci-Fi"]
}

# Genre cluster groups for diversity filtering
GENRE_CLUSTERS = [
    {"Action", "Adventure", "Thriller", "Crime"},
    {"Comedy", "Romance", "Musical"},
    {"Drama", "Documentary", "War", "Film-Noir"},
    {"Horror", "Mystery", "Sci-Fi"},
    {"Animation", "Children", "Fantasy"},
]

def get_mood_key(asv):
    valence = asv[0]
    arousal = asv[1]
    if valence >= 0.5 and arousal >= 0.5:
        return "high_val_high_aro"
    elif valence >= 0.5 and arousal < 0.5:
        return "high_val_low_aro"
    elif valence < 0.5 and arousal >= 0.5:
        return "low_val_high_aro"
    elif valence < 0.5 and arousal < 0.5:
        return "low_val_low_aro"
    return "neutral"

def get_genre_cluster(genres):
    """Return which cluster index a movie's genres belong to."""
    genre_set = set(genres)
    for i, cluster in enumerate(GENRE_CLUSTERS):
        if genre_set & cluster:
            return i
    return -1

def extract_year(title):
    """Extract year from MovieLens title format."""
    try:
        if "(" in title and ")" in title:
            year = title[title.rfind("(")+1:title.rfind(")")]
            if year.isdigit() and len(year) == 4:
                return int(year)
    except Exception:
        pass
    return None

def build_movie_vectors(movies):
    return np.stack(movies["vector"].values)

def get_recommendations(cue, asv, movies, top_n=10,
                        ratings=None, rating_boost=True):
    """
    Stage 5: Improved ranking with multiple scoring layers.
    
    Scoring breakdown:
    1. Cosine similarity  — genre vector alignment with CUE
    2. Mood genre boost   — 1.25x for mood-matched genres
    3. IMDB-style boost   — uses MovieLens avg ratings
    4. Recency boost      — newer movies score slightly higher
    5. Diversity filter   — avoid clustering in same genre group
    """

    movie_vectors = build_movie_vectors(movies)

    # Trim CUE to match movie vector size
    cue_trimmed = cue[:movie_vectors.shape[1]]
    if np.linalg.norm(cue_trimmed) == 0:
        cue_trimmed = np.ones(movie_vectors.shape[1]) / movie_vectors.shape[1]

    # 1. Base cosine similarity
    cue_2d      = cue_trimmed.reshape(1, -1)
    similarities = cosine_similarity(cue_2d, movie_vectors)[0]
    scores       = similarities.copy()

    # 2. Mood-genre affinity boost
    mood_key  = get_mood_key(asv)
    preferred = MOOD_GENRE_AFFINITY.get(mood_key, [])
    for i, genres in enumerate(movies["genres"]):
        if any(g in preferred for g in genres):
            scores[i] *= 1.25

    # 3. Average rating boost
    if rating_boost and ratings is not None:
        avg_ratings = (ratings.groupby("movieId")["rating"]
                               .mean()
                               .reset_index())
        avg_ratings.columns = ["movieId", "avg_rating"]
        movies_r = movies.merge(avg_ratings, on="movieId", how="left")
        movies_r["avg_rating"] = movies_r["avg_rating"].fillna(3.0)
        rating_factor = (movies_r["avg_rating"] / 5.0).values
        # Stronger rating influence: 0.8 to 1.1 multiplier
        scores *= (0.8 + 0.3 * rating_factor)

    # 4. Recency boost — newer movies get up to 10% boost
    current_year = 2024
    for i, title in enumerate(movies["title"]):
        year = extract_year(title)
        if year:
            age = max(0, current_year - year)
            # Movies from last 10 years: full boost, older: diminishing
            recency_factor = max(0.9, 1.0 - (age * 0.002))
            scores[i] *= recency_factor

    # 5. Get top candidates (2x top_n to allow diversity filtering)
    candidate_indices = np.argsort(scores)[::-1][:top_n * 4]

    # 6. Diversity filter — max 2 movies per genre cluster
    # 6. Diversity filter — max 3 per cluster, fall back if not enough
    cluster_counts = {}
    final_indices  = []
    overflow       = []

    for idx in candidate_indices:
        genres  = movies.iloc[idx]["genres"]
        cluster = get_genre_cluster(genres)
        count   = cluster_counts.get(cluster, 0)
        if cluster == -1 or count < 3:
            final_indices.append(idx)
            cluster_counts[cluster] = count + 1
        else:
            overflow.append(idx)

    # If diversity filter left us short, fill with overflow
    if len(final_indices) < top_n:
        needed = top_n - len(final_indices)
        final_indices += overflow[:needed]

    final_indices = final_indices[:top_n]

    # Build results
    results = []
    for idx in final_indices:
        row = movies.iloc[idx]
        results.append({
            "title":      row["title"],
            "genres":     row["genres"],
            "score":      round(float(scores[idx]), 4),
            "similarity": round(float(similarities[idx]), 4),
            "mood_match": any(g in preferred for g in row["genres"])
        })

    return results, mood_key, preferred


def apply_feedback(movies, recommendations, movie_title, star_rating):
    """
    AOOA stand-in: adjust scores based on star rating feedback.
    1-2 stars = negative, 3 = neutral, 4-5 = positive
    """
    feedback_weight = (star_rating - 3) / 2.0

    rated_movie = next(
        (r for r in recommendations if r["title"] == movie_title), None
    )
    if not rated_movie:
        return recommendations

    rated_genres = set(rated_movie["genres"])

    updated = []
    for rec in recommendations:
        shared     = len(rated_genres & set(rec["genres"]))
        adjustment = 1.0 + (feedback_weight * 0.1 * shared)
        updated.append({
            **rec,
            "score": round(rec["score"] * adjustment, 4)
        })

    updated.sort(key=lambda x: x["score"], reverse=True)
    return updated


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from modules.data_loader         import load_data, encode_movies
    from modules.context_encoder     import encode_context
    from modules.emotion_encoder     import encode_emotion
    from modules.user_representation import build_dur
    from modules.fusion_attention    import build_cue

    movies, ratings = load_data()
    movies, _       = encode_movies(movies)

    csv = encode_context("mobile", "netflix", "night", "weekend", 45)
    asv = encode_emotion(0.7, 0.4, 0.1, 8.0, "it was okay I guess")
    dur = build_dur(csv, asv)
    cue, _ = build_cue(csv, asv, dur)

    recs, mood_key, preferred = get_recommendations(
        cue, asv, movies, top_n=10, ratings=ratings
    )

    print(f"Mood: {mood_key}")
    print(f"Preferred genres: {preferred}")
    print(f"\nTop 10 Recommendations:")
    for i, r in enumerate(recs, 1):
        tag = "✓" if r["mood_match"] else ""
        print(f"  {i}. {r['title']}")
        print(f"     Score: {r['score']}  Similarity: {r['similarity']} {tag}")
        print(f"     Genres: {', '.join(r['genres'])}")