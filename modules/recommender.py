import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Genre-to-mood mapping (stand-in for learned associations)
MOOD_GENRE_AFFINITY = {
    # (valence_range, arousal_range) : preferred genres
    "high_val_high_aro":  ["Action", "Adventure", "Comedy", "Animation"],
    "high_val_low_aro":   ["Romance", "Musical", "Fantasy", "Children"],
    "low_val_high_aro":   ["Thriller", "Horror", "Crime", "Mystery"],
    "low_val_low_aro":    ["Drama", "Documentary", "Film-Noir", "War"],
    "neutral":            ["Comedy", "Drama", "Adventure", "Sci-Fi"]
}

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

def build_movie_vectors(movies, all_genres):
    """Convert movies dataframe into numpy vectors."""
    vectors = np.stack(movies["vector"].values)
    return vectors

def get_recommendations(cue, asv, movies, top_n=10, 
                         ratings=None, rating_boost=True):
    """
    Stage 5: Rank movies using CUE via cosine similarity.
    Stand-in for TADRN.
    """

    all_genres = sorted(list(set(
        g for genres in movies["genres"] for g in genres
    )))
    movie_vectors = build_movie_vectors(movies, all_genres)

    # Trim CUE to match movie vector size for cosine similarity
    cue_trimmed = cue[:len(all_genres)]
    if np.linalg.norm(cue_trimmed) == 0:
        cue_trimmed = np.ones(len(all_genres)) / len(all_genres)

    # Compute cosine similarity between user and all movies
    cue_2d = cue_trimmed.reshape(1, -1)
    similarities = cosine_similarity(cue_2d, movie_vectors)[0]

    # Mood-based genre boost (stand-in for ECCM influence)
    mood_key   = get_mood_key(asv)
    preferred  = MOOD_GENRE_AFFINITY.get(mood_key, [])

    boosted = similarities.copy()
    for i, genres in enumerate(movies["genres"]):
        if any(g in preferred for g in genres):
            boosted[i] *= 1.25

    # Optional: boost highly rated movies
    if rating_boost and ratings is not None:
        avg_ratings = (ratings.groupby("movieId")["rating"]
                               .mean()
                               .reset_index())
        avg_ratings.columns = ["movieId", "avg_rating"]
        movies_r = movies.merge(avg_ratings, on="movieId", how="left")
        movies_r["avg_rating"] = movies_r["avg_rating"].fillna(3.0)
        rating_factor = (movies_r["avg_rating"] / 5.0).values
        boosted *= (0.85 + 0.15 * rating_factor)

    # Get top N
    top_indices = np.argsort(boosted)[::-1][:top_n]
    results = []
    for idx in top_indices:
        row = movies.iloc[idx]
        results.append({
            "title":      row["title"],
            "genres":     row["genres"],
            "score":      round(float(boosted[idx]), 4),
            "similarity": round(float(similarities[idx]), 4),
            "mood_match": any(g in preferred for g in row["genres"])
        })

    return results, mood_key, preferred


def apply_feedback(movies, recommendations, movie_title, star_rating):
    """
    AOOA stand-in: adjust scores based on star rating feedback.
    1-2 stars = negative, 3 = neutral, 4-5 = positive
    """
    feedback_weight = (star_rating - 3) / 2.0  # range: -1 to +1

    rated_movie = next(
        (r for r in recommendations if r["title"] == movie_title), None
    )
    if not rated_movie:
        return recommendations

    rated_genres = set(rated_movie["genres"])

    # Boost or penalize movies sharing genres with the rated one
    updated = []
    for rec in recommendations:
        shared = len(rated_genres & set(rec["genres"]))
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
    import numpy as np

    movies, ratings = load_data()
    movies, _       = encode_movies(movies)

    csv = encode_context("mobile", "netflix", "night", "weekend", 45)
    asv = encode_emotion(0.7, 0.4, 0.1, 8.0, "it was okay I guess")
    dur = build_dur(csv, asv)
    cue, weights = build_cue(csv, asv, dur)

    recs, mood_key, preferred = get_recommendations(
        cue, asv, movies, top_n=5, ratings=ratings
    )

    print(f"Mood detected: {mood_key}")
    print(f"Preferred genres: {preferred}")
    print(f"\nTop 5 Recommendations:")
    for i, r in enumerate(recs, 1):
        mood_tag = "✓ mood match" if r["mood_match"] else ""
        print(f"  {i}. {r['title']}")
        print(f"     Genres: {', '.join(r['genres'])}")
        print(f"     Score: {r['score']}  {mood_tag}")