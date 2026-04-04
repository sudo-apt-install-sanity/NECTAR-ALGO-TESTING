import numpy as np
import sys
sys.path.append(".")

from modules.data_loader         import load_data, encode_movies
from modules.context_encoder     import encode_context, describe_csv
from modules.emotion_encoder     import encode_emotion, describe_asv
from modules.user_representation import build_dur
from modules.fusion_attention    import build_cue
from modules.recommender         import get_recommendations, apply_feedback
from database.queries            import save_session, get_session_history_vectors

def run_pipeline(device, platform, time_slot, day, session_duration,
                 skip_rate, completion_rate, rewatch_ratio,
                 dwell_time, review_text,
                 user_id=None, top_n=10):
    """
    Full NECTAR pipeline: raw user data → movie recommendations.
    If user_id is provided, loads session history from DB and saves
    the current session after running.
    """

    # Load data
    movies, ratings = load_data()
    movies, _       = encode_movies(movies)

    # Stage 1: Context encoding
    csv = encode_context(device, platform, time_slot, day, session_duration)
    csv_description = describe_csv(csv)

    # Stage 2: Emotion encoding
    asv = encode_emotion(skip_rate, completion_rate,
                         rewatch_ratio, dwell_time, review_text)
    asv_description = describe_asv(asv)

    # Load session history from DB if user is logged in
    session_history = None
    if user_id is not None:
        session_history = get_session_history_vectors(user_id, limit=5)
        if len(session_history) == 0:
            session_history = None

    # Stage 3: Dynamic user representation
    dur = build_dur(csv, asv, session_history)

    # Stage 4: Fusion attention
    cue, attention_weights = build_cue(csv, asv, dur)

    # Stage 5: Recommendations
    recommendations, mood_key, preferred_genres = get_recommendations(
        cue, asv, movies, top_n=top_n, ratings=ratings
    )

    # Save session to DB if user is logged in
    session_id = None
    if user_id is not None:
        inputs = {
            "device": device, "platform": platform,
            "time_slot": time_slot, "day": day,
            "session_duration": session_duration,
            "skip_rate": skip_rate, "completion_rate": completion_rate,
            "rewatch_ratio": rewatch_ratio, "dwell_time": dwell_time,
            "review_text": review_text
        }
        session_id = save_session(user_id, inputs, mood_key, csv, asv)

    return {
        "csv":               csv,
        "csv_description":   csv_description,
        "asv":               asv,
        "asv_description":   asv_description,
        "dur_shape":         dur.shape,
        "cue_shape":         cue.shape,
        "attention_weights": attention_weights,
        "mood_key":          mood_key,
        "preferred_genres":  preferred_genres,
        "recommendations":   recommendations,
        "movies":            movies,
        "session_id":        session_id,
    }


if __name__ == "__main__":
    result = run_pipeline(
        device="mobile",        platform="netflix",
        time_slot="night",      day="weekend",
        session_duration=45,    skip_rate=0.7,
        completion_rate=0.4,    rewatch_ratio=0.1,
        dwell_time=8.0,         review_text="it was okay I guess",
        user_id=1
    )
    print("=== NECTAR PIPELINE OUTPUT ===\n")
    print(f"Session saved with ID: {result['session_id']}")
    print(f"Mood: {result['mood_key']}")
    print(f"Top recommendations:")
    for i, r in enumerate(result['recommendations'][:5], 1):
        print(f"  {i}. {r['title']} (score: {r['score']})")