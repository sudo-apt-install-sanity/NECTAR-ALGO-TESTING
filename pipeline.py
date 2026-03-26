import numpy as np
import sys
sys.path.append(".")

from modules.data_loader         import load_data, encode_movies
from modules.context_encoder     import encode_context, describe_csv
from modules.emotion_encoder     import encode_emotion, describe_asv
from modules.user_representation import build_dur
from modules.fusion_attention    import build_cue
from modules.recommender         import get_recommendations, apply_feedback

def run_pipeline(device, platform, time_slot, day, session_duration,
                 skip_rate, completion_rate, rewatch_ratio,
                 dwell_time, review_text, session_history=None, top_n=10):
    """
    Full NECTAR pipeline: raw user data → movie recommendations.
    Returns all intermediate outputs for UI display.
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

    # Stage 3: Dynamic user representation
    dur = build_dur(csv, asv, session_history)

    # Stage 4: Fusion attention
    cue, attention_weights = build_cue(csv, asv, dur)

    # Stage 5: Recommendations
    recommendations, mood_key, preferred_genres = get_recommendations(
        cue, asv, movies, top_n=top_n, ratings=ratings
    )

    return {
        "csv":                csv,
        "csv_description":    csv_description,
        "asv":                asv,
        "asv_description":    asv_description,
        "dur_shape":          dur.shape,
        "cue_shape":          cue.shape,
        "attention_weights":  attention_weights,
        "mood_key":           mood_key,
        "preferred_genres":   preferred_genres,
        "recommendations":    recommendations,
        "movies":             movies,
    }


if __name__ == "__main__":
    result = run_pipeline(
        device="mobile",        platform="netflix",
        time_slot="night",      day="weekend",
        session_duration=45,    skip_rate=0.7,
        completion_rate=0.4,    rewatch_ratio=0.1,
        dwell_time=8.0,         review_text="it was okay I guess"
    )

    print("=== NECTAR PIPELINE OUTPUT ===\n")
    print(f"[Stage 1] Context signals: {result['csv_description']}")
    print(f"\n[Stage 2] Emotion state:")
    for k, v in result['asv_description'].items():
        print(f"  {k}: {v}")
    print(f"\n[Stage 3] DUR shape: {result['dur_shape']}")
    print(f"\n[Stage 4] CUE shape: {result['cue_shape']}")
    print(f"  Attention weights: {result['attention_weights']}")
    print(f"\n[Stage 5] Mood: {result['mood_key']}")
    print(f"  Preferred genres: {result['preferred_genres']}")
    print(f"\nTop Recommendations:")
    for i, r in enumerate(result['recommendations'], 1):
        tag = " ✓" if r['mood_match'] else ""
        print(f"  {i}. {r['title']} (score: {r['score']}){tag}")