import numpy as np
import sys
sys.path.append(".")

# Try to load trained model — fall back to rules if unavailable
_model = None

def _get_model():
    global _model
    if _model is None:
        try:
            from models.emotion_model import load_model
            _model = load_model()
            if _model is not None:
                print("EmotionNet loaded — using trained model for VAD inference")
            else:
                print("EmotionNet not found — using rule-based VAD scoring")
        except Exception as e:
            print(f"EmotionNet unavailable ({e}) — using rule-based VAD scoring")
            _model = False  # Mark as unavailable
    return _model if _model is not False else None


def _rule_based_vad(skip_rate, completion_rate, rewatch_ratio, dwell_time):
    """Original rule-based VAD scoring as fallback."""
    skip_score     = 1.0 - skip_rate
    completion_score = completion_rate
    rewatch_score  = min(rewatch_ratio * 2, 1.0)
    dwell_score    = min(dwell_time / 30.0, 1.0)

    valence   = np.mean([skip_score, completion_score,
                         rewatch_score, dwell_score])
    arousal   = np.mean([completion_score, dwell_score, rewatch_score])
    dominance = np.mean([skip_score, completion_score])
    return valence, arousal, dominance


def _model_based_vad(skip_rate, completion_rate,
                     rewatch_ratio, dwell_time, model):
    """
    Use trained EmotionNet for VAD inference.
    Maps behavioral signals to the same 6-feature space
    the model was trained on.
    """
    from models.emotion_model import predict_vad

    avg_rating_proxy  = completion_rate * 5.0
    rating_std_proxy  = abs(skip_rate - completion_rate) * 2.0
    num_ratings_proxy = 1.0
    trend_proxy       = (rewatch_ratio - skip_rate + 4) / 8.0
    time_span_proxy   = min(dwell_time / 30.0, 1.0)
    recency_proxy     = 0.7  # assume reasonably recent

    features = [
        avg_rating_proxy / 5.0,
        min(rating_std_proxy / 2.0, 1.0),
        num_ratings_proxy,
        trend_proxy,
        time_span_proxy,
        recency_proxy,
    ]

    valence, arousal, dominance = predict_vad(model, features)
    return valence, arousal, dominance


def encode_emotion(skip_rate, completion_rate, rewatch_ratio,
                   dwell_time, review_text=None):
    """
    Stage 2: Convert user behavior into an Affective State Vector (ASV).

    Uses trained EmotionNet if available, falls back to rule-based scoring.
    Stand-in for BEEN (TIPA + SE + ELM).
    """

    # ── VAD inference ─────────────────────────────────────────
    model = _get_model()
    if model is not None:
        valence, arousal, dominance = _model_based_vad(
            skip_rate, completion_rate, rewatch_ratio, dwell_time, model
        )
    else:
        valence, arousal, dominance = _rule_based_vad(
            skip_rate, completion_rate, rewatch_ratio, dwell_time
        )

    # ── Sentiment from review text ────────────────────────────
    sentiment_score = 0.5
    if review_text:
        text = review_text.lower()
        positive_words = ["good", "great", "love", "amazing",
                          "awesome", "enjoyed", "fantastic", "fun"]
        negative_words = ["bad", "boring", "awful", "terrible",
                          "okay", "meh", "guess", "waste", "slow"]
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        if pos_count + neg_count > 0:
            sentiment_score = pos_count / (pos_count + neg_count)

    # ── Build ASV ─────────────────────────────────────────────
    asv = np.array([
        valence,
        arousal,
        dominance,
        sentiment_score,
        (valence + sentiment_score) / 2,
        1.0 - arousal,
        valence * arousal,
        (1.0 - valence) * (1.0 - arousal),
    ])

    return asv


def describe_asv(asv):
    """Human readable emotional state from ASV."""
    valence   = asv[0]
    arousal   = asv[1]
    dominance = asv[2]
    sentiment = asv[3]

    if valence > 0.6 and arousal > 0.6:
        mood = "Engaged and positive — great recommendations opportunity"
    elif valence > 0.6 and arousal < 0.4:
        mood = "Content but passive — suggest easy watching"
    elif valence < 0.4 and arousal > 0.6:
        mood = "Restless or frustrated — needs something compelling"
    elif valence < 0.4 and arousal < 0.4:
        mood = "Disengaged and low — suggest comfort genres"
    else:
        mood = "Neutral — general recommendations apply"

    return {
        "mood_summary": mood,
        "valence":      round(float(valence),   2),
        "arousal":      round(float(arousal),   2),
        "dominance":    round(float(dominance), 2),
        "sentiment":    round(float(sentiment), 2),
        "inference":    "Neural (EmotionNet)" if _get_model() else "Rule-based"
    }


if __name__ == "__main__":
    print("=== Testing with trained model ===\n")

    test_cases = [
        (0.7, 0.4, 0.1, 8.0,  "it was okay I guess",  "Disengaged user"),
        (0.1, 0.9, 0.5, 20.0, "amazing loved it",     "Highly engaged"),
        (0.5, 0.5, 0.0, 10.0, "",                      "Neutral user"),
        (0.9, 0.2, 0.0, 3.0,  "boring waste of time", "Very disengaged"),
    ]

    for skip, comp, rew, dwell, review, label in test_cases:
        asv  = encode_emotion(skip, comp, rew, dwell, review)
        desc = describe_asv(asv)
        print(f"{label}")
        print(f"  Inference:  {desc['inference']}")
        print(f"  Mood:       {desc['mood_summary']}")
        print(f"  Valence:    {desc['valence']} | "
              f"Arousal: {desc['arousal']} | "
              f"Dominance: {desc['dominance']}")
        print(f"  Sentiment:  {desc['sentiment']}")
        print()