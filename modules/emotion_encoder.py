import numpy as np

def encode_emotion(skip_rate, completion_rate, rewatch_ratio, 
                   dwell_time, review_text=None):
    """
    Stage 2: Convert user behavior into an Affective State Vector (ASV).
    
    Stand-in for BEEN (TIPA + SE + ELM).
    Real version uses LSTM for behavior + BERT for text.
    Here we use rule-based scoring + simple sentiment.
    """

    # --- TIPA stand-in: score behavior signals ---
    # Each signal contributes to valence (positive/negative)
    # and arousal (calm/excited)

    # High skip rate = negative, restless
    skip_score = 1.0 - skip_rate  # invert: high skip = low score

    # High completion = positive, engaged
    completion_score = completion_rate

    # Rewatch = very positive, high engagement
    rewatch_score = min(rewatch_ratio * 2, 1.0)

    # Dwell time: normalize (assume max meaningful dwell = 30s)
    dwell_score = min(dwell_time / 30.0, 1.0)

    # Derive valence (mood positivity: 0=negative, 1=positive)
    valence = np.mean([skip_score, completion_score, 
                       rewatch_score, dwell_score])

    # Derive arousal (engagement level: 0=passive, 1=active)
    arousal = np.mean([completion_score, dwell_score, rewatch_score])

    # Derive dominance (control: low skip + high completion = in control)
    dominance = np.mean([skip_score, completion_score])

    # --- SE stand-in: simple keyword sentiment from review ---
    sentiment_score = 0.5  # neutral default

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

    # --- ELM stand-in: combine into unified emotion vector ---
    # VAD model: Valence, Arousal, Dominance + sentiment
    asv = np.array([
        valence,
        arousal,
        dominance,
        sentiment_score,
        # Derived composite signals
        (valence + sentiment_score) / 2,   # overall positivity
        1.0 - arousal,                      # passivity
        valence * arousal,                  # engaged-positive
        (1.0 - valence) * (1.0 - arousal), # disengaged-negative
    ])

    return asv

def describe_asv(asv):
    """Human readable emotional state from ASV."""
    valence, arousal, dominance, sentiment = asv[0], asv[1], asv[2], asv[3]

    mood = ""
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
        "valence": round(float(valence), 2),
        "arousal": round(float(arousal), 2),
        "dominance": round(float(dominance), 2),
        "sentiment": round(float(sentiment), 2)
    }

if __name__ == "__main__":
    # Simulate a user who has been skipping a lot, low completion
    asv = encode_emotion(
        skip_rate=0.7,
        completion_rate=0.4,
        rewatch_ratio=0.1,
        dwell_time=8.0,
        review_text="it was okay I guess"
    )

    print(f"ASV shape: {asv.shape}")
    print(f"ASV vector: {asv}")
    print()

    description = describe_asv(asv)
    print(f"Mood Summary: {description['mood_summary']}")
    print(f"  Valence   (pos/neg): {description['valence']}")
    print(f"  Arousal   (engagement): {description['arousal']}")
    print(f"  Dominance (control): {description['dominance']}")
    print(f"  Sentiment (from text): {description['sentiment']}")