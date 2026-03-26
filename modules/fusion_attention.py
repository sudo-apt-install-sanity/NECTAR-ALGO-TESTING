import numpy as np

def build_cue(csv, asv, dur):
    """
    Stage 4: Fuse CSV + ASV + DUR into Comprehensive User Embedding (CUE).
    
    Stand-in for NFAM + GMAU.
    Real version uses multi-head cross-attention + gating.
    Here we use weighted fusion + simple gating.
    """

    # --- NFAM stand-in: compute signal reliability scores ---
    # How much should we trust each signal right now?

    # Context signal strength: how many context fields are active
    context_strength = np.sum(csv > 0) / len(csv)

    # Emotion signal strength: how extreme the emotional state is
    # (neutral = low strength, strong emotion = high strength)
    emotion_strength = float(np.std(asv))
    emotion_strength = min(emotion_strength * 4, 1.0)

    # Temporal signal strength: always moderate in prototype
    temporal_strength = 0.6

    # Normalize weights so they sum to 1
    total = context_strength + emotion_strength + temporal_strength
    w_context = context_strength / total
    w_emotion = emotion_strength / total
    w_temporal = temporal_strength / total

    # --- GMAU stand-in: gate each signal ---
    # If strength is too low, suppress it entirely
    gate_threshold = 0.15

    gate_context  = 1.0 if w_context  > gate_threshold else 0.0
    gate_emotion  = 1.0 if w_emotion  > gate_threshold else 0.0
    gate_temporal = 1.0 if w_temporal > gate_threshold else 0.0

    # Apply gates
    gated_csv = csv * gate_context
    gated_asv = asv * gate_emotion
    gated_dur = dur * gate_temporal

    # --- Fuse into CUE ---
    # Weighted concatenation of all three signals
    cue = np.concatenate([
        gated_csv * w_context,
        gated_asv * w_emotion,
        gated_dur * w_temporal
    ])

    # Store attention weights for UI display
    attention_weights = {
        "context":  round(float(w_context),  3),
        "emotion":  round(float(w_emotion),  3),
        "temporal": round(float(w_temporal), 3),
        "gate_context":  gate_context,
        "gate_emotion":  gate_emotion,
        "gate_temporal": gate_temporal
    }

    return cue, attention_weights


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from modules.context_encoder     import encode_context
    from modules.emotion_encoder     import encode_emotion
    from modules.user_representation import build_dur
    import numpy as np

    csv = encode_context("mobile", "netflix", "night", "weekend", 45)
    asv = encode_emotion(0.7, 0.4, 0.1, 8.0, "it was okay I guess")

    past = [
        np.concatenate([
            encode_context("laptop", "netflix", "evening", "weekday", 90),
            encode_emotion(0.2, 0.8, 0.3, 15.0, "really enjoyed this")
        ])
    ]
    dur = build_dur(csv, asv, past)
    cue, weights = build_cue(csv, asv, dur)

    print(f"CUE shape: {cue.shape}")
    print(f"\nAttention weights:")
    for k, v in weights.items():
        print(f"  {k}: {v}")
    print(f"\nCUE (first 20 values): {cue[:20].round(3)}")