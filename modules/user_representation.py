import numpy as np

def build_dur(csv, asv, session_history=None):
    """
    Stage 3: Combine CSV + ASV into a Dynamic User Representation (DUR).
    
    Stand-in for ECCM + TDM (Bi-directional ConvLSTM).
    Real version computes emotion-context correlation matrix
    and tracks temporal drift with BiConvLSTM.
    Here we use outer product for correlation + 
    moving average for temporal drift.
    """

    # --- ECCM stand-in: outer product of CSV and ASV ---
    # Captures co-dependencies between context and emotion dimensions
    # e.g. "night + mobile" co-occurring with "low valence"
    correlation_matrix = np.outer(csv, asv)

    # Flatten the matrix into a vector
    correlation_vec = correlation_matrix.flatten()

    # --- TDM stand-in: temporal drift via session history ---
    # If we have past sessions, average them to capture drift
    if session_history and len(session_history) > 0:
        # Stack past DUR snapshots and compute weighted average
        # More recent sessions get higher weight
        history_array = np.array(session_history)
        weights = np.linspace(0.5, 1.0, len(history_array))
        weights /= weights.sum()
        temporal_context = np.average(history_array, axis=0, weights=weights)
    else:
        # No history — use current combined signal as baseline
        temporal_context = np.concatenate([csv, asv])

    # Normalize temporal context to same length as correlation_vec
    # by projecting down with average pooling
    chunk_size = max(1, len(correlation_vec) // len(temporal_context))
    temporal_vec = np.array([
        np.mean(temporal_context[i:i+1]) 
        for i in range(len(temporal_context))
    ])

    # --- Build DUR: correlation + direct signals + temporal ---
    # Keep correlation_vec compressed (take every nth element)
    step = max(1, len(correlation_vec) // 32)
    compressed_correlation = correlation_vec[::step][:32]

    dur = np.concatenate([
        csv,                    # raw context (18 dims)
        asv,                    # raw emotion (8 dims)
        compressed_correlation, # emotion-context correlations (32 dims)
        temporal_vec,           # temporal drift signal
    ])

    return dur

def describe_dur(dur, csv_size=18, asv_size=8):
    """Break down what's in the DUR."""
    return {
        "total_dimensions": len(dur),
        "context_portion": dur[:csv_size],
        "emotion_portion": dur[csv_size:csv_size+asv_size],
        "correlation_portion": dur[csv_size+asv_size:csv_size+asv_size+32],
        "temporal_portion": dur[csv_size+asv_size+32:]
    }

if __name__ == "__main__":
    # Import previous stages
    import sys
    sys.path.append(".")
    from modules.context_encoder import encode_context
    from modules.emotion_encoder import encode_emotion

    # Build CSV and ASV from previous stages
    csv = encode_context(
        device="mobile",
        platform="netflix",
        time_slot="night",
        day="weekend",
        session_duration=45
    )

    asv = encode_emotion(
        skip_rate=0.7,
        completion_rate=0.4,
        rewatch_ratio=0.1,
        dwell_time=8.0,
        review_text="it was okay I guess"
    )

    # Simulate 3 past sessions for temporal context
    past_session_1 = np.concatenate([
        encode_context("laptop", "netflix", "evening", "weekday", 90),
        encode_emotion(0.2, 0.8, 0.3, 15.0, "really enjoyed this")
    ])
    past_session_2 = np.concatenate([
        encode_context("mobile", "youtube", "night", "weekday", 30),
        encode_emotion(0.5, 0.5, 0.0, 10.0, "it was fine")
    ])
    past_session_3 = np.concatenate([
        encode_context("smart_tv", "amazon", "afternoon", "weekend", 120),
        encode_emotion(0.1, 0.9, 0.2, 20.0, "amazing movie loved it")
    ])

    session_history = [past_session_1, past_session_2, past_session_3]

    # Build DUR
    dur = build_dur(csv, asv, session_history)
    breakdown = describe_dur(dur)

    print(f"CSV shape: {csv.shape}")
    print(f"ASV shape: {asv.shape}")
    print(f"DUR shape: {dur.shape}")
    print(f"\nDUR breakdown:")
    print(f"  Context dims:     {len(breakdown['context_portion'])}")
    print(f"  Emotion dims:     {len(breakdown['emotion_portion'])}")
    print(f"  Correlation dims: {len(breakdown['correlation_portion'])}")
    print(f"  Temporal dims:    {len(breakdown['temporal_portion'])}")
    print(f"\nDUR vector (first 20 values): {dur[:20].round(3)}")