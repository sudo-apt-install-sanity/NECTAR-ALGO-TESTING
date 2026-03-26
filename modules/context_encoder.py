import numpy as np

# All possible values for each context field
DEVICES = ["mobile", "tablet", "laptop", "smart_tv"]
PLATFORMS = ["netflix", "youtube", "amazon", "disney"]
TIME_SLOTS = ["morning", "afternoon", "evening", "night"]
DAYS = ["weekday", "weekend"]

def encode_context(device, platform, time_slot, day, session_duration):
    """
    Stage 1: Convert raw context into a Contextual State Vector (CSV).
    
    This is our simplified stand-in for HCE (GCE + TCT).
    Real version would use Graph Neural Networks + Transformers.
    Here we use one-hot encoding + normalization.
    """

    # One-hot encode each categorical field
    device_vec = _one_hot(device, DEVICES)
    platform_vec = _one_hot(platform, PLATFORMS)
    time_vec = _one_hot(time_slot, TIME_SLOTS)
    day_vec = _one_hot(day, DAYS)

    # Normalize session duration (assume max session = 300 mins)
    duration_vec = np.array([min(session_duration / 300.0, 1.0)])

    # Concatenate everything into one vector
    csv = np.concatenate([device_vec, platform_vec, time_vec, day_vec, duration_vec])

    # Add simple context interaction signals (stand-in for GCE relationships)
    # e.g. night + mobile is a specific pattern worth capturing
    is_night_mobile = 1.0 if (time_slot == "night" and device == "mobile") else 0.0
    is_weekend_tv = 1.0 if (day == "weekend" and device == "smart_tv") else 0.0
    is_morning_laptop = 1.0 if (time_slot == "morning" and device == "laptop") else 0.0

    interaction_vec = np.array([is_night_mobile, is_weekend_tv, is_morning_laptop])

    # Final CSV = one-hot vectors + duration + interaction signals
    csv = np.concatenate([csv, interaction_vec])

    return csv

def _one_hot(value, options):
    """Convert a categorical value into a one-hot vector."""
    vec = np.zeros(len(options))
    if value.lower() in options:
        vec[options.index(value.lower())] = 1.0
    return vec

def describe_csv(csv):
    """Human readable breakdown of what's in the CSV."""
    labels = (
        [f"device:{d}" for d in DEVICES] +
        [f"platform:{p}" for p in PLATFORMS] +
        [f"time:{t}" for t in TIME_SLOTS] +
        [f"day:{d}" for d in DAYS] +
        ["session_duration"] +
        ["interact:night_mobile", "interact:weekend_tv", "interact:morning_laptop"]
    )
    active = [labels[i] for i, v in enumerate(csv) if v > 0]
    return active

if __name__ == "__main__":
    # Simulate a user context
    csv = encode_context(
        device="mobile",
        platform="netflix",
        time_slot="night",
        day="weekend",
        session_duration=45
    )

    print(f"CSV shape: {csv.shape}")
    print(f"CSV vector: {csv}")
    print(f"\nActive context signals:")
    for signal in describe_csv(csv):
        print(f"  ✓ {signal}")