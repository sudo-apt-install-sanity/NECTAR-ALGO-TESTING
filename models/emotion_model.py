import torch
import torch.nn as nn
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.pt")

class EmotionNet(nn.Module):
    """
    Small feedforward network that maps behavioral features
    to VAD (Valence, Arousal, Dominance) scores.

    Input:  6 behavioral features
    Output: 3 emotion dimensions (all in 0-1 range)
    """
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=3):
        super(EmotionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in 0-1 range
        )

    def forward(self, x):
        return self.network(x)


def load_model():
    """Load trained model if it exists, else return None."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = EmotionNet()
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"),
                       weights_only=True)
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Could not load emotion model: {e}")
        return None


def predict_vad(model, features):
    """
    Run inference on a single feature vector.
    features: list or array of 6 values
    Returns: (valence, arousal, dominance) as floats
    """
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
    vad = output.squeeze().numpy()
    return float(vad[0]), float(vad[1]), float(vad[2])


if __name__ == "__main__":
    # Quick architecture test
    model = EmotionNet()
    dummy = torch.randn(4, 6)
    out   = model(dummy)
    print(f"Model architecture: OK")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: {out.min():.3f} - {out.max():.3f}")
    print(f"\nModel parameters: "
          f"{sum(p.numel() for p in model.parameters())}")