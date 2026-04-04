import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(".")

from models.emotion_model import EmotionNet, MODEL_PATH

def build_training_data(ratings_path="ml-latest-small/ratings.csv"):
    """
    Engineer behavioral features from MovieLens ratings data.
    Each 'session' is a window of ratings per user.
    We use rating patterns as proxies for emotional state.
    """
    print("Loading ratings...")
    ratings = pd.read_csv(ratings_path)
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings = ratings.sort_values(["userId", "timestamp"])

    print("Engineering features...")
    X = []  # Features
    y = []  # VAD labels

    for user_id, group in ratings.groupby("userId"):
        group = group.reset_index(drop=True)

        # Slide a window of 10 ratings at a time
        window_size = 10
        for start in range(0, len(group) - window_size, 5):
            window = group.iloc[start:start + window_size]

            # ── Feature engineering ───────────────────────
            avg_rating    = window["rating"].mean()
            rating_std    = window["rating"].std()
            num_ratings   = len(window)

            # Rating trend: are ratings going up or down?
            if len(window) > 1:
                first_half  = window.iloc[:5]["rating"].mean()
                second_half = window.iloc[5:]["rating"].mean()
                rating_trend = second_half - first_half
            else:
                rating_trend = 0.0

            # Genre diversity proxy: time span of ratings
            time_span = (window["timestamp"].max() -
                         window["timestamp"].min()).total_seconds()
            time_span_norm = min(time_span / 86400, 1.0)  # cap at 1 day

            # Recency: how recent is this window (0=old, 1=recent)
            total_span = (group["timestamp"].max() -
                          group["timestamp"].min()).total_seconds()
            if total_span > 0:
                window_pos = (window["timestamp"].mean() -
                              group["timestamp"].min()).total_seconds()
                recency = window_pos / total_span
            else:
                recency = 0.5

            features = [
                avg_rating / 5.0,           # normalize to 0-1
                min(rating_std / 2.0, 1.0), # normalize
                num_ratings / window_size,  # always 1.0 here
                (rating_trend + 4) / 8.0,   # normalize -4 to +4 → 0-1
                time_span_norm,
                recency,
            ]

            # ── Label engineering (VAD proxies) ───────────
            # Valence: driven by average rating
            valence = avg_rating / 5.0

            # Arousal: driven by rating variability + speed
            arousal = min(
                (rating_std / 2.0) * 0.5 +
                (1.0 - time_span_norm) * 0.5,
                1.0
            )

            # Dominance: driven by trend + recency
            dominance = (
                (rating_trend + 4) / 8.0 * 0.5 +
                recency * 0.5
            )

            X.append(features)
            y.append([valence, arousal, dominance])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Training samples: {len(X)}")
    print(f"Feature shape:    {X.shape}")
    print(f"Label shape:      {y.shape}")
    print(f"Label ranges:")
    print(f"  Valence:   {y[:,0].min():.3f} - {y[:,0].max():.3f}")
    print(f"  Arousal:   {y[:,1].min():.3f} - {y[:,1].max():.3f}")
    print(f"  Dominance: {y[:,2].min():.3f} - {y[:,2].max():.3f}")

    return X, y


def train(epochs=50, batch_size=256, lr=0.001):
    """Train the EmotionNet on engineered features."""

    # Build data
    X, y = build_training_data()

    # Train/val split (80/20)
    split     = int(len(X) * 0.8)
    X_train   = torch.tensor(X[:split])
    y_train   = torch.tensor(y[:split])
    X_val     = torch.tensor(X[split:])
    y_val     = torch.tensor(y[split:])

    # Model, loss, optimizer
    model     = EmotionNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    print(f"\nTraining EmotionNet...")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Epochs:        {epochs}")
    print(f"Batch size:    {batch_size}")
    print()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────
        model.train()
        train_losses = []
        indices      = torch.randperm(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step()

        avg_train = np.mean(train_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            saved = "✓ saved"
        else:
            saved = ""

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train:.4f} | "
                  f"Val Loss: {val_loss:.4f} {saved}")

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    # ── Final evaluation ──────────────────────────────────
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        final_pred = model(X_val)
        mae = torch.mean(torch.abs(final_pred - y_val)).item()
        print(f"Final MAE: {mae:.4f}")

    # Sample predictions
    print(f"\nSample predictions vs actual:")
    print(f"{'':20} {'V_pred':>8} {'V_true':>8} | "
          f"{'A_pred':>8} {'A_true':>8} | "
          f"{'D_pred':>8} {'D_true':>8}")
    for i in range(5):
        p = final_pred[i].numpy()
        t = y_val[i].numpy()
        print(f"Sample {i+1:14} "
              f"{p[0]:8.3f} {t[0]:8.3f} | "
              f"{p[1]:8.3f} {t[1]:8.3f} | "
              f"{p[2]:8.3f} {t[2]:8.3f}")


if __name__ == "__main__":
    train(epochs=50, batch_size=256, lr=0.001)