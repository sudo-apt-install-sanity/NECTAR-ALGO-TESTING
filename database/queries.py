import json
import numpy as np
from database.db import get_connection

# ── User functions ────────────────────────────────────────────

def get_all_users():
    """Return list of all usernames."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users ORDER BY username")
    rows = cursor.fetchall()
    conn.close()
    return [row["username"] for row in rows]

def get_or_create_user(username):
    """Get user_id for username, creating the user if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if row:
        user_id = row["user_id"]
    else:
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        user_id = cursor.lastrowid
    conn.close()
    return user_id

# ── Session functions ─────────────────────────────────────────

def save_session(user_id, inputs, mood_detected, csv_vector, asv_vector):
    """Save a pipeline run as a session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions (
            user_id, device, platform, time_slot, day,
            session_duration, skip_rate, completion_rate,
            rewatch_ratio, dwell_time, review_text,
            mood_detected, csv_vector, asv_vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        inputs["device"],
        inputs["platform"],
        inputs["time_slot"],
        inputs["day"],
        inputs["session_duration"],
        inputs["skip_rate"],
        inputs["completion_rate"],
        inputs["rewatch_ratio"],
        inputs["dwell_time"],
        inputs.get("review_text", ""),
        mood_detected,
        json.dumps(csv_vector.tolist()),
        json.dumps(asv_vector.tolist()),
    ))
    conn.commit()
    session_id = cursor.lastrowid
    conn.close()
    return session_id

def get_user_sessions(user_id, limit=10):
    """Get recent sessions for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM sessions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_session_history_vectors(user_id, limit=5):
    """
    Get past session vectors for temporal modeling (TDM).
    Returns a list of numpy arrays — one per past session.
    """
    sessions = get_user_sessions(user_id, limit=limit)
    history = []
    for s in sessions:
        try:
            csv = np.array(json.loads(s["csv_vector"]))
            asv = np.array(json.loads(s["asv_vector"]))
            combined = np.concatenate([csv, asv])
            history.append(combined)
        except Exception:
            continue
    return history

# ── Feedback functions ────────────────────────────────────────

def save_feedback(session_id, movie_title, star_rating):
    """Save a star rating for a movie in a session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (session_id, movie_title, star_rating)
        VALUES (?, ?, ?)
    """, (session_id, movie_title, star_rating))
    conn.commit()
    conn.close()

def get_user_feedback(user_id):
    """Get all feedback a user has given across all sessions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.movie_title, f.star_rating, f.created_at
        FROM feedback f
        JOIN sessions s ON f.session_id = s.session_id
        WHERE s.user_id = ?
        ORDER BY f.created_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]