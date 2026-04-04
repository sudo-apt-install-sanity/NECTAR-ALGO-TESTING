import sqlite3
import json
import numpy as np
from database.db import get_connection

def get_mood_distribution(user_id):
    """Count how many sessions had each mood for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT mood_detected, COUNT(*) as count
        FROM sessions
        WHERE user_id = ?
        GROUP BY mood_detected
        ORDER BY count DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return {row["mood_detected"]: row["count"] for row in rows}

def get_genre_frequency(user_id):
    """
    Count how often each genre appeared in recommendations
    by looking at feedback history.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.movie_title, f.star_rating
        FROM feedback f
        JOIN sessions s ON f.session_id = s.session_id
        WHERE s.user_id = ?
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_rating_distribution(user_id):
    """Count how many ratings of each star value a user gave."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT f.star_rating, COUNT(*) as count
        FROM feedback f
        JOIN sessions s ON f.session_id = s.session_id
        WHERE s.user_id = ?
        GROUP BY f.star_rating
        ORDER BY f.star_rating
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return {row["star_rating"]: row["count"] for row in rows}

def get_session_activity(user_id):
    """Get session timestamps and behavior signals over time."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            created_at,
            skip_rate,
            completion_rate,
            rewatch_ratio,
            mood_detected,
            device,
            platform
        FROM sessions
        WHERE user_id = ?
        ORDER BY created_at ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_attention_trends(user_id):
    """
    Get average attention weights per session by
    reading the ASV vectors stored in sessions.
    Returns valence and arousal trends over time.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT created_at, asv_vector
        FROM sessions
        WHERE user_id = ?
        ORDER BY created_at ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()

    trends = []
    for row in rows:
        try:
            asv = json.loads(row["asv_vector"])
            trends.append({
                "created_at": row["created_at"],
                "valence":    round(asv[0], 3),
                "arousal":    round(asv[1], 3),
                "dominance":  round(asv[2], 3),
            })
        except Exception:
            continue
    return trends

def get_platform_device_breakdown(user_id):
    """Count sessions per device and platform."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT device, platform, COUNT(*) as count
        FROM sessions
        WHERE user_id = ?
        GROUP BY device, platform
        ORDER BY count DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_users_summary():
    """Summary stats across all users for admin view."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            u.username,
            COUNT(DISTINCT s.session_id) as total_sessions,
            COUNT(DISTINCT f.feedback_id) as total_ratings,
            MAX(s.created_at) as last_active
        FROM users u
        LEFT JOIN sessions s ON u.user_id = s.user_id
        LEFT JOIN feedback f ON s.session_id = f.session_id
        GROUP BY u.user_id
        ORDER BY total_sessions DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]