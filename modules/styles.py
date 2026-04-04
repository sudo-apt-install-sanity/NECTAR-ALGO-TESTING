NETFLIX_CSS = """
<style>
/* ── Global ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #0a0a0a;
    color: #ffffff;
}

/* ── Hide Streamlit default elements in user view ─────── */
.user-view .stSidebar {
    background-color: #111111;
}

/* ── Hero Banner ──────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 32px;
    border: 1px solid #ffffff15;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1rem;
    color: #ffffff80;
    margin-top: 8px;
}

.hero-mood {
    display: inline-block;
    background: #ffffff15;
    border: 1px solid #ffffff25;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.85rem;
    color: #ffffffcc;
    margin-top: 16px;
}

.hero-mood-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4ade80;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Movie Cards ──────────────────────────────────────── */
.movie-card {
    background: #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #ffffff10;
    transition: transform 0.2s ease, border-color 0.2s ease;
    height: 100%;
}

.movie-card:hover {
    border-color: #ffffff30;
}

.movie-card-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}

.movie-card-poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
}

.movie-card-body {
    padding: 12px;
}

.movie-card-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 4px 0;
    line-height: 1.3;
}

.movie-card-genres {
    font-size: 0.75rem;
    color: #ffffff60;
    margin-bottom: 8px;
}

.movie-card-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 500;
}

.badge-mood {
    background: #4ade8020;
    color: #4ade80;
    border: 1px solid #4ade8040;
}

.badge-imdb {
    background: #f59e0b20;
    color: #f59e0b;
    border: 1px solid #f59e0b40;
}

.badge-score {
    background: #818cf820;
    color: #818cf8;
    border: 1px solid #818cf840;
}

/* ── Section Headers ──────────────────────────────────── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #ffffff;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #ffffff15;
}

/* ── View Toggle ──────────────────────────────────────── */
.view-toggle {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
}

/* ── Mood Badge Colors ────────────────────────────────── */
.mood-high_val_high_aro { color: #4ade80; }
.mood-high_val_low_aro  { color: #60a5fa; }
.mood-low_val_high_aro  { color: #f87171; }
.mood-low_val_low_aro   { color: #a78bfa; }
.mood-neutral           { color: #94a3b8; }

/* ── Plot text ────────────────────────────────────────── */
.movie-plot {
    font-size: 0.78rem;
    color: #ffffff90;
    line-height: 1.5;
    margin-top: 6px;
}

/* ── Scrollable row ───────────────────────────────────── */
.stHorizontalBlock {
    gap: 12px;
}
</style>
"""

MOOD_COLORS = {
    "high_val_high_aro": "#4ade80",
    "high_val_low_aro":  "#60a5fa",
    "low_val_high_aro":  "#f87171",
    "low_val_low_aro":   "#a78bfa",
    "neutral":           "#94a3b8",
}

MOOD_LABELS = {
    "high_val_high_aro": "Energetic & Positive",
    "high_val_low_aro":  "Calm & Content",
    "low_val_high_aro":  "Tense & Restless",
    "low_val_low_aro":   "Reflective & Low",
    "neutral":           "Neutral",
}

def mood_color(mood_key):
    return MOOD_COLORS.get(mood_key, "#94a3b8")

def mood_label(mood_key):
    return MOOD_LABELS.get(mood_key, "Unknown")