import streamlit as st
import numpy as np
import sys
sys.path.append(".")

from pipeline import run_pipeline
from modules.recommender import apply_feedback
from modules.omdb_client import enrich_recommendations
from database.db import initialize_db
from database.queries import (
    get_all_users, get_or_create_user,
    get_user_sessions, get_user_feedback,
    save_feedback
)

# Initialize DB on startup
initialize_db()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NECTAR",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 NECTAR")
st.caption("Neuro-Emotion and Context-aware Temporal Adaptive Recommender")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:

    # ── User selection ────────────────────────────────────────
    st.header("👤 User")
    existing_users = get_all_users()
    user_options   = existing_users + ["+ Add new user"]
    selected       = st.selectbox("Select user", user_options)

    if selected == "+ Add new user":
        new_username = st.text_input("Enter username")
        if st.button("Create user") and new_username.strip():
            get_or_create_user(new_username.strip())
            st.success(f"User '{new_username}' created")
            st.rerun()
        st.stop()

    username = selected
    user_id  = get_or_create_user(username)
    st.success(f"Logged in as: **{username}**")
    st.divider()

    # ── Stage 1 inputs ────────────────────────────────────────
    st.subheader("Stage 1 — Situational Context")
    device    = st.selectbox("Device",   ["mobile","tablet","laptop","smart_tv"])
    platform  = st.selectbox("Platform", ["netflix","youtube","amazon","disney"])
    time_slot = st.selectbox("Time",     ["morning","afternoon","evening","night"])
    day       = st.selectbox("Day type", ["weekday","weekend"])
    session_dur = st.slider("Session duration (mins)", 5, 300, 45)

    st.divider()

    # ── Stage 2 inputs ────────────────────────────────────────
    st.subheader("Stage 2 — Behavior Signals")
    skip_rate       = st.slider("Skip rate",       0.0, 1.0, 0.3)
    completion_rate = st.slider("Completion rate", 0.0, 1.0, 0.7)
    rewatch_ratio   = st.slider("Rewatch ratio",   0.0, 1.0, 0.1)
    dwell_time      = st.slider("Dwell time (secs)", 1.0, 30.0, 10.0)
    review_text     = st.text_input("Last review (optional)",
                                    placeholder="e.g. really enjoyed it")
    top_n           = st.slider("Number of recommendations", 5, 20, 10)
    run_btn         = st.button("🚀 Get Recommendations",
                                use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────
if run_btn:
    with st.spinner("Running NECTAR pipeline..."):
        result = run_pipeline(
            device=device, platform=platform,
            time_slot=time_slot, day=day,
            session_duration=session_dur,
            skip_rate=skip_rate, completion_rate=completion_rate,
            rewatch_ratio=rewatch_ratio, dwell_time=dwell_time,
            review_text=review_text, user_id=user_id, top_n=top_n
        )
    st.session_state["result"]          = result
    st.session_state["recs"]            = result["recommendations"]
    st.session_state["session_id"]      = result["session_id"]
    st.session_state["enriched_recs"]   = None
    st.session_state["last_session_id"] = None

# ── Main panel ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Pipeline Output",
                             "🕓 Session History",
                             "⭐ My Feedback"])

# ── Tab 1: Pipeline output ────────────────────────────────────
with tab1:
    if "result" not in st.session_state:
        st.info("Set your context and behavior signals in the sidebar, "
                "then click Get Recommendations.")
    else:
        result = st.session_state["result"]
        recs   = st.session_state["recs"]

        # Stage outputs
        st.header("Pipeline Stage Outputs")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Stage 1 — CSV")
            st.caption("Contextual State Vector")
            for signal in result["csv_description"]:
                st.success(f"✓ {signal}")
            with st.expander("Raw vector"):
                st.code(str(result["csv"].round(3)))

        with col2:
            st.subheader("Stage 2 — ASV")
            st.caption("Affective State Vector")
            mood = result["asv_description"]
            st.info(mood["mood_summary"])
            st.metric("Valence",   mood["valence"])
            st.metric("Arousal",   mood["arousal"])
            st.metric("Dominance", mood["dominance"])
            st.metric("Sentiment", mood["sentiment"])
            with st.expander("Raw vector"):
                st.code(str(result["asv"].round(3)))

        with col3:
            st.subheader("Stage 3 — DUR")
            st.caption("Dynamic User Representation")
            st.metric("Vector dimensions", result["dur_shape"][0])
            st.write("Combines context + emotion + temporal drift")
            st.write("Captures how user state evolves over sessions")

        with col4:
            st.subheader("Stage 4 — CUE")
            st.caption("Comprehensive User Embedding")
            st.metric("Vector dimensions", result["cue_shape"][0])
            aw = result["attention_weights"]
            st.write("**Attention weights:**")
            st.progress(aw["context"],
                        text=f"Context:  {aw['context']}")
            st.progress(aw["emotion"],
                        text=f"Emotion:  {aw['emotion']}")
            st.progress(aw["temporal"],
                        text=f"Temporal: {aw['temporal']}")
            st.write("**Gates:**")
            st.write(f"Context:  "
                     f"{'🟢 Open' if aw['gate_context']  else '🔴 Closed'}")
            st.write(f"Emotion:  "
                     f"{'🟢 Open' if aw['gate_emotion']  else '🔴 Closed'}")
            st.write(f"Temporal: "
                     f"{'🟢 Open' if aw['gate_temporal'] else '🔴 Closed'}")

        st.divider()

        # ── Recommendations ───────────────────────────────────
        st.header("Stage 5 — Recommendations")
        st.caption(
            f"Mood: **{result['mood_key'].replace('_',' ')}** "
            f"| Preferred genres: "
            f"{', '.join(result['preferred_genres'])}"
        )

        # Enrich top 5 with OMDB data (cached per session)
        if not st.session_state.get("enriched_recs") or \
           st.session_state.get("last_session_id") != st.session_state.get("session_id"):
            with st.spinner("Fetching movie details..."):
                st.session_state["enriched_recs"] = enrich_recommendations(recs[:5])
                st.session_state["last_session_id"] = st.session_state.get("session_id")

        enriched  = st.session_state["enriched_recs"]
        remaining = recs[5:]

        # Top 5 — rich cards with posters
        for i, rec in enumerate(enriched):
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 3, 2])
                with c1:
                    if rec.get("poster") and rec["poster"] != "N/A":
                        st.image(rec["poster"], width=100)
                    else:
                        st.write("🎬")
                with c2:
                    tag = "🎯 Mood match" if rec["mood_match"] else ""
                    st.markdown(f"**{i+1}. {rec['title']}** {tag}")
                    st.caption(", ".join(rec["genres"]))
                    if rec.get("plot") and rec["plot"] != "No description available.":
                        st.write(rec["plot"][:120] + "...")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if rec.get("rating") and rec["rating"] != "N/A":
                            st.caption(f"⭐ IMDB: {rec['rating']}")
                    with col_b:
                        if rec.get("director") and rec["director"] != "N/A":
                            st.caption(f"🎬 {rec['director']}")
                    with col_c:
                        if rec.get("runtime") and rec["runtime"] != "N/A":
                            st.caption(f"⏱ {rec['runtime']}")
                with c3:
                    st.metric("Score",      rec["score"])
                    st.metric("Similarity", rec["similarity"])
                    stars = st.feedback("stars", key=f"stars_{i}")
                    if stars is not None:
                        star_val  = stars + 1
                        rated_key = f"rated_{i}_{st.session_state.get('session_id')}"
                        if not st.session_state.get(rated_key):
                            if st.session_state.get("session_id"):
                                save_feedback(
                                    st.session_state["session_id"],
                                    rec["title"],
                                    star_val
                                )
                            st.session_state[rated_key] = True
                        updated = apply_feedback(
                            result["movies"], recs,
                            rec["title"], star_val
                        )
                        st.session_state["recs"] = updated
                        st.rerun()

        # Remaining movies — simple cards
        if remaining:
            st.subheader("More recommendations")
            for i, rec in enumerate(remaining, start=5):
                with st.container(border=True):
                    c1, c2, c3 = st.columns([4, 2, 2])
                    with c1:
                        tag = "🎯 Mood match" if rec["mood_match"] else ""
                        st.markdown(f"**{i+1}. {rec['title']}** {tag}")
                        st.caption(", ".join(rec["genres"]))
                    with c2:
                        st.metric("Score",      rec["score"])
                        st.metric("Similarity", rec["similarity"])
                    with c3:
                        stars = st.feedback("stars", key=f"stars_{i}")
                        if stars is not None:
                            star_val  = stars + 1
                            rated_key = f"rated_{i}_{st.session_state.get('session_id')}"
                            if not st.session_state.get(rated_key):
                                if st.session_state.get("session_id"):
                                    save_feedback(
                                        st.session_state["session_id"],
                                        rec["title"],
                                        star_val
                                    )
                                st.session_state[rated_key] = True
                            updated = apply_feedback(
                                result["movies"], recs,
                                rec["title"], star_val
                            )
                            st.session_state["recs"] = updated
                            st.rerun()

# ── Tab 2: Session history ────────────────────────────────────
with tab2:
    st.header(f"Session History — {username}")
    sessions = get_user_sessions(user_id, limit=10)

    if not sessions:
        st.info("No sessions yet. Run the pipeline to start building history.")
    else:
        st.caption(f"{len(sessions)} recent sessions")
        for s in sessions:
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**{s['created_at']}**")
                    st.write(f"Device: {s['device']} | "
                             f"Platform: {s['platform']}")
                with c2:
                    st.write(f"Time: {s['time_slot']} | Day: {s['day']}")
                    st.write(f"Skip: {s['skip_rate']} | "
                             f"Completion: {s['completion_rate']}")
                with c3:
                    st.write(f"Mood: **{s['mood_detected']}**")

# ── Tab 3: Feedback history ───────────────────────────────────
with tab3:
    st.header(f"Feedback History — {username}")
    feedback = get_user_feedback(user_id)

    if not feedback:
        st.info("No feedback yet. Rate some movies to see your history here.")
    else:
        st.caption(f"{len(feedback)} ratings given")
        for f in feedback:
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**{f['movie_title']}**")
                with c2:
                    st.write("⭐" * f["star_rating"])
                with c3:
                    st.write(f"{f['created_at']}")