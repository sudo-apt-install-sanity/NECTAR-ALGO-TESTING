import streamlit as st
import numpy as np
import sys
sys.path.append(".")

from pipeline import run_pipeline
from modules.recommender import apply_feedback
from modules.omdb_client import enrich_recommendations
from modules.styles import NETFLIX_CSS, mood_color, mood_label
from modules.analytics import (
    get_mood_distribution, get_rating_distribution,
    get_session_activity, get_attention_trends,
    get_platform_device_breakdown, get_all_users_summary
)
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

# Inject CSS
st.markdown(NETFLIX_CSS, unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────
if "view" not in st.session_state:
    st.session_state["view"] = "user"

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 NECTAR")

    # View toggle
    st.markdown("---")
    view_col1, view_col2 = st.columns(2)
    with view_col1:
        if st.button("🎬 User View",
                     use_container_width=True,
                     type="primary" if st.session_state["view"] == "user" else "secondary"):
            st.session_state["view"] = "user"
            st.rerun()
    with view_col2:
        if st.button("🔬 Research",
                     use_container_width=True,
                     type="primary" if st.session_state["view"] == "research" else "secondary"):
            st.session_state["view"] = "research"
            st.rerun()

    st.markdown("---")

    # User selection
    st.markdown("**👤 User**")
    existing_users = get_all_users()
    user_options   = existing_users + ["+ Add new user"]
    selected       = st.selectbox("Select user", user_options,
                                  label_visibility="collapsed")

    if selected == "+ Add new user":
        new_username = st.text_input("Enter username")
        if st.button("Create user") and new_username.strip():
            get_or_create_user(new_username.strip())
            st.success(f"User '{new_username}' created")
            st.rerun()
        st.stop()

    username = selected
    user_id  = get_or_create_user(username)
    st.success(f"**{username}**")
    st.markdown("---")

    # Context inputs
    st.markdown("**⚙️ Context**")
    device    = st.selectbox("Device",   ["mobile","tablet","laptop","smart_tv"])
    platform  = st.selectbox("Platform", ["netflix","youtube","amazon","disney"])
    time_slot = st.selectbox("Time",     ["morning","afternoon","evening","night"])
    day       = st.selectbox("Day type", ["weekday","weekend"])
    session_dur = st.slider("Session duration (mins)", 5, 300, 45)

    st.markdown("**🧠 Behavior**")
    skip_rate       = st.slider("Skip rate",       0.0, 1.0, 0.3)
    completion_rate = st.slider("Completion rate", 0.0, 1.0, 0.7)
    rewatch_ratio   = st.slider("Rewatch ratio",   0.0, 1.0, 0.1)
    dwell_time      = st.slider("Dwell time (secs)", 1.0, 30.0, 10.0)
    review_text     = st.text_input("Last review",
                                    placeholder="e.g. really enjoyed it")
    top_n           = st.slider("Recommendations", 5, 20, 10)
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

# ═══════════════════════════════════════════════════════════════
# USER VIEW — Netflix-style
# ═══════════════════════════════════════════════════════════════
if st.session_state["view"] == "user":

    if "result" not in st.session_state:
        # Landing state
        st.markdown("""
        <div class="hero-banner">
            <p class="hero-title">🎬 NECTAR</p>
            <p class="hero-subtitle">
                Your emotionally intelligent movie companion.
                Set your context and mood in the sidebar, then click
                <strong>Get Recommendations</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header">How it works</div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**1. Set Context**")
            st.caption("Tell NECTAR what device you're on, what time it is, and how long you've been watching.")
        with c2:
            st.markdown("**2. Signal Behavior**")
            st.caption("Adjust the sliders to reflect how engaged you've been — skip rate, completion, rewatch.")
        with c3:
            st.markdown("**3. Get Recommendations**")
            st.caption("NECTAR infers your emotional state and finds movies that match who you are right now.")
        with c4:
            st.markdown("**4. Give Feedback**")
            st.caption("Rate movies to help NECTAR learn your preferences over time.")

    else:
        result = st.session_state["result"]
        recs   = st.session_state["recs"]
        mood   = result["mood_key"]

        # Enrich top 5
        if not st.session_state.get("enriched_recs") or \
           st.session_state.get("last_session_id") != st.session_state.get("session_id"):
            with st.spinner("Fetching movie details..."):
                st.session_state["enriched_recs"] = enrich_recommendations(recs[:5])
                st.session_state["last_session_id"] = st.session_state.get("session_id")

        enriched  = st.session_state["enriched_recs"]
        remaining = recs[5:]

        # Hero banner
        mc = mood_color(mood)
        ml = mood_label(mood)
        st.markdown(f"""
        <div class="hero-banner">
            <p class="hero-title">Good to see you, {username}</p>
            <p class="hero-subtitle">
                Here's what NECTAR thinks you'll enjoy right now.
            </p>
            <span class="hero-mood">
                <span class="hero-mood-dot" style="background:{mc}"></span>
                Detected mood: <strong>{ml}</strong>
                &nbsp;·&nbsp; Preferred: {', '.join(result['preferred_genres'])}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Top picks grid
        st.markdown('<div class="section-header">🎯 Top Picks For You</div>',
                    unsafe_allow_html=True)

        cols = st.columns(5)
        for i, rec in enumerate(enriched):
            with cols[i]:
                # Poster
                if rec.get("poster") and rec["poster"] != "N/A":
                    st.image(rec["poster"], use_container_width=True)
                else:
                    st.markdown("""
                    <div class="movie-card-poster-placeholder">🎬</div>
                    """, unsafe_allow_html=True)

                # Title and genres
                st.markdown(f"**{rec['title']}**")
                st.caption(", ".join(rec["genres"]))

                # Badges
                badges = ""
                if rec["mood_match"]:
                    badges += '<span class="badge badge-mood">🎯 Mood</span> '
                if rec.get("rating") and rec["rating"] != "N/A":
                    badges += f'<span class="badge badge-imdb">⭐ {rec["rating"]}</span>'
                if badges:
                    st.markdown(badges, unsafe_allow_html=True)

                # Plot
                if rec.get("plot") and rec["plot"] != "No description available.":
                    st.caption(rec["plot"][:90] + "...")

                # Star rating
                stars = st.feedback("stars", key=f"u_stars_{i}")
                if stars is not None:
                    star_val  = stars + 1
                    rated_key = f"rated_{i}_{st.session_state.get('session_id')}"
                    if not st.session_state.get(rated_key):
                        if st.session_state.get("session_id"):
                            save_feedback(
                                st.session_state["session_id"],
                                rec["title"], star_val
                            )
                        st.session_state[rated_key] = True
                    updated = apply_feedback(
                        result["movies"], recs, rec["title"], star_val
                    )
                    st.session_state["recs"] = updated
                    st.rerun()

        # More recommendations
        if remaining:
            st.markdown('<div class="section-header">More You Might Like</div>',
                        unsafe_allow_html=True)
            rem_cols = st.columns(min(len(remaining), 5))
            for i, rec in enumerate(remaining[:5]):
                with rem_cols[i]:
                    st.markdown(f"**{rec['title']}**")
                    st.caption(", ".join(rec["genres"]))
                    if rec["mood_match"]:
                        st.markdown('<span class="badge badge-mood">🎯 Mood</span>',
                                    unsafe_allow_html=True)
                    st.metric("Score", rec["score"])
                    stars = st.feedback("stars", key=f"u_rem_stars_{i}")
                    if stars is not None:
                        star_val  = stars + 1
                        rated_key = f"rated_rem_{i}_{st.session_state.get('session_id')}"
                        if not st.session_state.get(rated_key):
                            if st.session_state.get("session_id"):
                                save_feedback(
                                    st.session_state["session_id"],
                                    rec["title"], star_val
                                )
                            st.session_state[rated_key] = True
                        updated = apply_feedback(
                            result["movies"], recs, rec["title"], star_val
                        )
                        st.session_state["recs"] = updated
                        st.rerun()

# ═══════════════════════════════════════════════════════════════
# RESEARCH VIEW — Pipeline details
# ═══════════════════════════════════════════════════════════════
else:
    st.title("🔬 NECTAR — Research View")
    st.caption("Full pipeline outputs, vectors, and analytics")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Pipeline Output",
                                       "🕓 Session History",
                                       "⭐ My Feedback",
                                       "📈 Analytics"])

    # ── Tab 1 ─────────────────────────────────────────────────
    with tab1:
        if "result" not in st.session_state:
            st.info("Run the pipeline from the sidebar to see outputs.")
        else:
            result = st.session_state["result"]
            recs   = st.session_state["recs"]

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
                inference = mood.get("inference", "Rule-based")
                if "Neural" in inference:
                    st.success(f"🧠 {inference}")
                else:
                    st.warning(f"📐 {inference}")
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
            st.header("Stage 5 — Recommendations")
            st.caption(
                f"Mood: **{result['mood_key'].replace('_',' ')}** "
                f"| Preferred genres: "
                f"{', '.join(result['preferred_genres'])}"
            )

            # Enrich
            if not st.session_state.get("enriched_recs") or \
               st.session_state.get("last_session_id") != \
               st.session_state.get("session_id"):
                with st.spinner("Fetching movie details..."):
                    st.session_state["enriched_recs"] = \
                        enrich_recommendations(recs[:5])
                    st.session_state["last_session_id"] = \
                        st.session_state.get("session_id")

            enriched  = st.session_state["enriched_recs"]
            remaining = recs[5:]

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
                        if rec.get("plot") and \
                           rec["plot"] != "No description available.":
                            st.write(rec["plot"][:120] + "...")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if rec.get("rating") and rec["rating"] != "N/A":
                                st.caption(f"⭐ IMDB: {rec['rating']}")
                        with col_b:
                            if rec.get("director") and \
                               rec["director"] != "N/A":
                                st.caption(f"🎬 {rec['director']}")
                        with col_c:
                            if rec.get("runtime") and \
                               rec["runtime"] != "N/A":
                                st.caption(f"⏱ {rec['runtime']}")
                    with c3:
                        st.metric("Score",      rec["score"])
                        st.metric("Similarity", rec["similarity"])
                        stars = st.feedback("stars", key=f"r_stars_{i}")
                        if stars is not None:
                            star_val  = stars + 1
                            rated_key = f"r_rated_{i}_" \
                                        f"{st.session_state.get('session_id')}"
                            if not st.session_state.get(rated_key):
                                if st.session_state.get("session_id"):
                                    save_feedback(
                                        st.session_state["session_id"],
                                        rec["title"], star_val
                                    )
                                st.session_state[rated_key] = True
                            updated = apply_feedback(
                                result["movies"], recs,
                                rec["title"], star_val
                            )
                            st.session_state["recs"] = updated
                            st.rerun()

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
                            stars = st.feedback("stars", key=f"r_rem_{i}")
                            if stars is not None:
                                star_val  = stars + 1
                                rated_key = f"r_rem_{i}_" \
                                            f"{st.session_state.get('session_id')}"
                                if not st.session_state.get(rated_key):
                                    if st.session_state.get("session_id"):
                                        save_feedback(
                                            st.session_state["session_id"],
                                            rec["title"], star_val
                                        )
                                    st.session_state[rated_key] = True
                                updated = apply_feedback(
                                    result["movies"], recs,
                                    rec["title"], star_val
                                )
                                st.session_state["recs"] = updated
                                st.rerun()

    # ── Tab 2 ─────────────────────────────────────────────────
    with tab2:
        st.header(f"Session History — {username}")
        sessions = get_user_sessions(user_id, limit=10)
        if not sessions:
            st.info("No sessions yet.")
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

    # ── Tab 3 ─────────────────────────────────────────────────
    with tab3:
        st.header(f"Feedback History — {username}")
        feedback = get_user_feedback(user_id)
        if not feedback:
            st.info("No feedback yet.")
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

    # ── Tab 4 ─────────────────────────────────────────────────
    with tab4:
        st.header(f"Analytics — {username}")
        sessions = get_session_activity(user_id)

        if not sessions:
            st.info("No data yet. Run the pipeline a few times.")
        else:
            total_sessions = len(sessions)
            rating_dist    = get_rating_distribution(user_id)
            total_ratings  = sum(rating_dist.values())
            avg_rating     = (
                sum(k * v for k, v in rating_dist.items()) / total_ratings
                if total_ratings > 0 else 0
            )
            mood_dist = get_mood_distribution(user_id)
            top_mood  = max(mood_dist, key=mood_dist.get) if mood_dist else "N/A"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Sessions",  total_sessions)
            m2.metric("Total Ratings",   total_ratings)
            m3.metric("Avg Star Rating", round(avg_rating, 2))
            m4.metric("Dominant Mood",
                      top_mood.replace("_", " ") if top_mood != "N/A" else "N/A")

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Mood Distribution")
                if mood_dist:
                    for k, v in mood_dist.items():
                        pct = v / sum(mood_dist.values())
                        st.write(f"**{k.replace('_',' ')}**")
                        st.progress(pct,
                                    text=f"{v} sessions ({pct*100:.0f}%)")

            with col2:
                st.subheader("Star Rating Distribution")
                if rating_dist:
                    for star in range(1, 6):
                        count = rating_dist.get(star, 0)
                        total = sum(rating_dist.values())
                        pct   = count / total if total > 0 else 0
                        st.write(f"{'⭐' * star}")
                        st.progress(pct,
                                    text=f"{count} ({pct*100:.0f}%)")

            st.divider()
            st.subheader("Emotion Trends Over Sessions")
            trends = get_attention_trends(user_id)
            if len(trends) > 1:
                import pandas as pd
                df_trends = pd.DataFrame(trends)
                df_trends["session"] = range(1, len(df_trends) + 1)
                st.line_chart(
                    df_trends.set_index("session")[
                        ["valence", "arousal", "dominance"]
                    ]
                )
                st.caption("Valence = mood positivity | "
                           "Arousal = engagement | Dominance = control")
            else:
                st.info("Need at least 2 sessions to show trends.")

            st.divider()
            st.subheader("Device & Platform Usage")
            breakdown = get_platform_device_breakdown(user_id)
            if breakdown:
                for b in breakdown:
                    st.write(f"**{b['device']}** on **{b['platform']}** — "
                             f"{b['count']} session"
                             f"{'s' if b['count'] > 1 else ''}")

            st.divider()
            st.subheader("All Users Summary")
            summary = get_all_users_summary()
            if summary:
                import pandas as pd
                st.dataframe(pd.DataFrame(summary),
                             use_container_width=True)