import streamlit as st
import numpy as np
import sys
sys.path.append(".")

from pipeline import run_pipeline
from modules.recommender import apply_feedback

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NECTAR",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 NECTAR")
st.caption("Neuro-Emotion and Context-aware Temporal Adaptive Recommender")
st.divider()

# ── Sidebar: User Context Inputs ─────────────────────────────
with st.sidebar:
    st.header("⚙️ User Context")
    st.subheader("Stage 1 — Situational Context")
    device       = st.selectbox("Device",    ["mobile","tablet","laptop","smart_tv"])
    platform     = st.selectbox("Platform",  ["netflix","youtube","amazon","disney"])
    time_slot    = st.selectbox("Time",      ["morning","afternoon","evening","night"])
    day          = st.selectbox("Day type",  ["weekday","weekend"])
    session_dur  = st.slider("Session duration (mins)", 5, 300, 45)

    st.divider()
    st.subheader("Stage 2 — Behavior Signals")
    skip_rate       = st.slider("Skip rate",        0.0, 1.0, 0.3)
    completion_rate = st.slider("Completion rate",  0.0, 1.0, 0.7)
    rewatch_ratio   = st.slider("Rewatch ratio",    0.0, 1.0, 0.1)
    dwell_time      = st.slider("Dwell time (secs)",1.0, 30.0, 10.0)
    review_text     = st.text_input("Last review (optional)",
                                    placeholder="e.g. really enjoyed it")
    top_n           = st.slider("Number of recommendations", 5, 20, 10)
    run_btn         = st.button("🚀 Get Recommendations", use_container_width=True)

# ── Run pipeline ─────────────────────────────────────────────
if run_btn:
    with st.spinner("Running NECTAR pipeline..."):
        result = run_pipeline(
            device=device, platform=platform,
            time_slot=time_slot, day=day,
            session_duration=session_dur,
            skip_rate=skip_rate, completion_rate=completion_rate,
            rewatch_ratio=rewatch_ratio, dwell_time=dwell_time,
            review_text=review_text, top_n=top_n
        )
    st.session_state["result"] = result
    st.session_state["recs"]   = result["recommendations"]

# ── Display results ───────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    recs   = st.session_state["recs"]

    # ── Stage outputs ─────────────────────────────────────────
    st.header("Pipeline Stage Outputs")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Stage 1 — CSV")
        st.caption("Contextual State Vector")
        desc = result["csv_description"]
        for signal in desc:
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
        st.progress(aw["context"],  text=f"Context:  {aw['context']}")
        st.progress(aw["emotion"],  text=f"Emotion:  {aw['emotion']}")
        st.progress(aw["temporal"], text=f"Temporal: {aw['temporal']}")
        st.write("**Gates:**")
        st.write(f"Context gate:  {'🟢 Open' if aw['gate_context']  else '🔴 Closed'}")
        st.write(f"Emotion gate:  {'🟢 Open' if aw['gate_emotion']  else '🔴 Closed'}")
        st.write(f"Temporal gate: {'🟢 Open' if aw['gate_temporal'] else '🔴 Closed'}")

    st.divider()

    # ── Recommendations ───────────────────────────────────────
    st.header("Stage 5 — Recommendations")
    st.caption(
        f"Mood detected: **{result['mood_key'].replace('_',' ')}** "
        f"| Preferred genres: {', '.join(result['preferred_genres'])}"
    )

    for i, rec in enumerate(recs):
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 2, 2])
            with c1:
                mood_tag = "🎯 Mood match" if rec["mood_match"] else ""
                st.markdown(f"**{i+1}. {rec['title']}** {mood_tag}")
                st.caption(", ".join(rec["genres"]))
            with c2:
                st.metric("Score",      rec["score"])
                st.metric("Similarity", rec["similarity"])
            with c3:
                stars = st.feedback("stars", key=f"stars_{i}")
                if stars is not None:
                    star_val = stars + 1  # streamlit returns 0-4
                    updated = apply_feedback(
                        result["movies"], recs,
                        rec["title"], star_val
                    )
                    st.session_state["recs"] = updated
                    st.success(f"Feedback recorded ({star_val}★)")
                    st.rerun()