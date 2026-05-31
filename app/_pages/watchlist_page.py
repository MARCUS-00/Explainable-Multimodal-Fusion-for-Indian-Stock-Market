import os

import pandas as pd
import streamlit as st

from prediction.watchlist import generate_watchlist
from prediction.adaptive_gate import get_current_threshold, explain_threshold
from app.components.output_card import render_card
from config.settings import MERGED_CSV, WATCHLIST_MIN_CONFIDENCE


@st.cache_data(ttl=3600)
def _load_merged():
    if not os.path.exists(MERGED_CSV):
        return None
    return pd.read_csv(MERGED_CSV, parse_dates=["Date"])


def _show_regime_info():
    """Display the adaptive regime badge. Non-critical - silently skipped on error."""
    try:
        df_pre = _load_merged()
        if df_pre is None:
            return
        threshold, regime_score, regime_label = get_current_threshold(df_pre)
        explanation = explain_threshold(threshold, regime_score, regime_label)
        badges = {"BULL": "🟢", "BEAR": "🔴", "NEUTRAL": "🟡"}
        badge = badges.get(regime_label, "⚪")
        first_line = explanation.splitlines()[0]
        rest_lines = "\n".join(explanation.splitlines()[1:])
        st.info(badge + " **" + first_line + "**\n\n" + rest_lines)
    except Exception:
        pass


def render_watchlist_page():
    st.header("Daily Watchlist")
    _show_regime_info()
    # Adaptive gate toggle: when enabled, the app derives a dynamic threshold
    adaptive = st.checkbox("Adaptive gate (use regime-aware threshold)", value=True)
    # Slider lower bound set to the configured meaningful confidence threshold.
    # Disable manual slider when adaptive gate is active.
    min_conf = st.slider("Min confidence", WATCHLIST_MIN_CONFIDENCE, 0.75, WATCHLIST_MIN_CONFIDENCE, disabled=adaptive)
    dir_choice = st.radio(
        "Direction",
        ["UP only", "DOWN only", "Both"],
        index=0,
        horizontal=True,
        help="UP only = long-only watchlist (default). DOWN only = short signals. Both = all directions.",
    )
    _dir_map = {"UP only": "up", "DOWN only": "down", "Both": "both"}
    if st.button("Generate"):
        with st.spinner("Running predictions ..."):
            df = _load_merged()
            if df is None:
                st.error("merged_final.csv not found. Run: python features/build_features.py")
                return
            try:
                wl = generate_watchlist(
                    df,
                    min_confidence=None if adaptive else min_conf,
                    direction=_dir_map[dir_choice],
                )
            except Exception as e:
                st.error(f"Error: {e}")
                return
            if wl.empty:
                st.error("No predictions. Train models first.")
                return

            st.success(f"Predictions for {len(wl)} stocks")
            cols = ["Stock", "Prediction", "Expected_Movement", "Confidence",
                    "Confidence_Level", "Recommendation", "Last_Close"]
            visible = [c for c in cols if c in wl.columns]
            st.dataframe(wl[visible], use_container_width=True)

            st.subheader("Cards")
            for _, row in wl.iterrows():
                xai_raw = str(row.get("XAI_Factors", ""))
                xai = [b.strip() for b in xai_raw.split("|") if b.strip()]
                render_card({**row.to_dict(), "XAI_Factors": xai})
    else:
        st.info("Click Generate to run predictions.")
