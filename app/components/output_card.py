import streamlit as st


def render_card(result: dict) -> None:
    """Render the prediction card with stock, direction, confidence, and XAI."""
    stock = result.get("Stock", "")
    prediction = result.get("Prediction", "")
    expected = result.get("Expected_Movement", "")
    confidence = result.get("Confidence", "")
    recommendation = result.get("Recommendation", "")
    last_close = result.get("Last_Close", "")
    last_date = result.get("Last_Date", "")

    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            direction_icon = "🟢" if prediction == "UP" else "🔴"
            st.markdown(f"## {stock} {direction_icon} {prediction}")
            st.markdown(
                f"<div style='font-size:1.4rem;font-weight:700;color:#22c55e'>Expected movement: {expected}</div>",
                unsafe_allow_html=True,
            )
            rec_color = "#16a34a" if str(recommendation).upper().startswith("BUY") else "#dc2626"
            st.markdown(
                f"<div style='font-weight:700;color:{rec_color}'>Recommendation: {recommendation}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Close:** {last_close}")
            st.markdown(f"**Date:** {last_date}")
        with right:
            try:
                if isinstance(confidence, str):
                    conf_num = float(confidence.rstrip("%").strip())
                    conf_pct = conf_num if "%" in confidence else conf_num * 100.0
                else:
                    conf_pct = float(confidence) * 100.0
                if conf_pct >= 70:
                    conf_color = "#16a34a"
                elif conf_pct >= 60:
                    conf_color = "#d97706"
                else:
                    conf_color = "#dc2626"
                st.markdown(
                    "<div style='text-align:right;font-size:0.9rem;opacity:0.7'>Confidence</div>"
                    f"<div style='text-align:right;font-size:2.4rem;font-weight:700;color:{conf_color}'>{conf_pct:.1f}%</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(
                    "<div style='text-align:right;font-size:0.9rem;opacity:0.7'>Confidence</div>"
                    f"<div style='text-align:right;font-size:2.4rem;font-weight:700;color:#16a34a'>{confidence}</div>",
                    unsafe_allow_html=True,
                )

    factors = result.get("XAI_Factors", []) or []
    if factors:
        with st.expander("📊 Key Factors (XAI)"):
            for f in factors:
                st.markdown(f"- {f}")
