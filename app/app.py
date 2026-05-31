import os
import sys

# Streamlit runs scripts with the script's directory on sys.path, not the
# project root. This ensures all project packages (app, config, models, …)
# are importable regardless of how / where Streamlit is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="Stock Prediction", page_icon=":chart_with_upwards_trend:",
                   layout="wide")

st.markdown(
    """
    <style>
    html, body, .stApp {
        font-size: 18px;
    }
    .stApp {
        max-width: 1400px;
        padding-top: 2rem;
        margin: 0 auto;
    }
    .main {
        max-width: 1400px;
        padding-top: 2rem;
    }
    h1 { font-size: 2.6rem; font-weight: 700; margin-bottom: 0.75rem; }
    h2 { font-size: 2.0rem; font-weight: 700; margin-bottom: 0.75rem; }
    h3 { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.75rem; }
    [data-testid="stSidebar"] {
        width: 280px;
        padding: 1rem;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem;
    }
    .stButton button {
        padding: 0.6rem 1.6rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        min-width: 140px;
    }
    .stDataFrame tbody tr { min-height: 44px; }
    .stDataFrame thead th { font-weight: 600; }
    .stDataFrame tbody td { font-size: 0.95rem; }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
    }
    .stInfo, .stSuccess, .stError {
        padding: 1rem 1.25rem;
        font-size: 1rem;
        border-radius: 8px;
    }
    .stContainer {
        background: var(--secondary-background-color);
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from app._pages.watchlist_page import render_watchlist_page
from app._pages.single_stock_page import render_single_stock_page

def main() -> None:
    st.title("Stock Prediction System")
    st.caption("Nifty 50 - XGBoost + LSTM Ensemble (AUC-weighted) - Explainable AI")

    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("Go to", ["Watchlist", "Single Stock"], index=0)

    if page == "Watchlist":
        render_watchlist_page()
    else:
        render_single_stock_page()


if __name__ == "__main__":
    main()