import os

import pandas as pd
import streamlit as st

from prediction.single_stock import predict_single
from app.components.output_card import render_card
from config.settings             import MERGED_CSV, STOCKS


@st.cache_data(ttl=3600)
def _load_merged():
    if not os.path.exists(MERGED_CSV):
        return None
    return pd.read_csv(MERGED_CSV, parse_dates=["Date"])


def render_single_stock_page():
    st.header("Single Stock Prediction")
    symbol = st.selectbox("Select stock:", sorted(STOCKS), key="single_stock_select")
    if st.button("Predict", key="single_stock_predict_button"):
        with st.spinner(f"Predicting {symbol} ..."):
            df = _load_merged()
            if df is None:
                st.error("merged_final.csv not found. "
                         "Run: python features/build_features.py")
                return
            try:
                result = predict_single(symbol, df)
            except Exception as e:
                st.error(f"Error: {e}")
                return
            if "error" in result:
                st.error(result["error"])
                return
            render_card(result)
