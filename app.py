# app.py (root)
import streamlit as st
import pandas as pd
from datetime import date, timedelta

# add src/ to import path (works on Streamlit Cloud + locally)
import os, sys
APP_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_prices
from analyzer import compute_metrics, make_pdf_report
from plotter import price_chart, spectrum_chart, signals_panel

st.set_page_config(page_title="SignalScope", layout="wide")
st.title("RehlySignal Analytics")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker (e.g., URA, CCJ, ^GSPC):", value="URA")
    years_back = st.slider("Years of history", 1, 10, 5)
    max_period_days = st.slider("Max FFT period (days)", 30, 400, 200)
    smooth = st.checkbox("Smooth FFT spectrum", value=True)
    run_button = st.button("Run / Refresh", type="primary")

@st.cache_data(show_spinner=False, ttl=3600)
def get_data(ticker: str, years_back: int) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    return load_prices(ticker, start.isoformat(), end.isoformat())

if run_button or "df" not in st.session_state:
    try:
        st.session_state["df"] = get_data(ticker, years_back)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

df = st.session_state.get("df")
if df is None or df.empty:
    st.warning("No data returned. Try a different ticker or shorter range.")
    st.stop()

metrics = compute_metrics(df, max_period_days=max_period_days, smooth=smooth)

col_price, col_fft = st.columns([2, 1.4])
with col_price:
    st.subheader("Price & Returns")
    st.plotly_chart(price_chart(df, metrics), use_container_width=True)

with col_fft:
    st.subheader("Dominant Cycles (FFT)")
    st.plotly_chart(spectrum_chart(metrics), use_container_width=True)

st.subheader("Signal Snapshot")
signals_panel(metrics)

st.divider()
st.caption("Generate a lightweight one-page PDF with current charts & metrics.")
if st.button("Export PDF report"):
    try:
        pdf_bytes = make_pdf_report(ticker, df, metrics)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"SignalScope_{ticker}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")


