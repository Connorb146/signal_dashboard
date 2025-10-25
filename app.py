# app.py (root) — MASSIVE UPGRADE
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# path to src
import os, sys
APP_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_prices
from analyzer import compute_metrics, make_pdf_report
from plotter import (
    kpi_strip, candle_chart, price_chart, returns_hist,
    spectrum_chart, entropy_chart, rsi_chart, vol_chart,
    drawdown_chart, signals_panel, corr_matrix_chart,
    xcorr_panel, heatmap_table
)

# ---------- page config ----------
st.set_page_config(page_title="RehlySignal Analytics", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
      .stMetric {background: rgba(255,255,255,0.03); border: 1px solid rgba(100,100,100,0.2); border-radius: 12px; padding: 8px;}
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] { padding: 10px 14px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- sidebar ----------
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Primary ticker", value="URA")
    years_back = st.slider("Years of history", 1, 15, 5)
    max_period_days = st.slider("Max Cycle Strength period (days)", 30, 500, 220)
    smooth = st.checkbox("Smooth Cycle Strength spectrum", value=True)
    # compare tickers (comma-separated)
    compare_list = st.text_area("Compare tickers (comma-separated)", "URA, CCJ, ^GSPC")
    # small preset universe for heatmap
    universe = st.text_area("Momentum Grid universe (comma-separated)", "URA, CCJ, UUUU, ^GSPC, ^NDX, XLE, XLU, GLD, SLV")
    run_button = st.button("Run / Refresh", type="primary")

# ---------- data cache ----------
@st.cache_data(show_spinner=False, ttl=3600)
def get_data(ticker: str, years_back: int) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    return load_prices(ticker, start.isoformat(), end.isoformat())

@st.cache_data(show_spinner=False, ttl=3600)
def get_many(tickers: list[str], years_back: int) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        t = t.strip()
        if not t: 
            continue
        try:
            out[t] = get_data(t, years_back)
        except Exception:
            out[t] = pd.DataFrame()
    return out

# ---------- load primary ----------
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

# ---------- compute metrics for primary ----------
metrics = compute_metrics(df, max_period_days=max_period_days, smooth=smooth)

# ---------- header & KPI strip ----------
title_col, kpi_col = st.columns([1.2, 1.8])
with title_col:
    st.title("RehlySignal Analytics")
    st.caption("Physics-inspired signals: cycles, entropy, regimes, and correlations.")

with kpi_col:
    kpi_strip(df, metrics)

# ---------- tabs ----------
tab_over, tab_cycles, tab_reg, tab_cmp, tab_heat, tab_rep = st.tabs(
    ["Overview", "Cycles", "Regimes", "Compare", "Momentum Grid", "Report"]
)

# ===== Overview =====
with tab_over:
    c1, c2 = st.columns([1.6, 1.4])
    with c1:
        st.subheader("Price (candles) + MA")
        st.plotly_chart(candle_chart(df), use_container_width=True)
    with c2:
        st.subheader("Returns Distribution")
        st.plotly_chart(returns_hist(df), use_container_width=True)

    st.subheader("Market Pulse")
    signals_panel(metrics)

# ===== Cycles =====
with tab_cycles:
    cc1, cc2 = st.columns([1.7, 1.3])
    with cc1:
        st.subheader("Cycle Strength")
        st.plotly_chart(spectrum_chart(metrics), use_container_width=True)
    with cc2:
        st.subheader("Rolling Market Disorder")
        st.plotly_chart(entropy_chart(metrics), use_container_width=True)

# ===== Regimes =====
with tab_reg:
    st.subheader("Momentum & Volatility")
    rc1, rc2 = st.columns([1.6, 1.4])
    with rc1:
        st.plotly_chart(rsi_chart(df), use_container_width=True)
    with rc2:
        st.plotly_chart(vol_chart(df), use_container_width=True)

    st.subheader("Drawdown Curve")
    st.plotly_chart(drawdown_chart(df), use_container_width=True)

# ===== Compare =====
with tab_cmp:
    st.subheader("Correlation Matrix")

    # Rebuild inputs INSIDE the tab scope
    tickers = [t.strip() for t in compare_list.split(",") if t.strip()]
    if len(tickers) < 2:
        st.info("Add 2+ tickers in the sidebar (comma-separated) to compare.")
        st.stop()

    many = get_many(tickers, years_back)

    aligned = None
    series_list = []

    for t, d in many.items():
        if d is None or d.empty:
            continue
        if "Date" not in d.columns or "Close" not in d.columns:
            continue

        base = d.set_index("Date")["Close"]
        # If some pandas versions yield a DataFrame, squeeze to Series
        if isinstance(base, pd.DataFrame):
            base = base.iloc[:, 0]

        s = pd.Series(base.astype(float), name=t)
        series_list.append(s)

    if series_list:
        aligned = pd.concat(series_list, axis=1, join="outer").sort_index()

    if aligned is not None and not aligned.empty:
        aligned = aligned.ffill().dropna(how="all")
        st.plotly_chart(corr_matrix_chart(aligned), use_container_width=True)

        st.subheader("Pairwise Cross-Correlation (lead/lag)")
        a1, a2 = st.columns([1, 1])
        with a1:
            a = st.selectbox("Series A", options=list(aligned.columns), index=0)
        with a2:
            b = st.selectbox("Series B", options=list(aligned.columns),
                             index=min(1, len(aligned.columns) - 1))

        if a in aligned.columns and b in aligned.columns:
            df_a = aligned[[a]].rename(columns={a: "Close"}).reset_index(names="Date")
            df_b = aligned[[b]].rename(columns={b: "Close"}).reset_index(names="Date")
            xcorr_panel(df_a, df_b, a, b)
    else:
        st.info("Provide 2+ valid tickers in the sidebar to compute correlations.")

# ===== Heatmap =====
with tab_heat:
    st.subheader("Rolling Return Heatmap (1/3/6/12 months)")
    ulist = [t.strip() for t in universe.split(",") if t.strip()]
    uni_data = get_many(ulist, years_back)
    tbl = heatmap_table(uni_data)
    if tbl is not None:
        st.dataframe(tbl, use_container_width=True, height=min(600, 60 + 28*len(tbl)))

# ===== Report =====
with tab_rep:
    st.caption("Generate a one-page PDF with current metrics.")
    if st.button("Export PDF report"):
        try:
            pdf_bytes = make_pdf_report(ticker, df, metrics)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"RehlySignal_{ticker}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

