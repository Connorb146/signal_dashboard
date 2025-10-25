# app.py — RehlySignal Pro (Investor-Friendly)
from src.analyzer import compute_metrics, make_pdf_report, generate_ai_summary
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- path to src (package style) ---
from src.data_loader import load_prices
from src.analyzer import compute_metrics, make_pdf_report
from src.plotter import (
    kpi_strip, candle_chart, price_chart, returns_hist,
    spectrum_chart, entropy_chart, rsi_chart, vol_chart,
    drawdown_chart, signals_panel, corr_matrix_chart,
    xcorr_panel, heatmap_table
)

# ---------- page config ----------
st.set_page_config(page_title="RehlySignal Pro", layout="wide")
st.markdown("""
<style>
/* Global Layout */
.block-container {
    padding-top: 0rem;
    padding-bottom: 2rem;
    max-width: 1400px;
    background: linear-gradient(135deg, rgba(10,10,15,1) 0%, rgba(25,25,35,1) 100%);
    color: #F5F5F5;
    font-family: 'Inter', sans-serif;
}

/* Header Bar */
.rehly-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(30,30,40,0.7);
    border: 1px solid rgba(90,90,120,0.2);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    border-radius: 16px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.6rem;
    backdrop-filter: blur(10px);
}
.rehly-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00FFC2;
    letter-spacing: 0.5px;
}
.rehly-subtitle {
    font-size: 0.9rem;
    color: #AAAAAA;
    margin-top: -4px;
}
.rehly-logo {
    font-weight: 700;
    font-size: 1rem;
    color: #00FFC2;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    color: #DDD;
    border-radius: 10px;
    padding: 10px 16px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.1);
    transform: scale(1.02);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg,#00FFC2 0%,#00B4D8 100%);
    color: #0B0B0B !important;
    font-weight: 700;
}

/* Metrics & KPIs */
.stMetric {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 8px;
    transition: background 0.2s ease;
}
.stMetric:hover {
    background: rgba(255,255,255,0.08);
}

/* Buttons */
div.stButton > button:first-child {
    background: linear-gradient(90deg,#00FFC2 0%,#00B4D8 100%);
    color: #000;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 0 12px #00FFC2;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)


# ---------- sidebar (plain-English labels) ----------
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Symbol", value="URA", help="Any Yahoo Finance-compatible symbol (e.g., URA, CCJ, ^GSPC).")
    years_back = st.slider("History window (years)", 1, 15, 5, help="How far back to analyze.")
    max_period_days = st.slider("Cycle max length (days)", 30, 500, 220,
                                help="Upper bound for recurring rhythm detection.")
    smooth = st.checkbox("Smooth cycle view", value=True, help="Light smoothing for easier reading.")
    compare_list = st.text_area("Compare symbols (comma-separated)", "URA, CCJ, ^GSPC",
                                help="Used in the Compare tab to show correlations and lead/lag.")
    market_focus = st.selectbox(
    "Uranium market focus",
    ["Exploration & Juniors", "Fuel & Utilities", "ETFs & Macro"],
    index=2
)

if market_focus == "Exploration & Juniors":
    universe = "UUUU, NXE, DNN, UEC, BOE.AX, GLO.TO"
elif market_focus == "Fuel & Utilities":
    universe = "CCJ, LEU, KAP.L, AEC.V, BWXT, ^GSPC"
else:
    universe = "URA, URNM, ^GSPC, XLE, XLU, GLD, SLV"

    run_button = st.button("Run / Refresh", type="primary")

# ---------- data ----------
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

# load primary
if run_button or "df" not in st.session_state:
    try:
        st.session_state["df"] = get_data(ticker, years_back)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

df = st.session_state.get("df")
if df is None or df.empty:
    st.warning("No data returned. Try a different symbol or shorter range.")
    st.stop()

# metrics
metrics = compute_metrics(df, max_period_days=max_period_days, smooth=smooth)

# ---------- header ----------
title_col, kpi_col = st.columns([1.2, 1.8])
with title_col:
    st.title("RehlySignal Pro")
    st.caption("Plain-English analytics for timing, momentum, and risk.")
with kpi_col:
    kpi_strip(df, metrics)  # KPI names use investor language

# ---------- tabs ----------
tab_over, tab_cycles, tab_reg, tab_cmp, tab_heat, tab_rep = st.tabs(
    ["Overview", "Rhythm (Cycles)", "Risk & Regimes", "Compare", "Momentum Grid", "Report"]
)

# ===== Overview =====
with tab_over:
    c1, c2 = st.columns([1.6, 1.4])
    with c1:
        st.subheader("Price (candles) + Moving Average")
    st.plotly_chart(price_with_regimes(df, metrics), use_container_width=True)
    with c2:
        st.subheader("Daily Move Distribution")
        st.plotly_chart(returns_hist(df), use_container_width=True)

    st.subheader("Market Pulse")
    st.caption("Quick read of trend stretch (Overheat/Cooldown), cycle rhythm, and market disorder.")
    signals_panel(metrics)
    # --- AI summary section ---
st.markdown("### Market Narrative")
try:
    summary = generate_ai_summary(metrics, ticker)
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px; padding: 1rem; font-size: 1rem; line-height: 1.5;'>
        {summary}
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Unable to generate summary: {e}")


# ===== Rhythm (Cycles) =====
with tab_cycles:
    cc1, cc2 = st.columns([1.7, 1.3])
    with cc1:
        st.subheader("Cycle Strength")
        st.caption("Highlights recurring rhythm in buying/selling flows.")
        st.plotly_chart(spectrum_chart(metrics), use_container_width=True)
    with cc2:
        st.subheader("Market Disorder (Rolling)")
        st.caption("Higher disorder = choppier market; lower = cleaner trends.")
        st.plotly_chart(entropy_chart(metrics), use_container_width=True)

# ===== Risk & Regimes =====
with tab_reg:
    st.subheader("Momentum & Volatility")
    rc1, rc2 = st.columns([1.6, 1.4])
    with rc1:
        st.caption("RSI ~ Overheat/Cooldown meter (classic 70/30 lines).")
        st.plotly_chart(rsi_chart(df), use_container_width=True)
    with rc2:
        st.caption("Annualized realized volatility (higher = larger swings).")
        st.plotly_chart(vol_chart(df), use_container_width=True)

    st.subheader("Drawdown Curve")
    st.caption("Peak-to-trough of price; helps visualize pain during pullbacks.")
    st.plotly_chart(drawdown_chart(df), use_container_width=True)

# ===== Compare =====
with tab_cmp:
    st.subheader("Correlation Matrix")
    tickers = [t.strip() for t in compare_list.split(",") if t.strip()]
    if len(tickers) < 2:
        st.info("Add 2+ symbols in the sidebar to compare.")
    else:
        many = get_many(tickers, years_back)

        aligned = None
        series_list = []
        for t, d in many.items():
            if d is None or d.empty: 
                continue
            if "Date" not in d.columns or "Close" not in d.columns:
                continue
            base = d.set_index("Date")["Close"]
            if isinstance(base, pd.DataFrame):  # safety
                base = base.iloc[:, 0]
            s = pd.Series(base.astype(float), name=t)
            series_list.append(s)

        if series_list:
            aligned = pd.concat(series_list, axis=1, join="outer").sort_index()

        if aligned is not None and not aligned.empty:
            aligned = aligned.ffill().dropna(how="all")
            st.plotly_chart(corr_matrix_chart(aligned), use_container_width=True)

            st.subheader("Lead/Lag Finder (Cross-Corr of Returns)")
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
            st.info("Provide 2+ valid symbols to compute correlations.")

# ===== Momentum Grid =====
with tab_heat:
    st.subheader("Performance Grid (1/3/6/12 months)")
    ulist = [t.strip() for t in universe.split(",") if t.strip()]
    uni_data = get_many(ulist, years_back)
    tbl = heatmap_table(uni_data)
    if tbl is not None:
        st.dataframe(tbl, use_container_width=True, height=min(600, 60 + 28*len(tbl)))
    else:
        st.info("No valid symbols returned.")

# ===== Report =====
with tab_rep:
    st.caption("Download a one-page PDF with current Market Pulse & Cycle Strength.")
    if st.button("Export PDF"):
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
