# src/plotter.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict

def price_chart(df: pd.DataFrame, metrics: Dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    if len(df) > 20:
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Close"].rolling(20).mean(),
            mode="lines",
            name="MA(20)"
        ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=30), height=420, title="Price")
    return fig

def spectrum_chart(metrics: Dict):
    periods = metrics["periods"]
    spec = metrics["spectrum"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=spec, mode="lines", name="FFT Power"))
    fig.update_xaxes(title="Period (days)")
    fig.update_yaxes(title="Power")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=30), height=420)
    return fig

def signals_panel(metrics: Dict):
    dom = metrics.get("dominant_period_days", float("nan"))
    ent = metrics.get("entropy_bits", float("nan"))
    froth = metrics.get("froth_score", float("nan"))
    z_now = metrics.get("zscore", [np.nan])[-1]

    cols = st.columns(4)
    cols[0].metric("Dominant Period (d)", f"{dom:.1f}" if np.isfinite(dom) else "n/a")
    cols[1].metric("Entropy (bits)", f"{ent:.3f}" if np.isfinite(ent) else "n/a")
    cols[2].metric("Z-score (20d)", f"{z_now:.3f}" if np.isfinite(z_now) else "n/a")
    cols[3].metric("Froth Score", f"{froth:.3f}" if np.isfinite(froth) else "n/a")

    if np.isfinite(froth):
        if froth >= 0.8:
            st.warning("Signal: HIGH — potential froth regime. Validate before acting.")
        elif froth >= 0.5:
            st.info("Signal: MODERATE — watch for confirmation.")
        else:
            st.success("Signal: LOW — calm/neutral regime.")
