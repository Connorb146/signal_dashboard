# src/plotter.py — MASSIVE UPGRADE HELPERS
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict

# ---------- small utils ----------
def _pct_change(s: pd.Series, periods: int) -> pd.Series:
    return s.pct_change(periods=periods)

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _realized_vol(series: pd.Series, window: int = 20) -> pd.Series:
    r = np.log(series).diff()
    return (r.rolling(window).std() * np.sqrt(252)) * 100.0

def _drawdown(series: pd.Series) -> pd.Series:
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd

def kpi_strip(df: pd.DataFrame, metrics: Dict):
    close = df["Close"]
    last = float(close.iloc[-1])

    def safe_pct(s: pd.Series, periods: int):
        try:
            val = s.pct_change(periods=periods).iloc[-1]
            if isinstance(val, (pd.Series, pd.DataFrame)):
                val = val.squeeze()
            return float(val)
        except Exception:
            return np.nan

    chg_1m = safe_pct(close, 21) * 100
    chg_3m = safe_pct(close, 63) * 100
    chg_1y = safe_pct(close, 252) * 100
    froth = float(metrics.get("froth_score", np.nan))

    cols = st.columns(4)
    cols[0].metric("Last", f"{last:,.2f}")
    cols[1].metric("1M", f"{chg_1m:+.1f}%" if np.isfinite(chg_1m) else "n/a")
    cols[2].metric("3M", f"{chg_3m:+.1f}%" if np.isfinite(chg_3m) else "n/a")
    cols[3].metric("Froth", f"{froth:.3f}" if np.isfinite(froth) else "n/a")

# ---------- charts ----------
def candle_chart(df: pd.DataFrame):
    # if no OHLC, fall back to line
    if not {"Open","High","Low","Close"}.issubset(df.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
        if len(df) > 20:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"].rolling(20).mean(), name="MA(20)"))
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=30), title="Price")
        return fig

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC"
    )])
    if len(df) > 20:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"].rolling(20).mean(), mode="lines", name="MA(20)"
        ))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=30))
    return fig

def price_chart(df: pd.DataFrame, metrics: Dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    if len(df) > 20:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"].rolling(20).mean(), mode="lines", name="MA(20)"))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=30), height=420, title="Price")
    return fig

def returns_hist(df: pd.DataFrame):
    r = np.log(df["Close"]).diff().dropna()
    fig = px.histogram(r, nbins=50)
    fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=30), title="Log Returns Histogram")
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

def entropy_chart(metrics: Dict):
    # make a rolling entropy over returns for display (approx)
    ret = metrics.get("returns", np.array([]))
    if ret.size < 50:
        return go.Figure()
    s = pd.Series(ret)
    roll = s.rolling(60).apply(lambda x: _entropy_np(x.values), raw=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(roll)), y=roll, mode="lines", name="Rolling Entropy (60)"))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=30))
    return fig

def _entropy_np(v: np.ndarray, bins: int = 30):
    v = v[np.isfinite(v)]
    if v.size < 10: return np.nan
    hist, _ = np.histogram(v, bins=bins, density=True)
    p = hist[hist>0]
    return float(-np.sum(p*np.log2(p)))

def rsi_chart(df: pd.DataFrame):
    rsi = _rsi(df["Close"].astype(float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=rsi, mode="lines", name="RSI(14)"))
    fig.add_hline(y=70, line_dash="dot")
    fig.add_hline(y=30, line_dash="dot")
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=30), title="RSI")
    return fig

def vol_chart(df: pd.DataFrame):
    vol = _realized_vol(df["Close"].astype(float), window=20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=vol, mode="lines", name="Realized Vol (20d, %/yr)"))
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=30), title="Realized Volatility")
    return fig

def drawdown_chart(df: pd.DataFrame):
    dd = _drawdown(df["Close"].astype(float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=dd*100.0, mode="lines", name="Drawdown (%)"))
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=30), title="Drawdown")
    return fig

def signals_panel(metrics: Dict):
    dom = metrics.get("dominant_period_days", float("nan"))
    ent = metrics.get("entropy_bits", float("nan"))
    froth = metrics.get("froth_score", float("nan"))
    z = metrics.get("zscore", np.array([np.nan]))
    z_now = z[-1] if z.size else np.nan

    cols = st.columns(4)
    cols[0].metric("Dominant Period (d)", f"{dom:.1f}" if np.isfinite(dom) else "n/a")
    cols[1].metric("Entropy (bits)", f"{ent:.3f}" if np.isfinite(ent) else "n/a")
    cols[2].metric("Z-score (20d)", f"{z_now:.3f}" if np.isfinite(z_now) else "n/a")
    cols[3].metric("Froth Score", f"{froth:.3f}" if np.isfinite(froth) else "n/a")

    if np.isfinite(froth):
        if froth >= 0.8:
            st.warning("Regime: HIGH — potential froth. Validate before acting.")
        elif froth >= 0.5:
            st.info("Regime: MODERATE — watch for confirmation.")
        else:
            st.success("Regime: LOW — calm / neutral.")

# ---------- compare / correlations ----------
def corr_matrix_chart(aligned_close: pd.DataFrame):
    # drop columns with all NaN
    ac = aligned_close.dropna(axis=1, how="all")
    # use daily returns correlation
    rets = np.log(ac).diff().dropna(how="all")
    c = rets.corr()
    fig = px.imshow(c, aspect="auto", text_auto=True, origin="lower",
                    color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=30))
    return fig

def xcorr_panel(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str, max_lag: int = 60):
    # align
    s1 = df_a.set_index("Date")["Close"].astype(float)
    s2 = df_b.set_index("Date")["Close"].astype(float)
    jo = pd.concat([s1, s2], axis=1, join="inner").dropna()
    if jo.empty:
        st.info("No overlapping history.")
        return
    r1 = np.log(jo.iloc[:,0]).diff().dropna().to_numpy()
    r2 = np.log(jo.iloc[:,1]).diff().dropna().to_numpy()
    L = min(len(r1), len(r2))
    r1, r2 = r1[-L:], r2[-L:]

    lags = np.arange(-max_lag, max_lag+1)
    vals = []
    for lag in lags:
        if lag < 0:
            v = np.corrcoef(r1[:lag], r2[-lag:])[0,1]
        elif lag > 0:
            v = np.corrcoef(r1[lag:], r2[:-lag])[0,1]
        else:
            v = np.corrcoef(r1, r2)[0,1]
        vals.append(v)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=vals, name="xcorr"))
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=30),
                      title=f"Cross-Correlation {name_a} vs {name_b} (returns)")
    st.plotly_chart(fig, use_container_width=True)

def heatmap_table(uni_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    def tail_pct_float(s: pd.Series, periods: int) -> float:
        """Return the last percent change as a float (or np.nan) — never a Series."""
        try:
            if not isinstance(s, pd.Series) or s.empty:
                return np.nan
            pc = s.pct_change(periods=periods)
            # grab scalar safely
            val = pc.iloc[-1] if len(pc) else np.nan
            # squeeze if some pandas versions hand us a 0-D/1-elem object/Series
            if isinstance(val, (pd.Series, pd.DataFrame)):
                val = np.array(val).ravel()[-1]
            return float(val)
        except Exception:
            return np.nan

    rows = []
    for t, df in uni_data.items():
        if df is None or df.empty:
            continue
        if "Date" not in df.columns or "Close" not in df.columns:
            continue

        s = df.set_index("Date")["Close"]
        # if some loaders ever return 1-col DataFrame, coerce to Series
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.astype(float)

        r1  = tail_pct_float(s, 21)   # ~1M
        r3  = tail_pct_float(s, 63)   # ~3M
        r6  = tail_pct_float(s, 126)  # ~6M
        r12 = tail_pct_float(s, 252)  # ~12M

        rows.append({
            "Ticker": t,
            "1M %": None if not np.isfinite(r1)  else round(r1 * 100, 1),
            "3M %": None if not np.isfinite(r3)  else round(r3 * 100, 1),
            "6M %": None if not np.isfinite(r6)  else round(r6 * 100, 1),
            "12M %": None if not np.isfinite(r12) else round(r12 * 100, 1),
        })

    if not rows:
        return None

    tbl = pd.DataFrame(rows).set_index("Ticker").sort_index()
    return tbl

    # color gradient via pandas Styler is heavy in Streamlit; keep as plain dataframe for speed
    return tbl

