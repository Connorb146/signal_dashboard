# src/plotter.py — Investor-friendly labels + robust scalars
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict

# ---------- utilities ----------
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

def _entropy_np(v: np.ndarray, bins: int = 30):
    v = v[np.isfinite(v)]
    if v.size < 10: return np.nan
    hist, _ = np.histogram(v, bins=bins, density=True)
    p = hist[hist>0]
    return float(-np.sum(p*np.log2(p)))

# ---------- KPI strip (plain-English) ----------
def kpi_strip(df: pd.DataFrame, metrics: Dict):
    close = df["Close"]
    last = float(close.iloc[-1])

    def safe_pct(s: pd.Series, periods: int):
        try:
            val = s.pct_change(periods=periods).iloc[-1]
            if isinstance(val, (pd.Series, pd.DataFrame)):
                val = np.array(val).ravel()[-1]
            return float(val)
        except Exception:
            return np.nan

    chg_1m = safe_pct(close, 21) * 100
    chg_3m = safe_pct(close, 63) * 100
    froth = float(metrics.get("froth_score", np.nan))

    cols = st.columns(4)
    cols[0].metric("Last Price", f"{last:,.2f}")
    cols[1].metric("1-Month Change", f"{chg_1m:+.1f}%" if np.isfinite(chg_1m) else "n/a")
    cols[2].metric("3-Month Change", f"{chg_3m:+.1f}%" if np.isfinite(chg_3m) else "n/a")
    cols[3].metric("Market Tension", f"{froth:.3f}" if np.isfinite(froth) else "n/a")

# ---------- charts (titles use investor language) ----------
def candle_chart(df: pd.DataFrame):
    if not {"Open","High","Low","Close"}.issubset(df.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
        if len(df) > 20:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"].rolling(20).mean(), name="MA(20)"))
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=30), title="Price (line)")
        return fig

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC"
    )])
    if len(df) > 20:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"].rolling(20).mean(), mode="lines", name="MA(20)"
        ))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=30), title="Price (candles) + MA")
    return fig

def price_chart(df: pd.DataFrame, metrics: Dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    if len(df) > 20:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"].rolling(20).mean(), mode="lines", name="MA(20)"))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=30), height=420, title="Price")
    return fig

def price_with_regimes(df: pd.DataFrame, metrics: Dict):
    """Price chart with dominant rhythm overlay + entropy-based background."""
    close = df["Close"].astype(float).to_numpy()
    dates = df["Date"]

    fig = go.Figure()

    # Entropy-based background shading (regime coloring)
    ret = metrics.get("returns", np.array([]))
    if ret.size >= len(dates):
        ent = pd.Series(ret).rolling(60).apply(lambda x: _entropy_np(x.values))
        ent_norm = (ent - ent.min()) / (ent.max() - ent.min())
        colors = ["rgba(0,255,194," + str(c * 0.3 + 0.1) + ")" for c in ent_norm.fillna(0)]
        fig.add_trace(go.Bar(x=dates, y=np.nanmax(close) * 0.02,
                             marker_color=colors, opacity=0.2,
                             name="Regime shading", hoverinfo="skip"))

    # Price line
    fig.add_trace(go.Scatter(x=dates, y=close, mode="lines", name="Close", line=dict(color="#00FFC2")))

    # Dominant rhythm overlay (approx sine wave)
    dom = metrics.get("dominant_period_days", np.nan)
    if np.isfinite(dom) and dom > 10:
        phase = np.linspace(0, 2*np.pi*len(dates)/dom, len(dates))
        wave = np.sin(phase) * (np.std(close) * 0.1) + np.mean(close)
        fig.add_trace(go.Scatter(x=dates, y=wave, mode="lines",
                                 name=f"Cycle Overlay (~{dom:.0f} d)", line=dict(color="#FFD700", width=1.5, dash="dot")))

    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=30),
                      title="Price with Rhythm & Regime Context",
                      yaxis_title="Price")
    return fig


def returns_hist(df: pd.DataFrame):
    r = np.log(df["Close"]).diff().dropna()
    fig = px.histogram(r, nbins=50, labels={"value":"Daily log return"})
    fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=30), title="Daily Move Distribution")
    return fig

def spectrum_chart(metrics: Dict):
    periods = metrics["periods"]
    spec = metrics["spectrum"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=spec, mode="lines", name="Cycle Power",
                             hovertemplate="Length (days): %{x:.0f}<br>Strength: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="Cycle length (days)")
    fig.update_yaxes(title="Strength")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=30), height=420)
    return fig

def entropy_chart(metrics: Dict):
    # rolling disorder proxy from returns
    ret = metrics.get("returns", np.array([]))
    if ret.size < 50:
        return go.Figure()
    s = pd.Series(ret)
    roll = s.rolling(60).apply(lambda x: _entropy_np(x.values), raw=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(roll)), y=roll, mode="lines",
                             name="Market Disorder (60)", hovertemplate="Index: %{y:.3f}<extra></extra>"))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=30), title="Market Disorder (Rolling)")
    return fig

def rsi_chart(df: pd.DataFrame):
    rsi = _rsi(df["Close"].astype(float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=rsi, mode="lines", name="RSI(14)"))
    fig.add_hline(y=70, line_dash="dot")
    fig.add_hline(y=30, line_dash="dot")
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=30), title="Overheat/Cooldown Meter (RSI)")
    return fig

def vol_chart(df: pd.DataFrame):
    vol = _realized_vol(df["Close"].astype(float), window=20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=vol, mode="lines", name="Realized Vol (20d, %/yr)"))
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=30), title="Volatility (Realized, Annualized)")
    return fig

def drawdown_chart(df: pd.DataFrame):
    dd = _drawdown(df["Close"].astype(float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=dd*100.0, mode="lines", name="Drawdown (%)"))
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=30), title="Drawdown (Peak-to-Trough)")
    return fig

def signals_panel(metrics: Dict):
    dom = metrics.get("dominant_period_days", float("nan"))
    ent = metrics.get("entropy_bits", float("nan"))
    froth = metrics.get("froth_score", float("nan"))
    z = metrics.get("zscore", np.array([np.nan]))
    z_now = z[-1] if z.size else np.nan

    cols = st.columns(4)
    cols[0].metric("Cycle Rhythm (days)", f"{dom:.1f}" if np.isfinite(dom) else "n/a")
    cols[1].metric("Market Disorder", f"{ent:.3f}" if np.isfinite(ent) else "n/a")
    cols[2].metric("Overheat/Cooldown (Z)", f"{z_now:.3f}" if np.isfinite(z_now) else "n/a")
    cols[3].metric("Market Tension", f"{froth:.3f}" if np.isfinite(froth) else "n/a")

    if np.isfinite(froth):
        if froth >= 0.8:
            st.warning("Regime: HIGH tension — expect unstable moves; reduce size or wait for confirmation.")
        elif froth >= 0.5:
            st.info("Regime: MODERATE tension — opportunities with caution; validate with price action.")
        else:
            st.success("Regime: LOW tension — cleaner trends; momentum strategies may work better.")

# ---------- correlations ----------
def corr_matrix_chart(aligned_close: pd.DataFrame):
    ac = aligned_close.dropna(axis=1, how="all")
    rets = np.log(ac).diff().dropna(how="all")
    c = rets.corr()
    fig = px.imshow(c, aspect="auto", text_auto=True, origin="lower",
                    color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    labels=dict(color="Correlation"))
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=30), title="Return Correlations")
    return fig

def xcorr_panel(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str, max_lag: int = 60):
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
                      title=f"Lead/Lag — {name_a} vs {name_b} (returns)")
    st.plotly_chart(fig, use_container_width=True)

# ---------- momentum grid (robust floats) ----------
def heatmap_table(uni_data: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    def tail_pct_float(s: pd.Series, periods: int) -> float:
        try:
            if not isinstance(s, pd.Series) or s.empty:
                return np.nan
            pc = s.pct_change(periods=periods)
            val = pc.iloc[-1] if len(pc) else np.nan
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
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.astype(float)

        r1  = tail_pct_float(s, 21)
        r3  = tail_pct_float(s, 63)
        r6  = tail_pct_float(s, 126)
        r12 = tail_pct_float(s, 252)

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
