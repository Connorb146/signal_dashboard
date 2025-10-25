# src/analyzer.py
import io
import numpy as np
import pandas as pd
from typing import Dict
from scipy.fft import rfft, rfftfreq
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def _rolling_zscore(x: np.ndarray, window: int = 20) -> np.ndarray:
    if len(x) < window + 1:
        return np.full_like(x, np.nan, dtype=float)
    s = pd.Series(x, copy=False)
    return ((s - s.rolling(window).mean()) / s.rolling(window).std()).to_numpy()

def _shannon_entropy(values: np.ndarray, bins: int = 50) -> float:
    v = values[~np.isnan(values)]
    if v.size < 10:
        return np.nan
    hist, _ = np.histogram(v, bins=bins, density=True)
    p = hist[hist > 0]
    return float(-np.sum(p * np.log2(p)))

def compute_metrics(df: pd.DataFrame, max_period_days: int = 200, smooth: bool = True) -> Dict:
    """Return dict with returns, zscore, FFT spectrum/periods, entropy, froth score, etc."""
    out: Dict = {}
    close = df["Close"].to_numpy(dtype=float)

    # log returns
    ret = np.diff(np.log(close), prepend=np.log(close[0]))
    out["returns"] = ret
    out["zscore"] = _rolling_zscore(ret, window=20)

    # FFT on demeaned close
    demean = close - np.nanmean(close)
    N = len(demean)
    spec = np.abs(rfft(demean))
    freqs = rfftfreq(N, d=1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

    mask = periods <= max_period_days
    spec = spec[mask]
    periods = periods[mask]

    if smooth and spec.size > 5:
        k = max(3, int(0.01 * len(spec)))
        if k % 2 == 0: k += 1
        spec = np.convolve(spec, np.ones(k)/k, mode='same')

    finite = np.isfinite(periods)
    dom_period = float(periods[finite][np.argmax(spec[finite])]) if np.any(finite) else float("nan")

    out["spectrum"] = spec
    out["periods"] = periods
    out["dominant_period_days"] = dom_period
    out["close"] = close

    ent = _shannon_entropy(ret, bins=50)
    out["entropy_bits"] = ent

    z_now = np.nan_to_num(out["zscore"][-1], nan=0.0)
    ent_norm = 0.0 if np.isnan(ent) else ent / 6.0
    cyc_norm = 0.0 if np.isnan(dom_period) else min(dom_period / max_period_days, 1.0)
    out["froth_score"] = float(0.5 * ent_norm + 0.3 * abs(z_now) + 0.2 * cyc_norm)
    return out

def make_pdf_report(ticker: str, df: pd.DataFrame, metrics: Dict) -> bytes:
    """Simple 1-page PDF summary."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 0.75 * inch
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"SignalScope Report — {ticker}")
    y -= 0.4 * inch

    c.setFont("Helvetica", 11)
    def line(txt):
        nonlocal y
        c.drawString(margin, y, txt)
        y -= 0.24 * inch

    last_close = float(df["Close"].iloc[-1])
    dom = metrics.get("dominant_period_days", float("nan"))
    ent = metrics.get("entropy_bits", float("nan"))
    froth = metrics.get("froth_score", float("nan"))
    z_now = metrics.get("zscore", [np.nan])[-1]

    line(f"Last Close: {last_close:,.2f}")
    line(f"Dominant FFT Period (days): {dom:.1f}" if np.isfinite(dom) else "Dominant FFT Period: n/a")
    line(f"Entropy (bits) of returns: {ent:.3f}" if np.isfinite(ent) else "Entropy: n/a")
    line(f"Latest rolling Z-score (20d): {z_now:.3f}" if np.isfinite(z_now) else "Rolling Z-score: n/a")
    line(f"Composite Froth Score: {froth:.3f}" if np.isfinite(froth) else "Froth Score: n/a")

    c.setFont("Helvetica-Oblique", 9)
    line("Notes: FFT period is a proxy for recurring cycles; entropy ~ distributional disorder.")
    line("Exploratory analytics only. Validate signals before acting.")

    c.showPage()
    c.save()
    return buf.getvalue()
