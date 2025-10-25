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

    # Close as clean 1-D float array
    close_raw = df["Close"].to_numpy()
    # drop NaNs
    m = np.isfinite(close_raw)
    close = close_raw[m].astype(float).reshape(-1)

    # If not enough data, return minimal metrics
    if close.size < 2:
        out.update({
            "returns": np.array([], dtype=float),
            "zscore": np.array([], dtype=float),
            "spectrum": np.array([], dtype=float),
            "periods": np.array([], dtype=float),
            "dominant_period_days": float("nan"),
            "entropy_bits": float("nan"),
            "froth_score": float("nan"),
            "close": close,
        })
        return out

    # --- Returns (use 1-D prepend slice to avoid AxisError) ---
    log_close = np.log(np.clip(close, 1e-12, None))
    ret = np.diff(log_close, prepend=log_close[:1])  # prepend has same shape dim
    out["returns"] = ret

    # --- Rolling Z-score on returns ---
    out["zscore"] = _rolling_zscore(ret, window=20)

    # --- FFT on demeaned close ---
    demean = close - np.nanmean(close)
    N = demean.size

    if N < 4:
        # Too short for meaningful FFT
        out.update({
            "spectrum": np.array([], dtype=float),
            "periods": np.array([], dtype=float),
            "dominant_period_days": float("nan"),
        })
    else:
        spec = np.abs(rfft(demean))
        freqs = rfftfreq(N, d=1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

        mask = np.isfinite(periods) & (periods <= max_period_days)
        spec = spec[mask]
        periods = periods[mask]

        if smooth and spec.size > 5:
            k = max(3, int(0.01 * len(spec)))
            if k % 2 == 0:
                k += 1
            spec = np.convolve(spec, np.ones(k)/k, mode='same')

        if periods.size:
            dom_period = float(periods[np.argmax(spec)])
        else:
            dom_period = float("nan")

        out["spectrum"] = spec
        out["periods"] = periods
        out["dominant_period_days"] = dom_period

    # --- Entropy & Froth score ---
    ent = _shannon_entropy(ret, bins=50)
    out["entropy_bits"] = ent

    z_now = np.nan_to_num(out["zscore"][-1] if out["zscore"].size else np.nan, nan=0.0)
    ent_norm = 0.0 if np.isnan(ent) else ent / 6.0
    dom = out.get("dominant_period_days", float("nan"))
    cyc_norm = 0.0 if np.isnan(dom) else min(dom / max_period_days, 1.0)
    out["froth_score"] = float(0.5 * ent_norm + 0.3 * abs(z_now) + 0.2 * cyc_norm)

    out["close"] = close
    return out
def generate_ai_summary(metrics: dict, ticker: str) -> str:
    """Turn numeric metrics into a short human summary."""
    froth = float(metrics.get("froth_score", np.nan))
    ent = float(metrics.get("entropy_bits", np.nan))
    dom = float(metrics.get("dominant_period_days", np.nan))
    z = metrics.get("zscore", np.array([np.nan]))
    z_now = float(z[-1]) if isinstance(z, (list, np.ndarray)) and len(z) else np.nan

    lines = []

    # Rhythm / cycle
    if np.isfinite(dom):
        if dom < 40:
            lines.append(f"Short-term rhythm (~{dom:.0f} days) — quick cycles dominate {ticker}.")
        elif dom < 100:
            lines.append(f"Medium rhythm (~{dom:.0f} days) — typical swing length for {ticker}.")
        else:
            lines.append(f"Long rhythm (~{dom:.0f} days) — slower accumulation/distribution phases.")
    else:
        lines.append("Cycle rhythm unclear — data too noisy for consistent pattern.")

    # Disorder
    if np.isfinite(ent):
        if ent < 3.5:
            lines.append("Market disorder low — price action relatively orderly; trends easier to ride.")
        elif ent < 5.0:
            lines.append("Market disorder moderate — some choppiness; selective setups.")
        else:
            lines.append("Market disorder high — chaotic movement; momentum strategies riskier.")
    else:
        lines.append("No valid entropy data for this range.")

    # Z-score interpretation
    if np.isfinite(z_now):
        if z_now > 1:
            lines.append(f"Currently stretched upward (z ≈ {z_now:.2f}) — possible overheat near-term.")
        elif z_now < -1:
            lines.append(f"Currently stretched downward (z ≈ {z_now:.2f}) — possible rebound setup.")
        else:
            lines.append("Price near equilibrium; neither extended nor compressed.")
    else:
        lines.append("Z-score unavailable.")

    # Froth / tension
    if np.isfinite(froth):
        if froth > 0.8:
            lines.append("Tension high — volatility spikes likely; avoid chasing new trends.")
        elif froth > 0.5:
            lines.append("Tension moderate — rotation or pause phases likely soon.")
        else:
            lines.append("Tension low — environment favors steady trend continuation.")
    else:
        lines.append("No tension reading available.")

    return " ".join(lines)


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
