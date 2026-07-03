"""
signal.py — simple moving-average crossover signal.

The idea is easy to understand:
  - Compute a SHORT moving average (e.g. 8 bars) and a LONG moving average
    (e.g. 24 bars) over the closing prices.
  - When the short MA is above the long MA, the recent trend is up → "buy".
  - When the short MA is below the long MA, the recent trend is down → "sell".

We also produce a *confidence* value between 0 and 1.  Confidence is derived
from how far apart the two MAs are relative to the recent price level: a
larger spread means a stronger, more convincing crossover.

This is intentionally simple.  A real system would add filters, regime
detection, and walk-forward validation.  The goal here is a clear, testable
baseline that the agentic layer can reason about.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Moving-average window lengths in bars.
# With 1-hour bars, SHORT = 8 hours (~one trading day) and LONG = 24 hours.
# ---------------------------------------------------------------------------
SHORT_WINDOW: int = 8
LONG_WINDOW:  int = 24

# Confidence is capped at 1.0 and scaled by this normalisation factor.
# It represents the MA spread (as % of price) that we consider "maximum
# conviction".  0.5 % spread → confidence 1.0 after sigmoid scaling.
CONFIDENCE_SCALE: float = 0.005   # 0.5 % of price


def compute_signal(bars: pd.DataFrame) -> tuple[str, float]:
    """
    Compute the MA-crossover signal from a bar DataFrame.

    Parameters
    ----------
    bars : DataFrame with a 'close' column, indexed by timestamp.

    Returns
    -------
    direction  : "buy" or "sell"
    confidence : float in [0, 1]
    """
    if len(bars) < LONG_WINDOW:
        raise ValueError(
            f"Need at least {LONG_WINDOW} bars to compute signal; "
            f"got {len(bars)}."
        )

    close = bars["close"]

    # --- Step 1: compute both moving averages ---
    # pandas .rolling().mean() automatically handles the warm-up period;
    # the first LONG_WINDOW - 1 values will be NaN.
    short_ma = close.rolling(window=SHORT_WINDOW).mean()
    long_ma  = close.rolling(window=LONG_WINDOW).mean()

    # --- Step 2: look at the most recent bar ---
    latest_short = short_ma.iloc[-1]
    latest_long  = long_ma.iloc[-1]
    latest_close = close.iloc[-1]

    # --- Step 3: determine direction ---
    # If the fast MA is above the slow MA, prices have recently risen faster
    # than the longer-term average → bullish crossover.
    direction = "buy" if latest_short > latest_long else "sell"

    # --- Step 4: compute confidence ---
    # Spread is how much the two MAs differ, expressed as a fraction of price.
    # E.g. short=210, long=209, close=210 → spread = 1/210 ≈ 0.0048.
    raw_spread = abs(latest_short - latest_long) / latest_close

    # Map the spread to [0, 1] using a sigmoid-like clamp.
    # We use a simple linear clip: spread / CONFIDENCE_SCALE, capped at 1.0.
    # You could substitute a proper sigmoid if you want smoother gradients.
    confidence = min(raw_spread / CONFIDENCE_SCALE, 1.0)

    # Round for readability.
    confidence = round(confidence, 4)

    return direction, confidence
