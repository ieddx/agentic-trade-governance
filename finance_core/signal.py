"""
signal.py — simple moving-average crossover signal.

The idea is easy to understand:
  - Compute a SHORT moving average (e.g. 8 bars) and a LONG moving average
    (e.g. 24 bars) over the closing prices.
  - When the short MA is above the long MA, the recent trend is up → "buy".
  - When the short MA is below the long MA, the recent trend is down → "sell".

Confidence is volatility-normalised: the raw MA separation (as % of price) is
divided by the recent hourly return volatility so that the same dollar gap
means less in a high-vol market than in a calm one.  The result is mapped to
[0, MAX_CONFIDENCE=0.85] — never 1.0, because a maxed-out score would flag as
overfitting to the governance agent.

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

# Number of recent bars used to estimate volatility (matches market_context.py).
VOL_LOOKBACK: int = 24

# A normalised score of NORM_SCALE maps to MAX_CONFIDENCE.
# normalised_score = ma_separation_fraction / recent_volatility_fraction.
#
# History: initially 2.0, which caused all typical scores (2–3.5×) to clip
# at MAX_CONFIDENCE — confidence never discriminated between weak and strong
# signals.  Raised to 4.0 so typical scores span the full 0.4–0.85 range.
# At NORM_SCALE=4.0, you need a separation equal to 4× the typical hourly
# move to reach maximum confidence, which is a genuinely strong crossover.
NORM_SCALE: float = 4.0

# Confidence is capped below 1.0.  A score of 1.0 would indicate perfect
# certainty, which is implausible for a two-MA crossover and is penalised
# by the governance agent as a sign of overfitting.
MAX_CONFIDENCE: float = 0.85

# Fallback volatility used when there are too few bars to estimate vol; kept
# low so confidence degrades gracefully rather than blowing up on thin data.
_FALLBACK_VOL: float = 0.001   # 0.1 % per bar


def compute_signal(bars: pd.DataFrame) -> tuple[str, float, dict]:
    """
    Compute the MA-crossover signal from a bar DataFrame.

    Parameters
    ----------
    bars : DataFrame with a 'close' column, indexed by timestamp.

    Returns
    -------
    direction  : "buy" or "sell"
    confidence : float in [0, MAX_CONFIDENCE]
    breakdown  : dict of deterministic intermediate values used to derive
                 confidence.  These are computed entirely in code — no LLM
                 involvement.  Agents interpret these facts but never generate
                 them.
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
    latest_short = float(short_ma.iloc[-1])
    latest_long  = float(long_ma.iloc[-1])
    latest_close = float(close.iloc[-1])

    # --- Step 3: determine direction ---
    direction = "buy" if latest_short > latest_long else "sell"

    # --- Step 4: raw MA separation as a fraction of price ---
    ma_separation = abs(latest_short - latest_long) / latest_close  # unitless fraction

    # --- Step 5: recent hourly volatility (std dev of % returns) ---
    # Use VOL_LOOKBACK+1 rows so pct_change() produces VOL_LOOKBACK observations,
    # matching the approach in market_context.py.
    vol_tail = close.tail(VOL_LOOKBACK + 1)
    if len(vol_tail) >= 2:
        recent_vol = float(vol_tail.pct_change().dropna().std())
    else:
        recent_vol = _FALLBACK_VOL

    if recent_vol <= 0:
        recent_vol = _FALLBACK_VOL

    # --- Step 6: volatility-normalised score and final confidence ---
    # How many typical hourly moves does the MA gap represent?
    # A large normalised score means the signal stands well above the noise.
    normalized_score = ma_separation / recent_vol

    # Map [0, NORM_SCALE] → [0, MAX_CONFIDENCE].  Values above NORM_SCALE are
    # clipped at MAX_CONFIDENCE so confidence never reaches 1.0.
    confidence = min(normalized_score / NORM_SCALE * MAX_CONFIDENCE, MAX_CONFIDENCE)
    confidence = round(confidence, 4)

    # --- Step 7: build the breakdown dict ---
    # All values are deterministic facts derived from price data.  No LLM
    # is involved in computing them; agents interpret but never generate them.
    breakdown: dict = {
        "ma_separation_pct":    round(ma_separation * 100, 4),   # % of price
        "recent_volatility_pct": round(recent_vol * 100, 4),     # % per bar (hourly)
        "normalized_score":     round(normalized_score, 4),       # separation / vol
        "confidence":           confidence,
    }

    return direction, confidence, breakdown
