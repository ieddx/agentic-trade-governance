"""
market_context.py — read-only market analysis derived from cached bar data.

get_market_context() computes a small set of numerical indicators from a
ticker's hourly OHLCV bars and returns them as a plain dict.  It reuses the
bar-loading infrastructure in data_loader.py (cache-first) so no extra API
calls are made.

IMPORTANT: this module is purely analytical and read-only.  It reads prices
and volumes from the local cache; it never places orders, modifies positions,
or calls any trading endpoint.
"""

from finance_core.data_loader import load_bars

# Number of recent hourly bars used for every rolling calculation.
# 24 bars = one full trading day, matching TARGET_HORIZON_BARS in ticket.py
# so the context window aligns with the trade's maximum holding period.
LOOKBACK_BARS = 24


def get_market_context(ticker: str, bars=None) -> dict:
    """
    Compute recent market context for *ticker* from its hourly bar data.

    Reuses load_bars() (cache-first) so calling this function after a
    successful data_loader run costs nothing extra.

    Parameters
    ----------
    ticker : equity symbol
    bars   : optional pre-loaded bar DataFrame.  Pass a pre-loaded DataFrame
             from a coordinator so this function and produce_ticket() analyse
             the same snapshot — critical for temporal consistency.

    Returned keys
    -------------
    ticker
        The symbol this context is for.

    lookback_bars
        Number of hourly bars the metrics are computed over (always
        LOOKBACK_BARS so callers know the window without reading source code).

    current_price
        Most recent closing price — the same value used as the Ticket entry.

    recent_volatility_pct
        Standard deviation of hourly percentage returns over the window,
        expressed as a percentage (not a decimal).
        Why it matters: high per-hour volatility means the price can
        easily swing past a tight stop-loss on noise alone.  A risk reviewer
        may want to widen stops or reduce size in high-vol regimes.

    volume_ratio
        Latest bar's volume divided by mean volume over the window.
        > 1.5 → unusually high participation (possible news / catalyst).
        < 0.5 → thin market with wide spreads and higher slippage risk.

    latest_volume / avg_volume
        Raw values behind volume_ratio for full transparency.

    momentum_pct
        Percentage price change from the first to the last bar in the window.
        Positive means the price has risen over the lookback period.
        For a BUY signal, positive momentum confirms the trend; negative
        momentum means the signal is counter-trend (higher risk).

    recent_high / recent_low
        Absolute intrabar high and low over the window.

    price_position
        Where current_price sits in the [recent_low, recent_high] range,
        linearly scaled to [0, 1].
        0.0 = exactly at the recent low, 1.0 = exactly at the recent high.
        A buy near 1.0 means entering near the top of the recent range —
        significant mean-reversion risk if the range-high acts as resistance.
    """
    if bars is None:
        bars, _ = load_bars(symbol=ticker)

    # Take LOOKBACK_BARS + 1 rows so that pct_change() (which loses one row)
    # still produces exactly LOOKBACK_BARS return observations.
    tail = bars.tail(LOOKBACK_BARS + 1)

    close  = tail["close"]
    volume = tail["volume"]
    high   = tail["high"]
    low    = tail["low"]

    # --- Volatility: std dev of hourly % returns ---
    hourly_returns        = close.pct_change().dropna()   # LOOKBACK_BARS observations
    recent_volatility_pct = round(float(hourly_returns.std() * 100), 4)

    # --- Volume trend ---
    latest_volume = int(volume.iloc[-1])
    avg_volume    = float(volume.iloc[:-1].mean())        # exclude the latest bar from avg
    volume_ratio  = round(float(latest_volume / avg_volume), 3) if avg_volume > 0 else 1.0

    # --- Momentum: total % move from start to end of window ---
    oldest_close  = float(close.iloc[0])
    current_price = float(close.iloc[-1])
    momentum_pct  = round((current_price - oldest_close) / oldest_close * 100, 4)

    # --- Price position in recent high-low range (0 = low end, 1 = high end) ---
    recent_high = float(high.max())
    recent_low  = float(low.min())
    price_range = recent_high - recent_low
    price_position = (
        round((current_price - recent_low) / price_range, 4)
        if price_range > 0
        else 0.5   # flat market; position is ambiguous, default to midpoint
    )

    return {
        "ticker":                ticker,
        "lookback_bars":         LOOKBACK_BARS,
        "current_price":         round(current_price, 4),
        "recent_volatility_pct": recent_volatility_pct,
        "volume_ratio":          volume_ratio,
        "latest_volume":         latest_volume,
        "avg_volume":            round(avg_volume, 0),
        "momentum_pct":          momentum_pct,
        "recent_high":           round(recent_high, 4),
        "recent_low":            round(recent_low, 4),
        "price_position":        price_position,
    }
