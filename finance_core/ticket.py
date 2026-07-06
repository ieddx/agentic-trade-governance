"""
ticket.py — defines the Ticket dataclass.

A Ticket is the single structured object that the deterministic finance core
hands to the agentic layer.  The finance core decides *what* to trade and *at
what price levels*; the agentic layer decides *whether and how* to act on it
(risk checks, position sizing overrides, human approval, execution routing).

Keeping these two concerns in separate modules makes it easy to:
  - unit-test the finance core without any agent or LLM in the loop.
  - swap out the signal logic without touching governance code.
  - log / audit every trade decision as a plain Python object.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Triple-barrier parameters (Lopez de Prado, "Advances in Financial Machine
# Learning", chapter 3).
# ---------------------------------------------------------------------------

# Maximum number of hourly bars we are willing to hold the position.
TARGET_HORIZON_BARS: int = 24    # 24 hours

# ---------------------------------------------------------------------------
# Volatility-scaled barrier distances.
#
# WHY we changed from fixed bps to volatility-scaled:
#   The original fixed stop (75 bps) was narrower than a single hour's typical
#   price move (~80–105 bps in recent AAPL/MSFT data).  A stop inside one bar's
#   typical range gets hit by noise even when the underlying trend is intact.
#   By anchoring the stop to N multiples of recent hourly volatility we ensure
#   the barrier clears the noise floor before the signal can be evaluated.
#
# Target is set to 2× the stop distance, preserving the 2:1 reward:risk ratio.
#
# NOTE: dollar risk per trade now varies with volatility — in high-vol regimes
#   the stop is wider and each trade risks more money.  Position sizing is NOT
#   yet adjusted for this (size is still notional/entry).  Volatility-scaled
#   sizing (e.g. sizing to a fixed dollar risk) is flagged as future work.
#
# History (kept for traceability):
#   TAKE_PROFIT_BPS = 150.0   # +1.50 % — replaced by 2× vol-scaled stop
#   STOP_LOSS_BPS   =  75.0   # -0.75 % — replaced by STOP_VOL_MULTIPLE × vol
# ---------------------------------------------------------------------------

# Stop distance = STOP_VOL_MULTIPLE × recent hourly volatility.
# At 2.0× a 1%-per-hour market, the stop is 2% below entry — roughly two
# hours of typical movement, giving the signal room to breathe before stopping.
STOP_VOL_MULTIPLE: float = 2.0

# Target is always 2× the stop so reward:risk stays at 2:1.
TARGET_VOL_MULTIPLE: float = STOP_VOL_MULTIPLE * 2.0


@dataclass
class Ticket:
    """
    A Ticket is the interface between the deterministic finance core and the
    agentic governance layer.

    Fields
    ------
    ticker      : str   — the equity symbol, e.g. "AAPL".
    direction   : str   — "buy" or "sell".
    entry       : float — suggested entry price (last close at signal time).
    stop        : float — stop-loss price; computed from STOP_LOSS_BPS.
    target      : float — take-profit price; computed from TAKE_PROFIT_BPS.
    confidence  : float — signal strength in [0, 1].  1 = maximum conviction.
    size        : int   — number of shares (naïve $10 000 / entry default).
    """

    ticker: str
    direction: str        # "buy" | "sell"
    entry: float
    stop: float
    target: float
    confidence: float     # 0.0 – MAX_CONFIDENCE (never 1.0 — see signal.py)
    size: int
    # Deterministic intermediate values used to compute confidence.  All values
    # are derived from price data in code; no LLM is involved in producing them.
    confidence_breakdown: dict = field(default_factory=dict)


def build_ticket(
    ticker: str,
    direction: str,
    entry: float,
    confidence: float,
    recent_vol_pct: float,
    notional: float = 10_000.0,
    confidence_breakdown: Optional[dict] = None,
) -> Ticket:
    """
    Construct a Ticket from the minimal inputs.

    Stop and target distances scale with recent hourly volatility so barriers
    sit outside the noise floor.  See STOP_VOL_MULTIPLE for the rationale.

    Parameters
    ----------
    ticker          : equity symbol
    direction       : "buy" or "sell"
    entry           : current price used as the entry reference
    confidence      : signal confidence in [0, 1]
    recent_vol_pct  : recent hourly volatility as a percentage (e.g. 1.05 for
                      1.05 %/hr).  Drives the stop and target distances.
    notional        : dollar amount used to size the position (default $10 000)
    """
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")

    # Convert vol from percentage to fraction, then apply multiples.
    stop_frac   = STOP_VOL_MULTIPLE   * (recent_vol_pct / 100.0)
    target_frac = TARGET_VOL_MULTIPLE * (recent_vol_pct / 100.0)

    if direction == "buy":
        stop   = entry * (1.0 - stop_frac)
        target = entry * (1.0 + target_frac)
    else:
        stop   = entry * (1.0 + stop_frac)
        target = entry * (1.0 - target_frac)

    # Simple share-count: how many whole shares can we buy for the notional?
    size = max(1, int(notional / entry))

    # Augment the breakdown with the barrier distances that were actually used,
    # so the governance agent (and any audit log) can see exactly what volatility
    # drove the stop/target levels.  We copy the dict to avoid mutating the
    # caller's object.
    breakdown = dict(confidence_breakdown) if confidence_breakdown else {}
    breakdown["stop_vol_multiple"]    = STOP_VOL_MULTIPLE
    breakdown["stop_distance_pct"]    = round(stop_frac * 100, 4)
    breakdown["target_distance_pct"]  = round(target_frac * 100, 4)

    return Ticket(
        ticker=ticker,
        direction=direction,
        entry=round(entry, 4),
        stop=round(stop, 4),
        target=round(target, 4),
        confidence=round(confidence, 4),
        size=size,
        confidence_breakdown=breakdown,
    )
