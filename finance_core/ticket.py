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

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Triple-barrier parameters (Lopez de Prado, "Advances in Financial Machine
# Learning", chapter 3).  All distances are in basis points (1 bp = 0.01 %).
# ---------------------------------------------------------------------------

# How far *above* entry we set the take-profit barrier.
TAKE_PROFIT_BPS: float = 150.0   # +1.50 %

# How far *below* entry we set the stop-loss barrier.
STOP_LOSS_BPS: float = 75.0      # -0.75 %

# Maximum number of hourly bars we are willing to hold the position.
TARGET_HORIZON_BARS: int = 24    # 24 hours


def _bps_to_multiplier(bps: float) -> float:
    """Convert basis points to a price multiplier.  150 bps -> 1.015."""
    return bps / 10_000.0


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
    confidence: float     # 0.0 – 1.0
    size: int


def build_ticket(
    ticker: str,
    direction: str,
    entry: float,
    confidence: float,
    notional: float = 10_000.0,
) -> Ticket:
    """
    Construct a Ticket from the minimal inputs.

    The stop and target levels are derived from the global triple-barrier
    constants so that every ticket in the system uses the same risk geometry.

    Parameters
    ----------
    ticker      : equity symbol
    direction   : "buy" or "sell"
    entry       : current price used as the entry reference
    confidence  : signal confidence in [0, 1]
    notional    : dollar amount used to size the position (default $10 000)
    """
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")

    profit_mult = _bps_to_multiplier(TAKE_PROFIT_BPS)
    loss_mult   = _bps_to_multiplier(STOP_LOSS_BPS)

    if direction == "buy":
        target = entry * (1 + profit_mult)   # price goes up to profit
        stop   = entry * (1 - loss_mult)     # price goes down to stop out
    else:
        target = entry * (1 - profit_mult)   # price goes down to profit
        stop   = entry * (1 + loss_mult)     # price goes up to stop out

    # Simple share-count: how many whole shares can we buy for the notional?
    size = max(1, int(notional / entry))

    return Ticket(
        ticker=ticker,
        direction=direction,
        entry=round(entry, 4),
        stop=round(stop, 4),
        target=round(target, 4),
        confidence=round(confidence, 4),
        size=size,
    )
