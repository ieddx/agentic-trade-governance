"""
core.py — entry point for the deterministic finance core.

This module is the "main" of the finance_core package.  It:
  1. Loads historical AAPL bars (from cache or Alpaca).
  2. Runs the MA-crossover signal to get a direction and confidence score.
  3. If confidence clears a minimum threshold, builds a Ticket and prints it.
  4. Returns the Ticket (or None) so the agentic layer can consume it.

The agentic layer (agentic_layer/) will call produce_ticket() and then decide
what to do: run risk checks, ask an LLM for a second opinion, request human
approval, or route the order to a broker.  All of that is *outside* this file.
"""

from __future__ import annotations

from finance_core.data_loader import load_bars
from finance_core.signal import compute_signal
from finance_core.ticket import Ticket, build_ticket

# ---------------------------------------------------------------------------
# Minimum confidence required to emit a Ticket.
# Below this threshold the signal is too weak to act on; core returns None.
# ---------------------------------------------------------------------------
MIN_CONFIDENCE: float = 0.20   # 20 % of the CONFIDENCE_SCALE spread


def produce_ticket(symbol: str = "AAPL", bars=None) -> Ticket | None:
    """
    Run the full finance-core pipeline and return a Ticket if the signal is
    strong enough, otherwise return None.

    Parameters
    ----------
    symbol : equity ticker to analyse (default "AAPL")
    bars   : optional pre-loaded bar DataFrame.  When a coordinator pre-fetches
             bars and passes them here, the whole pipeline (signal, research,
             governance) operates on the same data snapshot — no temporal drift
             if the cache refreshes mid-run.  Pass None to load bars here.
    """
    # --- Step 1: load bars ---
    # Use pre-loaded bars when provided by a coordinator; otherwise load now.
    if bars is None:
        bars = load_bars(symbol=symbol)

    # --- Step 2: compute the signal ---
    # signal.py returns (direction, confidence, breakdown).
    direction, confidence, breakdown = compute_signal(bars)

    print(
        f"[core] Signal: direction={direction!r}, confidence={confidence:.4f} "
        f"(threshold={MIN_CONFIDENCE})"
    )

    # --- Step 3: gate on confidence ---
    if confidence < MIN_CONFIDENCE:
        print("[core] Confidence below threshold — no Ticket produced.")
        return None

    # --- Step 4: build the Ticket ---
    # Entry price is the most recent closing price.
    entry_price = float(bars["close"].iloc[-1])

    ticket = build_ticket(
        ticker=symbol,
        direction=direction,
        entry=entry_price,
        confidence=confidence,
        recent_vol_pct=breakdown["recent_volatility_pct"],
        confidence_breakdown=breakdown,
    )

    print("[core] Ticket produced:")
    print(f"  {ticket}")

    return ticket


# ---------------------------------------------------------------------------
# Allow running this file directly:  python -m finance_core.core
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from finance_core.data_loader import load_bars as _load
    _bars, _feed = _load()
    print(f"[__main__] feed: {_feed}")
    produce_ticket(bars=_bars)
