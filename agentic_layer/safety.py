"""
safety.py — pre-execution safety harness for trade tickets.

WHY THIS EXISTS
---------------
A deterministic finance core and an LLM governance agent can both produce
plausible-looking output that is structurally wrong: a stop placed on the
wrong side of entry, a confidence value outside [0,1], a dollar risk that
exceeds any sensible per-trade limit.  Neither unit tests nor LLM reasoning
catch all of these — unit tests only cover what was anticipated, and LLMs can
hallucinate or misread numbers.

validate_ticket() is the last hard gate before a ticket enters the human-
approval loop.  It checks structural invariants that must hold regardless of
market conditions, signal strength, or governance opinion.  If any check
fails the workflow exits immediately — no Gemini quota is spent on a broken
ticket, and no human is asked to approve something the system already knows is
wrong.

WHAT THIS DOES NOT DO
---------------------
This harness validates structure, not strategy.  It will not catch a ticket
that is structurally valid but has a bad signal (that is the governance agent's
job).  It also does not size positions, manage portfolio-level risk, or check
margin — those belong in a full execution layer that this system does not have.

REAL-WORLD NOTE
---------------
In a production trading system, a harness like this is typically:
  - Run by the order management system (OMS) independently of any strategy code
  - Backed by a signed pre-trade risk check from the prime broker
  - Required by regulation (SEC Rule 15c3-5 in the US, comparable rules
    elsewhere) to prevent erroneous orders from reaching the exchange

Here it is implemented at the coordinator level as the closest analogue in a
paper-trading / research system.
"""

from __future__ import annotations

from finance_core.ticket import Ticket

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

# Hard cap on dollar risk per trade (stop distance × share count).
# In a live system this would be set per-account based on capital and drawdown
# limits; $500 is a conservative default for a research / paper-trade system.
MAX_DOLLAR_RISK: float = 500.0

# Allowlist of symbols this system is permitted to trade.
# Keeps the system from accidentally issuing tickets for thinly-traded names,
# OTC stocks, or symbols entered via typo.  Any ticker not in this set is
# rejected regardless of signal strength or governance opinion.
ALLOWED_TICKERS: frozenset = frozenset({
    "AAPL", "MSFT", "NVDA", "TSLA",
    "GOOGL", "AMZN", "META",
    "SPY", "QQQ",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_ticket(ticket: Ticket) -> list[str]:
    """
    Run structural safety checks on *ticket*.

    Returns a list of human-readable violation strings.  An empty list means
    all checks passed.  The coordinator should refuse to proceed if this list
    is non-empty.

    Checks
    ------
    1. Ticker is in the ALLOWED_TICKERS allowlist.
    2. Size is a positive integer.
    3. Stop is on the correct side of entry.
    4. Target is on the correct side of entry.
    5. Confidence is within [0, 1].
    6. Dollar risk (stop distance × size) is below MAX_DOLLAR_RISK.
    """
    violations: list[str] = []

    # --- 1. Ticker allowlist ---
    if ticket.ticker not in ALLOWED_TICKERS:
        violations.append(
            f"Ticker '{ticket.ticker}' is not in the safety allowlist "
            f"({', '.join(sorted(ALLOWED_TICKERS))})."
        )

    # --- 2. Positive integer size ---
    if not isinstance(ticket.size, int) or ticket.size < 1:
        violations.append(
            f"Size must be a positive integer; got {ticket.size!r}."
        )

    # --- 3. Stop on correct side of entry ---
    if ticket.direction == "buy" and ticket.stop >= ticket.entry:
        violations.append(
            f"BUY stop (${ticket.stop}) must be below entry (${ticket.entry})."
        )
    elif ticket.direction == "sell" and ticket.stop <= ticket.entry:
        violations.append(
            f"SELL stop (${ticket.stop}) must be above entry (${ticket.entry})."
        )

    # --- 4. Target on correct side of entry ---
    if ticket.direction == "buy" and ticket.target <= ticket.entry:
        violations.append(
            f"BUY target (${ticket.target}) must be above entry (${ticket.entry})."
        )
    elif ticket.direction == "sell" and ticket.target >= ticket.entry:
        violations.append(
            f"SELL target (${ticket.target}) must be below entry (${ticket.entry})."
        )

    # --- 5. Confidence in [0, 1] ---
    if not (0.0 <= ticket.confidence <= 1.0):
        violations.append(
            f"Confidence {ticket.confidence} is outside [0, 1]."
        )

    # --- 6. Dollar risk cap ---
    dollar_risk = abs(ticket.entry - ticket.stop) * ticket.size
    if dollar_risk > MAX_DOLLAR_RISK:
        violations.append(
            f"Dollar risk ${dollar_risk:.2f} exceeds hard cap "
            f"${MAX_DOLLAR_RISK:.2f} "
            f"(stop distance ${abs(ticket.entry - ticket.stop):.4f} "
            f"× {ticket.size} shares)."
        )

    return violations
