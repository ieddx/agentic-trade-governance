"""
test_ticket.py — smoke tests for the Ticket dataclass and build_ticket helper.

These tests run without any network access or API keys.  They verify that:
  - build_ticket returns a valid Ticket object.
  - Stop and target prices are computed from the correct barrier constants.
  - Direction validation rejects bad input.

Run with:  pytest tests/
"""

import pytest
from finance_core.ticket import (
    Ticket,
    build_ticket,
    TAKE_PROFIT_BPS,
    STOP_LOSS_BPS,
)


def test_ticket_is_dataclass():
    """Ticket can be constructed directly (no magic required)."""
    t = Ticket(
        ticker="AAPL",
        direction="buy",
        entry=200.0,
        stop=198.5,
        target=203.0,
        confidence=0.65,
        size=50,
    )
    assert t.ticker == "AAPL"
    assert t.direction == "buy"


def test_build_ticket_buy():
    """For a buy ticket, target > entry > stop."""
    t = build_ticket(ticker="AAPL", direction="buy", entry=200.0, confidence=0.75)

    assert isinstance(t, Ticket)
    assert t.direction == "buy"
    assert t.entry == 200.0
    assert t.target > t.entry, "target must be above entry for a buy"
    assert t.stop < t.entry,   "stop must be below entry for a buy"
    assert t.size >= 1


def test_build_ticket_sell():
    """For a sell ticket, target < entry < stop."""
    t = build_ticket(ticker="AAPL", direction="sell", entry=200.0, confidence=0.55)

    assert t.direction == "sell"
    assert t.target < t.entry, "target must be below entry for a sell"
    assert t.stop > t.entry,   "stop must be above entry for a sell"


def test_take_profit_distance():
    """Target is exactly TAKE_PROFIT_BPS above entry (buy side)."""
    entry = 100.0
    t = build_ticket(ticker="TEST", direction="buy", entry=entry, confidence=0.5)
    expected_target = round(entry * (1 + TAKE_PROFIT_BPS / 10_000), 4)
    assert t.target == expected_target


def test_stop_loss_distance():
    """Stop is exactly STOP_LOSS_BPS below entry (buy side)."""
    entry = 100.0
    t = build_ticket(ticker="TEST", direction="buy", entry=entry, confidence=0.5)
    expected_stop = round(entry * (1 - STOP_LOSS_BPS / 10_000), 4)
    assert t.stop == expected_stop


def test_invalid_direction():
    """build_ticket raises ValueError for an unknown direction."""
    with pytest.raises(ValueError, match="direction must be"):
        build_ticket(ticker="AAPL", direction="hold", entry=200.0, confidence=0.5)


def test_confidence_range():
    """Confidence on the returned ticket matches the input."""
    t = build_ticket(ticker="AAPL", direction="buy", entry=200.0, confidence=0.42)
    assert 0.0 <= t.confidence <= 1.0
    assert t.confidence == 0.42


def test_size_proportional_to_notional():
    """Size increases as notional increases (coarse check)."""
    t_small = build_ticket("AAPL", "buy", 200.0, 0.5, notional=1_000.0)
    t_large = build_ticket("AAPL", "buy", 200.0, 0.5, notional=100_000.0)
    assert t_large.size > t_small.size
