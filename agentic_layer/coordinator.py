"""
coordinator.py — orchestrates the full agentic governance workflow.

This is the top-level entry point for running a complete trade review cycle.
It wires together three stages that were previously independent:

  STAGE 1  finance_core   — load market data, compute signal, produce Ticket
  STAGE 2  research_agent — interpret market conditions, produce ResearchReport
  STAGE 3  governance_agent — weigh ticket + research, produce GovernanceDecision

WHY THE COORDINATOR CONTROLS DATA LOADING
------------------------------------------
Both Stage 1 (produce_ticket) and Stage 2 (research_ticket) need bar data.
If each loaded bars independently, there is a risk of temporal inconsistency:
  - The cache could refresh between the two calls, giving them different
    snapshots of the market.
  - In a live-data setting this would mean the trade signal was computed on
    bars up to time T, while the research metrics covered bars up to T+ε.
  - The governance agent would then be comparing numbers derived from
    different market states — potentially approving a trade whose market
    context has already shifted.

The coordinator loads bars ONCE and passes the same DataFrame to both stages,
guaranteeing that every piece of analysis in a single workflow run describes
the same market moment.

FLOW
----
  run_governance_workflow(ticker)
      │
      ├─ LOAD   load_bars(ticker)           → bars  (one snapshot)
      │
      ├─ STAGE1 produce_ticket(bars=bars)   → Ticket   (or None if no signal)
      │
      ├─ STAGE2 research_ticket(bars=bars)  → ResearchReport
      │
      ├─ STAGE3 review_ticket(ticket,
      │                research_report=report) → GovernanceDecision
      │
      └─ RETURN WorkflowResult(ticket, research_report, governance_decision)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# finance_core: data loading + ticket production
from finance_core.data_loader import load_bars
from finance_core.core import produce_ticket
from finance_core.ticket import Ticket

# agentic_layer: research and governance agents
from agentic_layer.research_agent import research_ticket, ResearchReport
from agentic_layer.governance_agent import review_ticket, GovernanceDecision


# ---------------------------------------------------------------------------
# WorkflowResult — bundles all three stage outputs into one return value.
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """
    The combined output of a single run of run_governance_workflow().

    Fields
    ------
    ticker              : the equity symbol that was analysed.
    ticket              : the trade proposal from the finance core.
    research_report     : market-context analysis from the research agent.
    governance_decision : approve/veto decision from the governance agent.
    """
    ticker:              str
    ticket:              Ticket
    research_report:     ResearchReport
    governance_decision: GovernanceDecision


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_governance_workflow(
    ticker: str = "AAPL",
    as_of: Optional[datetime] = None,
) -> WorkflowResult | None:
    """
    Run the complete three-stage governance workflow for *ticker*.

    Returns a WorkflowResult on success, or None if the finance core produced
    no ticket (signal strength below MIN_CONFIDENCE threshold).

    Parameters
    ----------
    ticker : equity symbol to analyse (default "AAPL")
    as_of  : tz-aware datetime; when provided, bars end at this point so the
             full workflow runs as if executed at that historical moment.
             When None, current market data is used.
    """
    _banner("DATA LOAD")

    # -----------------------------------------------------------------------
    # Load bars ONCE.  Both produce_ticket and research_ticket accept a
    # pre-loaded DataFrame via their `bars` parameter so neither will call
    # load_bars() again.  This is the snapshot-consistency guarantee described
    # in the module docstring.
    # -----------------------------------------------------------------------
    bars, feed_used = load_bars(symbol=ticker, as_of=as_of)
    window_label = f"as-of {as_of.isoformat()}" if as_of else "current"
    print(f"[coordinator] Loaded {len(bars)} bars for {ticker} "
          f"(feed: {feed_used}, window: {window_label}).")

    # -----------------------------------------------------------------------
    # Stage 1 — deterministic signal + Ticket
    # -----------------------------------------------------------------------
    _banner("STAGE 1 — FINANCE CORE")

    ticket = produce_ticket(symbol=ticker, bars=bars)
    if ticket is None:
        print(
            "[coordinator] Finance core produced no ticket "
            "(signal confidence below threshold).  Workflow aborted."
        )
        return None

    # -----------------------------------------------------------------------
    # Stage 2 — market research
    # The research agent receives the same `bars` snapshot that produced the
    # ticket.  Metrics therefore describe exactly the market state the signal
    # was computed on.
    # -----------------------------------------------------------------------
    _banner("STAGE 2 — RESEARCH AGENT")

    report = research_ticket(ticket, bars=bars)

    # -----------------------------------------------------------------------
    # Stage 3 — governance decision
    # The research report is passed alongside the ticket so the governance
    # prompt includes both the trade's risk math AND the market context.
    # -----------------------------------------------------------------------
    _banner("STAGE 3 — GOVERNANCE AGENT")

    decision = review_ticket(ticket, research_report=report)

    # Bundle everything and print a human-readable summary.
    result = WorkflowResult(
        ticker=ticker,
        ticket=ticket,
        research_report=report,
        governance_decision=decision,
    )
    _print_summary(result, feed_used=feed_used)
    return result


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    """Print a section separator."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _print_summary(result: WorkflowResult, feed_used: str = "unknown") -> None:
    """Print a clean, readable end-of-run summary of all three stages."""
    t  = result.ticket
    rr = result.research_report
    gd = result.governance_decision

    _banner("WORKFLOW SUMMARY")

    # --- Ticket ---
    print("TICKET")
    print(f"  Ticker:     {t.ticker}")
    print(f"  Direction:  {t.direction.upper()}")
    print(f"  Entry:      ${t.entry}")
    print(f"  Stop:       ${t.stop}")
    print(f"  Target:     ${t.target}")
    print(f"  Confidence: {t.confidence:.2%}")
    print(f"  Size:       {t.size} shares")
    bd = t.confidence_breakdown
    if bd:
        print(f"  Confidence breakdown (deterministic, no LLM):")
        print(f"    MA separation:    {bd.get('ma_separation_pct')}% of price")
        print(f"    Recent vol:       {bd.get('recent_volatility_pct')}% / hr")
        print(f"    Normalized score: {bd.get('normalized_score')} (sep / vol)")
        print(f"  Barrier distances (vol-scaled):")
        print(f"    Stop:   {bd.get('stop_distance_pct')}%  "
              f"({bd.get('stop_vol_multiple')}× vol)")
        print(f"    Target: {bd.get('target_distance_pct')}%  (2× stop, 2:1 R:R)")
    print(f"  Data feed:  {feed_used}")

    # --- Research ---
    print()
    print("RESEARCH REPORT")
    m = rr.metrics
    print(f"  Volatility:     {m.get('recent_volatility_pct')}% / hour")
    print(f"  Volume ratio:   {m.get('volume_ratio')}x avg")
    print(f"  Momentum:       {m.get('momentum_pct')}% over {m.get('lookback_bars')} bars")
    print(f"  Price position: {m.get('price_position')} "
          f"(range ${m.get('recent_low')} – ${m.get('recent_high')})")
    print(f"  Summary:  {rr.summary}")
    if rr.concerns:
        print(f"  Concerns:")
        for c in rr.concerns:
            print(f"    • {c}")
    else:
        print("  Concerns: (none)")

    # --- Governance ---
    print()
    print("GOVERNANCE DECISION")
    status = "APPROVED ✓" if gd.approved else "VETOED ✗"
    print(f"  Status:    {status}")
    print(f"  Reasoning: {gd.reasoning}")
    if gd.flags:
        print(f"  Flags:")
        for f in gd.flags:
            print(f"    • {f}")
    else:
        print("  Flags:     (none)")


# ---------------------------------------------------------------------------
# Entry point: python -m agentic_layer.coordinator [--ticker SYM] [--as-of DATE]
#
# --ticker SYMBOL          Which stock to analyse (default: AAPL).
# --as-of "YYYY-MM-DD HH:00"  Run as if it were this historical moment.
#   Accepts "YYYY-MM-DD HH:MM", "YYYY-MM-DD HH:00", or "YYYY-MM-DD" (date
#   alone defaults to 16:00 ET — NYSE close — on that day).
#   When omitted, current SIP data is used.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import zoneinfo

    _ET = zoneinfo.ZoneInfo("America/New_York")

    parser = argparse.ArgumentParser(
        description="Run the agentic trade-governance workflow."
    )
    parser.add_argument(
        "--ticker", default="AAPL",
        help="Equity symbol to analyse (default: AAPL)",
    )
    parser.add_argument(
        "--as-of", dest="as_of", default=None,
        metavar="DATETIME",
        help=(
            'Historical cutoff, e.g. "2025-09-15 14:00" or "2025-09-15". '
            "Date-only defaults to 16:00 ET (NYSE close). "
            "When omitted, current SIP data is used."
        ),
    )
    args = parser.parse_args()

    as_of_dt: Optional[datetime] = None
    if args.as_of:
        raw = args.as_of.strip()
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:00", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt)
                if fmt == "%Y-%m-%d":
                    parsed = parsed.replace(hour=16, minute=0)
                break
            except ValueError:
                continue
        if parsed is None:
            parser.error(
                f"--as-of value {raw!r} could not be parsed. "
                'Use "YYYY-MM-DD HH:MM" or "YYYY-MM-DD".'
            )
        # Interpret as Eastern time (the natural timezone for US market hours).
        as_of_dt = parsed.replace(tzinfo=_ET)

    run_governance_workflow(ticker=args.ticker, as_of=as_of_dt)
