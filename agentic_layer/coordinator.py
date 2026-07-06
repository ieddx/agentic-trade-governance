"""
coordinator.py — orchestrates the full agentic governance workflow.

This is the top-level entry point for running a complete trade review cycle.
It wires together five stages in strict order:

  STAGE 1  finance_core    — load market data, compute signal, produce Ticket
  STAGE 2  safety          — structural validation before any Gemini quota is spent
  STAGE 3  research_agent  — interpret market conditions via MCP, produce ResearchReport
  STAGE 4  governance_agent — weigh ticket + research, produce GovernanceDecision
  STAGE 5  human gate      — explicit terminal confirmation before any execution
  AUDIT    audit            — append-only log of the full run (always runs)

WHY THE COORDINATOR CONTROLS DATA LOADING
------------------------------------------
Both Stage 1 (produce_ticket) and Stage 3 (research_ticket) need bar data.
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

HUMAN-IN-THE-LOOP GATE (Stage 5)
----------------------------------
The governance agent's "approved" decision does NOT auto-execute a trade.
Execution always requires explicit human confirmation via a terminal prompt.

Why this matters in real trading systems:
  1. Regulatory requirement.  The SEC, FINRA, and comparable bodies require
     "meaningful human review" for algorithmic trading systems, particularly
     for novel or high-risk signals.  An LLM approval alone does not satisfy
     this standard.
  2. Model failure mode.  An LLM that approves a trade may be hallucinating,
     misreading numbers, or reasoning from stale context.  A human glancing at
     the summary has a last-resort chance to catch an obvious error.
  3. Accountability.  A human pressing "y" creates a named decision point in
     the audit log — who approved what, when, and after seeing which summary.
     "The algorithm did it" is not an acceptable audit response.

In this paper-trading system the execution step is intentionally not
implemented.  The human gate is wired in anyway so the approval flow is tested
and documented before any real execution is connected.

FLOW
----
  run_governance_workflow(ticker, as_of)
      │
      ├─ LOAD    load_bars(ticker)            → bars  (one snapshot)
      │
      ├─ STAGE1  produce_ticket(bars=bars)    → Ticket   (or None → exit)
      │
      ├─ STAGE2  validate_ticket(ticket)      → violations
      │           any violations → audit + exit
      │
      ├─ STAGE3  research_ticket(bars=bars)   → ResearchReport
      │
      ├─ STAGE4  review_ticket(ticket,
      │                research_report=report) → GovernanceDecision
      │
      ├─ STAGE5  _human_gate(ticket, report,
      │                decision)              → "approved" | "rejected"
      │
      └─ AUDIT   log_run(...)                 → logs/audit.jsonl
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# finance_core: data loading + ticket production
from finance_core.data_loader import load_bars
from finance_core.core import produce_ticket
from finance_core.ticket import Ticket

# agentic_layer: safety, research, governance, audit
from agentic_layer.safety import validate_ticket, MAX_DOLLAR_RISK, ALLOWED_TICKERS
from agentic_layer.research_agent import research_ticket, ResearchReport
from agentic_layer.governance_agent import review_ticket, GovernanceDecision
from agentic_layer.audit import log_run


# ---------------------------------------------------------------------------
# WorkflowResult — bundles all stage outputs into one return value.
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
    human_decision      : "approved" | "rejected" — the final human gate.
    """
    ticker:              str
    ticket:              Ticket
    research_report:     ResearchReport
    governance_decision: GovernanceDecision
    human_decision:      str


# ---------------------------------------------------------------------------
# Human-in-the-loop gate
# ---------------------------------------------------------------------------

def _human_gate(
    ticket: Ticket,
    report: ResearchReport,
    decision: GovernanceDecision,
) -> str:
    """
    Present a concise summary and ask for explicit human confirmation.

    Returns "approved" or "rejected".

    HUMAN-IN-THE-LOOP RATIONALE
    ----------------------------
    This gate runs regardless of what the governance agent decided.  Governance
    "approved" does NOT auto-execute.  The governance agent may be wrong, and a
    human looking at the summary has the final say.

    In a real trading system this pattern is enforced at the infrastructure
    level (the execution API requires a signed human token), not just in
    application code — so a bug in the coordinator cannot bypass it.  Here it
    is implemented in the coordinator as the paper-trade analogue.
    """
    dollar_risk = abs(ticket.entry - ticket.stop) * ticket.size
    gov_status  = "APPROVED ✓" if decision.approved else "VETOED ✗"

    print()
    print("=" * 60)
    print("HUMAN APPROVAL GATE")
    print("=" * 60)
    print(f"  Ticker:          {ticket.ticker}")
    print(f"  Direction:       {ticket.direction.upper()}")
    print(f"  Size:            {ticket.size} shares")
    print(f"  Entry:           ${ticket.entry}")
    print(f"  Stop:            ${ticket.stop}  (−{abs(ticket.entry - ticket.stop):.4f})")
    print(f"  Target:          ${ticket.target}  (+{abs(ticket.target - ticket.entry):.4f})")
    print(f"  Dollar at risk:  ${dollar_risk:.2f}")
    print(f"  Confidence:      {ticket.confidence:.2%}")
    print()
    print(f"  Governance:      {gov_status}")
    print(f"  Reasoning:       {decision.reasoning}")
    if decision.flags:
        print(f"  Gov flags:")
        for f in decision.flags:
            print(f"    • {f}")
    if report.concerns:
        print(f"  Research concerns:")
        for c in report.concerns:
            print(f"    • {c}")
    print()
    print("  [PAPER MODE — no real order will be placed]")
    print()

    try:
        raw = input("  Approve execution? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raw = ""

    if raw == "y":
        print()
        print("  HUMAN OVERRIDE: approved")
        return "approved"
    else:
        print()
        print("  HUMAN OVERRIDE: rejected")
        return "rejected"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_governance_workflow(
    ticker: str = "AAPL",
    as_of: Optional[datetime] = None,
) -> Optional[WorkflowResult]:
    """
    Run the complete five-stage governance workflow for *ticker*.

    Returns a WorkflowResult on completion of the human gate, or None if the
    finance core produced no ticket (signal below MIN_CONFIDENCE) or safety
    validation failed.

    Parameters
    ----------
    ticker : equity symbol to analyse (default "AAPL")
    as_of  : tz-aware datetime; when provided, bars end at this point so the
             full workflow runs as if executed at that historical moment.
             When None, current market data is used.
    """
    as_of_str = as_of.isoformat() if as_of else None

    # Shared audit-log fields — populated incrementally as stages complete.
    _ticket_fields: Optional[dict]   = None
    _safety_passed: bool             = False
    _safety_violations: list[str]    = []
    _research_summary: Optional[str] = None
    _research_concerns: Optional[list] = None
    _gov_approved: Optional[bool]    = None
    _gov_reasoning: Optional[str]    = None
    _gov_flags: Optional[list]       = None
    _human_decision: str             = "not_reached"
    _error: Optional[str]            = None

    try:
        # -------------------------------------------------------------------
        # DATA LOAD
        # -------------------------------------------------------------------
        _banner("DATA LOAD")
        bars, feed_used = load_bars(symbol=ticker, as_of=as_of)
        window_label = f"as-of {as_of.isoformat()}" if as_of else "current"
        print(f"[coordinator] Loaded {len(bars)} bars for {ticker} "
              f"(feed: {feed_used}, window: {window_label}).")

        # -------------------------------------------------------------------
        # STAGE 1 — deterministic signal + Ticket
        # -------------------------------------------------------------------
        _banner("STAGE 1 — FINANCE CORE")
        ticket = produce_ticket(symbol=ticker, bars=bars)
        if ticket is None:
            print(
                "[coordinator] Finance core produced no ticket "
                "(signal confidence below threshold).  Workflow aborted."
            )
            log_run(
                ticker=ticker, as_of=as_of_str,
                ticket_fields=None,
                safety_passed=False, safety_violations=["no_ticket_produced"],
                research_summary=None, research_concerns=None,
                governance_approved=None, governance_reasoning=None,
                governance_flags=None,
                human_decision="not_reached",
            )
            return None

        _ticket_fields = dataclasses.asdict(ticket)

        # -------------------------------------------------------------------
        # STAGE 2 — safety validation
        # -------------------------------------------------------------------
        _banner("STAGE 2 — SAFETY VALIDATION")
        violations = validate_ticket(ticket)

        if violations:
            _safety_passed     = False
            _safety_violations = violations
            print("[coordinator] *** SAFETY VIOLATION — workflow aborted ***")
            for v in violations:
                print(f"  ✗  {v}")
            log_run(
                ticker=ticker, as_of=as_of_str,
                ticket_fields=_ticket_fields,
                safety_passed=False, safety_violations=violations,
                research_summary=None, research_concerns=None,
                governance_approved=None, governance_reasoning=None,
                governance_flags=None,
                human_decision="not_reached",
            )
            return None

        _safety_passed = True
        print("[coordinator] Safety validation passed ✓")
        print(f"  Ticker '{ticket.ticker}' is in the allowlist.")
        dollar_risk = abs(ticket.entry - ticket.stop) * ticket.size
        print(f"  Dollar risk ${dollar_risk:.2f} is within the ${MAX_DOLLAR_RISK:.0f} cap.")
        print(f"  Stop/target sides correct for {ticket.direction.upper()}.")
        print(f"  Confidence {ticket.confidence:.2%} is within [0, 1].")

        # -------------------------------------------------------------------
        # STAGE 3 — market research (via MCP)
        # -------------------------------------------------------------------
        _banner("STAGE 3 — RESEARCH AGENT (via MCP)")
        report = research_ticket(ticket, bars=bars)
        _research_summary  = report.summary
        _research_concerns = report.concerns

        # -------------------------------------------------------------------
        # STAGE 4 — governance decision
        # -------------------------------------------------------------------
        _banner("STAGE 4 — GOVERNANCE AGENT")
        decision = review_ticket(ticket, research_report=report)
        _gov_approved  = decision.approved
        _gov_reasoning = decision.reasoning
        _gov_flags     = decision.flags

        # -------------------------------------------------------------------
        # STAGE 5 — human gate (always runs, regardless of governance verdict)
        # -------------------------------------------------------------------
        _human_decision = _human_gate(ticket, report, decision)

        # -------------------------------------------------------------------
        # Summary print
        # -------------------------------------------------------------------
        result = WorkflowResult(
            ticker=ticker,
            ticket=ticket,
            research_report=report,
            governance_decision=decision,
            human_decision=_human_decision,
        )
        _print_summary(result, feed_used=feed_used)

    except Exception as exc:
        _error = f"{type(exc).__name__}: {exc}"
        print(f"[coordinator] Unhandled error: {_error}")
        raise

    finally:
        # Audit log always runs — even on exceptions.
        log_run(
            ticker=ticker,
            as_of=as_of_str,
            ticket_fields=_ticket_fields,
            safety_passed=_safety_passed,
            safety_violations=_safety_violations,
            research_summary=_research_summary,
            research_concerns=_research_concerns,
            governance_approved=_gov_approved,
            governance_reasoning=_gov_reasoning,
            governance_flags=_gov_flags,
            human_decision=_human_decision,
            error=_error,
        )
        print(f"\n[coordinator] Audit entry written → logs/audit.jsonl")

    return result


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _print_summary(result: WorkflowResult, feed_used: str = "unknown") -> None:
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

    # --- Human decision ---
    print()
    print("HUMAN DECISION")
    print(f"  {result.human_decision.upper()}")


# ---------------------------------------------------------------------------
# Entry point: python -m agentic_layer.coordinator [--ticker SYM] [--as-of DATE]
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
        as_of_dt = parsed.replace(tzinfo=_ET)

    run_governance_workflow(ticker=args.ticker, as_of=as_of_dt)
