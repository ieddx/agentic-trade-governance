"""
governance_agent.py — the first agent in the agentic layer.

ROLE IN THE SYSTEM
------------------
The finance_core package is fully deterministic: given the same bars it always
produces the same Ticket (entry, stop, target, size, confidence).  No LLM
touches those numbers.

This agent sits one layer above.  It receives a completed Ticket and asks
Google Gemini: "Is this trade reasonable from a risk-governance perspective?"
Gemini can APPROVE or VETO, and it must explain why.

IMPORTANT BOUNDARY — the agent must NOT recompute or modify stop, target, or
size.  Those values are owned by the finance core and are the source of truth.
The agent's only authority is the approve/veto decision and the reasoning.

FLOW
----
  Ticket (from finance_core)
      │
      ▼
  review_ticket()          ← this module
      │  builds a structured prompt describing the trade
      │  sends it to Gemini  (via _gemini.generate)
      │  parses the JSON response
      ▼
  GovernanceDecision
      │  approved (bool)
      │  reasoning (str)
      └─ flags    (list[str])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Shared Gemini helpers: model name, generate(), JSON fence-stripping + parse.
# All Gemini SDK plumbing (client creation, dotenv loading, GenerateContentConfig)
# lives in _gemini.py so this file stays focused on governance logic.
from agentic_layer._gemini import generate, parse_json_response, GEMINI_MODEL


# ---------------------------------------------------------------------------
# GovernanceDecision — the structured output of the agent.
# ---------------------------------------------------------------------------

@dataclass
class GovernanceDecision:
    """
    The result of running a Ticket through the governance agent.

    Fields
    ------
    approved  : True if the agent approves the trade, False if vetoed.
    reasoning : Full natural-language explanation from the LLM.
    flags     : List of specific risk concerns the LLM identified.  Empty if
                no concerns were raised (a trade can be approved with flags).
    """
    approved:  bool
    reasoning: str
    flags:     list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prompt(ticket, research_report=None) -> str:
    """
    Build the governance prompt for Gemini.

    When *research_report* is provided the prompt gains a second section with
    market-context metrics and the analyst's concerns, and the task list gains
    two additional questions asking Gemini to weigh those conditions.

    When *research_report* is None the prompt is identical to the pre-research
    version — backward compatible with callers that don't pass context.
    """
    stop_pct    = abs(ticket.entry - ticket.stop)   / ticket.entry * 100
    target_pct  = abs(ticket.target - ticket.entry) / ticket.entry * 100
    rr_ratio    = target_pct / stop_pct if stop_pct else 0
    dollar_risk = abs(ticket.entry - ticket.stop) * ticket.size

    # --- optional research-context block ---
    # Built as plain strings before the outer f-string so their content is
    # never confused with f-string placeholders.
    if research_report is not None:
        m = research_report.metrics
        concerns_str = (
            "\n".join(f"  • {c}" for c in research_report.concerns)
            if research_report.concerns else "  (none)"
        )
        research_section = (
            "\n--- MARKET RESEARCH CONTEXT"
            " (same data snapshot as the trade signal) ---\n"
            f"Analyst summary:\n  {research_report.summary}\n"
            "\nKey metrics (last "
            f"{m.get('lookback_bars', 24)} hourly bars):\n"
            f"  Hourly volatility:  {m.get('recent_volatility_pct')}%"
            "  (std dev of hourly % returns)\n"
            f"  Volume ratio:       {m.get('volume_ratio')}x"
            "  (latest bar vs avg; <0.5=thin, >1.5=elevated)\n"
            f"  Price momentum:     {m.get('momentum_pct')}%"
            "  (total move over the window)\n"
            f"  Price position:     {m.get('price_position')}"
            "  (0.0=at recent low, 1.0=at recent high)\n"
            f"  Recent range:       "
            f"${m.get('recent_low')} – ${m.get('recent_high')}\n"
            f"\nResearch concerns flagged:\n{concerns_str}\n"
        )
        research_task = (
            "\n5. Do the market conditions SUPPORT or UNDERMINE this trade?\n"
            "   • Does momentum align with the trade direction?\n"
            "   • Is volatility high enough that the 0.75% stop could"
            " be hit by noise?\n"
            "   • Does a price position near 1.0 create reversal risk"
            " for a buy near the recent high?\n"
            "6. Do any of the research concerns listed above materially"
            " change your decision?\n"
        )
    else:
        research_section = ""
        research_task    = ""

    prompt = f"""
You are a risk-governance reviewer for an algorithmic trading system.
Your role is to approve or veto a proposed trade on risk grounds.

You must NOT suggest changes to the entry price, stop price, target price, or
position size — those are fixed by the deterministic finance core and are not
yours to modify.  Your only authority is the approve/veto decision and the
reasoning behind it.

--- PROPOSED TRADE ---
Ticker:          {ticket.ticker}
Direction:       {ticket.direction.upper()}
Entry price:     ${ticket.entry:.4f}
Stop-loss:       ${ticket.stop:.4f}  ({stop_pct:.2f}% from entry)
Take-profit:     ${ticket.target:.4f}  ({target_pct:.2f}% from entry)
Reward:risk:     {rr_ratio:.2f}:1
Confidence:      {ticket.confidence:.2%}
Position size:   {ticket.size} shares
Dollar at risk:  ${dollar_risk:.2f}  (stop distance × shares)

--- RISK PARAMETERS (set by finance core, not modifiable) ---
Stop-loss distance:    0.75% below entry  (75 bps)
Take-profit distance:  1.50% above entry  (150 bps)
Max holding period:    24 hours
{research_section}
--- YOUR TASK ---
Reason about the following:
1. Is the reward:risk ratio acceptable for a short-term momentum trade?
2. Is the confidence score plausible, or does a maxed-out score (1.0) suggest
   the signal may be overfitting or poorly calibrated?
3. Is the dollar amount at risk reasonable for an algorithmic position?
4. Are there any other risk concerns worth flagging?
{research_task}
Return a JSON object with exactly these keys — no markdown fences:
  "approved"  : true or false
  "reasoning" : a concise paragraph explaining your decision
  "flags"     : a JSON array of short strings, one per concern (can be [])
""".strip()

    return prompt


def _parse_response(text: str) -> GovernanceDecision:
    """
    Parse Gemini's JSON response into a GovernanceDecision.

    Uses parse_json_response() from _gemini.py to strip fences and parse JSON.
    Falls back to a safe vetoed decision if parsing fails so a malformed
    LLM response never crashes the pipeline.
    """
    try:
        data = parse_json_response(text)
        return GovernanceDecision(
            approved=bool(data.get("approved", False)),
            reasoning=str(data.get("reasoning", "(no reasoning provided)")),
            flags=[str(f) for f in data.get("flags", [])],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return GovernanceDecision(
            approved=False,
            reasoning=(
                f"Governance agent failed to parse Gemini's response "
                f"({type(exc).__name__}: {exc}).\n\nRaw response:\n{text}"
            ),
            flags=["parse_error"],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def review_ticket(ticket, research_report=None) -> GovernanceDecision:
    """
    Send a Ticket to Gemini for risk-governance review.

    Parameters
    ----------
    ticket          : finance_core.ticket.Ticket
    research_report : agentic_layer.research_agent.ResearchReport, optional.
                      When provided, the market context (metrics, analyst
                      summary, flagged concerns) is embedded in the governance
                      prompt so Gemini weighs market conditions alongside the
                      trade's own risk math.
                      When None, only the trade parameters are evaluated —
                      identical to the pre-research behaviour.

    Returns
    -------
    GovernanceDecision with approved, reasoning, and flags.
    """
    ctx = "with research context" if research_report is not None else "no research context"
    prompt = _build_prompt(ticket, research_report)
    print(f"[governance] Sending ticket to Gemini ({GEMINI_MODEL}) for review ({ctx}) …")
    raw_text = generate(prompt)
    print("[governance] Gemini responded.")
    return _parse_response(raw_text)


# ---------------------------------------------------------------------------
# Manual smoke-test: python -m agentic_layer.governance_agent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from finance_core.core import produce_ticket

    print("=" * 60)
    print("STEP 1: generate a Ticket from the finance core")
    print("=" * 60)

    ticket = produce_ticket()
    if ticket is None:
        print("No ticket produced (confidence below threshold). Exiting.")
        raise SystemExit(0)

    print()
    print("=" * 60)
    print("STEP 2: send Ticket to governance agent")
    print("=" * 60)

    decision = review_ticket(ticket)

    print()
    print("=" * 60)
    print("TICKET")
    print("=" * 60)
    print(f"  Ticker:     {ticket.ticker}")
    print(f"  Direction:  {ticket.direction.upper()}")
    print(f"  Entry:      ${ticket.entry}")
    print(f"  Stop:       ${ticket.stop}")
    print(f"  Target:     ${ticket.target}")
    print(f"  Confidence: {ticket.confidence:.2%}")
    print(f"  Size:       {ticket.size} shares")

    print()
    print("=" * 60)
    print("GOVERNANCE DECISION")
    print("=" * 60)
    status = "APPROVED ✓" if decision.approved else "VETOED ✗"
    print(f"  Status:    {status}")
    print(f"  Reasoning: {decision.reasoning}")
    if decision.flags:
        print(f"  Flags:")
        for f in decision.flags:
            print(f"    • {f}")
    else:
        print("  Flags:     (none)")
