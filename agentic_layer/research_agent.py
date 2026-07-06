"""
research_agent.py — market research agent for the agentic layer.

ROLE IN THE SYSTEM
------------------
Before the governance agent approves or vetoes a trade, it helps to know what
the market has actually been doing recently.  This agent fetches quantitative
market context (volatility, volume trend, momentum, price position) and asks
Gemini to interpret those numbers in plain language for a risk reviewer.

The research report is designed to be fed into the governance agent alongside
the Ticket so that the governance decision is grounded in market conditions,
not just the trade parameters in isolation.

IMPORTANT BOUNDARY — like governance_agent.py, this agent is read-only.  It
interprets and reports; it does not change the Ticket's entry, stop, target,
or size.  Those values are owned by the finance core.

FLOW
----
  Ticket (from finance_core)
      │
      ▼
  research_ticket()
      │  _get_context_via_mcp(ticker) → raw metrics dict   [MCP tool call]
      │    └─ falls back to get_market_context() if MCP fails
      │  _build_prompt(ticker, metrics) → prompt string
      │  generate(prompt) → raw Gemini text                [LLM call]
      │  _parse_response(text, metrics) → ResearchReport
      ▼
  ResearchReport
      │  summary  (str)       — Gemini's plain-language analyst note
      │  metrics  (dict)      — raw numbers from get_market_context
      └─ concerns (list[str]) — specific conditions worth flagging

MCP INTEGRATION
---------------
The market-context metrics are fetched by calling mcp_server.server over the
Model Context Protocol (MCP) rather than importing get_market_context directly.
The MCP server runs as a subprocess (stdio transport); this agent uses a thin
client helper (mcp_server/client.py) that speaks the MCP protocol and returns
JSON.  If the MCP call fails for any reason, the agent falls back to the direct
import so the overall pipeline remains robust.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

# Shared Gemini helpers: model name, generate(), JSON fence-stripping + parse.
from agentic_layer._gemini import generate, parse_json_response, GEMINI_MODEL

# Direct import used as fallback when the MCP call fails.
from finance_core.market_context import get_market_context

# Path to the mcp client helper and the Python 3.11+ interpreter that runs it.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Prefer python3.11 (required by the mcp package); fall back to the current
# interpreter so tests that mock subprocess still work.
_MCP_PYTHON = os.environ.get("MCP_PYTHON", "python3.11")


# ---------------------------------------------------------------------------
# ResearchReport — the structured output of the research agent.
# ---------------------------------------------------------------------------

@dataclass
class ResearchReport:
    """
    The result of running a Ticket through the research agent.

    Fields
    ------
    summary  : Gemini's plain-language interpretation of recent market
               conditions, written as a concise analyst note (3-5 sentences).
    metrics  : Raw dict returned by get_market_context — the numeric context
               a human reviewer or the governance agent can inspect directly.
    concerns : Specific conditions worth flagging before entering the trade
               (e.g. "high hourly volatility", "price at recent high",
               "thin volume").  Empty list if conditions are unremarkable.
    """
    summary:  str
    metrics:  dict
    concerns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MCP market-context call
# ---------------------------------------------------------------------------

def _get_context_via_mcp(ticker: str, as_of: str = "") -> dict:
    """
    Fetch market context by calling the get_market_context MCP tool.

    Launches mcp_server.client as a subprocess under Python 3.11+, which
    in turn spawns mcp_server.server and communicates via the MCP stdio
    transport.  Returns the parsed dict on success.

    Raises RuntimeError if the subprocess exits non-zero or returns
    unparseable output — callers should fall back to the direct import.
    """
    cmd = [_MCP_PYTHON, "-m", "mcp_server.client", ticker]
    if as_of:
        cmd.append(as_of)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
        timeout=30,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"mcp_server.client exited {proc.returncode}: {proc.stderr.strip()}"
        )

    return json.loads(proc.stdout.strip())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prompt(ticker: str, metrics: dict) -> str:
    """
    Build a prompt asking Gemini to interpret market context metrics.

    We pass all the raw numbers explicitly and include a plain-English
    definition of each one so Gemini doesn't have to infer units or
    conventions.  The task is interpretation and concern-flagging only —
    Gemini is told explicitly not to suggest trade parameter changes.
    """
    # Render metrics as aligned key: value lines for readability in the prompt.
    metric_lines = "\n".join(f"  {k:<26} {v}" for k, v in metrics.items())

    prompt = f"""
You are a market-research analyst assisting a risk-governance team.
You have been given recent quantitative market metrics for {ticker} derived
from the last {metrics.get('lookback_bars', 24)} hourly bars of price and
volume data.

Your job is to interpret these metrics in plain language and flag any
conditions that a risk reviewer should be aware of before approving a trade.

--- MARKET METRICS ---
{metric_lines}

--- METRIC DEFINITIONS ---
recent_volatility_pct  std dev of hourly % returns (higher = bigger hourly swings)
volume_ratio           latest bar volume / avg volume  (>1 = high, <0.5 = thin)
momentum_pct           total % price change over the lookback window
price_position         0.0 = at recent low, 1.0 = at recent high

--- YOUR TASK ---
1. Write a concise analyst note (3-5 sentences) interpreting these conditions
   for someone about to enter a short-term trade in this stock.
2. List any specific concerns relevant to entering a trade right now
   (e.g. "very high hourly volatility", "price at recent high",
   "counter-trend momentum", "thin volume").
   Leave the list empty if conditions look unremarkable.

Do NOT suggest changes to any trade parameters (entry, stop, target, size).
Return ONLY a JSON object with exactly these keys — no markdown fences:
  "summary"  : your analyst note as a single string
  "concerns" : a JSON array of short concern strings (can be empty [])
""".strip()

    return prompt


def _parse_response(text: str, metrics: dict) -> ResearchReport:
    """
    Parse Gemini's JSON response into a ResearchReport.

    Falls back to a ResearchReport that surfaces the raw response text in
    summary if JSON parsing fails, so a malformed LLM reply never crashes
    the pipeline.  The raw metrics are always attached regardless.
    """
    try:
        data = parse_json_response(text)
        return ResearchReport(
            summary=str(data.get("summary", "(no summary provided)")),
            metrics=metrics,
            concerns=[str(c) for c in data.get("concerns", [])],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return ResearchReport(
            summary=(
                f"Research agent failed to parse Gemini's response "
                f"({type(exc).__name__}: {exc}).\n\nRaw response:\n{text}"
            ),
            metrics=metrics,
            concerns=["parse_error"],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def research_ticket(ticket, bars=None) -> ResearchReport:
    """
    Produce a ResearchReport for the ticker named in *ticket*.

    Steps:
      1. Pull raw market metrics from the bar data (no extra API calls).
      2. Ask Gemini to interpret those metrics as an analyst note.
      3. Return a ResearchReport with the summary, raw metrics, and concerns.

    Parameters
    ----------
    ticket : finance_core.ticket.Ticket
    bars   : optional pre-loaded bar DataFrame from a coordinator.  Forwarded
             to get_market_context() so that the research and the trade signal
             are computed on the same data snapshot.

    Returns
    -------
    ResearchReport
    """
    # Step 1: fetch market-context metrics via the MCP tool.
    # The MCP server is launched as a subprocess; on failure we fall back to
    # the direct import so the pipeline stays resilient.
    #
    # Note: when the coordinator passes pre-loaded bars (snapshot consistency),
    # those bars were already written to the per-symbol CSV cache by load_bars,
    # so the MCP server reads the same data when it calls load_bars internally.
    try:
        metrics = _get_context_via_mcp(ticker=ticket.ticker)
        print(f"[research] MCP tool called: get_market_context (ticker={ticket.ticker})")
    except Exception as exc:
        print(f"[research] MCP call failed ({exc}); falling back to direct import.")
        metrics = get_market_context(ticker=ticket.ticker, bars=bars)

    # Step 2: ask Gemini to narrate the numbers.
    prompt   = _build_prompt(ticker=ticket.ticker, metrics=metrics)
    print(f"[research] Sending {ticket.ticker} market context to Gemini ({GEMINI_MODEL}) …")
    raw_text = generate(prompt)
    print("[research] Gemini responded.")

    # Step 3: parse into structured output.
    return _parse_response(raw_text, metrics)


# ---------------------------------------------------------------------------
# Manual smoke-test: python -m agentic_layer.research_agent
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
    print("STEP 2: run the research agent")
    print("=" * 60)

    report = research_ticket(ticket)

    print()
    print("=" * 60)
    print("TICKET")
    print("=" * 60)
    print(f"  Ticker:     {ticket.ticker}")
    print(f"  Direction:  {ticket.direction.upper()}")
    print(f"  Entry:      ${ticket.entry}")
    print(f"  Confidence: {ticket.confidence:.2%}")

    print()
    print("=" * 60)
    print("MARKET METRICS")
    print("=" * 60)
    for k, v in report.metrics.items():
        print(f"  {k:<28} {v}")

    print()
    print("=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(f"  Summary:\n    {report.summary}")
    print()
    if report.concerns:
        print(f"  Concerns:")
        for c in report.concerns:
            print(f"    • {c}")
    else:
        print("  Concerns: (none)")
