"""
mcp_server/server.py — MCP server that exposes get_market_context as a tool.

WHAT IS MCP?
------------
The Model Context Protocol (MCP) is a standard protocol — originally defined
by Anthropic — that lets agents discover and call tools in a uniform way,
regardless of which agent framework or language the tool is implemented in.
An MCP server announces what tools it provides (name, description, parameter
schema) over a well-known JSON-RPC-over-stdio (or HTTP/SSE) transport.  Any
MCP client — a research agent, a governance agent, an entirely different
system — can connect and call those tools without knowing anything about the
underlying implementation.

WHY WRAP get_market_context AS AN MCP SERVER?
---------------------------------------------
1. Standardisation.  The finance_core.market_context module is already a clean
   pure-computation unit.  Wrapping it as an MCP tool makes it reachable by
   any MCP-speaking agent without copy-pasting import paths or adjusting call
   signatures for each consumer.

2. Reusability across agents.  Today the research agent calls this tool.
   Tomorrow a portfolio-risk agent, a reporting agent, or an entirely external
   system could call the same endpoint, with no changes to this server.

3. Protocol conformance.  MCP provides a typed, discoverable interface: the
   server advertises parameter types and descriptions that clients can inspect
   programmatically, making the tool self-documenting to any agent that
   connects.

TRANSPORT
---------
This server uses stdio transport — the client launches the server as a
subprocess and communicates via stdin/stdout.  This is the simplest option:
no network setup, no port management, no authentication overhead.  It is the
standard choice for local tool servers in the MCP ecosystem.

RUNNING STANDALONE (for testing)
---------------------------------
    python3.11 -m mcp_server.server

The server will wait for MCP messages on stdin and reply on stdout.  Use an
MCP-capable client or the `mcp` CLI to send test requests.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so finance_core can be imported
# even when this server is launched as a subprocess from a different cwd.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from finance_core.market_context import get_market_context


# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------

app = Server("market-context-server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_market_context",
            description=(
                "Compute recent market context for an equity ticker from its "
                "hourly OHLCV bar data. Returns volatility, volume ratio, "
                "momentum, price position, and related metrics over the last "
                "24 hourly bars. Pure computation — no trading actions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Equity symbol, e.g. 'AAPL'.",
                    },
                    "as_of": {
                        "type": "string",
                        "description": (
                            "Optional ISO-8601 datetime string for historical "
                            "mode, e.g. '2025-09-15T14:00:00-04:00'. "
                            "When omitted, current cached data is used."
                        ),
                        "default": "",
                    },
                },
                "required": ["ticker"],
            },
        )
    ]


@app.call_tool()
async def call_tool(
    name: str,
    arguments: dict,
) -> list[types.TextContent]:
    if name != "get_market_context":
        raise ValueError(f"Unknown tool: {name!r}")

    ticker = arguments["ticker"]
    as_of_str = arguments.get("as_of", "") or ""

    # Resolve optional as_of to a tz-aware datetime so load_bars can use it.
    as_of_dt = None
    if as_of_str:
        import datetime as dt
        import zoneinfo
        _ET = zoneinfo.ZoneInfo("America/New_York")
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                parsed = dt.datetime.strptime(as_of_str, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=_ET)
                as_of_dt = parsed
                break
            except ValueError:
                continue

    # Load bars separately so we can pass as_of; then feed them into
    # get_market_context to avoid a second load_bars call.
    from finance_core.data_loader import load_bars
    bars, _ = load_bars(symbol=ticker, as_of=as_of_dt)

    result: dict = get_market_context(ticker=ticker, bars=bars)

    return [types.TextContent(type="text", text=json.dumps(result))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(_main())
