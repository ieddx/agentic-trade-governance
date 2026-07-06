"""
mcp_server/client.py — thin MCP client for get_market_context.

Called as a subprocess by research_agent.py:

    python3.11 -m mcp_server.client TICKER [AS_OF]

Connects to mcp_server.server over stdio, calls the get_market_context tool,
and writes the resulting JSON dict to stdout.  Exits 0 on success, 1 on error
(error message on stderr).

This script runs under Python 3.11+ (required by the mcp package).  The
research agent (which may run under Python 3.9) launches it as a subprocess
and parses its stdout as JSON.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Project root on path so server.py can import finance_core even from a
# subprocess launched in a different working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _call(ticker: str, as_of: str) -> dict:
    server_params = StdioServerParameters(
        command=sys.executable,          # same python3.11 that runs this client
        args=["-m", "mcp_server.server"],
        cwd=_PROJECT_ROOT,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            arguments: dict = {"ticker": ticker}
            if as_of:
                arguments["as_of"] = as_of

            result = await session.call_tool("get_market_context", arguments)

            # result.content is a list; the server returns one TextContent item.
            raw = result.content[0].text  # type: ignore[attr-defined]
            return json.loads(raw)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3.11 -m mcp_server.client TICKER [AS_OF]", file=sys.stderr)
        sys.exit(1)

    ticker = sys.argv[1]
    as_of  = sys.argv[2] if len(sys.argv) > 2 else ""

    try:
        context = asyncio.run(_call(ticker, as_of))
        print(json.dumps(context))
    except Exception as exc:
        print(f"MCP client error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
