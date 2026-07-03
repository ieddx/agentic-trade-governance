# Agentic Trade Governance

A Python project that separates deterministic signal generation from
agentic governance and execution.

## Structure

| Package | Role |
|---|---|
| `finance_core` | Loads market data, computes signals, produces a `Ticket` |
| `agentic_layer` | Risk checks, LLM reasoning, human approval (coming soon) |
| `mcp_server` | MCP server exposing finance tools to external agents (coming soon) |

## Quickstart

```bash
cp .env.example .env   # add your Alpaca and Gemini keys
pip install -r requirements.txt
python -m finance_core.core
```
