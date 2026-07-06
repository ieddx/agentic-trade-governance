# Agentic Trade Governance: An LLM Risk-Review Layer for Algorithmic Trading

**Deterministic finance core proposes trades. LLM agents reason about them. A human always has the final gate.**

**Track:** Agents for Business

---

## 1. The Problem

I was looking at the output of a simple moving-average crossover signal and noticed something: the number on the screen (buy AAPL, confidence 1.0) told me nothing about whether that was actually a good moment to act. The signal fired. It always fires with the same confidence when the condition is met. It has no way to know that today's volatility is unusually high, that price is sitting at the top of its recent range, or that the stop-loss it just proposed is smaller than a typical hour's noise.

That gap is exactly what a human risk reviewer closes in real trading desks. Risk officers read the signal alongside market context and catch things the model can't see about itself. I chose to build that reviewer as a set of LLM agents instead of a person, because reading a set of numbers and reasoning about what they mean in context is a task language models are well suited for.

So I built a system with a strict boundary: code owns exact facts, LLMs own judgment. Nothing crosses that line in either direction.

## 2. Why Agents

Trading decisions have two distinct kinds of work in them. One kind is arithmetic: compute a moving average, compute a stop distance, size a position. This needs to be exact and reproducible. The other kind is interpretive: does this setup make sense given today's volatility, does the confidence score look plausible, is there a reason to hesitate that the numbers alone don't show. This second kind is judgment, and judgment benefits from weighing several factors at once and explaining the reasoning in plain language.

LLMs are strong at the second kind and weak at the first. Ask an LLM to compute a stop-loss and it will produce something plausible-looking that may be wrong. Ask it to explain why a stop-loss of 75 basis points looks too tight given recent volatility, and it does that well. So I split the work along that line: code computes, agents interpret. This is the design principle the whole system is built around.

## 3. Architecture

The system has two layers. The finance core (`finance_core/`) is pure code: it loads hourly bars from Alpaca, computes a moving-average crossover, and builds a `Ticket` with entry, stop, target, size, and a volatility-normalized confidence score. No LLM touches this layer. Every number in it is reproducible from the same input data.

The agentic layer (`agentic_layer/`) receives the ticket and runs it through a fixed pipeline, enforced in order by the coordinator (`coordinator.py`):

1. **Safety validation** (`safety.py`): deterministic checks, not LLM reasoning, because safety has to be reliable. Ticker allowlist, stop and target on the right side of entry, confidence bounds, a hard $500 dollar-risk cap.
2. **Research agent** (`research_agent.py`): fetches market context through an MCP tool call and asks Gemini (`gemini-2.5-flash`) to write a short analyst note flagging anything worth watching.
3. **Governance agent** (`governance_agent.py`): reads the ticket and the research note, then approves or vetoes with explicit reasoning and structured flags.
4. **Human gate**: the governance decision, dollar risk, and flags are shown to a person who types `y` or `N`. Governance approval never auto-executes.
5. **Audit log** (`audit.py`): every run appends one record to `logs/audit.jsonl`, written in a `finally` block so it fires even on errors.

Three rubric concepts live here directly: a multi-agent system (research and governance agents with distinct roles, orchestrated by the coordinator), an MCP server (`mcp_server/server.py` exposes `get_market_context` as a typed tool over stdio; `mcp_server/client.py` calls it from a subprocess), and layered security features (the allowlist, the human gate, and the audit trail). The coordinator loads the bar data once and passes the same snapshot to every stage, so no agent reasons about a market that has moved since another agent looked at it.

## 4. Findings: What the Agents Caught

This is the part of the project I care most about, because it shows the agents doing real work rather than rubber-stamping.

The first run exposed a defect immediately. Confidence was always 1.0. Any crossover, regardless of strength, scored the maximum. The governance agent flagged this on its own: a maxed-out score suggested the signal might be overfitting or poorly calibrated. That flag was correct. I replaced the binary confidence with a volatility-normalized score (moving-average separation divided by recent hourly volatility, capped at 0.85), and scores now spread out meaningfully. AAPL scores around 0.74 on a strong crossover. NVDA scores around 0.14 on a weak one.

The second defect was in the stop-loss. It was fixed at 75 basis points, but AAPL's typical hourly move was 80 to 105 basis points. The stop sat inside the noise of a single bar. The governance agent flagged this in its risk assessment. I replaced the fixed stop with a volatility-scaled one (`STOP_VOL_MULTIPLE = 2.0`, so stop equals twice recent hourly volatility), which gives AAPL a 210 basis point stop and TSLA, a more volatile name, a 364 basis point stop.

After both fixes, something interesting happened: the governance agent's veto reasons changed. Before, it was catching calibration artifacts. After, it started catching a real risk instead, flagging that AAPL was trading near the top of its 24-hour range, which raises reversal risk on a buy. The agent stopped complaining about my bugs and started doing its actual job. The agent catching a flaw I could then fix is the design working, not failing.

## 5. The Build

I used Google's Gemini API (`gemini-2.5-flash`) for both agents, on the free tier, which caps at 20 requests a day and shaped how I tested. Market data comes from Alpaca's paper-trading API, using the SIP feed with an IEX fallback when SIP data is delayed or unavailable.

Tool exposure runs over MCP, using the `mcp` Python package with stdio transport. That package needs Python 3.10 or newer, but the rest of the project runs on Python 3.9 (the system Python on my machine). Rather than fight that mismatch, I split it: the MCP server and client run as Python 3.11 subprocesses, and the coordinator on Python 3.9 talks to them over subprocess stdio and parses JSON back. This is a legitimate MCP pattern. Server and client are separate processes and can run different runtimes.

I used Claude Code throughout to write and wire up the implementation faster than I would have by hand, especially for boilerplate like the MCP server scaffolding and the audit log writer. But the architectural decisions were mine: where to draw the code-versus-LLM boundary, the single-snapshot design in the coordinator, and the rule that the human gate is always the last step regardless of what governance decides. An assistant can write a function. It can't decide where the trust boundary in a risk system should sit.

## 6. Design Decisions I Stand Behind

- **LLMs never compute risk math.** Stops, targets, position sizes, and confidence are all computed in code. LLMs are unreliable at exact arithmetic, and a wrong stop distance is a real dollar loss, not a rounding error.
- **Single data snapshot.** The coordinator loads bars once and reuses that snapshot for every stage. Without this, the cache could refresh mid-pipeline and different agents would reason about different markets.
- **Human always the final gate.** Governance approval does not execute anything. A person sees the dollar risk and the flags and has the last chance to catch something every other layer missed.
- **Deterministic security checks.** The ticker allowlist and dollar-risk cap live in plain code, not a prompt. An LLM asked whether a ticker is safe might approve one it shouldn't. A set lookup does not.

## 7. Limitations and Roadmap

The signal itself is a placeholder. A moving-average crossover is a teaching example, not something I'd trade real money on. The confidence score, while an improvement over a fixed 1.0, is a relative heuristic, not a statistically calibrated probability: 0.74 does not mean a 74 percent win rate. Alpaca's free tier also limits me to a 15-minute delayed SIP feed and a lower-quality IEX fallback. And the system stops at a decision. It never places an order.

Two things are next. A backtesting-calibration agent would run only offline, grading historical tickets on two axes: whether the trade actually hit target before stop, and whether the setup was sound independent of the outcome. It would suggest parameter changes in plain language based on aggregate patterns, gated by a human and validated on data it hasn't seen before it's adopted. The calibration math stays deterministic. The agent only interprets the aggregate results. A thin web UI would also replace the terminal prompt with a browser view of the ticket, the research note, and an approve or reject button. A production version of this system, particularly the signal and calibration components, would need to be developed privately with proprietary data. The architecture shown here is the reusable, publishable part.

## 8. Closing

What I take from this project is mostly about discipline: deciding early where code ends and judgment begins, and not letting that line blur under pressure to make something look more automated than it is. Watching the governance agent flag a confidence score that shouldn't have existed, and a stop-loss that was quietly broken, was the moment the project stopped being an exercise and started being useful. The system still ends the same way every time: a human reads the numbers and the reasoning, and decides. That's not a limitation I'm working around. It's the point.
