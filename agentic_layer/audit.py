"""
audit.py — append-only audit log for every governance workflow run.

WHY AN AUDIT LOG IS REQUIRED
-----------------------------
Any system that can influence real-money trading decisions must maintain a
tamper-evident record of every decision made, by whom (human or machine), and
on what basis.  Without this record:

  - Post-trade analysis cannot reconstruct what the system saw at decision time
  - Regulatory review (SEC, FINRA, MiFID II) has no evidence trail to audit
  - Incident investigation (a bad trade, a system bug) cannot determine root cause
  - Model performance cannot be measured against actual outcomes

This audit module satisfies that requirement at the simplest possible level:
one JSON object per line (JSONL), written to an append-only local file.
A production system would additionally:
  - Write to an immutable store (object storage with versioning, a write-once DB)
  - Sign each entry with a key held outside the trading system
  - Replicate to a separate environment so a system compromise cannot erase logs
  - Feed entries to a real-time monitoring pipeline

WHAT IS LOGGED
--------------
Every coordinator run appends one entry containing:
  - run_id         : UUID so entries can be correlated across systems
  - timestamp_utc  : ISO-8601 UTC timestamp of the run
  - ticker / as_of : the symbol and historical window requested
  - ticket         : all ticket fields at the time of the safety check
  - safety         : pass/fail + any violation strings
  - research       : summary + concerns from the research agent (or null)
  - governance     : approved + reasoning + flags (or null)
  - human_decision : "approved", "rejected", or "not_reached"
  - error          : exception string if the run crashed (or null)

LOG LOCATION
------------
logs/audit.jsonl in the project root.  The logs/ directory is created if it
does not exist.  logs/*.jsonl is in .gitignore — audit logs should not be
committed to version control (they may contain market data under license).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_LOG_DIR  = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "audit.jsonl"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(exist_ok=True)


def log_run(
    *,
    ticker: str,
    as_of: Optional[str],
    ticket_fields: Optional[dict],
    safety_passed: bool,
    safety_violations: list[str],
    research_summary: Optional[str],
    research_concerns: Optional[list[str]],
    governance_approved: Optional[bool],
    governance_reasoning: Optional[str],
    governance_flags: Optional[list[str]],
    human_decision: str,           # "approved" | "rejected" | "not_reached"
    error: Optional[str] = None,
) -> None:
    """
    Append one audit entry to logs/audit.jsonl.

    All parameters are keyword-only.  Pass None for fields that were not
    reached (e.g. research_summary when safety blocked the run).
    """
    _ensure_log_dir()

    entry = {
        "run_id":              str(uuid.uuid4()),
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
        "ticker":              ticker,
        "as_of":               as_of,
        "ticket":              ticket_fields,
        "safety": {
            "passed":          safety_passed,
            "violations":      safety_violations,
        },
        "research": None if research_summary is None else {
            "summary":         research_summary,
            "concerns":        research_concerns or [],
        },
        "governance": None if governance_approved is None else {
            "approved":        governance_approved,
            "reasoning":       governance_reasoning,
            "flags":           governance_flags or [],
        },
        "human_decision":      human_decision,
        "error":               error,
    }

    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
