"""
_gemini.py — shared Gemini helpers for every agent in the agentic_layer.

Both governance_agent.py and research_agent.py need the same three things:
  1. An authenticated google-genai Client built from GEMINI_API_KEY in .env
  2. A single call to generate a response from a text prompt
  3. A parser that strips markdown fences and returns a JSON dict

Putting these here means:
  - Changing the model affects all agents at once.
  - There's one place to add retry logic, logging, or token-counting later.
  - Each agent file stays focused on prompts and dataclasses, not SDK plumbing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# Project root: two levels up from agentic_layer/_gemini.py.
_PROJECT_ROOT = Path(__file__).parent.parent

# Single model name used by every agent.  Change here to upgrade everywhere.
# gemini-2.5-flash: fast, cost-effective, supports JSON output well.
GEMINI_MODEL = "gemini-2.5-flash"


def get_client() -> genai.Client:
    """
    Load .env from the project root and return an authenticated Gemini client.

    We call load_dotenv() here (inside a function rather than at import time)
    so that importing this module never has side effects.  The .env file is
    resolved relative to the project root so the path works regardless of the
    caller's working directory.
    """
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your Google AI Studio key."
        )
    return genai.Client(api_key=api_key)


def generate(prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """
    Send *prompt* to Gemini and return the raw response text.

    temperature=0.2 keeps outputs near-deterministic — important for getting
    reliably parseable JSON back rather than creative variations.
    max_tokens=2048 is enough for a structured JSON response with a paragraph
    of reasoning; increase if you need longer outputs.
    """
    client = get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


def parse_json_response(text: str) -> dict:
    """
    Strip optional markdown code fences from *text* then parse as JSON.

    Gemini is instructed to return bare JSON, but it occasionally wraps the
    output in ```json ... ``` fences.  This function handles both cases.

    Raises json.JSONDecodeError if the result is still not valid JSON.
    Callers should wrap the call in try/except and build a safe fallback value.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop every line that is purely a fence marker (``` or ```json).
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return json.loads(cleaned)
