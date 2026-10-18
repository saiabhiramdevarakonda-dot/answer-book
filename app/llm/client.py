"""Groq LLM Client Interface.

Provides a unified method to call Groq models and parse guaranteed JSON output.
Includes a deterministic fallback when GROQ_API_KEY is not configured.
"""

import json
import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from app.core.config import settings


def _clean_json_text(text: str) -> str:
    """Removes markdown code block formatting (```json ... ```) if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def get_llm_client():
    """Initializes and returns a ChatGroq instance if API key is set, else None."""
    if not settings.GROQ_API_KEY:
        return None
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=settings.GROQ_TEMPERATURE,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


import time

def call_groq_json(system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Sends prompts to Groq and parses the JSON response into a Python dictionary.

    Includes automatic retry on Groq rate limits (429).
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please configure GROQ_API_KEY in your environment or .env file.")

    client = get_llm_client()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    for attempt in range(max_retries):
        try:
            request_started = time.perf_counter()
            response = client.invoke(messages)
            elapsed = time.perf_counter() - request_started

            print(
                 f"[Groq latency] attempt={attempt + 1} "
                 f"duration={elapsed:.2f}s"
                 )
            content_text = _clean_json_text(str(response.content))
            return json.loads(content_text)
        except Exception as exc:
            err_str = str(exc)
            # Handle rate limit 429
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_time = 15 * (attempt + 1)
                # Try to parse wait time from error message if available
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0

                print(f"    [Rate limit hit: waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}]")
                time.sleep(wait_time)
                continue

            if attempt == max_retries - 1:
                raise ValueError(f"Failed to generate JSON after {max_retries} attempts: {exc}") from exc
            time.sleep(2)

    raise ValueError(f"Failed to obtain valid JSON from Groq after {max_retries} retries.")


