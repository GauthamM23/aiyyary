import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def chat(system: str, messages: list[dict], max_tokens: int = 2048) -> str:
    """Send a chat request to the configured LLM provider and return raw text."""
    llm_mode = os.getenv("LLM_MODE", "groq").strip().lower()

    if llm_mode == "claude":
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        raw_text = next((b.text for b in response.content if b.type == "text"), "")
    elif llm_mode == "groq":
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        raw_text = response.choices[0].message.content or ""
    else:
        raise RuntimeError(
            f"Unknown LLM_MODE '{llm_mode}'. Expected 'claude' or 'groq'."
        )

    return _strip_code_fences(raw_text)


def parse_json(text: str) -> dict:
    """Parse text as JSON, raising a clear RuntimeError with the raw text on failure."""
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse LLM response as JSON: {exc}\nRaw response:\n{text}"
        ) from exc
