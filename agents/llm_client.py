"""LLM client abstraction for AIYYARY.

Wraps Anthropic (Claude), Groq, and local Ollama chat completion calls behind
a single interface selected via the LLM_MODE environment variable, with
automatic retry on rate limits and JSON-repair support for structured
responses.
"""

import json
import os
import re

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_JSON_PAYLOAD_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    match = _CODE_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Detect rate-limit errors across the Anthropic and Groq SDKs without a hard import."""
    type_name = type(exc).__name__
    if "RateLimit" in type_name:
        return True
    status_code = getattr(exc, "status_code", None)
    return status_code == 429


class LLMClient:
    """Single entry point for all LLM calls used by AIYYARY agents."""

    def __init__(self):
        self.llm_mode = os.getenv("LLM_MODE", "groq").strip().lower()

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_claude(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_groq(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return response.choices[0].message.content or ""

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_ollama(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        import requests

        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    # Single dispatch point every agent (1, 2, 3) and every executor calls for
    # a raw LLM completion. Routes to Claude/Groq/Ollama based on LLM_MODE, so
    # switching providers for the whole pipeline is a one-line .env change —
    # no agent code ever calls a provider SDK directly.
    def chat(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        """Send a chat request to the configured LLM provider and return clean text."""
        if self.llm_mode == "claude":
            raw_text = self._call_claude(system, messages, max_tokens, temperature)
        elif self.llm_mode == "groq":
            raw_text = self._call_groq(system, messages, max_tokens, temperature)
        elif self.llm_mode == "ollama":
            raw_text = self._call_ollama(system, messages, max_tokens, temperature)
        else:
            raise RuntimeError(f"Unknown LLM_MODE '{self.llm_mode}'. Expected 'claude', 'groq', or 'ollama'.")

        return _strip_code_fences(raw_text)

    def parse_json(self, text: str) -> dict:
        """Parse text as JSON, raising ValueError with the raw response on failure.

        Tolerates a conversational preamble/suffix around the JSON payload
        (e.g. a repair-turn reply of "Here is the JSON response:\n\n{...}")
        by falling back to the {...}/[...] span in the text when a direct
        parse fails — so a repair attempt that actually contains valid JSON
        is treated as a success, not folded into a raised error alongside
        the original failed response.
        """
        cleaned = _strip_code_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            match = _JSON_PAYLOAD_RE.search(cleaned)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Failed to parse LLM response as JSON: {exc}\nRaw response:\n{text}"
            ) from exc

    # The workhorse every agent uses for its structured (JSON-returning) LLM
    # calls — reconciliation, verdicts, justifications, mitigations, roadmap,
    # ROI. Every downstream agent trusts the dict it gets back to be valid
    # JSON matching schema_description; this is what makes that guarantee
    # hold even when the model returns malformed JSON on the first attempt.
    def structured_chat(
        self,
        system: str,
        messages: list[dict],
        schema_description: str,
        max_tokens: int,
    ) -> dict:
        """Call chat() then parse_json(), with one automatic JSON-repair retry on failure."""
        raw_response = self.chat(system=system, messages=messages, max_tokens=max_tokens)

        try:
            return self.parse_json(raw_response)
        except ValueError:
            pass

        repair_prompt = (
            "Your previous response was not valid JSON. Here is what you returned: "
            f"{raw_response}. Please return only the JSON, exactly matching this schema: "
            f"{schema_description}."
        )
        repair_messages = [
            *messages,
            {"role": "assistant", "content": raw_response},
            {"role": "user", "content": repair_prompt},
        ]

        try:
            repair_response = self.chat(system=system, messages=repair_messages, max_tokens=max_tokens)
        except Exception as exc:
            raise ValueError(
                "LLM structured_chat failed: original response was not valid JSON, "
                f"and the repair attempt raised an error.\n"
                f"Original response:\n{raw_response}\n"
                f"Repair attempt error:\n{exc}"
            ) from exc

        try:
            return self.parse_json(repair_response)
        except ValueError as exc:
            raise ValueError(
                "LLM structured_chat failed after repair attempt.\n"
                f"Original response:\n{raw_response}\n"
                f"Repair attempt response:\n{repair_response}"
            ) from exc

    def inference_call(self, context: str, question: str, max_tokens: int = 300) -> str:
        """Lightweight single-question inference call. Returns plain text, not JSON."""
        system = (
            "You are a research analyst assisting a human researcher during a live "
            "business capability assessment interview. Answer concisely and practically."
        )
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        return self.chat(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=max_tokens,
        )
