"""Fireworks escalation path. ALL calls go through FIREWORKS_BASE_URL (required
by the rules) and only use models from ALLOWED_MODELS. Kept terse to minimize
the tokens that actually count toward the score.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _base_url() -> str:
    url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    return url.rstrip("/")


def allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def pick_model(kind: str = "general") -> str | None:
    """Choose a model from ALLOWED_MODELS by task kind.

    kind: "code" | "gemma" | "general". Falls back gracefully.
    """
    models = allowed_models()
    if not models:
        return None

    def find(*subs):
        for m in models:
            low = m.lower()
            if any(s in low for s in subs):
                return m
        return None

    if kind == "code":
        return find("code", "kimi") or find("minimax") or models[0]
    if kind == "gemma":
        # Prefer the cheapest Gemma variant for the "Gemma via Fireworks" prize.
        return find("nvfp4") or find("a4b") or find("gemma") or models[0]
    return find("minimax") or find("gemma") or models[0]


def chat(prompt: str, model: str, system: str | None = None,
         max_tokens: int = 512, temperature: float = 0.0,
         reasoning_effort: str | None = None, timeout: float = 28) -> tuple[str, int]:
    """Return (answer_text, total_tokens). Raises on hard network/API failure.

    reasoning_effort ("none" | "low" | "medium" | "high") lowers the thinking
    tokens on reasoning models such as minimax-m3. If a model rejects the field,
    we retry once without it.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    def _request(include_effort: bool) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if include_effort and reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        req = urllib.request.Request(
            f"{_base_url()}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {os.environ.get('FIREWORKS_API_KEY', '')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        data = _request(bool(reasoning_effort))
    except Exception:
        # Any failure with the reasoning parameter: retry once without it, so a
        # grading environment that handles reasoning_effort differently still
        # gets a normal answer instead of falling back to the weak local model.
        if reasoning_effort:
            data = _request(False)
        else:
            raise

    text = data["choices"][0]["message"].get("content", "") or ""
    text = _THINK_RE.sub("", text).strip()
    total = int(data.get("usage", {}).get("total_tokens", 0))
    return text, total
