"""Zero-token task classification.

We do not need perfect labels. We only need to know which handler strategy to
try first. Categories map to strategies in agent.py. Heuristics are intentionally
conservative: if unsure, we return "general" and let the prove-or-escalate
pipeline decide.
"""
from __future__ import annotations

import re

CATEGORIES = (
    "math",
    "logic",
    "code_generation",
    "code_debugging",
    "sentiment",
    "ner",
    "summarization",
    "factual",
    "general",
)

_RE = {
    "code_debugging": re.compile(
        r"\b(bug|debug|fix|error|wrong|incorrect|broken|fails?)\b", re.I,
    ),
    "code_generation": re.compile(
        r"\b(write|implement|create|define|complete)\b.*\b(function|method|class|code|program|script)\b"
        r"|\bdef\s+\w+\s*\(",
        re.I,
    ),
    "sentiment": re.compile(r"\b(sentiment|positive or negative|classify.*(review|feeling|emotion|tone))\b", re.I),
    "ner": re.compile(r"named entit|extract.*(entit|person|organization|location|date)|\bNER\b", re.I),
    "summarization": re.compile(r"\b(summar(y|ise|ize)|tl;?dr|in (one|a single|two|three) sentence|condense)\b", re.I),
    "logic": re.compile(
        r"\b(each (own|has|is)|different (pet|color|house|drink|job)|who (owns|has|is|sits|finished)|"
        r"puzzle|deduce|logical(ly)?|constraint|sit(s)? in a row|arrange|order them|to the (left|right) of)\b",
        re.I,
    ),
    # Strong math signals: computation verbs, quantity questions, operators.
    "math_strong": re.compile(
        r"\b(calculate|compute|how many|how much|how far|how fast|how long|how old|"
        r"average speed|percent|remainder|sum of|total of|product of)\b"
        r"|%|\d\s*[-+*/x]\s*\d",
        re.I,
    ),
    # Weak math: a number plus a unit/quantity word plus a question.
    "math_units": re.compile(
        r"\d.*\b(km|cm|mm|miles?|mph|km/?h|kg|grams?|g|liters?|litres?|ml|"
        r"dollars?|cents?|minutes?|hours?|seconds?|days?|weeks?|years?|percent|items?)\b",
        re.I,
    ),
    "sentiment_alt": re.compile(r"\breview\b.*\b(great|bad|love|hate|terrible|excellent|but)\b", re.I),
}


def classify(prompt: str) -> str:
    """Return the best-guess category for routing. Order matters (specific first)."""
    p = prompt or ""
    has_digit = bool(re.search(r"\d", p))

    if _RE["code_debugging"].search(p) and ("def " in p or "return" in p or "function" in p.lower()):
        return "code_debugging"
    if _RE["code_generation"].search(p):
        return "code_generation"
    if _RE["ner"].search(p):
        return "ner"
    if _RE["sentiment"].search(p) or _RE["sentiment_alt"].search(p):
        return "sentiment"
    if _RE["summarization"].search(p):
        return "summarization"
    if _RE["logic"].search(p):
        return "logic"
    if _RE["math_strong"].search(p):
        return "math"
    if has_digit and _RE["math_units"].search(p):
        return "math"
    return "general"


if __name__ == "__main__":
    tests = [
        ("A train travels 60 km in 45 minutes. What is its average speed in km/h?", "math"),
        ("What is 15% of 340?", "math"),
        ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", "math"),
        ("Who wrote Romeo and Juliet?", "general"),
        ("Anna, Ben, and Cara sit in a row. Who is in the middle?", "logic"),
        ("Extract all named entities from: Tim Cook at Apple on September 12.", "ner"),
        ("Summarize in one sentence: the 30-mile network opened.", "summarization"),
    ]
    for prompt, expected in tests:
        got = classify(prompt)
        print(f"{'ok' if got == expected else 'XX'} {got:14} (want {expected:14}) <- {prompt[:55]}")
