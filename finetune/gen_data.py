"""Generate a verified fine-tuning dataset, aligned to how the agent actually
prompts at inference.

Two steps per category:
  1. Generate diverse task prompts in the exact shape the agent receives.
  2. Generate each answer USING the agent's real system prompt, so the output
     format matches inference by construction. Math and code are verified by
     execution; bad ones are dropped.

Run with your Fireworks key:
    FIREWORKS_API_KEY=... python3 finetune/gen_data.py --per-category 150

Output: finetune/train.jsonl  (fields: system, user, assistant, category)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fireworks_client as fw
import code_exec
import solvers  # reuse the agent's real system prompts

os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
os.environ.setdefault(
    "ALLOWED_MODELS",
    "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code",
)
GEN = fw.pick_model("general")       # MiniMax: best generalist
CODE_GEN = fw.pick_model("code")     # Kimi: best coder

# System prompt used at inference for each classifier category (must match agent).
SYS = {
    "general": solvers._LANG_SYS["general"],
    "sentiment": solvers._LANG_SYS["sentiment"],
    "ner": solvers._LANG_SYS["ner"],
    "summarization": solvers._LANG_SYS["summarization"],
    "code_debugging": solvers._CODEDEBUG_SYS,
    "code_generation": solvers._CODEGEN_SYS,
    "math": solvers._MATH_SYS,          # program-of-thought
    "logic": solvers._LOGIC_REASON,
}

# Step 1 instructions: produce ONLY a JSON array of task-prompt strings, in the
# exact shape a user would send. No answers here.
PROMPT_INSTRUCT = {
    "general": "Write {n} diverse factual questions (science, history, geography, culture, sports, tech). Some may ask two things at once.",
    "sentiment": "Write {n} diverse 'Classify the sentiment: <review>' tasks with varied tone (positive, negative, neutral, mixed, sarcastic).",
    "ner": "Write {n} diverse tasks of the form 'Extract all named entities and their types: <sentence>' with entity-rich sentences (people, orgs, locations, dates).",
    "summarization": "Write {n} diverse tasks of the form 'Summarize in one sentence: <passage>' or 'Summarize in under 12 words: <passage>', each with a real 3-5 sentence passage.",
    "code_debugging": "Write {n} diverse tasks of the form 'This function has a bug: <one-line python def>. Find and fix it.' Put a realistic bug in the function. Keep each function on one line using semicolons.",
    "code_generation": "Write {n} diverse 'Write a Python function ...' specs (string, list, math, simple algorithms). Describe the function name and behavior clearly.",
    "math": "Write {n} diverse math word problems (percentages, rates, multi-step arithmetic, unit conversion) each with a single correct numeric answer.",
    "logic": "Write {n} diverse constraint/deduction puzzles (ordering, assignment, truth-teller) each with a single correct short answer.",
}

_ARR_SYS = ("Output ONLY a JSON array of strings. Each string is one complete task prompt. "
            "No answers, no numbering, no markdown.")


def _json_array(text: str) -> list[str]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b == -1:
        return []
    try:
        arr = json.loads(text[a:b + 1])
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []


def gen_prompts(cat: str, n: int) -> list[str]:
    try:
        text, _ = fw.chat(PROMPT_INSTRUCT[cat].format(n=n), GEN, system=_ARR_SYS,
                          max_tokens=2200, temperature=0.9, reasoning_effort="none")
    except Exception as e:
        print(f"  prompt-gen error: {e}", file=sys.stderr)
        return []
    return _json_array(text)


def answer(cat: str, prompt: str) -> str | None:
    """Answer using the agent's real system prompt, so format matches inference."""
    effort = "low" if cat in ("math", "logic") else "none"
    mt = 512 if cat in ("math", "code_generation", "code_debugging", "logic") else 256
    model = CODE_GEN if cat in ("code_generation", "code_debugging") else GEN
    try:
        text, _ = fw.chat(prompt, model, system=SYS[cat], max_tokens=mt, reasoning_effort=effort)
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    if cat in ("code_generation", "code_debugging"):
        code = code_exec.extract_code(text)
        if "def " not in code:
            return None
        ok, _ = code_exec.run_program(code + "\nprint('OK')", timeout=8)
        if not ok:
            return None
    elif cat == "math":
        # PoT answer: must be a runnable program that prints a number.
        ok, out = code_exec.run_program(text, timeout=8)
        if not ok or not re.search(r"-?\d", out):
            return None
    return text


def generate_category(cat: str, target: int) -> list[dict]:
    out, seen, rounds = [], set(), 0
    while len(out) < target and rounds < target // 10 + 10:
        rounds += 1
        for p in gen_prompts(cat, 12):
            if p in seen:
                continue
            seen.add(p)
            a = answer(cat, p)
            if a:
                out.append({"system": SYS[cat], "user": p, "assistant": a, "category": cat})
            if len(out) >= target:
                break
        print(f"  {cat}: {len(out)}/{target}", file=sys.stderr)
        time.sleep(0.2)
    return out[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=150)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "train.jsonl"))
    args = ap.parse_args()
    rows = []
    for cat in SYS:
        print(f"== {cat} ==", file=sys.stderr)
        rows.extend(generate_category(cat, args.per_category))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} examples to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
