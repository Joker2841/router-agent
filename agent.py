"""Container entry point for AMD Hackathon Track 1.

Contract: read /input/tasks.json, write /output/results.json, exit 0.

Strategy: the judged box is small (4 GB RAM, 2 vCPU, no GPU), so a large local
model cannot finish 19 tasks in the 10 minute limit. We therefore run a small,
fast local model for the categories it handles well at zero tokens, and escalate
the harder categories to a Fireworks model. The set of escalated categories is
configurable so we can tune the accuracy and token trade-off per build.

Environment:
  ESCALATE_CATEGORIES  comma separated categories to send to Fireworks
  MODE                 "moonshot" forces everything local (for offline testing)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import classifier
import solvers
from local_llm import get_llm

MODE = os.environ.get("MODE", "hybrid").lower()
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Time guard. If local inference falls behind, escalate the rest so we always
# finish inside the 10 minute limit even if the grading box is slow.
_START = time.time()
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "480"))


def _behind_schedule() -> bool:
    return (time.time() - _START) > TIME_BUDGET_S

# Categories to escalate to Fireworks. Everything else is answered locally.
# Default keeps the fast language tasks local and sends the hard ones out.
_DEFAULT_ESCALATE = "factual,math,logic,code_generation,code_debugging,general"
ESCALATE_CATEGORIES = {
    c.strip() for c in os.environ.get("ESCALATE_CATEGORIES", _DEFAULT_ESCALATE).split(",") if c.strip()
}

# Reasoning level per category. Categories not listed use REASONING_EFFORT (none).
# Only the genuinely hard categories reason, to keep the token score low.
# Tunable by env so we can find the cheapest level that stays correct.
_REASONING_BY_CAT = {
    "math": os.environ.get("REASONING_MATH", "low"),
    "logic": os.environ.get("REASONING_LOGIC", "medium"),
}
# Output budget per category. Reasoning categories need room to think and answer.
_MAXTOK_BY_CAT = {
    "math": 512, "logic": 1024,
    "code_generation": 512, "code_debugging": 512,
}


def _fireworks_available() -> bool:
    if MODE == "moonshot":
        return False
    try:
        import fireworks_client as fw
    except Exception:
        return False
    return bool(fw.allowed_models()) and bool(os.environ.get("FIREWORKS_API_KEY"))


def run_primary(cat: str, prompt: str, llm):
    if cat == "math":
        return solvers.solve_math(prompt, llm)
    if cat == "logic":
        return solvers.solve_logic(prompt, llm)
    if cat == "code_generation":
        return solvers.solve_code_generation(prompt, llm)
    if cat == "code_debugging":
        return solvers.solve_code_debugging(prompt, llm)
    return solvers.solve_language(prompt, llm, cat)


def handle(task: dict, llm) -> tuple[str, str, int]:
    prompt = task.get("prompt", "")
    cat = classifier.classify(prompt)
    fw_ok = _fireworks_available()

    # Preferred path for hard categories: escalate to Fireworks.
    if cat in ESCALATE_CATEGORIES and fw_ok:
        esc = _escalate(prompt, cat)
        if esc is not None:
            return esc

    # Time guard: if we are running low on time, escalate instead of spending it
    # on slow local inference, so we never blow the 10 minute limit.
    if fw_ok and _behind_schedule():
        esc = _escalate(prompt, cat)
        if esc is not None:
            return esc

    # Local path (zero tokens).
    res = run_primary(cat, prompt, llm)
    if res is not None and res.answer:
        return res.answer, res.source, res.fw_tokens

    # Local produced nothing usable: escalate as a last resort.
    if fw_ok:
        esc = _escalate(prompt, cat)
        if esc is not None:
            return esc
    return "", "empty", 0


def _escalate(prompt: str, cat: str):
    """Call Fireworks with the cheapest sufficient allowed model, kept terse."""
    try:
        import fireworks_client as fw
    except Exception:
        return None
    if not fw.allowed_models() or not os.environ.get("FIREWORKS_API_KEY"):
        return None

    kind = "code" if cat in ("code_generation", "code_debugging") else "general"
    if os.environ.get("ESCALATE_TO_GEMMA") == "1":
        kind = "gemma"
    model = fw.pick_model(kind)
    if not model:
        return None

    # Per category reasoning and token budget. Simple categories answer directly
    # with no thinking (cheap). Hard categories reason, and need a big enough
    # budget that the model finishes thinking AND emits the answer.
    default_effort = os.environ.get("REASONING_EFFORT", "none") or None
    effort = _REASONING_BY_CAT.get(cat, default_effort)
    max_tokens = _MAXTOK_BY_CAT.get(cat, 200)

    sys_prompt = "Answer correctly and as briefly as possible. Output only the answer."
    try:
        text, tokens = fw.chat(prompt, model, system=sys_prompt,
                               max_tokens=max_tokens, reasoning_effort=effort)
        # A reasoning model can burn the whole budget thinking and return nothing.
        # If that happens, retry once with no thinking so we still get an answer.
        if not text and effort not in (None, "none"):
            text, extra = fw.chat(prompt, model, system=sys_prompt,
                                   max_tokens=256, reasoning_effort="none")
            tokens += extra
        return text, f"fireworks:{model}", tokens
    except Exception:
        return None


def main() -> int:
    t0 = time.time()
    try:
        tasks = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read {INPUT_PATH}: {e}", file=sys.stderr)
        return 1
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])

    llm = get_llm()
    results = []
    total_tokens = 0
    for task in tasks:
        tid = task.get("task_id")
        try:
            answer, source, tokens = handle(task, llm)
        except Exception as e:
            answer, source, tokens = "", f"error:{e}", 0
        total_tokens += tokens
        results.append({"task_id": tid, "answer": answer})
        print(f"[{tid}] {source} tokens={tokens}", file=sys.stderr)

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {len(results)} tasks, {total_tokens} Fireworks tokens, "
          f"{time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
