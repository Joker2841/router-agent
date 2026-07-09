"""Container entrypoint for AMD Hackathon Track 1.

Contract: read /input/tasks.json -> write /output/results.json -> exit 0.

Pipeline (prove-or-escalate, local-first, zero Fireworks tokens whenever possible):
  1. classify the task (0 tokens)
  2. run the matching local handler:
       - math/logic  -> program-of-thought sampled for self-consistency + executed
       - code        -> generated locally, compile-checked
       - language    -> concise local generation
  3. if the local answer is VERIFIED, ship it (0 tokens).
  4. otherwise escalate to Fireworks, but only for hard categories, and in
     moonshot mode only when there is no usable local answer at all.

MODE (env):
  moonshot (default) -> escalate only hard tasks with NO verified local answer.
  hybrid             -> also escalate unverified answers in the escalation set.
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

MODE = os.environ.get("MODE", "moonshot").lower()
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

HARD = {"math", "logic", "code_generation", "code_debugging"}
LANG = {"sentiment", "ner", "summarization", "factual", "general"}
# Categories to escalate when unverified (hybrid mode).
ESCALATE_UNVERIFIED = set(HARD)
if os.environ.get("ESCALATE_FACTUAL") == "1":
    ESCALATE_UNVERIFIED.add("factual")

# Categories to never trust locally (always escalate when Fireworks is
# available). Empty by default = pure moonshot, 0 tokens. If the real hidden-set
# score shows a category dragging us down, set e.g.
# UNTRUSTED_CATEGORIES="logic" and provide a Fireworks key to escalate it.
_ut = os.environ.get("UNTRUSTED_CATEGORIES", "")
UNTRUSTED = {c.strip() for c in _ut.split(",") if c.strip()}


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
    res = run_primary(cat, prompt, llm)

    # 1) Verified local answer -> ship it, 0 tokens (unless the category is
    #    flagged untrusted, where even a "verified" local answer isn't reliable).
    if res is not None and res.verified and cat not in UNTRUSTED:
        return res.answer, res.source, 0

    # 2) Decide whether to escalate.
    want_escalation = False
    if cat in UNTRUSTED:
        want_escalation = True
    elif cat in HARD and (res is None or MODE == "hybrid"):
        want_escalation = True
    elif MODE == "hybrid" and cat in ESCALATE_UNVERIFIED:
        want_escalation = True

    if want_escalation:
        esc = _escalate(prompt, cat)
        if esc is not None:
            return esc

    # 3) Local fallback (0 tokens): keep whatever we have, or try a plain answer.
    if res is None:
        res = solvers.solve_language(prompt, llm, cat if cat in LANG else "general")
    if res is None:
        return "", "empty", 0
    return res.answer, res.source, res.fw_tokens


def _escalate(prompt: str, cat: str):
    """Call Fireworks with the cheapest sufficient allowed model. Terse."""
    try:
        import fireworks_client as fw
    except Exception:
        return None
    if not fw.allowed_models() or not os.environ.get("FIREWORKS_API_KEY"):
        return None

    kind = "code" if cat in ("code_generation", "code_debugging") else "general"
    if os.environ.get("ESCALATE_TO_GEMMA") == "1":
        kind = "gemma"  # for the "Gemma via Fireworks" prize
    model = fw.pick_model(kind)
    if not model:
        return None
    sys_prompt = "Answer correctly and as briefly as possible. Output only the answer."
    try:
        text, tokens = fw.chat(prompt, model, system=sys_prompt, max_tokens=600)
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
