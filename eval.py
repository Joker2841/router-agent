"""Local accuracy harness. Runs each task through the LOCAL pipeline only (no
Fireworks) and grades it, so we can estimate: (a) how many of 19 we'd get with a
pure zero-token agent, and (b) which categories need escalation.

Usage:
  LOCAL_MODEL_PATH=models/model.gguf python3 eval.py
  LOCAL_MODEL_PATH=models/model.gguf POT_SAMPLES=2 python3 eval.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import classifier
import solvers
import code_exec
from local_llm import get_llm


def _num(s: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(m[-1]) if m else None


def grade(answer: str, check: dict) -> bool:
    a = (answer or "")
    al = a.lower()
    t = check["type"]
    if t == "numeric":
        n = _num(a)
        return n is not None and abs(n - float(check["value"])) < 1e-6
    if t == "contains_all":
        return all(str(v).lower() in al for v in check["values"])
    if t == "contains_any":
        return any(str(v).lower() in al for v in check["values"])
    if t == "regex":
        return re.search(check["pattern"], a) is not None
    if t == "code_tests":
        ok, _ = code_exec.run_tests(a, check["function_name"], check["tests"])
        return ok
    return False


def main() -> int:
    tasks = json.loads(Path("eval_set.json").read_text(encoding="utf-8"))
    llm = get_llm()

    by_cat: dict[str, list[bool]] = {}
    rows = []
    t0 = time.time()
    for task in tasks:
        prompt = task["prompt"]
        gold_cat = task.get("category", "?")
        pred_cat = classifier.classify(prompt)
        res = _run(pred_cat, prompt, llm)
        answer = res.answer if res else ""
        ok = grade(answer, task["check"])
        verified = bool(res and res.verified)
        by_cat.setdefault(gold_cat, []).append(ok)
        rows.append((task["task_id"], gold_cat, pred_cat, verified, ok, answer.replace("\n", " ")[:60]))

    dt = time.time() - t0
    print(f"\n{'id':5} {'cat':16} {'routed':16} {'verif':5} {'ok':3} answer")
    print("-" * 90)
    for tid, gc, pc, v, ok, ans in rows:
        print(f"{tid:5} {gc:16} {pc:16} {'Y' if v else '.':5} {'OK' if ok else 'XX':3} {ans}")

    total = sum(1 for r in rows if r[4])
    n = len(rows)
    print("\n== per-category ==")
    for cat, results in sorted(by_cat.items()):
        c = sum(results)
        print(f"  {cat:16} {c}/{len(results)}")
    scaled = total / n * 19
    print(f"\nLOCAL accuracy: {total}/{n} = {total/n*100:.1f}%  "
          f"(~{scaled:.1f}/19 -> {'PASS' if total/n >= 0.80 else 'BELOW'} 80% gate on local alone)")
    print(f"Wall time: {dt:.1f}s ({dt/n:.1f}s/task)")
    fails = [r for r in rows if not r[4]]
    if fails:
        print("\nWould need escalation / fixing:")
        for tid, gc, pc, v, ok, ans in fails:
            print(f"  {tid} ({gc}, routed={pc}): {ans}")
    return 0


def _run(cat, prompt, llm):
    if cat == "math":
        return solvers.solve_math(prompt, llm)
    if cat == "logic":
        return solvers.solve_logic(prompt, llm)
    if cat == "code_generation":
        return solvers.solve_code_generation(prompt, llm)
    if cat == "code_debugging":
        return solvers.solve_code_debugging(prompt, llm)
    return solvers.solve_language(prompt, llm, cat)


if __name__ == "__main__":
    sys.exit(main())
