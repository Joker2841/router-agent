"""Judge our answers the way the real harness does: with a strong LLM checking
correctness, not lenient keyword matching. This tells us which categories the
local 1B truly passes so we stop being fooled by our own grader.

Usage (needs your Fireworks key):
    FIREWORKS_API_KEY=... python3 grade_llm.py _local_out/results.json
"""
import json
import os
import sys
from pathlib import Path

import fireworks_client as fw

os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
os.environ.setdefault("ALLOWED_MODELS", "accounts/fireworks/models/minimax-m3")
JUDGE = fw.pick_model("general")


def expected_intent(check: dict) -> str:
    t = check["type"]
    if t == "numeric":
        return f"The correct final answer is the number {check['value']}."
    if t == "contains_all":
        return "A correct answer must include all of: " + ", ".join(map(str, check["values"])) + "."
    if t == "contains_any":
        return "A correct answer must reflect: " + " / ".join(map(str, check["values"])) + "."
    if t == "regex":
        return f"A correct answer must satisfy the pattern: {check['pattern']}."
    if t == "code_tests":
        return (f"The code must define {check['function_name']} and return correct results "
                f"for these cases: {check['tests']}.")
    return "Answer must be correct."


JUDGE_SYS = ("You are a strict grader. Given a task, the correctness requirement, and a "
             "candidate answer, decide if the candidate is correct. Reply with exactly one "
             "word: YES or NO.")


def judge(prompt: str, check: dict, answer: str) -> bool:
    q = (f"TASK:\n{prompt}\n\nCORRECTNESS REQUIREMENT:\n{expected_intent(check)}\n\n"
         f"CANDIDATE ANSWER:\n{answer}\n\nIs the candidate answer correct? Reply YES or NO.")
    try:
        text, _ = fw.chat(q, JUDGE, system=JUDGE_SYS, max_tokens=8, reasoning_effort="none")
    except Exception as e:
        print(f"judge error: {e}", file=sys.stderr)
        return False
    return text.strip().upper().startswith("Y")


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "_local_out/results.json"
    gold_file = os.environ.get("GOLD_FILE", "eval_set.json")
    results = json.loads(Path(results_path).read_text())
    gold = {t["task_id"]: t for t in json.loads(Path(gold_file).read_text())}
    answers = {r["task_id"]: r.get("answer", "") for r in results}

    by_cat, passed = {}, 0
    print(f"{'id':6} {'cat':16} {'ok':3}")
    print("-" * 32)
    for tid, t in gold.items():
        ok = judge(t["prompt"], t["check"], answers.get(tid, ""))
        passed += ok
        by_cat.setdefault(t["category"], []).append(ok)
        print(f"{tid:6} {t['category']:16} {'YES' if ok else 'NO'}")

    n = len(gold)
    print("\nper category (LLM-judged):")
    for cat, r in sorted(by_cat.items()):
        print(f"  {cat:16} {sum(r)}/{len(r)}")
    print(f"\nJUDGE ACCURACY {passed}/{n} = {passed/n*100:.1f}%")


if __name__ == "__main__":
    main()
