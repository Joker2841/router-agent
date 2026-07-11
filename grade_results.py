"""Grade a container's /output/results.json against the eval_set.json answers.

Usage:
    python3 grade_results.py _local_out/results.json
"""
import json
import re
import sys
from pathlib import Path

import code_exec


def _num(s: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(m[-1]) if m else None


def grade(answer: str, check: dict) -> bool:
    a = answer or ""
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


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "_local_out/results.json"
    import os
    gold_file = os.environ.get("GOLD_FILE", "eval_set.json")
    results = json.loads(Path(results_path).read_text())
    gold = {t["task_id"]: t for t in json.loads(Path(gold_file).read_text())}
    answers = {r["task_id"]: r.get("answer", "") for r in results}

    by_cat = {}
    passed = 0
    print(f"{'id':6} {'cat':16} {'ok':3} answer")
    print("-" * 80)
    for tid, t in gold.items():
        ok = grade(answers.get(tid, ""), t["check"])
        passed += ok
        by_cat.setdefault(t["category"], []).append(ok)
        ans = answers.get(tid, "").replace("\n", " ")[:52]
        print(f"{tid:6} {t['category']:16} {'OK' if ok else 'XX':3} {ans}")

    n = len(gold)
    print("\nper category:")
    for cat, r in sorted(by_cat.items()):
        print(f"  {cat:16} {sum(r)}/{len(r)}")
    print(f"\nACCURACY {passed}/{n} = {passed/n*100:.1f}%  "
          f"(19-task equivalent ~{passed/n*19:.1f}/19, gate needs 16)")


if __name__ == "__main__":
    main()
