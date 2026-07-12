"""All-local check for the categories the official samples miss:
logic, code_generation, code_debugging. Run with MODE=moonshot so nothing
escalates -- this measures pure local (zero-token) capability.

    MODE=moonshot LOCAL_MODEL_PATH=models/model.gguf python3 test_uncovered.py
"""
import os
os.environ["MODE"] = "moonshot"

import agent
import code_exec
from local_llm import get_llm

llm = get_llm()


def solve(prompt):
    return agent.handle({"prompt": prompt}, llm)


def check_code(name, cases, answer):
    tests = [{"args": list(a), "expected": e} for a, e in cases]
    return code_exec.run_tests(answer, name, tests)


CODE_GEN = [
    ("Write a Python function is_palindrome(s) that returns True if s is a "
     "palindrome, ignoring case and spaces.",
     "is_palindrome",
     [(["A man a plan a canal Panama"], True), (["hello"], False), (["racecar"], True)]),
    ("Write a Python function count_vowels(s) that returns the number of vowels "
     "(a, e, i, o, u) in s, case-insensitive.",
     "count_vowels",
     [(["hello"], 2), (["AEIOU"], 5), (["xyz"], 0)]),
    ("Write a Python function second_largest(nums) that returns the second "
     "largest distinct value in a list of integers.",
     "second_largest",
     [([[1, 2, 3, 4]], 3), ([[5, 5, 4]], 4), ([[10, 20]], 10)]),
]

CODE_DEBUG = [
    ("This function has a bug:\n"
     "def factorial(n):\n    r = 0\n    for i in range(1, n + 1):\n        r *= i\n    return r\n"
     "Find and fix it.",
     "factorial",
     [([5], 120), ([0], 1), ([1], 1)]),
    ("This function has a bug:\n"
     "def get_max(lst):\n    m = 0\n    for x in lst:\n        if x > m:\n            m = x\n    return m\n"
     "Find and fix it so it works for lists of negative numbers.",
     "get_max",
     [([[-5, -2, -9]], -2), ([[3, 7, 1]], 7)]),
]

LOGIC = [
    ("Alice, Bob, and Carol finished a race. Alice did not finish first. Carol "
     "finished before Bob. Bob did not finish last. Who finished first, second, "
     "and third?",
     ["carol", "bob", "alice"]),
    ("Five houses in a row are numbered 1 to 5. The red house is immediately to "
     "the right of the blue house. The red house is house 3. Which house number "
     "is blue?",
     ["2"]),
    ("Tom is taller than Sara. Sara is taller than Mike. Mike is taller than Jen. "
     "Who is the shortest?",
     ["jen"]),
]


def main():
    cg = cd = lg = 0

    print("== CODE GENERATION ==")
    for prompt, name, cases in CODE_GEN:
        ans, src, _ = solve(prompt)
        ok, out = check_code(name, cases, ans)
        cg += ok
        print(f"  [{src}] {'PASS' if ok else 'FAIL'} {name}  ({out})")

    print("== CODE DEBUGGING ==")
    for prompt, name, cases in CODE_DEBUG:
        ans, src, _ = solve(prompt)
        ok, out = check_code(name, cases, ans)
        cd += ok
        print(f"  [{src}] {'PASS' if ok else 'FAIL'} {name}  ({out})")

    print("== LOGIC ==")
    for prompt, keys in LOGIC:
        ans, src, _ = solve(prompt)
        low = ans.lower()
        ok = all(k in low for k in keys)
        lg += ok
        print(f"  [{src}] {'PASS' if ok else 'FAIL'} :: {ans!r}")

    print(f"\ncode_gen {cg}/{len(CODE_GEN)}  "
          f"code_debug {cd}/{len(CODE_DEBUG)}  logic {lg}/{len(LOGIC)}")


if __name__ == "__main__":
    main()
