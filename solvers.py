"""Prove-or-escalate handlers. Every handler produces a ZERO-token local answer.

verified=True means we have real evidence the answer is correct:
  - math/logic: independent program-of-thought samples AGREED (self-consistency)
  - code: the generated function compiled/ran without error
verified=False is a lower-confidence local answer the agent may escalate.

Key lesson from real 1B output: a program that merely *executes* is not a
correct answer (a buggy program still prints a number). So math/logic sample the
solver program several times and only trust an answer when the runs agree.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass

import code_exec

POT_SAMPLES = int(os.environ.get("POT_SAMPLES", "3"))


@dataclass
class Result:
    answer: str
    source: str          # deterministic | local-pot | local-llm | fireworks
    verified: bool = False
    fw_tokens: int = 0


_MATH_SYS = (
    "You are a precise math solver. Write a short, self-contained Python 3 program "
    "that computes the answer and prints ONLY the final answer with print(), which is "
    "the number, no words, no units, no explanation. Wrap it in ```python fences."
)

_LOGIC_SYS = (
    "You solve logic/deduction puzzles by brute force. Write a self-contained "
    "Python 3 program that enumerates all possibilities, keeps the ones satisfying "
    "every stated constraint, and prints ONLY the direct answer to the question "
    "asked (for 'who' questions print the name). No explanation. Use ```python fences."
)

_CODEGEN_SYS = (
    "You are an expert Python developer. Return ONLY the requested function in a "
    "single ```python code block. Correct, handle edge cases, no explanation."
)

_CODEDEBUG_SYS = (
    "You fix bugs in code. Return ONLY the corrected function in a single "
    "```python code block. No explanation."
)

_LANG_SYS = {
    "sentiment": "Classify the sentiment (positive, negative, neutral, or mixed) and give a one-line justification. Be concise. No markdown.",
    "ner": "Extract every named entity and label its type (Person, Organization, Location, Date). One per line as 'entity - Type'. No preamble, no markdown.",
    "summarization": "Summarize following the exact length/format constraint in the request. Output only the summary, nothing else.",
    "factual": "Answer accurately and concisely. State only facts you are confident about; do not invent specifics. No markdown.",
    "general": "Answer accurately and concisely. No markdown.",
}


def _clean(s: str) -> str:
    return (s or "").strip()


def _is_bad_output(s: str) -> bool:
    """Reject program output that is clearly not an answer (e.g. a printed
    Python object repr like '<generator object solve at 0x...>')."""
    s = (s or "").strip()
    if not s:
        return True
    if s.startswith("<") and s.endswith(">"):
        return True
    if "object at 0x" in s or "<generator" in s or "<function" in s or "<map object" in s:
        return True
    return False


def _norm(s: str) -> str:
    """Normalize a PoT answer for consensus comparison."""
    s = _clean(s).lower()
    # Treat 144 and 144.0 as equal.
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return " ".join(s.split())


def _pot_consensus(prompt: str, llm, system: str, timeout: float = 10.0):
    """Sample the solver program POT_SAMPLES times; return (answer, agreement).

    agreement is the count of samples that produced the winning answer. We only
    trust the answer when at least a majority agree.
    """
    outputs = []
    for i in range(max(1, POT_SAMPLES)):
        temp = 0.0 if i == 0 else 0.5  # first pass greedy, rest diverse
        gen = llm.generate(prompt, max_tokens=400, temperature=temp, system=system)
        if not gen:
            continue
        ok, out = code_exec.run_program(gen, timeout=timeout)
        if ok and out and len(out) < 300 and not _is_bad_output(out):
            outputs.append((out.strip(), _norm(out)))
    if not outputs:
        return None, 0
    tally = Counter(n for _, n in outputs)
    winner_norm, count = tally.most_common(1)[0]
    # Pick a raw representative for the winning normalized answer.
    answer = next(raw for raw, n in outputs if n == winner_norm)
    return answer, count


_MATH_DIRECT = "Solve the problem step by step, then output ONLY the final numeric answer on the last line."

# Measured: the 4B reasons deduction puzzles better in prose than by writing
# brute-force code (PoT gave wrong/garbage answers). So logic = reason, then
# state a clean final answer we can extract.
_LOGIC_REASON = ("Solve this logic puzzle. Reason briefly step by step, then on the "
                 "final line write exactly 'Answer: <your answer>' where the answer is "
                 "the direct response to the question (a name, letter, or short phrase).")


def _majority_need() -> int:
    return 1 if POT_SAMPLES <= 1 else (POT_SAMPLES // 2 + 1)


def _final_answer(text: str) -> str:
    """Pull the answer after 'Answer:' if present, else the last non-empty line."""
    t = _clean(text)
    m = re.search(r"answer\s*[:\-]\s*(.+)", t, re.I)
    if m:
        line = m.group(1).strip().splitlines()[0].strip(" *.\t")
        if line:
            return line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else t


def solve_math(prompt: str, llm) -> Result | None:
    answer, agree = _pot_consensus(prompt, llm, _MATH_SYS)
    if answer is not None:
        return Result(answer=answer, source="local-pot", verified=(agree >= _majority_need()))
    # PoT produced nothing usable -> direct answer (unverified, may escalate).
    direct = _clean(llm.generate(prompt, max_tokens=256, temperature=0.0, system=_MATH_DIRECT))
    return Result(answer=_final_answer(direct), source="local-llm", verified=False) if direct else None


def solve_logic(prompt: str, llm) -> Result | None:
    gen = _clean(llm.generate(prompt, max_tokens=350, temperature=0.0, system=_LOGIC_REASON))
    if not gen:
        return None
    return Result(answer=_final_answer(gen), source="local-llm", verified=False)


def solve_code_generation(prompt: str, llm) -> Result | None:
    gen = _clean(llm.generate(prompt, max_tokens=512, temperature=0.0, system=_CODEGEN_SYS))
    if not gen:
        return None
    code = code_exec.extract_code(gen)
    verified = False
    if "def " in code:
        ok, _ = code_exec.run_program(code + "\nprint('OK')", timeout=8)
        verified = ok
    return Result(answer=gen, source="local-llm", verified=verified)


def solve_code_debugging(prompt: str, llm) -> Result | None:
    gen = _clean(llm.generate(prompt, max_tokens=512, temperature=0.0, system=_CODEDEBUG_SYS))
    if not gen:
        return None
    code = code_exec.extract_code(gen)
    verified = False
    if "def " in code:
        ok, _ = code_exec.run_program(code + "\nprint('OK')", timeout=8)
        verified = ok
    return Result(answer=gen, source="local-llm", verified=verified)


# Tight per-category output caps -> shorter generations -> faster on CPU.
# (Local output length costs 0 tokens but real wall-clock time; the 10-min /
# 19-task budget makes this matter, especially for the 4B model.)
_LANG_MAXTOK = {
    "sentiment": 110,
    "ner": 200,
    "summarization": 170,
    "factual": 230,
    "general": 230,
}


def solve_language(prompt: str, llm, category: str) -> Result | None:
    sys = _LANG_SYS.get(category, _LANG_SYS["general"])
    mt = _LANG_MAXTOK.get(category, 230)
    gen = _clean(llm.generate(prompt, max_tokens=mt, temperature=0.0, system=sys))
    if not gen:
        return None
    return Result(answer=gen, source="local-llm", verified=False)
