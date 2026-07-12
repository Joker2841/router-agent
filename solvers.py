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
    "that computes the answer and prints it with print(). Read every number exactly: "
    "a count like '640 units' means 640, not a percentage. If the question asks for "
    "more than one quantity, print EVERY requested quantity on its own line with a "
    "short label, e.g. print('sugar cups:', x); print('total cost:', y). If it asks "
    "for a single value, print just that number. No explanation. Wrap it in "
    "```python fences."
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
    "factual": "Answer accurately and completely in about 3 to 5 sentences. Address every part of the question and briefly explain, but do not write an essay. No filler, no markdown.",
    "general": "Answer accurately and completely in about 3 to 5 sentences. Address every part of the question and briefly explain, but do not write an essay. No filler, no markdown.",
}


def _clean(s: str) -> str:
    return (s or "").strip()


def _is_bad_output(s: str) -> bool:
    """Reject program output that is clearly not an answer (e.g. a printed
    Python object repr like '<generator object solve at 0x...>')."""
    s = (s or "").strip()
    if not s:
        return True
    if s.lower() in ("none", "null", "[]", "()", "{}", "no solution", "nan"):
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


def _nums(s: str):
    return re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))


def _num_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return False


def solve_math(prompt: str, llm) -> Result | None:
    """Cross-verify: solve via program-of-thought AND direct reasoning. Trust the
    local answer (verified) only when the two independent methods agree. Two
    different methods rarely repeat the same systematic error, so a disagreement
    is a reliable 'escalate this one' signal."""
    # Method 1: program-of-thought (write a program, execute it)
    gen = llm.generate(prompt, max_tokens=320, temperature=0.0, system=_MATH_SYS)
    pot = None
    if gen:
        ok, out = code_exec.run_program(gen, timeout=10)
        if ok and out and len(out) < 200 and not _is_bad_output(out):
            pot = out.strip()

    # Method 2: direct step-by-step reasoning
    direct_txt = _clean(llm.generate(prompt, max_tokens=256, temperature=0.0, system=_MATH_DIRECT))

    if pot is not None:
        pn, dn = _nums(pot), _nums(direct_txt)
        agree = bool(pn) and bool(dn) and _num_eq(pn[-1], dn[-1])
        return Result(answer=pot, source="local-pot", verified=agree)
    if direct_txt:
        return Result(answer=_final_answer(direct_txt), source="local-llm", verified=False)
    return None


_LOGIC_POT = (
    "You solve logic and deduction puzzles by exhaustive search. Write a self-contained "
    "Python 3 program that:\n"
    "1) Enumerates every possibility with itertools (permutations of items to positions, "
    "or itertools.product over attribute choices).\n"
    "2) Keeps only assignments that satisfy EVERY clue. Translate each clue carefully: "
    "'A before B' means pos[A] < pos[B]; 'immediately right of' means pos differs by "
    "exactly 1; 'not last' means pos != the maximum; 'taller than' defines an ordering.\n"
    "3) Reads what the question actually asks (e.g. who is SHORTEST vs tallest, which "
    "house NUMBER) and prints ONLY that: a single name, number, or the ordering asked "
    "for. Use print(). No explanation. Wrap it in ```python fences."
)


def solve_logic(prompt: str, llm) -> Result | None:
    """Logic as code: the model writes brute-force enumeration, we execute it, and
    self-consistency across samples gives a verified/escalate signal (like math)."""
    answer, agree = _pot_consensus(prompt, llm, _LOGIC_POT, timeout=10)
    # Trust the enumeration only when a majority of samples agree on a real answer.
    if (answer is not None and not _is_bad_output(answer)
            and agree >= _majority_need()):
        return Result(answer=answer, source="local-pot", verified=True)
    # Otherwise fall back to prose reasoning, which is more reliable on this model
    # for puzzles where the model writes buggy constraint code.
    direct = _clean(llm.generate(prompt, max_tokens=300, temperature=0.0, system=_LOGIC_REASON))
    if direct:
        return Result(answer=_final_answer(direct), source="local-llm", verified=False)
    # Last resort: whatever the enumeration produced, if anything.
    if answer is not None and not _is_bad_output(answer):
        return Result(answer=answer, source="local-pot", verified=False)
    return None


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
    "sentiment": 90,
    "ner": 160,
    "summarization": 130,
    "factual": 200,
    "general": 200,
}


_NUMWORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _parse_num(s: str):
    s = s.strip().lower()
    if s in _NUMWORD:
        return _NUMWORD[s]
    return int(s) if s.isdigit() else None


def _enforce_summary_format(prompt: str, text: str) -> str:
    """Deterministically force the summary to match the prompt's format rule:
    exact bullet count with per-bullet word limit, exact sentence count, or a
    total word cap. Content comes from the model; this only fixes the shape."""
    p = prompt.lower()
    text = text.strip()
    mb = re.search(r"(one|two|three|four|five|six|\d+)\s+bullet", p)
    ms = re.search(r"(?:exactly\s+)?(one|two|three|four|five|six|\d+)\s+sentence", p)
    mw = re.search(r"(?:under|no longer than|no more than|less than|at most|within)\s+(\d+)\s+words?", p)

    if mb:
        n = _parse_num(mb.group(1))
        wlim = int(mw.group(1)) if mw else None
        lines = [re.sub(r"^[\-\*•\d\.\)\s]+", "", ln).strip()
                 for ln in text.splitlines() if ln.strip()]
        if n and len(lines) < n:  # not enough bullets: split by sentence
            lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(lines)) if s.strip()]
        lines = [ln for ln in lines if ln][:n] if n else lines
        if wlim:
            lines = [" ".join(ln.split()[:wlim]).rstrip(".,;:") for ln in lines]
        return "\n".join("- " + ln for ln in lines) if lines else text

    if ms:
        n = _parse_num(ms.group(1))
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return " ".join(sents[:n]) if n else text

    if mw:
        return " ".join(text.split()[:int(mw.group(1))])
    return text


def solve_summarization(prompt: str, llm) -> Result | None:
    gen = _clean(llm.generate(prompt, max_tokens=300, temperature=0.0,
                              system=_LANG_SYS["summarization"]))
    if not gen:
        return None
    return Result(answer=_enforce_summary_format(prompt, gen), source="local-llm", verified=False)


_SPACY = None
_SPACY_MAP = {"PERSON": "Person", "ORG": "Organization", "GPE": "Location",
              "LOC": "Location", "FAC": "Location", "NORP": "Organization",
              "DATE": "Date", "TIME": "Date"}


def _get_spacy():
    global _SPACY
    if _SPACY is None:
        try:
            import spacy
            _SPACY = spacy.load("en_core_web_md")
        except Exception:
            _SPACY = False
    return _SPACY or None


def solve_ner(prompt: str, llm) -> Result | None:
    """Deterministic NER via spaCy (reliable, zero tokens). Falls back to the
    local model if spaCy is unavailable or finds nothing."""
    nlp = _get_spacy()
    if nlp is not None:
        text = prompt.rsplit(":", 1)[-1].strip() if ":" in prompt else prompt
        try:
            doc = nlp(text or prompt)
            pairs, seen = [], set()
            for ent in doc.ents:
                t = _SPACY_MAP.get(ent.label_)
                name = ent.text.strip()
                if t and name and (name, t) not in seen:
                    seen.add((name, t))
                    pairs.append(f"{name} - {t}")
            if pairs:
                return Result(answer="\n".join(pairs), source="local-spacy", verified=True)
        except Exception:
            pass
    return solve_language(prompt, llm, "ner")


def solve_language(prompt: str, llm, category: str) -> Result | None:
    sys = _LANG_SYS.get(category, _LANG_SYS["general"])
    mt = _LANG_MAXTOK.get(category, 230)
    gen = _clean(llm.generate(prompt, max_tokens=mt, temperature=0.0, system=sys))
    if not gen:
        return None
    return Result(answer=gen, source="local-llm", verified=False)
