"""Sandboxed Python execution for verified answers (zero Fireworks tokens).

Two uses:
  1. run_program(code)      -> execute a self-contained script, capture stdout
  2. run_tests(code, fn, tests) -> run a generated function against test cases

Everything runs in a separate `python3` subprocess with a hard timeout so a
runaway generation can never hang the 10-minute container budget.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import os
import textwrap

# Matches a fenced code block, optionally tagged ```python
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Resource guard prepended to every executed script: cap CPU seconds + address
# space so a bad program dies instead of eating the whole box.
_GUARD = textwrap.dedent(
    """
    import resource, sys
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    except Exception:
        pass
    """
)


def extract_code(text: str) -> str:
    """Pull the first fenced code block from an LLM answer, or return the raw text."""
    if not text:
        return ""
    m = _CODE_BLOCK_RE.search(text)
    return (m.group(1) if m else text).strip()


def _run(script: str, timeout: float = 12.0) -> tuple[bool, str]:
    """Run `script` in a fresh subprocess. Returns (ok, stdout-or-stderr)."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(_GUARD + "\n" + script)
            path = f.name
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Minimal env: no network creds, no surprises.
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        return False, (proc.stderr or proc.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"EXEC_ERROR: {e}"
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def run_program(code: str, timeout: float = 12.0) -> tuple[bool, str]:
    """Execute a self-contained script that prints its answer to stdout."""
    code = extract_code(code)
    if not code:
        return False, "NO_CODE"
    return _run(code, timeout=timeout)


def run_tests(answer_text: str, function_name: str, tests: list[dict], timeout: float = 12.0):
    """Execute a generated function against tests.

    tests: list of {"args": [...], "expected": ...}
    Returns (all_passed: bool, detail: str).
    """
    code = extract_code(answer_text)
    if not code or not function_name:
        return False, "NO_CODE_OR_FN"
    harness = (
        code
        + "\n\n"
        + "import json as _json\n"
        + f"_tests = {tests!r}\n"
        + f"_fn = {function_name}\n"
        + "_ok = True\n"
        + "for _t in _tests:\n"
        + "    try:\n"
        + "        _r = _fn(*_t['args'])\n"
        + "        if _r != _t['expected']:\n"
        + "            _ok = False\n"
        + "    except Exception:\n"
        + "        _ok = False\n"
        + "print('PASS' if _ok else 'FAIL')\n"
    )
    ok, out = _run(harness, timeout=timeout)
    return (ok and out.endswith("PASS")), out


if __name__ == "__main__":
    # Smoke test
    print(run_program("print(sum(range(10)))"))
    print(run_tests("def add(a,b):\n    return a+b", "add",
                    [{"args": [1, 2], "expected": 3}, {"args": [5, 5], "expected": 10}]))
    print(run_program("while True:\n    pass", timeout=2))  # should TIMEOUT
