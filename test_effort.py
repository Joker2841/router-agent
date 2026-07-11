"""Compare reasoning_effort levels on minimax-m3 for tokens, speed, and accuracy.

Run from the project directory with your Fireworks key:
    FIREWORKS_API_KEY=your_key python3 test_effort.py
"""
import os
import time

import fireworks_client as fw

os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
MODEL = "accounts/fireworks/models/minimax-m3"
SYS = "Answer correctly and as briefly as possible. Output only the answer."

PROMPTS = {
    "factual": "What is the chemical symbol for gold, and what is its atomic number?",
    "math": "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
    "logic": ("Three boxes are labeled A, B, and C. Exactly one contains a prize. "
              "Label A says 'the prize is here'. Label B says 'the prize is not here'. "
              "Label C says 'the prize is in A'. Only one statement is true. "
              "Which box has the prize?"),
    "sentiment": "Classify the sentiment: 'I waited an hour and the food arrived cold. Never coming back.'",
}

for effort in [None, "none", "low", "medium"]:
    print(f"\n=== reasoning_effort={effort} ===")
    total = 0
    for name, prompt in PROMPTS.items():
        t = time.time()
        try:
            text, tok = fw.chat(prompt, MODEL, system=SYS, max_tokens=256, reasoning_effort=effort)
            total += tok
            print(f"{name:9} tok={tok:4}  {time.time() - t:5.1f}s  :: {text[:70]!r}")
        except Exception as e:
            print(f"{name:9} ERROR {e}")
    print(f"total tokens: {total}")
