"""Local Gemma wrapper built on llama.cpp and CPU, which costs zero Fireworks tokens.

The model file is bundled in the image under models/ and its path comes from
LOCAL_MODEL_PATH. We keep the interface tiny: .generate(prompt) and .chat().
Loading is lazy so container boot stays under 60s even if the first task is
handled deterministically and never touches the model.

Switch between the 1B and 4B models by changing LOCAL_MODEL_PATH. No code change is needed.
"""
from __future__ import annotations

import os


class LocalLLM:
    def __init__(self, model_path: str | None = None, n_ctx: int = 4096, n_threads: int | None = None):
        self.model_path = model_path or os.environ.get("LOCAL_MODEL_PATH", "models/model.gguf")
        self.n_ctx = int(os.environ.get("LOCAL_N_CTX", n_ctx))
        self.n_threads = n_threads or int(os.environ.get("LOCAL_N_THREADS", os.cpu_count() or 2))
        self._llm = None
        self._load_error = None

    @property
    def available(self) -> bool:
        if self._llm is not None:
            return True
        if self._load_error is not None:
            return False
        return os.path.exists(self.model_path)

    def _ensure(self):
        if self._llm is not None or self._load_error is not None:
            return
        import sys as _sys

        if not os.path.exists(self.model_path):
            self._load_error = FileNotFoundError(self.model_path)
            print(f"[local_llm] MODEL NOT FOUND at {self.model_path}", file=_sys.stderr)
            return
        try:
            from llama_cpp import Llama
        except Exception as e:
            self._load_error = e
            print(f"[local_llm] llama_cpp import failed: {e}", file=_sys.stderr)
            return

        common = dict(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=256,
            verbose=False,
        )
        # Prefer the GGUF's embedded chat template (Gemma 3 ships one); fall back
        # to the explicit "gemma" formatter only if auto-detection fails.
        for attempt in (dict(common), dict(common, chat_format="gemma")):
            try:
                self._llm = Llama(**attempt)
                print(f"[local_llm] loaded {self.model_path} "
                      f"(chat_format={attempt.get('chat_format', 'auto')})", file=_sys.stderr)
                return
            except Exception as e:
                self._load_error = e
                print(f"[local_llm] load attempt failed "
                      f"(chat_format={attempt.get('chat_format', 'auto')}): {e}", file=_sys.stderr)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0,
                 stop: list[str] | None = None, system: str | None = None) -> str:
        """Return the model's text answer, or "" if the model is unavailable."""
        self._ensure()
        if self._llm is None:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            out = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
            )
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""


class MockLLM:
    """Deterministic stand-in for local testing without a GGUF file.

    Returns canned, realistic outputs keyed on prompt content so we can exercise
    the program-of-thought + code paths on a machine with no model.
    """

    available = True

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0,
                 stop=None, system=None) -> str:
        p = prompt.lower()
        if "store has 240" in p or ("240" in p and "15%" in p):
            return "```python\nitems = 240\nitems -= items * 0.15\nitems -= 60\nprint(int(items))\n```"
        if "second-largest" in p or "second largest" in p:
            return ("```python\n"
                    "def second_largest(nums):\n"
                    "    u = sorted(set(nums))\n"
                    "    return u[-2] if len(u) >= 2 else None\n"
                    "```")
        if "get_max" in p:
            return ("```python\n"
                    "def get_max(nums):\n"
                    "    return max(nums)\n"
                    "```")
        if "sam" in p and "jo" in p and "lee" in p:
            # program-of-thought brute force for the pet puzzle
            return ("```python\n"
                    "import itertools\n"
                    "for c in itertools.permutations(['cat','dog','bird']):\n"
                    "    sam,jo,lee = c\n"
                    "    if sam!='bird' and jo=='dog':\n"
                    "        print('Lee owns the cat' if lee=='cat' else sam+' owns the cat')\n"
                    "```")
        if "capital of australia" in p:
            return "Canberra is the capital of Australia; it sits near Lake Burley Griffin."
        if "sentiment" in p:
            return "Mixed. Positive about battery life, negative about the fragile screen."
        if "named entit" in p or "maria sanchez" in p:
            return "Maria Sanchez (Person), Fireworks AI (Organization), Berlin (Location), last March (Date)."
        return "OK"


def get_llm() -> "LocalLLM | MockLLM":
    """Factory: real model if present, else Mock (for dev boxes without a GGUF)."""
    if os.environ.get("USE_MOCK_LLM") == "1":
        return MockLLM()
    llm = LocalLLM()
    if llm.available:
        return llm
    # No model file and not explicitly mocked: still return the real wrapper so
    # generate() returns "" and the pipeline escalates to Fireworks.
    return llm
