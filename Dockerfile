# AMD Hackathon Track 1 — linux/amd64, CPU-only, small image.
# Build (from WSL2, on an amd64 host this needs no --platform):
#   docker build -t <registry>/<user>/router-agent:latest .
FROM python:3.11-slim

WORKDIR /app

# Prebuilt CPU wheel for llama-cpp-python -> no compiler/toolchain in the image.
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
      -r requirements.txt

# spaCy + medium English model for deterministic, zero-token NER.
# click is pulled explicitly: spaCy's download CLI imports it and the resolver
# sometimes leaves it out, which breaks `spacy download`.
RUN pip install --no-cache-dir spacy click && python -m spacy download en_core_web_md

# Application code
COPY code_exec.py classifier.py local_llm.py solvers.py fireworks_client.py agent.py ./

# Bundled local model (downloaded via download_model.sh before building).
# Copy ONLY the active model to keep the image small (not the spare 1B/4B files).
COPY models/model.gguf ./models/model.gguf

# Defaults (harness overrides FIREWORKS_* and ALLOWED_MODELS at runtime).
# The judged box has only 2 vCPUs, so we bundle the fast 1B model and escalate
# the hard categories to Fireworks. n_ctx is small since tasks are short.
# Zero-token config: everything is answered locally. Fireworks stays reachable
# only as a catastrophic fallback (a task returns an empty answer, or the time
# guard trips) so a rare local failure never costs us the accuracy gate.
ENV MODE=hybrid \
    ESCALATE_CATEGORIES="" \
    VERIFY_ESCALATE_CATEGORIES="" \
    TIME_BUDGET_S=500 \
    REASONING_EFFORT=none \
    POT_SAMPLES=3 \
    LOCAL_MODEL_PATH=/app/models/model.gguf \
    LOCAL_N_CTX=2048 \
    LOCAL_N_THREADS=2 \
    PYTHONUNBUFFERED=1

CMD ["python3", "agent.py"]
