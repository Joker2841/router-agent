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

# Application code
COPY code_exec.py classifier.py local_llm.py solvers.py fireworks_client.py agent.py ./

# Bundled local model (downloaded via download_model.sh before building).
# Copy ONLY the active model to keep the image small (not the spare 1B/4B files).
COPY models/model.gguf ./models/model.gguf

# Defaults (harness overrides FIREWORKS_* and ALLOWED_MODELS at runtime).
# n_ctx kept small on purpose: tasks are short, and a smaller KV cache keeps us
# comfortably inside the 4 GB grading box (avoids OOM with the 4B model).
ENV MODE=moonshot \
    LOCAL_MODEL_PATH=/app/models/model.gguf \
    LOCAL_N_CTX=2048 \
    PYTHONUNBUFFERED=1

CMD ["python3", "agent.py"]
