# Zero-Token Routing Agent

A local-first, token-efficient routing agent for the AMD Developer Hackathon ACT II, Track 1 (Hybrid Token-Efficient Routing Agent).

The agent answers a wide range of tasks using a small Gemma model bundled inside the container, together with deterministic execution for tasks that can be checked. Every answer produced locally costs zero Fireworks tokens. The agent only calls a Fireworks model as a fallback when a local answer cannot be produced. Under the hackathon scoring rules, this targets a passing accuracy score at zero tokens, which is the best possible position on the leaderboard.

## How it works

The agent classifies each task with lightweight rules that cost nothing, then routes it to the right local handler.

1. Math and logic. The model reasons through the problem, and for arithmetic it writes a short Python program that the agent runs and verifies, so the answer is computed rather than guessed.
2. Code generation and debugging. The model produces the function, which the agent compiles to confirm it runs.
3. Language tasks. Sentiment, summarization, named entity recognition, and factual questions are answered directly by the local model.

If a local answer is missing, the agent can escalate to a Fireworks model. In practice this is rarely needed.

## Project structure

| File | Purpose |
| --- | --- |
| `agent.py` | Container entry point. Reads `/input/tasks.json`, runs the pipeline, writes `/output/results.json`. |
| `classifier.py` | Zero-token task classification. |
| `solvers.py` | Local handlers for each category, including program-of-thought and verified execution. |
| `code_exec.py` | Sandboxed Python execution with time and memory limits. |
| `local_llm.py` | Gemma wrapper built on llama.cpp, running on CPU. |
| `fireworks_client.py` | Fallback client for the Fireworks API. |
| `eval.py`, `eval_set.json` | A 20 task local benchmark with an automatic grader. |
| `Dockerfile` | Builds the submission image for linux/amd64. |

## Grading environment

The judged container runs on 4 GB of RAM, 2 vCPUs, and no GPU. The local model runs on CPU. The image bundles a quantized Gemma 3 model in GGUF format, and boots in well under the 60 second limit.

## Setup

The project runs from a Linux environment such as WSL2. Docker is required to build and run the container.

```bash
# 1. Download the local model into models/model.gguf
pip install -U huggingface_hub
./download_model.sh 4b

# 2. Run the local benchmark to check accuracy
python3 eval.py

# 3. Build and run the container the way the harness does
IMAGE=router-agent:local ./run_local.sh
```

## Building and publishing the image

```bash
docker login
IMAGE=docker.io/<your-user>/router-agent:latest ./build_and_push.sh
```

Make the registry repository public so the judging system can pull it. The image must include a linux/amd64 manifest, which `build_and_push.sh` handles.

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODE` | `moonshot` | `moonshot` targets zero tokens. `hybrid` escalates unverified hard tasks. |
| `LOCAL_MODEL_PATH` | `/app/models/model.gguf` | Path to the bundled model. |
| `LOCAL_N_CTX` | `2048` | Context length. Small on purpose to keep memory low. |
| `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` | injected | Provided by the harness at evaluation time. |

## License

MIT
