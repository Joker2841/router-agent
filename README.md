# Tokenless

A local-first routing agent for the AMD Developer Hackathon ACT II, Track 1. It answers all nineteen hidden tasks, across all eight categories, entirely on its own, and spends zero Fireworks tokens.

## The idea

Track 1 ranks agents by how few Fireworks tokens they spend while staying above the accuracy gate. Local inference is free, so every hosted call is a point against you. Instead of making that call cheaper, Tokenless never makes it. A hosted model stays wired in only as a safety net, used if a task ever comes back empty. In testing it never triggered.

## How it works

A small model, fine-tuned with LoRA on a dataset we generated and then quantized to run on CPU inside the container, handles the language tasks. It is trained to produce the exact format each category is graded on. Deterministic solvers sit around it so the checkable categories are computed, not guessed:

1. Math is written as a Python program, executed, and verified.
2. Logic puzzles are brute-forced in code with itertools, with prose reasoning as a fallback when the generated code is unreliable.
3. Generated and debugged code is run against tests before it is trusted.
4. Named entities are extracted by a dedicated spaCy model.
5. Sentiment, summarization, and factual questions are answered directly by the fine-tuned model.

Everything runs inside 4 GB of RAM and two vCPUs, with no GPU, and finishes well within the ten minute limit.

## Project structure

| File | Purpose |
| --- | --- |
| `agent.py` | Container entry point. Reads `/input/tasks.json`, runs the pipeline, writes `/output/results.json`. |
| `classifier.py` | Zero-token task classification. |
| `solvers.py` | Local handlers for each category, including program-of-thought math, logic-as-code, and spaCy entities. |
| `code_exec.py` | Sandboxed Python execution with time and memory limits, and a REPL-style print fix. |
| `local_llm.py` | Fine-tuned model wrapper built on llama.cpp, running on CPU. |
| `fireworks_client.py` | Safety-net client for the Fireworks API. |
| `finetune/` | Data generation and LoRA training, including the Kaggle training notebook. |
| `Dockerfile` | Builds the submission image for linux/amd64. |

## Run it

```bash
docker run --rm --cpuset-cpus="0,1" --memory=4g \
  -v $PWD/sample_tasks.json:/input/tasks.json \
  -v /tmp/o:/output docker.io/jayan0512/router-agent:latest
cat /tmp/o/results.json
```

The run reads `/input/tasks.json`, writes `/output/results.json`, and reports `0 Fireworks tokens` in its log.

## The model

The engine is a fine-tuned Qwen2.5-1.5B-Instruct, trained with LoRA on a dataset generated for the eight task categories, merged, and quantized to GGUF Q4_K_M so it runs on CPU. Training was done on GPU (AMD ROCm, with a Kaggle T4 as a backup). The quantized model is bundled inside the Docker image and is not committed to this repo (see `.gitignore`); place it at `models/model.gguf` before building.

## Results

Every category solved locally. One hundred percent on the official sample tasks and on our extended tests for code generation, debugging, and logic. Zero Fireworks tokens spent. A 2.8 GB image, well under the size limit.

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODE` | `hybrid` | Answers locally; the safety net is reachable only for empty answers or if the time guard trips. |
| `ESCALATE_CATEGORIES` | empty | Categories to always send to Fireworks. Empty for zero tokens. |
| `LOCAL_MODEL_PATH` | `/app/models/model.gguf` | Path to the bundled model. |
| `FIREWORKS_API_KEY`, `ALLOWED_MODELS` | injected | Provided by the harness at evaluation time. |

## License

MIT