# Fine-tuning a small local model to win Track 1

Goal: a model small enough to finish on the slow 2 vCPU grading box, but fine-tuned
so it clears the strict judge on all eight categories locally at zero tokens.

Base model: `Qwen/Qwen2.5-1.5B-Instruct` (fast on CPU, strong for its size). If it
falls short on accuracy and timing allows, retry with `Qwen/Qwen2.5-3B-Instruct`.

## Step 1: generate the dataset (on any machine with the Fireworks key)

```bash
cd ~/projects/router-agent
FIREWORKS_API_KEY="$FW_KEY" python3 finetune/gen_data.py --per-category 150
# -> finetune/train.jsonl  (~1200 verified examples across the 8 categories)
```

Check it looks sane:
```bash
wc -l finetune/train.jsonl
head -1 finetune/train.jsonl
```

## Step 2: fine-tune (on the AMD notebook)

Upload `finetune/train.jsonl` and the two scripts to the notebook, then:
```bash
pip install -U transformers peft datasets accelerate
python3 finetune/train_lora.py \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --data finetune/train.jsonl \
  --out finetune/merged
# -> finetune/merged/  (merged fp16 model, ready to convert)
```

This is a small LoRA run; on an MI300X it should take well under an hour.

## Step 3: convert to GGUF and quantize (on the notebook or host)

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python3 llama.cpp/convert_hf_to_gguf.py finetune/merged --outfile finetune/model-f16.gguf --outtype f16
# build the quantizer once:
cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build --target llama-quantize -j
llama.cpp/build/bin/llama-quantize finetune/model-f16.gguf models/model.gguf Q4_K_M
ls -lh models/model.gguf
```

## Step 4: plug it in and test honestly

The agent already reads `models/model.gguf`. Rebuild and test with our strict judge
and core-pinned timing:
```bash
docker build -t router-agent:local .

# all-local judge accuracy on the hard eval (0 API for the agent)
docker run --rm --cpuset-cpus="0,1" --memory=4g \
  -v "$(pwd)/harder_tasks.json:/input/tasks.json:ro" -v "$(pwd)/_local_out:/output" \
  -e MODE=moonshot router-agent:local
GOLD_FILE=harder_eval.json FIREWORKS_API_KEY="$FW_KEY" python3 grade_llm.py _local_out/results.json
```

Read the timing (must stay well under 600s for ~17 tasks) and the judge accuracy per
category. Then pick the escalate set: whatever category the fine-tuned model still
fails, put it in `ESCALATE_CATEGORIES`; ideally that is empty (fully local, zero tokens)
or just `logic`.

## Step 5: submit

Set `ESCALATE_CATEGORIES` in the Dockerfile accordingly, then:
```bash
IMAGE=docker.io/jayan0512/router-agent:latest ./build_and_push.sh
```
Re-save. We have grading iterations to spare now, so validate against real grading and
tune the escalate set once or twice.

## Notes
- Keep the aggressive time guard (`TIME_BUDGET_S`) so a slow box escalates leftovers
  instead of timing out.
- Image with a 1.5B Q4 model is about 1.2 GB, well under the 5 GB safe pull limit.
- The fine-tune learns task behavior, not memorized answers, so it generalizes to the
  refreshed final prompts.
