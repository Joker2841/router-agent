# Tokenless: submission form content

## Title
Tokenless

## Short Description (50 to 255 characters)
Most agents try to spend fewer tokens. Tokenless spends none. It answers all nineteen Track 1 tasks across eight categories locally, clears the accuracy gate, and calls a hosted model zero times by turning math and logic into code and running a fine-tuned model on CPU.

## Long Description (600 to 2000 characters, 100 words min)
Track 1 rewards the agent that spends the fewest Fireworks tokens while staying above the accuracy gate. Local inference is free, so every hosted call is a point against you. Most teams try to make that call cheaper. We decided to never make it at all. Tokenless answers all nineteen hidden tasks, across all eight categories, entirely on its own, for zero tokens.

The engine is a small model we fine-tuned with LoRA on a dataset we generated ourselves, then quantized to run on CPU inside the container. It is trained to produce the exact format each category is graded on, so there is no wasted output. Around it sit solvers that do not guess. Math is written as a Python program, executed, and verified. Logic puzzles are brute-forced in code, with prose reasoning as a backup when the generated code is shaky. Generated and debugged code is run against tests before we trust it. Named entities are extracted by a dedicated spaCy model that is deterministic and fast. A hosted model stays wired in as a safety net, used only if a task ever comes back empty. It never did.

None of this came easy. Bigger models scored higher but timed out on the two vCPU judging box, so we moved to a fast fine-tuned 1.5B. Math kept returning blank until we found the model was writing notebook-style code that computed the answer and never printed it, which we fixed in the runner and watched math jump from zero to perfect. Multi-part questions and logic each needed their own fix. The payoff is every category solved locally, zero tokens spent, in a 2.8 GB image that runs inside 4 GB of RAM and two vCPUs and finishes with time to spare.

## Categories / Event Tracks
Track 1: Token-Efficient Routing Agent

## Technologies Used
Python, Docker (linux/amd64), llama.cpp, GGUF Q4_K_M quantization, Qwen2.5-1.5B-Instruct, LoRA fine-tuning with PEFT and Transformers, spaCy (en_core_web_md) for named entity recognition, program-of-thought code execution, Fireworks AI (safety-net escalation), AMD ROCm and Kaggle T4 for training.
