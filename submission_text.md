# lablab.ai submission text (Track 1)

## Submission Title
Zero-Token Routing Agent

## Short Description
A local-first routing agent for Track 1. It solves math, logic, and code locally with a small Gemma model and verified execution, and answers language tasks on the spot. The goal is zero Fireworks tokens while staying above the accuracy gate.

## Long Description
Zero-Token Routing Agent is our entry for Track 1. It handles a wide range of tasks while spending as few Fireworks tokens as possible. The idea is simple. Not every task needs a paid model, so we do the work locally whenever we can, and only call a Fireworks model when we really have to.

The agent runs a small Gemma model, Gemma 3 4B, inside the container using llama.cpp on CPU. When a task arrives, we first classify it with plain rules that cost nothing. Math and logic problems are handled by having the model write a short Python program that we run and check, so the answer is proven instead of guessed. Code tasks are generated locally and compiled to make sure they actually work. Everyday language tasks like sentiment, summarization, named entity recognition, and factual questions are answered directly by the local model.

Because every one of these answers is produced inside the container, they count as zero tokens under the hackathon rules. That puts us in the best possible spot on the leaderboard, which ranks passing teams by how few tokens they use. If a local answer is ever missing, the agent can fall back to a Fireworks model, but in practice it almost never needs to.

We used AMD Developer Cloud for our development and testing runs, and the final container is built for the AMD judged environment, which has 4 GB of RAM and 2 vCPUs. When the agent does need to escalate, it calls Fireworks models that are served on AMD hardware. On our own 20 task test set the agent reached 95 percent accuracy at zero tokens, so it clears the 80 percent gate with room to spare while keeping the token count at zero.

## Categories / Event Tracks
Track 1: Hybrid Token-Efficient Routing Agent

## Technologies Used
Python, Docker, Gemma 3 (4B, GGUF), llama.cpp, Fireworks AI API, AMD Developer Cloud, AMD ROCm
