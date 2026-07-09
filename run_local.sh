#!/usr/bin/env bash
# Build the image and run it the way the judging harness does: mount an input
# file at /input/tasks.json, capture /output, and expect a valid results.json.
set -euo pipefail

IMAGE="${IMAGE:-router-agent:local}"
INPUT="${INPUT:-sample_input.json}"

docker build -t "$IMAGE" .

mkdir -p _local_out
docker run --rm \
  -v "$(pwd)/$INPUT:/input/tasks.json:ro" \
  -v "$(pwd)/_local_out:/output" \
  -e MODE="${MODE:-moonshot}" \
  -e FIREWORKS_API_KEY="${FIREWORKS_API_KEY:-}" \
  -e FIREWORKS_BASE_URL="${FIREWORKS_BASE_URL:-https://api.fireworks.ai/inference/v1}" \
  -e ALLOWED_MODELS="${ALLOWED_MODELS:-}" \
  "$IMAGE"

echo "Output:"
cat _local_out/results.json
