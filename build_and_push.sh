#!/usr/bin/env bash
# Build a linux/amd64 image and push it to a public registry.
# Run after download_model.sh has produced models/model.gguf.
#
# Usage:
#   IMAGE=docker.io/<your-user>/router-agent:latest ./build_and_push.sh
set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE, for example docker.io/<your-user>/router-agent:latest}"

if [[ ! -f models/model.gguf ]]; then
  echo "models/model.gguf missing. Run ./download_model.sh first." >&2
  exit 1
fi

docker buildx build --platform linux/amd64 -t "$IMAGE" --push .

echo "Pushed $IMAGE"
echo "Local image size (keep under 10 GB):"
docker images "$IMAGE" --format '{{.Size}}'
echo "Ensure the repository is public in your registry settings."
