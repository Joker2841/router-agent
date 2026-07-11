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

# Push a single linux/amd64 image in classic Docker v2 schema2 format.
# provenance=false drops attestation layers, and oci-mediatypes=false forces
# Docker media types instead of OCI, which some registry pullers reject.
docker buildx build --platform linux/amd64 --provenance=false \
  --output "type=image,name=$IMAGE,oci-mediatypes=false,push=true" .

echo "Pushed $IMAGE"
echo "Local image size (keep under 10 GB):"
docker images "$IMAGE" --format '{{.Size}}'
echo "Ensure the repository is public in your registry settings."
