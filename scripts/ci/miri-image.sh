#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ci/miri.env
command=${1:?usage: miri-image.sh reference|build|publish [IMAGE]}
inputs=(ci/miri.env scripts/nightly.txt docker/ci/miri/* scripts/ci/miri-image.sh)
channel=$(cat scripts/nightly.txt)
reference="ghcr.io/crambl/hdf5-pure-miri:$channel"
if [[ $command == reference ]]; then
    printf '%s\n' "$reference"
    exit 0
fi
case "$command" in build|publish) ;; *) exit 1 ;; esac
image=${2:-$reference}
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp --parents "${inputs[@]}" "$context"
docker build --platform linux/amd64 --file "$context/docker/ci/miri/Dockerfile" \
    --build-arg "BASE_IMAGE=$MIRI_PARENT" --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
    --tag "$image" "$context"
size=$(docker image inspect --format '{{.Size}}' "$image")
printf 'Miri image: %s\nImage size: %s bytes\nCompiler: %s\n' "$image" "$size" "$channel"
if [[ $command == publish ]]; then
    docker push "$image"
    printf 'Published image: %s\nCompiler: %s\n' "$image" "$channel" | tee -a "$GITHUB_STEP_SUMMARY"
fi
