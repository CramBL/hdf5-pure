#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ci/miri.env
command=${1:?usage: miri-image.sh fingerprint|build|publish [IMAGE]}
inputs=(ci/miri.env scripts/nightly.txt docker/ci/miri/* scripts/ci/miri-image.sh scripts/ci/run-miri.sh scripts/soundness.just)
fingerprint=$(sha256sum "${inputs[@]}" | sha256sum | cut -d' ' -f1)
if [[ $command == fingerprint ]]; then
    printf '%s\n' "$fingerprint"
    exit 0
fi
case "$command" in build|publish) ;; *) exit 1 ;; esac
image=${2:?missing image tag}
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp --parents "${inputs[@]}" "$context"
docker build --platform linux/amd64 --file "$context/docker/ci/miri/Dockerfile" \
    --build-arg "BASE_IMAGE=$MIRI_PARENT" --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
    --build-arg "INPUT_FINGERPRINT=$fingerprint" --tag "$image" "$context"
size=$(docker image inspect --format '{{.Size}}' "$image")
printf 'Miri image: %s\nImage size: %s bytes\nInputs: %s\n' "$image" "$size" "$fingerprint"
if [[ $command == publish ]]; then
    docker push "$image"
    digest=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image" |
        awk -v repository="${image%:*}@" 'index($0, repository) == 1 { print substr($0, length(repository) + 1) }')
    [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]
    {
        printf '### miri\n\n'
        printf '| Input | Value |\n|---|---|\n'
        printf '| Image | `%s@%s` |\n' "${image%:*}" "$digest"
        printf '| Tag | `%s` |\n' "$image"
        printf '| Platform | `linux/amd64` |\n'
        printf '| Parent | `%s` |\n' "$MIRI_PARENT"
        printf '| Inputs | `%s` |\n' "$fingerprint"
        printf '| Bytes | `%s` |\n' "$size"
        printf '| Nightly | `%s` |\n' "$(cat scripts/nightly.txt)"
        printf '| Rustup / just | `%s` / `%s` |\n' "$RUSTUP_VERSION" "$JUST_VERSION"
        printf '| Sysroot | `/opt/miri-sysroot` |\n'
        printf '| Commit | `%s` |\n' "$(git rev-parse HEAD)"
        printf '| Run | %s/%s/actions/runs/%s/attempts/%s |\n\n' \
            "$GITHUB_SERVER_URL" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"
    } | tee "$RUNNER_TEMP/miri-image.md" >> "$GITHUB_STEP_SUMMARY"
fi
