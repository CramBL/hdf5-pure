#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source ci/matio.env
command=${1:?usage: matio-image.sh fingerprint|build|publish [IMAGE]}
inputs=(ci/matio.env rust-toolchain.toml docker/ci/matio/* scripts/ci/install-mold.sh scripts/ci/matio-image.sh)
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
docker build --platform linux/amd64 --file "$context/docker/ci/matio/Dockerfile" \
    --build-arg "BASE_IMAGE=$MATIO_PARENT" --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
    --build-arg "INPUT_FINGERPRINT=$fingerprint" --tag "$image" "$context"
size=$(docker image inspect --format '{{.Size}}' "$image")
printf 'Matio image: %s\nImage size: %s bytes\nInputs: %s\n' "$image" "$size" "$fingerprint"
if [[ $command == publish ]]; then
    docker push "$image"
    digest=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image" |
        awk -v repository="${image%:*}@" 'index($0, repository) == 1 { print substr($0, length(repository) + 1) }')
    [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]
    {
        printf '### matio\n\n'
        printf '| Input | Value |\n|---|---|\n'
        printf '| Image | `%s@%s` |\n' "${image%:*}" "$digest"
        printf '| Tag | `%s` |\n' "$image"
        printf '| Platform | `linux/amd64` |\n'
        printf '| Parent | `%s` |\n' "$MATIO_PARENT"
        printf '| Inputs | `%s` |\n' "$fingerprint"
        printf '| Bytes | `%s` |\n' "$size"
        printf '| Just / nextest | `%s` / `%s` |\n' "$JUST_VERSION" "$NEXTEST_VERSION"
        printf '| Micromamba | `%s` |\n' "$MICROMAMBA_VERSION"
        printf '| libmatio | `%s` |\n' "$(sed -n 's|.*/\(libmatio-[^#]*\)#.*|\1|p' docker/ci/matio/linux-64.lock)"
        printf '| Mold | `%s` |\n' "$(bash scripts/ci/install-mold.sh --identity)"
        printf '| Reference lock | `%s` |\n' "$(sha256sum docker/ci/matio/linux-64.lock | cut -d' ' -f1)"
        printf '| Commit | `%s` |\n' "$(git rev-parse HEAD)"
        printf '| Run | %s/%s/actions/runs/%s/attempts/%s |\n\n' \
            "$GITHUB_SERVER_URL" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"
        docker run --rm "$image" rustc --version --verbose
    } | tee "$RUNNER_TEMP/matio-image.md" >> "$GITHUB_STEP_SUMMARY"
fi
