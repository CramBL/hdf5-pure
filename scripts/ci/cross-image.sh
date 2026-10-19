#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

command=${1:?usage: cross-image.sh fingerprint|build|publish FAMILY [IMAGE]}
family=${2:?missing image family}
case "$family" in i686|s390x) ;; *) echo "Unknown cross image: $family" >&2; exit 1 ;; esac

catalog() {
    awk -v table="[$1]" -v key="$2" '
        /^\[/ { selected = ($0 == table) }
        selected && $1 == key && $2 == "=" { sub(/^[^=]*= /, ""); gsub(/"/, ""); print }
    ' ci/images.toml
}

target=$(catalog "images.$family" target)
parent=$(catalog "images.$family.build" parent)
platform=$(catalog "images.$family.build" platform)
version=$(catalog cmake version)
url=$(catalog cmake url)
sha256=$(catalog cmake sha256)
test "$target" = "$family-unknown-linux-gnu" && test -n "$parent" && test "$platform" = linux/amd64
read -ra inputs <<< "$(catalog "images.$family.build" inputs | tr -d '[],')"
test "${#inputs[@]}" -gt 0
fingerprint=$({
    awk -v table="[images.$family.build]" '
        /^\[/ { selected = ($0 == table || $0 == "[cmake]") }
        selected { print }
    ' ci/images.toml
    sha256sum "${inputs[@]}"
} | sha256sum | cut -d' ' -f1)

case "$command" in
    fingerprint)
        printf '%s\n' "$fingerprint"
        ;;
    build|publish)
        image=${3:?missing image tag}
        context=$(mktemp -d)
        trap 'rm -rf "$context"' EXIT
        cp --parents "${inputs[@]}" "$context"
        docker build --platform "$platform" --file "$context/docker/ci/cross/Dockerfile" \
            --build-arg "BASE_IMAGE=$parent" --build-arg "CMAKE_VERSION=$version" \
            --build-arg "CMAKE_URL=$url" --build-arg "CMAKE_SHA256=$sha256" \
            --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
            --build-arg "INPUT_FINGERPRINT=$fingerprint" --tag "$image" "$context"
        docker run --rm "$image" bash /usr/local/libexec/verify-cmake.sh "$version"
        docker run --rm "$image" bash /usr/local/libexec/verify-linking.sh
        if [[ $command == publish ]]; then
            docker push "$image"
            digest=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image" |
                awk -v repository="${image%:*}@" 'index($0, repository) == 1 { print substr($0, length(repository) + 1) }')
            [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]]
            {
                printf '### %s\n\n' "$family"
                printf '| Input | Value |\n|---|---|\n'
                printf '| Image | `%s@%s` |\n' "${image%:*}" "$digest"
                printf '| Tag | `%s` |\n' "$image"
                printf '| Platform | `%s` |\n' "$platform"
                printf '| Parent | `%s` |\n' "$parent"
                printf '| Inputs | `%s` |\n' "$fingerprint"
                printf '| CMake | `%s` |\n' "$version"
                printf '| Mold | `%s` |\n' "$(bash scripts/ci/install-mold.sh --identity)"
                printf '| Commit | `%s` |\n' "$(git rev-parse HEAD)"
                printf '| Run | %s/%s/actions/runs/%s/attempts/%s |\n\n' \
                    "$GITHUB_SERVER_URL" "$GITHUB_REPOSITORY" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"
            } | tee "$RUNNER_TEMP/cross-$family.md" >> "$GITHUB_STEP_SUMMARY"
        fi
        ;;
    *) echo "Unknown command: $command" >&2; exit 1 ;;
esac
