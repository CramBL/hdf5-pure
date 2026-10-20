#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

command=${1:?usage: cross-image.sh build|publish FAMILY [IMAGE]}
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
case "$command" in
    build|publish)
        image=${3:?missing image tag}
        context=$(mktemp -d)
        trap 'rm -rf "$context"' EXIT
        cp --parents "${inputs[@]}" "$context"
        docker pull --platform "$platform" "$parent"
        docker image inspect --format '{{json .Config}}' "$parent" | jq --raw-output '
            (.Env[] | capture("^(?<key>[^=]+)=(?<value>.*)$") |
                "ENV \(.key)=\(.value | tojson)"),
            (.Labels | to_entries[] | "LABEL \(.key | tojson)=\(.value | tojson)"),
            "CMD \(.Cmd | tojson)"
        ' > "$context/parent-config"
        sed "/^FROM scratch$/r $context/parent-config" docker/ci/cross/Dockerfile \
            > "$context/docker/ci/cross/Dockerfile"
        docker build --platform "$platform" --file "$context/docker/ci/cross/Dockerfile" \
            --build-arg "BASE_IMAGE=$parent" --build-arg "CMAKE_VERSION=$version" \
            --build-arg "CMAKE_URL=$url" --build-arg "CMAKE_SHA256=$sha256" \
            --build-arg "SOURCE_REVISION=$(git rev-parse HEAD)" \
            --tag "$image" "$context"
        if [[ $command == publish ]]; then
            docker push "$image"
            printf 'Published cross image: %s\nCMake: %s\n' "$image" "$version" \
                | tee -a "$GITHUB_STEP_SUMMARY"
        fi
        ;;
    *) echo "Unknown command: $command" >&2; exit 1 ;;
esac
