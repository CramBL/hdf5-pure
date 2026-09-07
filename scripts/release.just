set working-directory := ".."

default:
    @just --list release

release *ARGS:
    ./scripts/release.sh {{ ARGS }}

check-release-script:
    ./scripts/check-release-script.sh
