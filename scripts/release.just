set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default:
    @just --list release

# The release pull request for VERSION: bump, changelog, package, branch, PR.
prepare VERSION *ARGS:
    {{ uv }} release prepare {{ VERSION }} {{ ARGS }}

# Publish VERSION from main: crates.io, the tag, the GitHub release. Resumable.
publish VERSION *ARGS:
    {{ uv }} release publish {{ VERSION }} {{ ARGS }}
