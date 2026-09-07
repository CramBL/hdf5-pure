set working-directory := ".."

default:
    @just --list release

# The release pull request for VERSION: bump, changelog, package, branch, PR.
prepare VERSION *ARGS:
    uv run scripts/release.py prepare {{ VERSION }} {{ ARGS }}

# Publish VERSION from main: crates.io, the tag, the GitHub release. Resumable.
publish VERSION *ARGS:
    uv run scripts/release.py publish {{ VERSION }} {{ ARGS }}
