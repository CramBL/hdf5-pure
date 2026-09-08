set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default:
    @just --list release

# Set VERSION in the manifests and the changelog, and package. Commits nothing.
prepare VERSION *ARGS:
    {{ uv }} release prepare {{ VERSION }} {{ ARGS }}

# Commit the prepared release on its branch, prompt, then push it and open the pull request.
pr VERSION:
    {{ uv }} release pr {{ VERSION }}

# Publish the manifest's version from main: crates.io, the tag, the GitHub release. Resumable.
publish *ARGS:
    {{ uv }} release publish {{ ARGS }}
