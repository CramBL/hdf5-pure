set working-directory := ".."

uv := "uv run --project scripts/python --locked"

default: build

# The site under site/, failing on a broken link or a page missing from the nav.
build *ARGS:
    {{ uv }} docs build --strict {{ ARGS }}

# A local preview at http://127.0.0.1:8000, rebuilt on every change.
serve *ARGS:
    {{ uv }} docs serve {{ ARGS }}
