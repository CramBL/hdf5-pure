set working-directory := ".."

default: build

# The site under site/, failing on a broken link or a page missing from the nav.
build *ARGS:
    uv run --locked scripts/docs.py build --strict {{ ARGS }}

# A local preview at http://127.0.0.1:8000, rebuilt on every change.
serve *ARGS:
    uv run --locked scripts/docs.py serve {{ ARGS }}

# Refresh the lockfile after changing the script's dependencies.
lock:
    uv lock --script scripts/docs.py
