set working-directory := ".."

# The tooling under scripts/python, with its dev group.
uv := "uv run --project scripts/python --locked --group dev"

default: fmt-check lint test

fmt:
    {{ uv }} ruff format scripts/python

fmt-check:
    {{ uv }} ruff format --check scripts/python

lint:
    {{ uv }} ruff check scripts/python

test *ARGS:
    {{ uv }} pytest scripts/python {{ ARGS }}

# Refresh uv.lock after changing pyproject.toml.
lock:
    uv lock --project scripts/python
