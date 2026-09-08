set working-directory := ".."

uv := "uv run --project scripts/python --locked"

# The gate CI runs: alerts on lines added since BASE, and every commit message in the range.
default BASE="origin/main": (added BASE)

# Download the style packages .vale.ini pins into .vale/styles/.
sync:
    {{ uv }} lint-prose sync

# The tracked Markdown files, the whole backlog.
docs:
    {{ uv }} lint-prose docs

# The comments of the Rust and Python sources, the whole backlog.
source:
    {{ uv }} lint-prose source

# The comments of the justfiles, TOML files and workflows, the whole backlog.
scripts:
    {{ uv }} lint-prose scripts

# Every commit message in RANGE.
commits RANGE="origin/main..HEAD":
    {{ uv }} lint-prose commits {{ RANGE }}

# Alerts on lines added since BASE, as GitHub annotations, failing on an error.
added BASE="origin/main":
    {{ uv }} lint-prose added {{ BASE }}

# Every surface, the whole backlog.
all: docs source scripts commits
