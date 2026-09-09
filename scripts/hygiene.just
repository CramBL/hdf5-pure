set working-directory := ".."

default: shear zizmor typos

shear:
    cargo shear

# Static analysis of the workflows under .github.
zizmor:
    zizmor .github

# Spelling, over every tracked file. typos.toml lists the words the format uses.
typos:
    typos

# Every link in the documentation and the doc comments, online. The Links
# workflow runs it monthly.
lychee:
    git ls-files -- {{ lychee_paths }} | lychee --files-from -

# The links on the lines added since BASE, online. No other URL is requested,
# so a host that is down elsewhere does not fail the run.
lychee-added BASE="origin/main":
    #!/usr/bin/env bash
    set -euo pipefail
    diff="$(git diff -U0 "{{ BASE }}...HEAD" -- {{ lychee_paths }})"
    added="$(printf '%s\n' "$diff" | grep -E '^\+' | grep -vE '^\+\+\+ ' || true)"
    if [ -z "$added" ]; then
        exit 0
    fi
    printf '%s\n' "$added" | lychee -

# The paths both lychee recipes read, as `git ls-files` and `git diff` take
# them. lychee reads a `.rs` file as text, so it requests the URLs in a doc
# comment.
lychee_paths := "'*.md' '*.rs'"
