# /// script
# requires-python = ">=3.11"
# dependencies = ["mkdocs-material==9.7.6"]
# ///
"""MkDocs with the site's dependencies, locked in `docs.py.lock`.

    uv run scripts/docs.py build --strict
    uv run scripts/docs.py serve
"""

from mkdocs.__main__ import cli

cli()
