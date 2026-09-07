"""MkDocs with the site's dependencies.

just docs::build
just docs::serve
"""

import os

from mkdocs.__main__ import cli

from hdf5_pure_scripts import repo_root


def main() -> None:
    os.chdir(repo_root())
    cli()
