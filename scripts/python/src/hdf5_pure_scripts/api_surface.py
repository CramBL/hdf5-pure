"""Report types reachable through the public API that cannot be named.

    just api::api-surface

A type a consumer receives by value but cannot spell (a private module's
struct behind a `pub fn`) cannot be stored in a field or named in a signature,
so every such type is re-exported from lib.rs. This scans rustdoc's JSON for
the crate, which needs a nightly toolchain, and lists what slipped through.
"""

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

from hdf5_pure_scripts import repo_root


def rustdoc_json() -> dict:
    if (
        "nightly"
        not in subprocess.run(
            ["rustup", "toolchain", "list"], check=True, text=True, capture_output=True
        ).stdout
    ):
        sys.exit("needs a nightly toolchain: rustup toolchain install nightly")
    # Not `target/doc`: a `build.target-dir` in the user's cargo config moves it.
    metadata = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    path = Path(json.loads(metadata)["target_directory"]) / "doc" / "hdf5_pure.json"
    path.unlink(missing_ok=True)
    subprocess.run(
        [
            "cargo",
            "+nightly",
            "rustdoc",
            "--all-features",
            "--lib",
            "--",
            "-Zunstable-options",
            "--output-format",
            "json",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not path.is_file():
        sys.exit(f"rustdoc wrote no JSON at {path}")
    return json.loads(path.read_text())


def nameable_items(doc: dict, index: dict) -> set[int]:
    """Every item a consumer can name: the crate root, whatever public modules
    and `pub use` re-exports lead to, and so on down. rustdoc's JSON holds only
    public items unless asked otherwise, so reaching an item at all is enough."""
    nameable: set[int] = set()

    def walk(item_id: int) -> None:
        item = index.get(item_id)
        if item is None or item_id in nameable:
            return
        nameable.add(item_id)
        use = item["inner"].get("use")
        if use is not None:
            if use["id"] is not None:
                walk(use["id"])
            return
        module = item["inner"].get("module")
        if module is not None:
            for child_id in module["items"]:
                walk(child_id)

    walk(doc["root"])
    return nameable


def referenced(node, out: set[int]) -> None:
    """Every local type id an item's definition mentions, bounds included."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("resolved_path", "trait") and isinstance(value, dict):
                if "id" in value:
                    out.add(value["id"])
            referenced(value, out)
    elif isinstance(node, list):
        for element in node:
            referenced(element, out)


def owners(index: dict) -> dict[int, str]:
    """The name to report an item under, for the fields and associated items
    whose own name says nothing on its own."""
    owner: dict[int, str] = {}
    for item in index.values():
        inner = item["inner"]
        # A unit struct's or unit variant's kind is the bare string "unit", and
        # a tuple struct's is {"tuple": [...]}; only the dict forms carry fields.
        members = []
        name = item.get("name")
        if "struct" in inner:
            kind = inner["struct"]["kind"]
            if isinstance(kind, dict):
                members = kind.get("tuple") or kind.get("plain", {}).get("fields", [])
        elif "variant" in inner:
            kind = inner["variant"]["kind"]
            if isinstance(kind, dict):
                members = kind.get("tuple") or kind.get("struct", {}).get("fields", [])
        elif "impl" in inner:
            members = inner["impl"]["items"]
            # An impl block has no name of its own; its items belong to the type.
            for_type = inner["impl"]["for"].get("resolved_path", {})
            name = for_type.get("path", "").rsplit("::", 1)[-1]
        for member_id in members or []:
            if member_id is not None and name:
                owner[member_id] = name
    return owner


def leaks(doc: dict) -> tuple[int, dict[str, set[str]]]:
    """Unnameable local types by path, each with the public items that reach it."""
    index = {int(k): v for k, v in doc["index"].items()}
    paths = {int(k): v for k, v in doc["paths"].items()}
    nameable = nameable_items(doc, index)
    owner = owners(index)
    found: dict[str, set[str]] = collections.defaultdict(set)
    for item_id, item in index.items():
        inner = item["inner"]
        # A trait's own node carries its supertrait bounds, where this crate
        # names a private `sealed::Sealed` on purpose: the point of that pattern
        # is that no one outside can implement the trait. Its methods are
        # scanned as usual.
        if "trait" in inner:
            continue
        ids: set[int] = set()
        referenced(inner, ids)
        for type_id in ids:
            if type_id in nameable or type_id not in paths:
                continue
            if paths[type_id].get("crate_id", 0) != 0:
                continue
            name = item.get("name") or "?"
            if item_id in owner:
                name = f"{owner[item_id]}::{name}"
            found["::".join(paths[type_id]["path"])].add(name)
    return len(nameable), found


def main() -> None:
    os.chdir(repo_root())
    reached, found = leaks(rustdoc_json())
    print(f"reached {reached} public items from the crate root")
    if not found:
        print("no unnameable types reachable through it")
        return
    print("\nreachable by value, impossible to name:\n")
    for type_path, users in sorted(found.items()):
        print(f"  {type_path}")
        print(f"      through: {', '.join(sorted(users))}")
    print(
        "\nRe-export each from lib.rs (sealing it #[non_exhaustive] where the format\n"
        "can grow), or take it out of the public signature that reaches it."
    )
    sys.exit(1)
