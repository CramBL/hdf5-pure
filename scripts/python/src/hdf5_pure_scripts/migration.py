# The one-time 0.47.0 ownership migration check, removed after the first hdf5-pure-core release.
import io
import json
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from hdf5_pure_scripts import repo_root, rustdoc_json
from hdf5_pure_scripts.api_profiles import CORE, FACADE, cargo_feature_args

MIGRATION_LINTS = {
    "enum_missing": "warn",
    "struct_missing": "warn",
    "pub_module_level_const_missing": "warn",
}


def migration_baseline(manifest: dict, baseline: str) -> bool:
    metadata = manifest["package"].get("metadata", {})
    migration = metadata.get("hdf5-pure-api-migration")
    lints = metadata.get("cargo-semver-checks", {}).get("lints", {})
    if migration is None:
        if any(name in lints for name in MIGRATION_LINTS):
            raise ValueError("migration lint settings remain without a version-scoped bridge")
        return False
    if baseline != migration["from-version"]:
        raise ValueError(
            f"the ownership migration bridge expired after {migration['from-version']}; "
            "remove it and check both published packages normally"
        )
    if lints != MIGRATION_LINTS:
        raise ValueError("migration lint settings differ from the three known ownership artifacts")
    return True


def root_paths(doc: dict) -> set[tuple[str, str]]:
    index = doc["index"]
    paths = doc["paths"]
    found = set()

    def walk(module_id: int, prefix: str) -> None:
        for child_id in index[str(module_id)]["inner"]["module"]["items"]:
            child = index[str(child_id)]
            use = child["inner"].get("use")
            if use is None:
                name = child["name"]
                target_id = child_id
                kind = next(iter(child["inner"]))
            else:
                if use["is_glob"] or use["id"] is None:
                    raise ValueError(f"cannot compare re-export {use['source']}")
                name = use["name"]
                target_id = use["id"]
                path = paths.get(str(target_id))
                kind = (
                    path["kind"] if path is not None else next(iter(index[str(target_id)]["inner"]))
                )
            item_path = f"{prefix}::{name}"
            if kind in {"enum", "struct", "constant"}:
                found.add((item_path, kind))
            if kind == "module" and str(target_id) in index:
                walk(target_id, item_path)

    walk(doc["root"], "hdf5_pure")
    return found


def public_root_items(doc: dict) -> dict[str, int]:
    index = doc["index"]
    return {
        index[str(item_id)]["inner"].get("use", {}).get("name")
        or index[str(item_id)]["name"]: index[str(item_id)]["inner"].get("use", {}).get("id")
        or item_id
        for item_id in index[str(doc["root"])]["inner"]["module"]["items"]
    }


def normalized(value, doc: dict):
    if isinstance(value, dict):
        if "resolved_path" in value:
            path = value["resolved_path"]
            resolved = doc.get("paths", {}).get(str(path.get("id")), {}).get("path")
            components = resolved or path["path"].split("::")
            if components[0] in {"hdf5_pure", "hdf5_pure_core"}:
                components = ["owner", *components[1:]]
            return {
                "resolved_path": {
                    "path": "::".join(components),
                    "args": normalized(path["args"], doc),
                }
            }
        return {key: normalized(part, doc) for key, part in value.items() if key != "id"}
    if isinstance(value, list):
        return [normalized(part, doc) for part in value]
    return value


def public_type_shape(doc: dict, item_id: int) -> dict[str, object]:
    index = doc["index"]
    item = index[str(item_id)]
    inner = item["inner"]
    kind = "struct" if "struct" in inner else "enum"
    definition = inner[kind]
    shape: dict[str, object] = {
        "kind": kind,
        "non_exhaustive": "non_exhaustive" in item["attrs"],
        "repr": next(
            (attr["repr"] for attr in item["attrs"] if isinstance(attr, dict) and "repr" in attr),
            None,
        ),
        "generics": normalized(definition["generics"], doc),
        "fields": {},
        "variants": {},
        "variant_order": [],
        "inherent": {},
        "traits": [],
    }

    def fields(field_ids: list[int | None], public_only: bool = True) -> dict[str, object]:
        return {
            index[str(field_id)]["name"] or str(position): normalized(
                index[str(field_id)]["inner"]["struct_field"], doc
            )
            for position, field_id in enumerate(field_ids)
            if field_id is not None
            and (not public_only or index[str(field_id)]["visibility"] == "public")
        }

    if kind == "struct":
        struct_kind = definition["kind"]
        if isinstance(struct_kind, dict):
            shape["fields"] = fields(
                struct_kind.get("tuple") or struct_kind.get("plain", {}).get("fields", [])
            )
    else:
        variants = {}
        variant_order = []
        for variant_id in definition["variants"]:
            variant = index[str(variant_id)]
            variant_order.append(variant["name"])
            variant_kind = variant["inner"]["variant"]["kind"]
            variant_fields = {}
            if isinstance(variant_kind, dict):
                variant_fields = fields(
                    variant_kind.get("tuple") or variant_kind.get("struct", {}).get("fields", []),
                    False,
                )
            variants[variant["name"]] = {
                "kind": "unit" if isinstance(variant_kind, str) else next(iter(variant_kind)),
                "fields": variant_fields,
                "discriminant": normalized(variant["inner"]["variant"]["discriminant"], doc),
            }
        shape["variants"] = variants
        shape["variant_order"] = variant_order
    inherent = {}
    traits = []
    for impl_id in definition["impls"]:
        implementation = index[str(impl_id)]["inner"]["impl"]
        trait = implementation["trait"]
        if trait is None:
            for member_id in implementation["items"]:
                member = index[str(member_id)]
                if member["visibility"] != "public":
                    continue
                member_inner = member["inner"]
                member_kind = next(iter(member_inner))
                member_value = member_inner[member_kind]
                if member_kind == "function":
                    member_value = {
                        "sig": member_value["sig"],
                        "generics": member_value["generics"],
                        "header": member_value["header"],
                    }
                inherent[member["name"]] = (member_kind, normalized(member_value, doc))
        elif (
            implementation["blanket_impl"] is None
            and not implementation["is_synthetic"]
            and implementation["for"].get("resolved_path", {}).get("id") == item_id
        ):
            traits.append(normalized(trait, doc))
    shape["inherent"] = inherent
    shape["traits"] = sorted(traits, key=lambda value: json.dumps(value, sort_keys=True))
    return shape


def migration_shapes(baseline: dict, candidate: dict, core: dict) -> list[str]:
    baseline_items = public_root_items(baseline)
    candidate_items = public_root_items(candidate)
    core_items = public_root_items(core)
    moved = root_paths(core)
    failures = []
    # Pre-extraction main defines InvalidObjectHeaderMessageSize. Published 0.47.0 does not.
    # The allowance covers that prior delta, not PR #635's extraction.
    allowed_added_variants = {"FormatError": {"InvalidObjectHeaderMessageSize"}}
    for path, kind in sorted(moved):
        name = path.rsplit("::", 1)[-1]
        if name not in baseline_items:
            if name in candidate_items:
                failures.append(f"{name}: new core root item")
            continue
        if name not in candidate_items:
            failures.append(f"{name}: missing from facade")
            continue
        if kind == "constant":
            old = baseline["index"][str(baseline_items[name])]["inner"]["constant"]
            new = core["index"][str(core_items[name])]["inner"]["constant"]
            if normalized(old, baseline) != normalized(new, core):
                failures.append(f"{name}: constant changed")
            continue
        old = public_type_shape(baseline, baseline_items[name])
        new = public_type_shape(core, core_items[name])
        for variant in allowed_added_variants.get(name, set()):
            new["variants"].pop(variant, None)
            if variant in new["variant_order"]:
                new["variant_order"].remove(variant)
        for part in old:
            if old[part] != new[part]:
                failures.append(f"{name}: {part} changed")
    return failures


def check_migration_paths(baseline: str, selected: dict) -> None:
    request = Request(
        f"https://crates.io/api/v1/crates/{FACADE}/{baseline}/download",
        headers={"User-Agent": "hdf5-pure-api-check"},
    )
    with tempfile.TemporaryDirectory(prefix="hdf5-pure-api-") as directory:
        root = Path(directory)
        with urlopen(request, timeout=20) as response:
            with tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz") as archive:
                for member in archive:
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(f"unexpected archive entry: {member.name}")
                    if not (root / member.name).resolve().is_relative_to(root):
                        raise ValueError(f"archive entry outside extraction root: {member.name}")
                    archive.extract(member, root)
        features = cargo_feature_args(selected)
        baseline_doc = rustdoc_json.build(
            root / f"{FACADE}-{baseline}" / "Cargo.toml", root / "baseline", features, False
        )
        candidate_doc = rustdoc_json.build(repo_root() / "Cargo.toml", root / "candidate", features)
        core_doc = rustdoc_json.build(
            repo_root() / "crates" / CORE / "Cargo.toml",
            root / "candidate",
            cargo_feature_args({"default_features": False, "features": selected["core_features"]}),
        )
        missing = root_paths(baseline_doc) - root_paths(candidate_doc)
        shape_failures = migration_shapes(baseline_doc, candidate_doc, core_doc)
    if missing:
        items = "\n".join(f"  {kind} {path}" for path, kind in sorted(missing))
        raise ValueError(f"published root paths disappeared during ownership migration:\n{items}")
    if shape_failures:
        raise ValueError(
            "types moved to hdf5-pure-core differ from the release:\n" + "\n".join(shape_failures)
        )
