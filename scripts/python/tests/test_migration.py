import io
import tarfile
from copy import deepcopy
from unittest.mock import patch

import pytest

from hdf5_pure_scripts import migration
from hdf5_pure_scripts.api_profiles import profile
from hdf5_pure_scripts.migration import (
    MIGRATION_LINTS,
    check_migration_paths,
    migration_baseline,
    migration_shapes,
    root_paths,
)


def test_migration_bridge_applies_only_to_the_release_before_core():
    manifest = {
        "package": {
            "metadata": {
                "cargo-semver-checks": {"lints": MIGRATION_LINTS},
                "hdf5-pure-api-migration": {"from-version": "0.47.0"},
            }
        }
    }
    assert migration_baseline(manifest, "0.47.0")
    with pytest.raises(ValueError, match="expired"):
        migration_baseline(manifest, "0.48.0")


def test_migration_lints_cannot_remain_without_the_bridge():
    manifest = {"package": {"metadata": {"cargo-semver-checks": {"lints": MIGRATION_LINTS}}}}
    with pytest.raises(ValueError, match="without a version-scoped bridge"):
        migration_baseline(manifest, "0.48.0")


def test_root_path_comparison_catches_unrelated_missing_types():
    baseline = {
        "root": 0,
        "index": {
            "0": {"inner": {"module": {"items": [1, 2]}}},
            "1": {"name": "Datatype", "inner": {"enum": {}}},
            "2": {"name": "File", "inner": {"struct": {}}},
        },
        "paths": {},
    }
    candidate = {
        "root": 0,
        "index": {
            "0": {"inner": {"module": {"items": [1]}}},
            "1": {
                "inner": {
                    "use": {
                        "name": "Datatype",
                        "source": "hdf5_pure_core::Datatype",
                        "id": 3,
                        "is_glob": False,
                    }
                }
            },
        },
        "paths": {"3": {"kind": "enum"}},
    }
    assert root_paths(baseline) - root_paths(candidate) == {("hdf5_pure::File", "struct")}


def core_shape_fixture():
    return {
        "root": 0,
        "index": {
            "0": {"name": "hdf5_pure", "inner": {"module": {"items": [1]}}},
            "1": {
                "name": "Thing",
                "attrs": [],
                "inner": {
                    "struct": {
                        "kind": {"plain": {"fields": [2]}},
                        "generics": {"params": [], "where_predicates": []},
                        "impls": [3, 5],
                    }
                },
            },
            "2": {
                "name": "value",
                "visibility": "public",
                "inner": {"struct_field": {"primitive": "u32"}},
            },
            "3": {
                "inner": {
                    "impl": {
                        "for": {"resolved_path": {"path": "Thing", "id": 1}},
                        "trait": None,
                        "items": [4],
                        "blanket_impl": None,
                        "is_synthetic": False,
                    }
                }
            },
            "4": {
                "name": "get",
                "visibility": "public",
                "inner": {
                    "function": {
                        "sig": {"inputs": [], "output": {"primitive": "u32"}},
                        "generics": {"params": [], "where_predicates": []},
                        "header": {"is_const": False},
                    }
                },
            },
            "5": {
                "inner": {
                    "impl": {
                        "for": {"resolved_path": {"path": "Thing", "id": 1}},
                        "trait": {"path": "Clone", "id": 9, "args": None},
                        "items": [],
                        "blanket_impl": None,
                        "is_synthetic": False,
                    }
                }
            },
        },
        "paths": {},
    }


@pytest.mark.parametrize(
    ("change", "part"),
    [
        (
            lambda doc: doc["index"]["1"]["inner"]["struct"]["kind"]["plain"]["fields"].clear(),
            "fields",
        ),
        (
            lambda doc: doc["index"]["4"]["inner"]["function"]["sig"].update(
                output={"primitive": "u64"}
            ),
            "inherent",
        ),
        (lambda doc: doc["index"]["1"]["inner"]["struct"]["impls"].remove(5), "traits"),
    ],
)
def test_migration_shape_detects_public_core_changes(change, part):
    baseline = core_shape_fixture()
    core = deepcopy(baseline)
    change(core)
    assert migration_shapes(baseline, baseline, core) == [f"Thing: {part} changed"]


def test_migration_shape_rejects_unexpected_public_method_addition():
    baseline = core_shape_fixture()
    core = deepcopy(baseline)
    core["index"]["6"] = deepcopy(core["index"]["4"])
    core["index"]["6"]["name"] = "added"
    core["index"]["3"]["inner"]["impl"]["items"].append(6)
    assert migration_shapes(baseline, baseline, core) == ["Thing: inherent changed"]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ({"kind": "transparent", "int": None}, None),
        ({"kind": "c", "int": None}, None),
        ({"kind": "rust", "int": "u8"}, {"kind": "rust", "int": "u16"}),
    ],
)
def test_migration_shape_detects_representation_changes(before, after):
    baseline = core_shape_fixture()
    baseline["index"]["1"]["attrs"] = [{"repr": {"align": None, "packed": None, **before}}]
    core = deepcopy(baseline)
    core["index"]["1"]["attrs"] = (
        [] if after is None else [{"repr": {"align": None, "packed": None, **after}}]
    )
    assert migration_shapes(baseline, baseline, core) == ["Thing: repr changed"]


def test_migration_shape_distinguishes_types_with_the_same_short_name():
    baseline = core_shape_fixture()
    output = baseline["index"]["4"]["inner"]["function"]["sig"]
    output["output"] = {"resolved_path": {"path": "Error", "id": 8, "args": None}}
    baseline["paths"]["8"] = {"path": ["core", "fmt", "Error"]}
    core = deepcopy(baseline)
    core["paths"]["8"]["path"] = ["std", "io", "Error"]
    assert migration_shapes(baseline, baseline, core) == ["Thing: inherent changed"]


def test_migration_shape_detects_changed_enum_variant_and_field():
    baseline = core_shape_fixture()
    baseline["index"]["1"]["inner"] = {
        "enum": {
            "generics": {"params": [], "where_predicates": []},
            "variants": [6],
            "impls": [],
        }
    }
    baseline["index"]["6"] = {
        "name": "Value",
        "inner": {"variant": {"kind": {"tuple": [7]}, "discriminant": None}},
    }
    baseline["index"]["7"] = {
        "name": None,
        "visibility": "default",
        "inner": {"struct_field": {"primitive": "u32"}},
    }
    core = deepcopy(baseline)
    core["index"]["7"]["inner"]["struct_field"] = {"primitive": "u64"}
    assert migration_shapes(baseline, baseline, core) == ["Thing: variants changed"]


def test_migration_shape_detects_variant_order_change():
    baseline = core_shape_fixture()
    baseline["index"]["1"]["inner"] = {
        "enum": {
            "generics": {"params": [], "where_predicates": []},
            "variants": [6, 7],
            "impls": [],
        }
    }
    for item_id, name in (("6", "Fixed"), ("7", "Unlimited")):
        baseline["index"][item_id] = {
            "name": name,
            "inner": {"variant": {"kind": "unit", "discriminant": None}},
        }
    core = deepcopy(baseline)
    core["index"]["1"]["inner"]["enum"]["variants"].reverse()
    assert migration_shapes(baseline, baseline, core) == ["Thing: variant_order changed"]


def test_migration_shape_detects_changed_constant_value():
    baseline = {
        "root": 0,
        "index": {
            "0": {"name": "hdf5_pure", "inner": {"module": {"items": [1]}}},
            "1": {
                "name": "LIMIT",
                "inner": {"constant": {"type": {"primitive": "u8"}, "value": "1"}},
            },
        },
        "paths": {},
    }
    core = deepcopy(baseline)
    core["index"]["1"]["inner"]["constant"]["value"] = "2"
    assert migration_shapes(baseline, baseline, core) == ["LIMIT: constant changed"]


def test_migration_builds_core_with_the_profile_core_features():
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz"):
        pass
    archive.seek(0)
    doc = core_shape_fixture()
    with (
        patch.object(migration, "urlopen", return_value=archive),
        patch.object(migration.rustdoc_json, "build", side_effect=[doc, doc, doc]) as build,
    ):
        check_migration_paths("0.47.0", profile("hdf5-pure", "no-std"))
    assert [call.args[2] for call in build.call_args_list] == [
        ["--no-default-features", "--features", "checksum"],
        ["--no-default-features", "--features", "checksum"],
        ["--no-default-features"],
    ]
