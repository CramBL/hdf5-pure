"""Feature-test coverage and selector propagation."""

import os
import subprocess
from collections import Counter

import pytest
import yaml

from hdf5_pure_scripts import repo_root

WORKFLOWS = repo_root() / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


FEATURE_CONFIGURATIONS = (
    "std,deflate",
    "default,checksum",
    "default,fast-deflate",
    "default,ndarray",
    "default,num-complex",
    "default,provenance",
    "default,serde",
    "default,zfp",
)


def cargo_commands(script: str, **environment: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\ncargo() { printf "%s\\n" "$*"; }\nexport -f cargo\n' + script,
        ],
        cwd=repo_root(),
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [line.split() for line in result.stdout.splitlines()]


@pytest.mark.parametrize("shard_count", [1, 3, 10], ids=["single", "current", "empty_shards"])
def test_feature_shards_cover_each_configuration_once(shard_count):
    step = next(
        step
        for step in load("ci.yml")["jobs"]["feature-tests"]["steps"]
        if "SHARD_INDEX" in step.get("env", {})
    )
    assignments = []
    for shard in range(shard_count):
        commands = cargo_commands(step["run"], SHARD_INDEX=str(shard), SHARD_COUNT=str(shard_count))
        features = [command[4] for command in commands[::2]]
        expected = []
        for configuration in features:
            prefix = ["hack", "-p", "hdf5-pure", "--features", configuration]
            expected.extend(
                [
                    [*prefix, "nextest", "run", "--locked", "--no-default-features"],
                    [*prefix, "test", "--locked", "--doc", "--no-default-features"],
                ]
            )
        assert commands == expected
        assignments.append(features)
    assert Counter(feature for shard in assignments for feature in shard) == Counter(
        FEATURE_CONFIGURATIONS
    )
    assert max(map(len, assignments)) - min(map(len, assignments)) <= 1


@pytest.mark.parametrize(
    "recipe, command",
    [
        ("test-each-feature", ["nextest", "run", "--locked"]),
        ("doctest-each-feature", ["test", "--locked", "--doc"]),
    ],
    ids=["tests", "doctests"],
)
def test_local_feature_recipes_cover_the_same_configurations(recipe, command):
    assert cargo_commands(f"just {recipe} unit7_selector") == [
        [
            "hack",
            "-p",
            "hdf5-pure",
            "--features",
            features,
            *command,
            "--no-default-features",
            "unit7_selector",
        ]
        for features in FEATURE_CONFIGURATIONS
    ]
