import argparse
import json
import subprocess
import sys
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from hdf5_pure_scripts import repo_root
from hdf5_pure_scripts.api_profiles import (
    FACADE,
    cargo_feature_args,
    crates,
    feature_args,
    profile,
)
from hdf5_pure_scripts.migration import check_migration_paths, migration_baseline


def released_version(package: str) -> str | None:
    request = Request(
        f"https://crates.io/api/v1/crates/{package}",
        headers={"User-Agent": "hdf5-pure-api-check"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.load(response)["crate"]["max_stable_version"]
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def comparison_baseline(
    package: str, latest: str | None, expected: str | None, candidate: str
) -> str:
    if latest is None:
        sys.exit(f"{package} has no published stable API baseline")
    if expected is None:
        return latest
    if latest not in {expected, candidate}:
        sys.exit(f"published API baseline {latest} differs from release tag {expected}")
    return expected


def check_package(package: str, baseline: str, release_type: str, features: list[str]) -> None:
    print(f"Package: {package}; baseline: {baseline}; allowed change: {release_type}", flush=True)
    subprocess.run(
        [
            "cargo",
            "semver-checks",
            "-p",
            package,
            "--baseline-version",
            baseline,
            "--release-type",
            release_type,
            "--color",
            "never",
            *features,
        ],
        check=True,
    )


def checks(package: str | None = None, profile_name: str | None = None) -> list[tuple[str, str]]:
    return [
        (name, item["name"])
        for name, profiles in crates().items()
        for item in profiles
        if package in (None, name) and profile_name in (None, item["name"])
    ]


def check_all(release_type: str, expected_baseline: str | None = None) -> None:
    for package, profile_name in checks():
        check_api(package, profile_name, release_type, expected_baseline)


def check_api(
    package: str, profile_name: str, release_type: str, expected_baseline: str | None = None
) -> None:
    selected = profile(package, profile_name)
    manifest = tomllib.loads((repo_root() / "Cargo.toml").read_text())
    candidate_version = manifest["workspace"]["package"]["version"]
    facade_baseline = comparison_baseline(
        FACADE, released_version(FACADE), expected_baseline, candidate_version
    )
    try:
        migration = migration_baseline(manifest, facade_baseline)
    except ValueError as error:
        sys.exit(str(error))
    if package == FACADE:
        check_facade(selected, facade_baseline, release_type, migration)
        return
    if migration:
        print(
            f"{package} {profile_name}: no API baseline until {FACADE} releases after "
            f"{facade_baseline}; the migration check compares its types with that release",
            flush=True,
        )
        return
    baseline = comparison_baseline(
        package, released_version(package), expected_baseline, candidate_version
    )
    check_package(package, baseline, release_type, feature_args(selected))


def check_facade(selected: dict, baseline: str, release_type: str, migration: bool) -> None:
    if migration:
        print(
            f"Ownership migration from {FACADE} {baseline}, the release before hdf5-pure-core",
            flush=True,
        )
        try:
            check_migration_paths(baseline, selected)
        except ValueError as error:
            sys.exit(str(error))
        subprocess.run(
            [
                "cargo",
                "test",
                "--locked",
                "-p",
                FACADE,
                "--test",
                "format_reexports",
                *cargo_feature_args(selected),
            ],
            check=True,
        )
    check_package(FACADE, baseline, release_type, feature_args(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", choices=list(crates()))
    parser.add_argument("--profile")
    parser.add_argument("--release-type", required=True, choices=("major", "minor", "patch"))
    args = parser.parse_args()
    selected = checks(args.package, args.profile)
    if not selected:
        parser.error(f"no API profile {args.profile!r} for {args.package or 'any crate'}")
    for package, profile_name in selected:
        check_api(package, profile_name, args.release_type)


if __name__ == "__main__":
    main()
