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


def check_all(
    release_type: str,
    expected_baseline: str | None = None,
    package: str | None = None,
    profile_name: str | None = None,
) -> None:
    selected = checks(package, profile_name)
    if not selected:
        sys.exit(f"no API profile {profile_name!r} for {package or 'any crate'}")
    manifest = tomllib.loads((repo_root() / "Cargo.toml").read_text())
    candidate_version = manifest["workspace"]["package"]["version"]
    baselines = {
        name: comparison_baseline(
            name, released_version(name), expected_baseline, candidate_version
        )
        for name in dict.fromkeys(name for name, _ in selected)
    }
    for name, selected_profile in selected:
        check_api(name, selected_profile, release_type, baselines[name])


def check_api(package: str, profile_name: str, release_type: str, baseline: str) -> None:
    selected = profile(package, profile_name)
    if package == FACADE:
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
    check_package(package, baseline, release_type, feature_args(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", choices=list(crates()))
    parser.add_argument("--profile")
    parser.add_argument("--release-type", required=True, choices=("major", "minor", "patch"))
    args = parser.parse_args()
    selected = checks(args.package, args.profile)
    if not selected:
        parser.error(f"no API profile {args.profile!r} for {args.package or 'any crate'}")
    check_all(args.release_type, package=args.package, profile_name=args.profile)


if __name__ == "__main__":
    main()
