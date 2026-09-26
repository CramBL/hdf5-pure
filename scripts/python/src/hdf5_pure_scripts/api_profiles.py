import tomllib

from hdf5_pure_scripts import repo_root

FACADE = "hdf5-pure"
CORE = "hdf5-pure-core"


def crates() -> dict[str, list[dict]]:
    config = tomllib.loads((repo_root() / "scripts/api-profiles.toml").read_text())
    return {item["package"]: item["profile"] for item in config["crate"]}


def profile(package: str, name: str) -> dict:
    for item in crates()[package]:
        if item["name"] == name:
            return item
    raise ValueError(f"unknown API profile for {package}: {name}")


def feature_args(item: dict) -> list[str]:
    args = ["--default-features" if item["default_features"] else "--only-explicit-features"]
    if item["features"]:
        args.extend(["--features", ",".join(item["features"])])
    return args


def cargo_feature_args(item: dict) -> list[str]:
    args = [] if item["default_features"] else ["--no-default-features"]
    if item["features"]:
        args.extend(["--features", ",".join(item["features"])])
    return args
