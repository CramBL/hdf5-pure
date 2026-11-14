"""Reproduces fixed-record measurements in isolated source and target directories."""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFS = {
    "pre-pr": "1ff3bf482fbe6a1cd009ac1021bc6ddb791bda4e",
    "reviewed": "8bf4a1d63d41c2e4b3db6025bf5dc1485db57a6c",
    "bounded": "HEAD",
    "zerocopy": "HEAD",
}


def run(args, cwd=ROOT, **kwargs):
    return subprocess.run(args, cwd=cwd, check=True, **kwargs)


def capture(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    for label, ref in REFS.items():
        source = output / "sources" / label
        source.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "worktree", "add", "--detach", str(source), ref])
        manifest = (source / "Cargo.toml").read_text()
        if "zerocopy = " not in manifest:
            manifest = manifest.replace(
                "[dev-dependencies]\n",
                '[dev-dependencies]\nzerocopy = { version = "=0.8.63", default-features = false, features = ["derive"] }\n',
                1,
            )
        if 'name = "bounded_byte_reads"' not in manifest:
            manifest += '\n[[bench]]\nname = "bounded_byte_reads"\nharness = false\n'
        (source / "Cargo.toml").write_text(manifest)
        shutil.copy2(ROOT / "Cargo.lock", source / "Cargo.lock")
        shutil.copy2(
            ROOT / "benches/bounded_byte_reads.rs",
            source / "benches/bounded_byte_reads.rs",
        )
        if label == "zerocopy":
            run(
                [
                    "git",
                    "apply",
                    str(ROOT / "scripts/experiments/zerocopy-records.patch"),
                ],
                source,
            )
            run(
                ["cargo", "metadata", "--format-version", "1"],
                source,
                stdout=subprocess.DEVNULL,
            )


def build(output):
    (output / "build-complete.json").unlink(missing_ok=True)
    for label in REFS:
        source = output / "sources" / label
        log = output / f"{label}-build.log"
        target = output / "targets" / label
        with log.open("w") as stream:
            run(
                [
                    "cargo",
                    "bench",
                    "--locked",
                    "--bench",
                    "bounded_byte_reads",
                    "--no-run",
                    "--target-dir",
                    str(target),
                ],
                source,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        binary = Path(
            re.search(
                r"Executable benches/bounded_byte_reads.rs \(([^)]+)\)", log.read_text()
            )[1]
        )
        destination = output / label
        destination.mkdir(exist_ok=True)
        shutil.copy2(binary, destination / "benchmark.bin")
        shutil.copy2(source / "Cargo.lock", destination / "Cargo.lock")
        (destination / "source.patch").write_text(
            capture(["git", "diff", "--binary"], source) + "\n"
        )
        shutil.copy2(
            source / "benches/bounded_byte_reads.rs", destination / "harness.rs"
        )
        settings = {
            "source_ref": capture(["git", "rev-parse", "HEAD"], source),
            "harness_sha256": digest(source / "benches/bounded_byte_reads.rs"),
            "lockfile_sha256": digest(source / "Cargo.lock"),
            "source_patch_sha256": digest(destination / "source.patch"),
            "binary_sha256": digest(binary),
            "compiler": capture(["rustc", "-vV"]),
            "cargo": capture(["cargo", "--version"]),
            "hardware": capture(["lscpu"]),
            "features": "root defaults and normal dev-dependencies",
            "profile": "Cargo bench defaults with project and home Cargo configuration",
            "config": (ROOT / ".cargo/config.toml").read_text(),
            "environment": {
                key: os.environ.get(key)
                for key in [
                    "RUSTFLAGS",
                    "CARGO_ENCODED_RUSTFLAGS",
                    "RUSTC_WRAPPER",
                    "CARGO_PROFILE_BENCH_LTO",
                    "CARGO_PROFILE_BENCH_CODEGEN_UNITS",
                ]
            },
        }
        (destination / "settings.json").write_text(
            json.dumps(settings, indent=2) + "\n"
        )
        for edges in ["normal", "features", "all"]:
            (destination / f"dependencies-{edges}.txt").write_text(
                capture(["cargo", "tree", "--locked", "--edges", edges], source) + "\n"
            )
    run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "--example",
            "fixed_record_comparison",
        ]
    )
    destination = output / "private"
    destination.mkdir(exist_ok=True)
    shutil.copy2(
        ROOT / "target/release/examples/fixed_record_comparison",
        destination / "benchmark.bin",
    )
    shutil.copytree(
        ROOT / "examples/fixed_record_comparison",
        destination / "source",
        dirs_exist_ok=True,
    )
    (destination / "settings.json").write_text(
        json.dumps(
            {
                "binary_sha256": digest(destination / "benchmark.bin"),
                "compiler": capture(["rustc", "-vV"]),
                "profile": "Cargo release defaults, equivalent to the bench profile in this workspace",
                "sources": {
                    str(path.relative_to(ROOT)): digest(path)
                    for path in [
                        *sorted(
                            (ROOT / "examples/fixed_record_comparison").glob("*.rs")
                        ),
                        ROOT / "crates/hdf5-pure-format/src/fixed_record/mod.rs",
                        ROOT / "crates/hdf5-pure-format/src/bytes.rs",
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    (output / "build-complete.json").write_text(
        json.dumps(
            {
                label: digest(output / label / "benchmark.bin")
                for label in [*REFS, "private"]
            },
            indent=2,
        )
        + "\n"
    )


def measure(output, cpu, labels):
    built = json.loads((output / "build-complete.json").read_text())
    for label in labels:
        assert digest(output / label / "benchmark.bin") == built[label], label
        destination = output / label
        args = [
            "taskset",
            "-c",
            str(cpu),
            str(destination / "benchmark.bin"),
            "--bench",
            "--save-baseline",
            label,
        ]
        environment = dict(os.environ, CRITERION_HOME=str(destination / "criterion"))
        started = time.time()
        with (destination / "run.log").open("w") as stream:
            run(args, ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        (destination / "timing.json").write_text(
            json.dumps(
                {
                    "command": args,
                    "binary_sha256": built[label],
                    "cpu": cpu,
                    "start": started,
                    "end": time.time(),
                    "warm_up_seconds": 0.5,
                    "measurement_seconds": 2,
                    "sample_size": 100,
                    "confidence_level": 0.95,
                    "nresamples": 100000,
                },
                indent=2,
            )
            + "\n"
        )
        collect(destination, label)


def collect(destination, label):
    rows = []
    for path in sorted((destination / "criterion").glob(f"**/{label}/estimates.json")):
        benchmark = json.loads((path.parent / "benchmark.json").read_text())
        estimates = json.loads(path.read_text())
        estimator = "slope" if estimates.get("slope") else "mean"
        estimate = estimates[estimator]
        interval = estimate["confidence_interval"]
        rows.append(
            {
                "case": benchmark["full_id"],
                "estimator": estimator,
                "ns": estimate["point_estimate"],
                "lower_ns": interval["lower_bound"],
                "upper_ns": interval["upper_bound"],
            }
        )
    assert len(rows) == (128 if label == "private" else 57), (label, len(rows))
    (destination / "estimates.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (destination / "estimates.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def distributions(destination, label, rng):
    result = {}
    for path in sorted((destination / "criterion").glob(f"**/{label}/sample.json")):
        case = json.loads((path.parent / "benchmark.json").read_text())["full_id"]
        sample = json.loads(path.read_text())
        pairs = [
            (x * x, x * y)
            for x, y in zip(sample["iters"], sample["times"], strict=True)
        ]
        slopes = []
        for _ in range(10_000):
            selected = rng.choices(pairs, k=len(pairs))
            slopes.append(
                sum(pair[1] for pair in selected) / sum(pair[0] for pair in selected)
            )
        result[case] = slopes
    return result


def changes(row, name, candidate, baseline, candidate_samples, baseline_samples):
    ratios = sorted(
        (x / y - 1) * 100
        for x, y in zip(candidate_samples, baseline_samples, strict=True)
    )
    row[name + "_percent"] = (candidate["ns"] / baseline["ns"] - 1) * 100
    row[name + "_lower_percent"] = ratios[250]
    row[name + "_upper_percent"] = ratios[9749]


def write_rows(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def compare(output):
    rng = random.Random(1236)
    labels = [*REFS, "private"]
    estimates = {
        label: {
            row["case"]: row
            for row in json.loads((output / label / "estimates.json").read_text())
        }
        for label in labels
    }
    samples = {label: distributions(output / label, label, rng) for label in labels}
    rows = []
    for case in sorted(estimates["pre-pr"]):
        row = {"case": case}
        for label in REFS:
            for key in ["ns", "lower_ns", "upper_ns"]:
                row[label + "_" + key] = estimates[label][case][key]
        for candidate in ["bounded", "zerocopy"]:
            for baseline in ["pre-pr", "reviewed"]:
                changes(
                    row,
                    candidate + "_vs_" + baseline,
                    estimates[candidate][case],
                    estimates[baseline][case],
                    samples[candidate][case],
                    samples[baseline][case],
                )
        changes(
            row,
            "zerocopy_vs_bounded",
            estimates["zerocopy"][case],
            estimates["bounded"][case],
            samples["zerocopy"][case],
            samples["bounded"][case],
        )
        rows.append(row)
    write_rows(output / "public-comparison.csv", rows)
    rows = []
    for case in sorted(estimates["private"]):
        if not case.endswith("/bounded"):
            continue
        prefix = case.removesuffix("/bounded")
        row = {"case": prefix}
        for label in ["bounded", "zerocopy"]:
            for key in ["ns", "lower_ns", "upper_ns"]:
                row[label + "_" + key] = estimates["private"][prefix + "/" + label][key]
        changes(
            row,
            "zerocopy_vs_bounded",
            estimates["private"][prefix + "/zerocopy"],
            estimates["private"][case],
            samples["private"][prefix + "/zerocopy"],
            samples["private"][case],
        )
        rows.append(row)
    write_rows(output / "private-comparison.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["prepare", "build", "measure", "collect", "compare"]
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument(
        "--labels", nargs="+", choices=[*REFS, "private"], default=[*REFS, "private"]
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(output)
    elif args.stage == "build":
        build(output)
    elif args.stage == "measure":
        measure(output, args.cpu, args.labels)
    elif args.stage == "collect":
        for label in [*REFS, "private"]:
            collect(output / label, label)
    else:
        compare(output)


if __name__ == "__main__":
    main()
