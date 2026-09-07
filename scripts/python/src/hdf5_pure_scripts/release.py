"""Prepare and publish a release. Versions are `X.Y.Z` or `X.Y.Z-rc.N`.

    just release::prepare 0.45.0 --summary-file notes.md
    just release::prepare 0.45.0-rc.1
    just release::publish 0.45.0

`prepare` runs on a clean checkout of main. It checks the version against the
tags and the manifest, reports the cycle's public-API delta, sets the version
in Cargo.toml and Cargo.lock, for a final release promotes the changelog's
[Unreleased] section into a dated one opened by the summary, packages with
`cargo publish --dry-run`, commits on `release/vX.Y.Z`, pushes the branch and
opens the pull request. A candidate leaves the changelog alone: [Unreleased]
stays open until the final release promotes it.

`publish` runs on the merged release commit, from the Release workflow or from
a machine with `cargo login` and `gh auth login` done. A commit on main has
passed every CI guard, which the branch ruleset requires to merge. It checks
that the public-API delta allows the version, then publishes to crates.io,
pushes the tag and creates the GitHub release. Each of those is skipped once
it exists, so re-running after a failure resumes.
Publishing comes first because it cannot be undone: a failure there leaves
nothing public behind.
"""

import argparse
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from hdf5_pure_scripts import repo_root

CRATE = "hdf5-pure"
SEMVER_FEATURES = "serde,zfp,provenance,ndarray,num-complex"


def die(message):
    sys.exit(f"error: {message}")


def note(message):
    print(f"\033[1m==>\033[0m {message}", flush=True)


def warn(message):
    print(f"\033[33m!\033[0m {message}", file=sys.stderr, flush=True)


def run(*args, stdin=None):
    subprocess.run(args, check=True, text=True, input=stdin)


def output(*args, check=True):
    return subprocess.run(args, check=check, text=True, capture_output=True).stdout.strip()


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    rc: int | None = None

    PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")

    @classmethod
    def parse(cls, text):
        m = cls.PATTERN.match(text)
        if not m:
            raise ValueError(f"version must be X.Y.Z or X.Y.Z-rc.N (got '{text}')")
        major, minor, patch, rc = m.groups()
        return cls(int(major), int(minor), int(patch), int(rc) if rc else None)

    def __str__(self):
        base = f"{self.major}.{self.minor}.{self.patch}"
        return base if self.rc is None else f"{base}-rc.{self.rc}"

    def __lt__(self, other):
        return self.key < other.key

    @property
    def key(self):
        # Every candidate comes before its final, and candidates order by N.
        return (self.major, self.minor, self.patch, sys.maxsize if self.rc is None else self.rc)

    @property
    def is_rc(self):
        return self.rc is not None

    @property
    def final(self):
        return Version(self.major, self.minor, self.patch)

    def release_type(self, previous):
        if self.major != previous.major:
            return "major"
        if self.minor != previous.minor:
            return "minor"
        return "patch"


def argument_version(text):
    try:
        return Version.parse(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def tagged_versions():
    versions = []
    for tag in output("git", "tag", "--list", "v*").split():
        try:
            versions.append(Version.parse(tag[1:]))
        except ValueError:
            pass
    return versions


def latest(versions):
    return max(versions, key=lambda v: v.key, default=None)


def manifest():
    return tomllib.loads(Path("Cargo.toml").read_text())["package"]


def set_version(manifest_text, version):
    """Cargo.toml or Cargo.lock with the crate's own version line, the one
    that follows its `name` line, set to `version`."""
    lines = manifest_text.splitlines(keepends=True)
    seen_name = False
    for i, line in enumerate(lines):
        if line.startswith(f'name = "{CRATE}"'):
            seen_name = True
        elif seen_name and line.startswith("version = "):
            lines[i] = f'version = "{version}"\n'
            return "".join(lines)
    raise ValueError(f"no version line for {CRATE}")


def worktree_clean():
    return output("git", "status", "--porcelain", "--untracked-files=no") == ""


def changelog_section(changelog, header):
    """The body under `## [header]`, up to the next section or the link block, or None."""
    m = re.search(
        rf"^## \[{re.escape(header)}\][^\n]*\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)",
        changelog,
        re.M | re.S,
    )
    return m.group(1) if m else None


def promote_changelog(changelog, version, previous, summary, repo_url, today):
    """The changelog with [Unreleased] emptied into a dated `version` section
    that opens with `summary`, and the compare links moved along."""
    head, marker, body = changelog.partition("## [Unreleased]\n")
    if not marker:
        raise ValueError("no [Unreleased] section")
    section = f"## [{version}] - {today.isoformat()}\n\n{summary.strip()}\n\n"
    text = head + marker + "\n" + section + body.lstrip("\n")
    links = (
        f"[Unreleased]: {repo_url}/compare/v{version}...HEAD\n"
        f"[{version}]: {repo_url}/compare/v{previous}...v{version}"
    )
    text, count = re.subn(r"^\[Unreleased\]: .*$", links, text, count=1, flags=re.M)
    if count != 1:
        raise ValueError("no [Unreleased]: link line to anchor the compare links to")
    return text


def required_bump(verdict):
    """The release type a cargo semver-checks verdict asks for, or None."""
    if "requires new major" in verdict:
        return "major"
    if "requires new minor" in verdict:
        return "minor"
    return None


def bump_allows(release_type, required):
    order = {"patch": 0, "minor": 1, "major": 2}
    return required is None or order[release_type] >= order[required]


CHANGELOG = Path("CHANGELOG.md")


def semver_verdict(baseline, release_type):
    """The `Summary` line cargo semver-checks prints, with the report on stderr.

    Dies when there is no verdict, because a tool that cannot run says so in
    words rather than in an exit status that one version shares with "found
    breaks" (cargo-semver-checks #337).
    """
    if subprocess.run(["cargo", "semver-checks", "--version"], capture_output=True).returncode != 0:
        die(
            "cargo-semver-checks is not installed (`cargo binstall cargo-semver-checks`); "
            "pass --skip-api-delta to go without it"
        )
    result = subprocess.run(
        [
            "cargo",
            "semver-checks",
            "--baseline-version",
            str(baseline),
            "--release-type",
            release_type,
            "--default-features",
            "--features",
            SEMVER_FEATURES,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stderr.write(result.stdout)
    m = re.search(r"Summary.*", result.stdout)
    if not m:
        die(
            f"cargo-semver-checks printed no verdict, so the public-API delta since v{baseline} "
            "went unchecked; its own error above says why, and a toolchain newer than it "
            "supports is the common cause (`cargo binstall cargo-semver-checks` refreshes it). "
            "Pass --skip-api-delta to release without the check"
        )
    return m.group(0)


def release_notes(version):
    changelog = CHANGELOG.read_text()
    if version.is_rc:
        return (
            f"Release candidate v{version}. "
            "The changelog stays under [Unreleased] until the final release.\n\n"
            + changelog_section(changelog, "Unreleased")
        )
    return changelog_section(changelog, str(version))


def prepare(args):
    version = args.version
    tag = f"v{version}"
    branch = f"release/{tag}"

    if output("git", "rev-parse", "--abbrev-ref", "HEAD") != "main":
        die("not on main; prepare a release from main")
    if not worktree_clean():
        die("working tree is dirty; commit or stash first")
    tags = tagged_versions()
    previous = latest(tags)
    previous_stable = latest([v for v in tags if not v.is_rc])
    if previous_stable is None:
        die("no vX.Y.Z tag found; run `git fetch --tags` (a shallow clone has none)")
    current = Version.parse(manifest()["version"])

    # Releases move forward. crates.io versions can be yanked but never reused,
    # so a typo like 0.4.5 for 0.45.0 has to stop here.
    if not previous < version:
        die(f"{version} does not come after the last tag (v{previous})")
    if previous.is_rc and previous.final != version.final:
        die(
            f"v{previous} is a candidate of {previous.final}; "
            f"finish or abandon that cycle before releasing {version}"
        )
    # The manifest reads the last final, or anything up to the version being
    # cut when a breaking pull request already bumped it.
    if current < previous_stable or version < current:
        die(
            f"Cargo.toml reads {current}, outside the last release ({previous_stable}) .. {version}"
        )
    if output("git", "tag", "--list", tag):
        die(f"tag {tag} already exists")
    if output("git", "branch", "--list", branch):
        die(f"branch {branch} already exists; a previous prepare is in flight")
    changelog = CHANGELOG.read_text()
    if not (changelog_section(changelog, "Unreleased") or "").strip():
        die("CHANGELOG.md [Unreleased] is empty; nothing to release")

    summary = args.summary
    if version.is_rc:
        if summary or args.summary_file:
            die("a candidate takes no summary; the changelog is promoted by the final release")
    else:
        if changelog_section(changelog, str(version)) is not None:
            die(f"CHANGELOG.md already has a [{version}] section")
        if args.summary_file:
            if not args.summary_file.is_file():
                die(f"summary file not found: {args.summary_file}")
            summary = args.summary_file.read_text()
        if not (summary or "").strip():
            die("a final release needs its summary paragraph: --summary or --summary-file")

    note(f"Preparing {tag} after v{previous}")

    # CI's semver job derives what to check from the manifest, so once a
    # breaking pull request has bumped it the job checks nothing for the rest
    # of the cycle. This report covers the whole cycle and is read against the
    # changelog: an empty report on a cycle that claims a breaking change is a
    # finding. `publish` runs the same check as a gate.
    if args.skip_api_delta:
        warn("Skipping the public-API delta report (--skip-api-delta)")
    else:
        note(f"Public API delta since v{previous_stable}")
        note(semver_verdict(previous_stable, version.final.release_type(previous_stable)))

    note(f"Setting the version to {version} in Cargo.toml and Cargo.lock")
    for path in (Path("Cargo.toml"), Path("Cargo.lock")):
        path.write_text(set_version(path.read_text(), version))

    if not version.is_rc:
        note(f"Promoting CHANGELOG.md [Unreleased] into [{version}]")
        promoted = promote_changelog(
            changelog, version, previous_stable, summary, manifest()["repository"], date.today()
        )
        CHANGELOG.write_text(promoted)

    note("Packaging with cargo publish --dry-run")
    run("cargo", "publish", "--dry-run", "--allow-dirty")

    note(f"Committing on {branch}")
    run("git", "checkout", "-q", "-b", branch)
    run("git", "add", "Cargo.toml", "Cargo.lock", "CHANGELOG.md")
    run("git", "commit", "-q", "-m", f"Release {tag}")

    if args.local:
        note(f"Prepared {tag} on {branch} (--local: not pushed)")
        return

    note(f"Pushing {branch} and opening the pull request")
    run("git", "push", "-q", "-u", "origin", branch)
    body = (
        release_notes(version)
        + f"\nAfter merging, run the Release workflow with version {version}.\n"
    )
    run(
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch,
        "--title",
        f"Release {tag}",
        "--body-file",
        "-",
        stdin=body,
    )
    if args.skip_api_delta:
        warn(f"{tag} was prepared without its public-API delta report (--skip-api-delta)")


def publish(args):
    version = args.version
    tag = f"v{version}"
    sha = output("git", "rev-parse", "HEAD")

    # The commit under HEAD is the release commit, on main, and what CI passed.
    current = manifest()["version"]
    if current != str(version):
        die(f"Cargo.toml reads {current}, not {version}; publish from the merged release commit")
    if not worktree_clean():
        die("working tree is dirty")
    run("git", "fetch", "-q", "origin", "main", "refs/tags/*:refs/tags/*")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"]).returncode != 0:
        die("HEAD is not on origin/main")
    if not version.is_rc and changelog_section(CHANGELOG.read_text(), str(version)) is None:
        die(f"CHANGELOG.md has no [{version}] section; was the release prepared?")

    # The whole cycle's public-API delta, gated: a verdict asking for a larger
    # bump than this release makes stops it.
    previous_stable = latest([v for v in tagged_versions() if not v.is_rc])
    if previous_stable is None:
        die("no vX.Y.Z tag found")
    release_type = version.final.release_type(previous_stable)
    if args.skip_api_delta:
        warn("Skipping the public-API delta gate (--skip-api-delta)")
    else:
        note(f"Public API delta since v{previous_stable}, for a {release_type} release")
        verdict = semver_verdict(previous_stable, release_type)
        if not bump_allows(release_type, required_bump(verdict)):
            die(f"{verdict}, but {version} is a {release_type} release after {previous_stable}")
        note(verdict)

    note("Packaging with cargo publish --dry-run")
    run("cargo", "publish", "--dry-run")

    if args.dry_run:
        note(f"Dry run: {tag} would be published from {sha}")
        return

    # crates.io. Yanked or not, a published version answers 200.
    request = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{CRATE}/{version}",
        headers={"User-Agent": f"{CRATE} release script"},
    )
    try:
        with urllib.request.urlopen(request):
            published = True
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        published = False
    if published:
        note(f"{CRATE} {version} is already on crates.io")
    else:
        note(f"Publishing {CRATE} {version} to crates.io")
        run("cargo", "publish")

    # The tag, at this commit. A tag elsewhere is a conflict, not a retry.
    existing = output("git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}^{{}}").split()
    if existing:
        if existing[0] != sha:
            die(f"{tag} already exists at {existing[0]}, not at {sha}")
        note(f"{tag} already exists")
    else:
        note(f"Tagging {tag}")
        run("git", "tag", "-a", tag, "-m", f"Release {tag}")
        run("git", "push", "-q", "origin", tag)

    if subprocess.run(["gh", "release", "view", tag], capture_output=True).returncode == 0:
        note(f"GitHub release {tag} already exists")
    else:
        note(f"Creating GitHub release {tag}")
        notes = release_notes(version)
        notes += (
            f"\n**Full changelog:** {manifest()['repository']}/compare/v{previous_stable}...{tag}\n"
        )
        flags = ["--prerelease"] if version.is_rc else []
        run(
            "gh",
            "release",
            "create",
            tag,
            "--verify-tag",
            "--title",
            tag,
            "--notes-file",
            "-",
            *flags,
            stdin=notes,
        )

    # A tag pushed with the workflow's token triggers no workflow, so the site
    # is rebuilt on request for a final release.
    if os.environ.get("GITHUB_ACTIONS") == "true" and not version.is_rc:
        note(f"Rebuilding the documentation site from {tag}")
        run("gh", "workflow", "run", "docs.yml", "--ref", tag)

    note(f"Done: {tag}")
    if args.skip_api_delta:
        warn(f"{tag} went out without its public-API delta being checked (--skip-api-delta)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser(
        "prepare", help="the release pull request: bump, changelog, package, branch, PR"
    )
    p.add_argument("version", type=argument_version)
    p.add_argument("--summary", help="the paragraph a final release's changelog section opens with")
    p.add_argument("--summary-file", type=Path, help="the same, read from a file")
    p.add_argument(
        "--skip-api-delta", action="store_true", help="prepare without the public-API delta report"
    )
    p.add_argument(
        "--local", action="store_true", help="stop after the commit: no push, no pull request"
    )
    p.set_defaults(func=prepare)

    p = commands.add_parser(
        "publish", help="publish from main: crates.io, the tag, the GitHub release"
    )
    p.add_argument("version", type=argument_version)
    p.add_argument(
        "--skip-api-delta", action="store_true", help="publish without the public-API delta gate"
    )
    p.add_argument("--dry-run", action="store_true", help="stop before the first public step")
    p.set_defaults(func=publish)

    args = parser.parse_args()
    os.chdir(repo_root())
    try:
        args.func(args)
    except (subprocess.CalledProcessError, ValueError) as e:
        die(f"{' '.join(e.cmd)} failed" if isinstance(e, subprocess.CalledProcessError) else e)
