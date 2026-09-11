"""Prepare and publish a release. Versions are `X.Y.Z` or `X.Y.Z-rc.N`.

    just release::prepare 0.45.0 --summary-file notes.md
    just release::prepare 0.45.0-rc.1
    just release::pr 0.45.0
    just release::publish

`prepare` runs on a clean checkout of main, where Cargo.toml reads the last
tag: no pull request touches the version. It checks the version against the
tags and the manifest, rejects a patch version when the cycle's changelog
marks a breaking change, reports the cycle's public-API delta, sets the
version in Cargo.toml and Cargo.lock, for a final release promotes the changelog's
[Unreleased] section into a dated one, opened by a summary paragraph when one
is given, and packages with `cargo publish --dry-run`. It commits nothing. A
candidate does not change the changelog: [Unreleased] stays open until the
final release promotes it.

`pr` commits those changes on `release/vX.Y.Z`, prompts for confirmation, then
pushes the branch and opens the pull request.

`publish` runs on the merged release commit, from the Release workflow on the
merge or from a machine with `cargo login` and `gh auth login` done. The
version is the manifest's. A commit on main has passed every CI guard, which
the branch ruleset requires to merge. It checks that the public-API delta
allows the version, then publishes to crates.io,
pushes the tag and creates the GitHub release. Each of those is skipped once
it exists, so re-running after a failure resumes. Publishing comes first
because it cannot be undone.
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
RELEASE_FILES = {"Cargo.toml", "Cargo.lock", "CHANGELOG.md"}
SEMVER_FEATURES = "serde,zfp,provenance,ndarray,num-complex"


def fail(message):
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


def changed_files():
    """The tracked files with uncommitted changes."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {entry[3:] for entry in status.split("\0") if entry}


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


BREAKING = "**Breaking:**"


def pull_request_release_type(changelog, changelog_diff):
    version = promoted_version(changelog_diff)
    promoted = changelog_section(changelog, version) if version else None
    if promoted is not None and BREAKING in promoted:
        return "minor"
    return cycle_release_type(changelog)


def cycle_release_type(changelog):
    """The release type the [Unreleased] section calls for: minor with a breaking entry.

    Under 0.x, a breaking change takes the next minor and everything else may
    go in a patch. The marker is the one the changelog convention requires on
    a breaking entry, so a pull request declares its bump by writing its entry.
    """
    section = changelog_section(changelog, "Unreleased") or ""
    return "minor" if BREAKING in section else "patch"


DATED_SECTION_ADDED = re.compile(r"^\+## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.M)


def promoted_version(changelog_diff):
    m = DATED_SECTION_ADDED.search(changelog_diff)
    return m.group(1) if m else None


def promote_changelog(changelog, version, previous, summary, repo_url, today):
    """The changelog with [Unreleased] emptied into a dated `version` section,
    opened by `summary` when there is one, and the compare links updated."""
    head, marker, body = changelog.partition("## [Unreleased]\n")
    if not marker:
        raise ValueError("no [Unreleased] section")
    section = f"## [{version}] - {today.isoformat()}\n\n"
    if summary and summary.strip():
        section += f"{summary.strip()}\n\n"
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
    """The release type a cargo semver-checks verdict requires, or None."""
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
        fail(
            "cargo-semver-checks is not installed (`cargo binstall cargo-semver-checks`), "
            "or pass --skip-api-delta"
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
        fail(
            f"cargo-semver-checks printed no verdict for the delta since v{baseline}; see its "
            "output above. A version older than the toolchain cannot read rustdoc's JSON, and "
            "`cargo binstall cargo-semver-checks` installs the current one. "
            "--skip-api-delta releases without the check"
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
        fail("not on main; prepare a release from main")
    if not worktree_clean():
        fail("working tree is dirty; commit or stash first")
    run("git", "fetch", "-q", "--tags")
    tags = tagged_versions()
    previous = latest(tags)
    previous_stable = latest([v for v in tags if not v.is_rc])
    if previous_stable is None:
        fail("no vX.Y.Z tag found; run `git fetch --tags` (a shallow clone has none)")
    current = Version.parse(manifest()["version"])

    # A version below the last tag is rejected: crates.io never reuses a
    # version, so a typo such as 0.4.5 for 0.45.0 would be permanent.
    if not previous < version:
        fail(f"{version} does not come after the last tag (v{previous})")
    if previous.is_rc and previous.final != version.final:
        fail(
            f"v{previous} is a candidate of {previous.final}; "
            f"finish or abandon that cycle before releasing {version}"
        )
    # Between releases the manifest reads the last tag, a candidate included.
    if current != previous:
        fail(f"Cargo.toml reads {current}; between releases it reads the last tag (v{previous})")
    if output("git", "tag", "--list", tag):
        fail(f"tag {tag} already exists")
    if output("git", "branch", "--list", branch):
        fail(f"branch {branch} already exists; delete it or finish that release first")
    changelog = CHANGELOG.read_text()
    if not (changelog_section(changelog, "Unreleased") or "").strip():
        fail("CHANGELOG.md [Unreleased] is empty; nothing to release")
    release_type = version.final.release_type(previous_stable)
    if not bump_allows(release_type, cycle_release_type(changelog)):
        fail(
            f"CHANGELOG.md [Unreleased] marks a change {BREAKING}, which takes the next minor; "
            f"{version} is a {release_type} release"
        )

    summary = args.summary
    if version.is_rc:
        if summary or args.summary_file:
            fail("a candidate takes no summary; the final release promotes the changelog")
    else:
        if changelog_section(changelog, str(version)) is not None:
            fail(f"CHANGELOG.md already has a [{version}] section")
        if args.summary_file:
            if not args.summary_file.is_file():
                fail(f"summary file not found: {args.summary_file}")
            summary = args.summary_file.read_text()

    note(f"Preparing {tag} after v{previous}")

    # CI's semver job checks each pull request against the release type the
    # changelog calls for. This report covers the whole cycle against the
    # version being cut, and `publish` runs the same check as a gate.
    if args.skip_api_delta:
        warn("Skipping the public-API delta report (--skip-api-delta)")
    else:
        note(f"Public API delta since v{previous_stable}")
        note(semver_verdict(previous_stable, release_type))

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

    note(f"Prepared {tag} in the working tree. Next: `just release::pr {version}`")
    run("git", "status", "--short")


def print_release_type(args):
    resolve = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{args.base}^{{commit}}"], capture_output=True
    )
    if resolve.returncode != 0:
        fail(f"no such commit: {args.base!r} (a shallow clone has no remote branches)")
    diff = output("git", "diff", f"{args.base}...HEAD", "--", str(CHANGELOG))
    print(pull_request_release_type(CHANGELOG.read_text(), diff))


def pr(args):
    """Commit the prepared release on its branch, prompt, then push it and open
    the pull request."""
    version = args.version
    tag = f"v{version}"
    branch = f"release/{tag}"

    if output("git", "rev-parse", "--abbrev-ref", "HEAD") != "main":
        fail("not on main; run this where `prepare` ran")
    current = manifest()["version"]
    if current != str(version):
        fail(
            f"Cargo.toml reads {current}, not {version}; "
            f"run `just release::prepare {version}` first"
        )
    changed = changed_files()
    if not changed:
        fail(f"nothing to commit; run `just release::prepare {version}` first")
    if unexpected := changed - RELEASE_FILES:
        fail(f"changed files outside the release: {', '.join(sorted(unexpected))}")
    if output("git", "branch", "--list", branch):
        fail(f"branch {branch} already exists; delete it or finish that release first")

    note(f"Committing on {branch}")
    run("git", "checkout", "-q", "-b", branch)
    run("git", "add", *sorted(changed))
    run("git", "commit", "-q", "-m", f"Release {tag}")
    run("git", "show", "--stat", "--oneline", "HEAD")

    if input(f"Push {branch} and open the pull request? [y/N] ").strip().lower() != "y":
        note(f"Not pushed. The commit is on {branch}")
        return
    run("git", "push", "-q", "-u", "origin", branch)
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
        "--body",
        "",
    )


def publish(args):
    current = manifest()["version"]
    version = args.version or Version.parse(current)
    tag = f"v{version}"
    sha = output("git", "rev-parse", "HEAD")

    # The commit under HEAD is the release commit, on main, and what CI passed.
    if current != str(version):
        fail(f"Cargo.toml reads {current}, not {version}; publish from the merged release commit")
    if not worktree_clean():
        fail("working tree is dirty")
    run("git", "fetch", "-q", "origin", "main", "refs/tags/*:refs/tags/*")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"]).returncode != 0:
        fail("HEAD is not on origin/main")
    if not version.is_rc and changelog_section(CHANGELOG.read_text(), str(version)) is None:
        fail(f"CHANGELOG.md has no [{version}] section; run `just release::prepare` first")

    # The whole cycle's public-API delta. A verdict that requires a larger bump
    # than this release makes stops the release.
    previous_stable = latest([v for v in tagged_versions() if not v.is_rc])
    if previous_stable is None:
        fail("no vX.Y.Z tag found")
    release_type = version.final.release_type(previous_stable)
    if args.skip_api_delta:
        warn("Skipping the public-API delta gate (--skip-api-delta)")
    else:
        note(f"Public API delta since v{previous_stable}, for a {release_type} release")
        verdict = semver_verdict(previous_stable, release_type)
        if not bump_allows(release_type, required_bump(verdict)):
            fail(f"{verdict}, but {version} is a {release_type} release after {previous_stable}")
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

    # The tag, at this commit. A tag at another commit is an error.
    existing = output("git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}^{{}}").split()
    if existing:
        if existing[0] != sha:
            fail(f"{tag} already exists at {existing[0]}, not at {sha}")
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

    note(f"Done: {tag}")
    if args.skip_api_delta:
        warn(f"{tag} was published without the public-API delta check (--skip-api-delta)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser(
        "prepare", help="set the version and the changelog in the working tree, and package"
    )
    p.add_argument("version", type=argument_version)
    p.add_argument("--summary", help="a paragraph to open a final release's changelog section")
    p.add_argument("--summary-file", type=Path, help="the same, read from a file")
    p.add_argument(
        "--skip-api-delta", action="store_true", help="prepare without the public-API delta report"
    )
    p.set_defaults(func=prepare)

    p = commands.add_parser(
        "release-type",
        help="print the release type this pull request's changelog calls for",
    )
    p.add_argument("--base", required=True, help="the ref the pull request diffs against")
    p.set_defaults(func=print_release_type)

    p = commands.add_parser(
        "pr", help="commit the prepared release on its branch, then push it and open the PR"
    )
    p.add_argument("version", type=argument_version)
    p.set_defaults(func=pr)

    p = commands.add_parser(
        "publish", help="publish from main: crates.io, the tag, the GitHub release"
    )
    p.add_argument("version", nargs="?", type=argument_version, help="defaults to the manifest's")
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
        fail(f"{' '.join(e.cmd)} failed" if isinstance(e, subprocess.CalledProcessError) else e)
