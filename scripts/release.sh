#!/usr/bin/env bash
#
# release.sh — cut a new hdf5-pure release.
#
# Automates the mechanical, easy-to-botch parts of a release so they come out
# identical every time:
#   * bump the version in Cargo.toml and Cargo.lock, or accept one a breaking
#     PR already bumped (see "Versioning" in RELEASES.md)
#   * promote the CHANGELOG's [Unreleased] section into a dated `## [X.Y.Z]`
#     section and refresh the two compare links at the bottom
#   * verify the crate still packages (`cargo publish --dry-run`)
#   * commit "Release vX.Y.Z" and create the annotated `vX.Y.Z` tag
#
# The one-paragraph editorial summary that leads each release's changelog
# section is yours to write: pass it with --summary / --summary-file, or the
# script inserts a TODO placeholder and reminds you to fill it in before the
# release is published.
#
# Public, irreversible steps are opt-in and never run by default. Without any
# of the flags below the script only touches your working tree and local git,
# then prints the remaining commands:
#   --commit       commit the release and create the tag (implied by the below)
#   --push         push main and the tag to origin      (implies --commit)
#   --gh-release   create the GitHub release from the changelog section
#                                                        (implies --push)
#   --publish      `cargo publish` to crates.io          (implies --commit)
#
# The public-API delta report is not opt-in. CI stops checking once the manifest
# carries a bump, so this is the one run that sees a whole cycle's delta, and a
# `cargo-semver-checks` that is missing or cannot run stops the release rather
# than being reported as nothing to see. That is worth an override for the window
# after a rustc release, when no published version of the tool may yet parse the
# new rustdoc format:
#   --skip-api-delta  release without checking the public-API delta
#
# Usage:
#   scripts/release.sh 0.21.0 --summary-file notes.md
#   scripts/release.sh 0.21.0 --summary "One-paragraph summary." --push --gh-release
#
set -euo pipefail

# Every mktemp lands here so an abort part-way through leaves nothing behind.
TMP_FILES=""
cleanup() { [ -z "$TMP_FILES" ] || rm -f $TMP_FILES; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
NEW_VERSION=""
SUMMARY=""
SUMMARY_FILE=""
DO_COMMIT=0
DO_PUSH=0
DO_GH=0
DO_PUBLISH=0
SKIP_API_DELTA=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '\033[1m==>\033[0m %s\n' "$*"; }
# For the things a release can proceed without but nobody should discover later.
# On stderr, like `die` and unlike `note`, so redirecting the run to a log leaves
# the warnings on the terminal rather than in the part nobody reads.
warn() { printf '\n\033[33m!\033[0m %s\n' "$*" >&2; }

usage() {
  # The leading comment block, minus the shebang and the `#` markers. Ends at
  # the first line that is not a comment, so it stays correct as the block grows.
  awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)     usage 0 ;;
    --summary)     SUMMARY="${2:?--summary needs a value}"; shift 2 ;;
    --summary-file) SUMMARY_FILE="${2:?--summary-file needs a path}"; shift 2 ;;
    --commit)      DO_COMMIT=1; shift ;;
    --push)        DO_PUSH=1; shift ;;
    --gh-release)  DO_GH=1; shift ;;
    --publish)     DO_PUBLISH=1; shift ;;
    --skip-api-delta) SKIP_API_DELTA=1; shift ;;
    -*)            die "unknown option: $1 (see --help)" ;;
    *)
      [ -z "$NEW_VERSION" ] || die "unexpected extra argument: $1"
      NEW_VERSION="$1"; shift ;;
  esac
done

[ -n "$NEW_VERSION" ] || usage 1

# Resolve step dependencies: a later step implies every earlier one.
[ "$DO_GH" -eq 1 ] && DO_PUSH=1
{ [ "$DO_PUSH" -eq 1 ] || [ "$DO_PUBLISH" -eq 1 ]; } && DO_COMMIT=1

# ---------------------------------------------------------------------------
# Locate the repo and load current state
# ---------------------------------------------------------------------------
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

CARGO_TOML="Cargo.toml"
CARGO_LOCK="Cargo.lock"
CHANGELOG="CHANGELOG.md"
TAG="v${NEW_VERSION}"

# Extract the [package] version and repository URL from Cargo.toml. The version
# lives on the first `version =` line after `name = "hdf5-pure"`.
CUR_VERSION="$(awk '/^name = "hdf5-pure"/{p=1} p&&/^version = /{gsub(/[",]/,"",$3); print $3; exit}' "$CARGO_TOML")"
REPO_URL="$(awk -F'"' '/^repository = /{print $2; exit}' "$CARGO_TOML")"
[ -n "$CUR_VERSION" ] || die "could not read current version from $CARGO_TOML"
[ -n "$REPO_URL" ] || die "could not read repository URL from $CARGO_TOML"

# The version this release follows comes from the latest version tag, not from
# Cargo.toml. Under this repo's convention (see "Versioning" in RELEASES.md) the
# manifest carries the version being *developed*, so once a breaking PR has
# merged it already reads $NEW_VERSION and is no longer the previous release.
#
# Exact `vX.Y.Z` only: the shell glob's `*` matches dots, so a looser pattern
# would admit `v0.24.0.1`, and `--sort=-v:refname` ranks `v0.25.0-rc1` above
# `v0.25.0` unless `versionsort.suffix` is configured. `sort -V | tail -1` reads
# its whole input, where `head -1` would SIGPIPE (silently, under `set -o
# pipefail`) on a repo with enough tags to fill the pipe buffer.
# `|| true`: with no matching tag `grep` exits 1, and under `set -o pipefail`
# that would abort the script here — before the check below can say why.
PREV_VERSION="$(git tag --list 'v*' \
  | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
  | sed 's/^v//' \
  | sort -V \
  | tail -1 || true)"

# Compare dotted versions. `sort -V` orders them; equal values sort to the same
# line, so `_ge` is "not strictly less".
version_ge() {
  [ "$1" = "$2" ] || [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
printf '%s\n' "$NEW_VERSION" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
  || die "version must be X.Y.Z (got '$NEW_VERSION')"

# No tag means no compare base. Falling back to Cargo.toml would be wrong under
# this repo's convention — the manifest is the version being developed, not the
# previous release — and would silently emit a `vX...vX` self-comparison link.
# A shallow or tag-pruned clone lands here; fetch the tags rather than guess.
[ -n "$PREV_VERSION" ] || die "no vX.Y.Z tag found; run \`git fetch --tags\` (a shallow clone has none)"

note "Releasing ${PREV_VERSION} -> ${NEW_VERSION}"

# Releases move forward. Without this a typo like `0.2.5` for `0.25.0` passes
# every other check and, with --publish, is permanent: crates.io versions can be
# yanked but never deleted or reused.
version_ge "$NEW_VERSION" "$PREV_VERSION" && [ "$NEW_VERSION" != "$PREV_VERSION" ] \
  || die "$NEW_VERSION does not come after the last release ($PREV_VERSION)"

# Cargo.toml may legitimately read the last release (nothing breaking has merged
# this cycle) or anything up to the version being cut (something has, possibly in
# a smaller step than this release). Outside that range the manifest and the
# requested version disagree about what is being cut, which is worth stopping on
# rather than overwriting.
version_ge "$CUR_VERSION" "$PREV_VERSION" && version_ge "$NEW_VERSION" "$CUR_VERSION" \
  || die "$CARGO_TOML reads $CUR_VERSION, outside the last release ($PREV_VERSION) .. $NEW_VERSION"

# A `## [X.Y.Z]` section for this version means a previous run already promoted
# the changelog. Re-running would emit a second one; the local tag check below
# misses this when the tag was deleted to retry a failed publish.
grep -q "^## \[${NEW_VERSION}\]" "$CHANGELOG" \
  && die "$CHANGELOG already has a [$NEW_VERSION] section; a previous run promoted it"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "not on main (on '$BRANCH'); release from main"

git diff --quiet && git diff --cached --quiet \
  || die "working tree is dirty; commit or stash first"

git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null \
  && die "tag ${TAG} already exists"

# The [Unreleased] section must contain at least one entry to release.
UNRELEASED_BODY="$(awk '
  /^## \[Unreleased\]/{grab=1; next}
  grab && /^## \[/{exit}
  grab && /[^[:space:]]/{print}
' "$CHANGELOG")"
[ -n "$UNRELEASED_BODY" ] || die "CHANGELOG [Unreleased] is empty; nothing to release"

# Resolve the summary paragraph.
if [ -n "$SUMMARY_FILE" ]; then
  [ -f "$SUMMARY_FILE" ] || die "summary file not found: $SUMMARY_FILE"
  SUMMARY="$(cat "$SUMMARY_FILE")"
fi
if [ -z "$SUMMARY" ]; then
  SUMMARY="TODO: one-paragraph summary of ${NEW_VERSION} (see prior releases for the house style)."
  SUMMARY_IS_TODO=1
else
  SUMMARY_IS_TODO=0
fi

TODAY="$(date +%Y-%m-%d)"

# ---------------------------------------------------------------------------
# 1. Report the cycle's public-API delta
# ---------------------------------------------------------------------------
# CI's semver-checks job derives what to check from the version already in the
# manifest, so once a breaking PR has bumped it the job reports "no semver update
# required" and skips every check for the rest of the cycle. `--release-type
# minor` overrides that derivation and forces the full run, which is worth having
# exactly once, here, where the accumulated delta can be read against the
# [Unreleased] section about to be promoted.
#
# Informational: a listed break is expected under a 0.x minor bump, and an empty
# report on a cycle whose changelog claims a breaking change is the real signal.
# Reading a report that way takes for granted that there is one to read, which is
# what the guards below establish before anything is read into it (issue #337).
#
# Placed ahead of the bump and the promotion so that stopping here leaves a clean
# tree to re-run in. Nothing those steps produce is an input: `--release-type`
# sets the release type instead of deriving it from the manifest version, and the
# baseline comes from the tag.
if [ "$SKIP_API_DELTA" -eq 1 ]; then
  warn "Skipping the public-API delta report (--skip-api-delta); nothing will have checked this cycle's accumulated public-API delta"
else
  SEMVER_FEATURES="serde,zfp,provenance,ndarray,num-complex"
  command -v cargo-semver-checks >/dev/null 2>&1 \
    || die "cargo-semver-checks is not installed, and this is the one run that checks a whole cycle's public-API delta (CI stops checking once the manifest carries a bump); install it (\`cargo install cargo-semver-checks --locked\`) or pass --skip-api-delta"

  # The feature list is deliberate rather than derived — it names the features
  # whose public API is worth checking, not every feature that exists, so the
  # test-only ones stay out. That makes it drift when a feature is renamed or
  # removed, and swallowing the failure would hide cargo's "feature does not
  # exist" error behind an empty report that reads exactly like a clean one.
  # Check the names against the manifest first, so the drift stops the release
  # and says so before the build cost is paid.
  for feat in $(printf '%s' "$SEMVER_FEATURES" | tr ',' ' '); do
    awk '/^\[features\]/{f=1;next} /^\[/{f=0} f && /^[a-zA-Z0-9_-]+ = /{print $1}' \
      "$CARGO_TOML" | grep -qx -- "$feat" \
      || die "release.sh checks feature '$feat', which $CARGO_TOML no longer defines; update SEMVER_FEATURES"
  done

  note "Public API delta since ${PREV_VERSION} (informational)"
  SEMVER_OUT="$(mktemp)"; TMP_FILES="$TMP_FILES $SEMVER_OUT"
  # The exit status cannot be the test, because what it means changed under this
  # script: 0.48.0 exits 1 both for a run that found breaks and for one that died
  # before checking anything, where 0.50.0 splits those into 100 and 101. A rule
  # written on either is a rule about one version window. `|| true` therefore
  # still says "a break is the expected outcome here", not "any outcome is fine".
  #
  # What both versions do is print a `Summary` line once they reach a verdict —
  # "no semver update required", or the bump the findings require — and a run
  # that never reached its checks has none to print. Asserted as present rather
  # than by looking for error text, so the release stops on a verdict this does
  # not recognize instead of on only the failures it has already met.
  # `--color always` because the pipe would otherwise strip the red and green
  # that make the verdict scannable.
  cargo semver-checks --baseline-version "$PREV_VERSION" --release-type minor \
    --default-features --features "$SEMVER_FEATURES" --color always 2>&1 \
    | tee "$SEMVER_OUT" || true
  grep -q 'Summary' "$SEMVER_OUT" \
    || die "cargo-semver-checks printed no verdict above, so the public-API delta since ${PREV_VERSION} went unchecked; its own error says why. A toolchain newer than the installed version supports is the common cause, and \`cargo install cargo-semver-checks --locked\` refreshes it. Pass --skip-api-delta to release without the check"
fi

# ---------------------------------------------------------------------------
# 2. Bump the version in Cargo.toml and Cargo.lock
# ---------------------------------------------------------------------------
# Rewrite the first `version =` line that follows the hdf5-pure package header.
# The same state machine works for Cargo.toml ([package]) and Cargo.lock
# ([[package]] hdf5-pure).
bump_version() {
  local file="$1" tmp
  tmp="$(mktemp)"; TMP_FILES="$TMP_FILES $tmp"
  awk -v new="$NEW_VERSION" '
    /^name = "hdf5-pure"/ { pkg=1 }
    pkg && /^version = / { sub(/"[^"]*"/, "\"" new "\""); pkg=0 }
    { print }
  ' "$file" > "$tmp"
  # Copy through the existing file rather than renaming the temp over it, so the
  # original inode and its 0644 mode survive (mktemp creates 0600).
  cat "$tmp" > "$file"
}
# Idempotent, and applied to each file independently: skipping both because
# Cargo.toml already reads $NEW_VERSION would leave a Cargo.lock still declaring
# the previous version in the release commit and tag.
if [ "$CUR_VERSION" = "$NEW_VERSION" ]; then
  note "Version already reads $NEW_VERSION (bumped by the PR that required it)"
else
  note "Bumping version in $CARGO_TOML and $CARGO_LOCK"
fi
bump_version "$CARGO_TOML"
bump_version "$CARGO_LOCK"

# ---------------------------------------------------------------------------
# 3. Promote [Unreleased] into a dated section and refresh the compare links
# ---------------------------------------------------------------------------
note "Updating $CHANGELOG"
CL_TMP="$(mktemp)"; TMP_FILES="$TMP_FILES $CL_TMP"
SUMMARY="$SUMMARY" awk \
  -v new="$NEW_VERSION" -v prev="$PREV_VERSION" -v date="$TODAY" -v repo="$REPO_URL" '
  # Insert the new version header + summary as the first content under
  # [Unreleased] (i.e. before the first non-blank line that follows it).
  /^## \[Unreleased\]/ { print; seen=1; next }
  seen && !done && /[^[:space:]]/ {
    print "## [" new "] - " date
    print ""
    print ENVIRON["SUMMARY"]
    print ""
    done=1
  }
  # Rewrite the [Unreleased] compare link and add the [X.Y.Z] link beneath it.
  /^\[Unreleased\]:/ {
    print "[Unreleased]: " repo "/compare/v" new "...HEAD"
    print "[" new "]: " repo "/compare/v" prev "...v" new
    next
  }
  { print }
' "$CHANGELOG" > "$CL_TMP"
cat "$CL_TMP" > "$CHANGELOG"

# The `[X.Y.Z]:` compare link is emitted only as a side effect of matching the
# `[Unreleased]:` line, so a missing or renamed anchor would drop both links
# silently and still produce a plausible-looking release.
grep -q "^\[${NEW_VERSION}\]: " "$CHANGELOG" \
  || die "$CHANGELOG has no [Unreleased]: link line to anchor the compare links to"
grep -q "^## \[${NEW_VERSION}\] - ${TODAY}\$" "$CHANGELOG" \
  || die "failed to promote [Unreleased] into a [$NEW_VERSION] section"

# ---------------------------------------------------------------------------
# 4. Verify the crate still packages cleanly with the new version
# ---------------------------------------------------------------------------
note "Verifying with cargo publish --dry-run"
cargo publish --dry-run --allow-dirty

# ---------------------------------------------------------------------------
# 5. Commit + tag (opt-in)
# ---------------------------------------------------------------------------
if [ "$DO_COMMIT" -eq 0 ]; then
  note "Prepared release files (not committed)."
  git --no-pager diff --stat -- "$CARGO_TOML" "$CARGO_LOCK" "$CHANGELOG"
  [ "$SUMMARY_IS_TODO" -eq 1 ] && \
    warn "Fill in the TODO summary in $CHANGELOG before committing."
  # Repeated here because the skip happened before a build and a package check,
  # far enough up that it is no longer on screen.
  [ "$SKIP_API_DELTA" -eq 1 ] && \
    warn "This release's public-API delta was never checked (--skip-api-delta)."
  cat <<EOF

Next steps:
  git add $CARGO_TOML $CARGO_LOCK $CHANGELOG
  git commit -m "Release $TAG"
  git tag -a "$TAG" -m "Release $TAG"
  git push origin main && git push origin "$TAG"
  # GitHub release notes = the changelog section for this version
  cargo publish
EOF
  exit 0
fi

if [ "$SUMMARY_IS_TODO" -eq 1 ]; then
  die "refusing to commit with a TODO summary; pass --summary/--summary-file"
fi

note "Committing and tagging $TAG"
git add "$CARGO_TOML" "$CARGO_LOCK" "$CHANGELOG"
git commit -m "Release $TAG"
git tag -a "$TAG" -m "Release $TAG"

# ---------------------------------------------------------------------------
# 6. Push (opt-in)
# ---------------------------------------------------------------------------
if [ "$DO_PUSH" -eq 1 ]; then
  note "Pushing main and $TAG"
  git push origin main
  git push origin "$TAG"
fi

# ---------------------------------------------------------------------------
# 7. GitHub release (opt-in) — notes are the changelog section for this version
# ---------------------------------------------------------------------------
if [ "$DO_GH" -eq 1 ]; then
  note "Creating GitHub release $TAG"
  NOTES_TMP="$(mktemp)"; TMP_FILES="$TMP_FILES $NOTES_TMP"
  awk -v ver="$NEW_VERSION" '
    index($0, "## [" ver "]") == 1 { grab=1; next }  # start after this version header
    grab && /^## \[/ { exit }                        # stop at the next version header
    grab { print }
  ' "$CHANGELOG" > "$NOTES_TMP"
  printf '\n**Full changelog:** %s/compare/v%s...v%s\n' "$REPO_URL" "$PREV_VERSION" "$NEW_VERSION" >> "$NOTES_TMP"
  gh release create "$TAG" --title "$TAG" --notes-file "$NOTES_TMP"
  rm -f "$NOTES_TMP"
fi

# ---------------------------------------------------------------------------
# 8. crates.io publish (opt-in)
# ---------------------------------------------------------------------------
if [ "$DO_PUBLISH" -eq 1 ]; then
  note "Publishing to crates.io"
  cargo publish
fi

note "Done: $TAG"

# Last, so it is the line still on screen. An `if` rather than the `&&` form the
# earlier reminders use: as the final command of the script, a false test would
# be the script's own exit status.
if [ "$SKIP_API_DELTA" -eq 1 ]; then
  warn "$TAG went out without its public-API delta being checked (--skip-api-delta)."
fi
