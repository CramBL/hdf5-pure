"""Tests for `hdf5_pure_scripts.prose`: the parsing and formatting the gate is built from.

No test runs Vale or reaches the network: `_VALE_JSON` and `_CONTRASTIVE_JSON`
are captured replies.
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from hdf5_pure_scripts import prose

_VALE_JSON = """\
{
  "justfile": [
    {
      "Span": [3, 10],
      "Check": "Hdf5Pure.Overused",
      "Message": "Overused in generated prose: 'seamless'.",
      "Severity": "error",
      "Line": 12
    },
    {
      "Span": [17, 17],
      "Check": "Hdf5Pure.Semicolon",
      "Message": "Semicolon: split into two sentences.",
      "Severity": "error",
      "Line": 40
    }
  ]
}
"""

_CONTRASTIVE_JSON = """\
{
  "stdin.commit": [
    {
      "Span": [22, 32],
      "Check": "Hdf5Pure.Contrastive",
      "Message": "Contrastive 'rather than': state the current behaviour without the alternative.",
      "Severity": "error",
      "Line": 3
    }
  ]
}
"""

_STUB_ENGINE = prose.Engine(argv=["vale"], description="vale")

_REPLACEMENT_MESSAGE = "A replacement subject\n\nA replacement body.\n"

_BODY = "An appended note.\n"

_JUST_SOURCE = """\
# Lint the tracked Markdown files.
[doc("Every surface, the whole backlog.")]
docs:
    git ls-files -z '*.md'
"""

_TOML_SOURCE = """\
# The crate's own suite.
[package]
name = "hdf5-pure"
#Tight comment.
"""

_YAML_SOURCE = """\
# A pull request is gated against its base.
jobs:
  prose:
    runs-on: ubuntu-latest # trailing comments stay with the code
"""


def test_annotation_anchors_a_file_alert_on_its_line() -> None:
    alert = prose.parse_alerts(_VALE_JSON)[0]
    assert alert.annotation() == (
        "::error file=justfile,line=12::Hdf5Pure.Overused: Overused in generated prose: 'seamless'."
    )


def test_annotation_anchors_a_commit_alert_on_its_hash() -> None:
    alert = prose.parse_alerts(_VALE_JSON, where="8250d2ea", commit=True)[0]
    assert alert.annotation() == (
        "::error title=commit 8250d2ea::Hdf5Pure.Overused: Overused in generated prose: 'seamless'."
    )


@pytest.mark.parametrize(
    ("hunk", "expected"),
    [
        ("@@ -1,0 +5,3 @@", {5, 6, 7}),
        ("@@ -1 +5 @@", {5}),
        ("@@ -4,2 +3,0 @@", set()),
    ],
)
def test_added_lines_reads_a_hunk_header(hunk: str, expected: set[int]) -> None:
    diff = f"--- a/README.md\n+++ b/README.md\n{hunk}\n+added\n"
    assert prose.added_lines(diff).get("README.md", set()) == expected


def test_added_lines_keys_each_hunk_on_the_file_above_it() -> None:
    diff = """\
--- a/README.md
+++ b/README.md
@@ -1,0 +2,1 @@
+one
--- a/justfile
+++ b/justfile
@@ -8,0 +9,2 @@
+two
+three
"""
    assert prose.added_lines(diff) == {"README.md": {2}, "justfile": {9, 10}}


def test_keep_added_drops_an_alert_off_an_added_line() -> None:
    alerts = prose.parse_alerts(_VALE_JSON)
    kept = prose.keep_added(alerts, {"justfile": {12}})
    assert [alert.check for alert in kept] == ["Hdf5Pure.Overused"]


def test_parse_alerts_names_a_stdin_reply_after_the_file_it_read() -> None:
    alerts = prose.parse_alerts(_VALE_JSON.replace("justfile", "stdin.md"), where="scripts/x.just")
    assert [alert.where for alert in alerts] == ["scripts/x.just", "scripts/x.just"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            _JUST_SOURCE,
            "Lint the tracked Markdown files.\nEvery surface, the whole backlog.\n\n\n",
        ),
        (_TOML_SOURCE, "The crate's own suite.\n\n\nTight comment.\n"),
        (_YAML_SOURCE, "A pull request is gated against its base.\n\n\n\n"),
    ],
)
def test_comment_text_keeps_the_comments_and_blanks_the_rest(source: str, expected: str) -> None:
    assert prose.comment_text(source) == expected


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("vale version 3.20.0\n", "3.20.0"),
        ("vale version v3.20.0\n", "3.20.0"),
        ("no version here\n", None),
    ],
)
def test_normalize_version_reads_the_dotted_version(reported: str, expected: str | None) -> None:
    assert prose.normalize_version(reported) == expected


def test_summary_counts_what_the_run_covered_and_found() -> None:
    totals = prose.RunTotals(files=4, lines=217, commits=1, errors=5, warnings=0)
    assert totals.summary("origin/main") == (
        "vale: 4 files, 217 added lines, 1 commit checked: 5 errors, 0 warnings"
    )


def test_summary_says_nothing_to_check_when_the_range_is_empty() -> None:
    totals = prose.RunTotals(files=0, lines=0, commits=0, errors=0, warnings=0)
    assert totals.summary("HEAD") == (
        "vale: nothing to check since HEAD: "
        "no added lines in a linted file, and no commits in the range"
    )


def _run_git(root: Path, args: Sequence[str]) -> None:
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=Test", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _commit_empty(root: Path, message: str) -> None:
    _run_git(root, ["commit", "--quiet", "--allow-empty", "-m", message])


def _repository_with_one_commit(root: Path) -> None:
    _run_git(root, ["init", "--quiet"])
    _commit_empty(root, "Root")


def test_script_files_reads_every_toml_file_and_only_the_workflow_yaml(tmp_path: Path) -> None:
    for rel in (
        "justfile",
        "Cargo.toml",
        "crates/test-util/Cargo.toml",
        "lychee.toml",
        "scripts/api.just",
        ".github/workflows/ci.yml",
        ".config/nextest.yml",
        "README.md",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# A comment.\n")
    _run_git(tmp_path, ["init", "--quiet"])

    assert prose.script_files(tmp_path) == [
        ".github/workflows/ci.yml",
        "Cargo.toml",
        "crates/test-util/Cargo.toml",
        "justfile",
        "lychee.toml",
        "scripts/api.just",
    ]


def test_merge_base_exits_with_one_line_when_the_base_ref_does_not_resolve(tmp_path: Path) -> None:
    _repository_with_one_commit(tmp_path)

    with pytest.raises(SystemExit) as raised:
        prose._merge_base_with_head(tmp_path, "origin/main")

    assert str(raised.value) == (
        "error: base ref origin/main does not resolve: fetch it, or pass another base"
    )


def test_commits_in_drops_a_fixup_subject_and_keeps_a_body_that_quotes_one(
    tmp_path: Path,
) -> None:
    _repository_with_one_commit(tmp_path)
    _commit_empty(tmp_path, "fixup! Root")
    _commit_empty(tmp_path, "Quote a fixup subject\n\nfixup! Root is what this body says.")
    _commit_empty(tmp_path, f"squash! Root\n\n{_BODY}")
    _commit_empty(tmp_path, f"amend! Root\n\n{_REPLACEMENT_MESSAGE}")

    commits = prose.commits_in(tmp_path, "HEAD")

    assert [commit.subject for commit in commits] == [
        "amend! Root",
        "squash! Root",
        "Quote a fixup subject",
        "Root",
    ]
    assert commits[0].message == f"amend! Root\n\n{_REPLACEMENT_MESSAGE}"
    assert commits[-1].message == "Root\n"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (f"amend! Root\n\n{_REPLACEMENT_MESSAGE}", _REPLACEMENT_MESSAGE),
        ("amend! Root\n", ""),
        (f"squash! Root\n\n{_BODY}", f"\n\n{_BODY}"),
        (f"A plain subject\n\n{_BODY}", f"A plain subject\n\n{_BODY}"),
    ],
)
def test_message_to_lint_returns_the_text_that_lands_on_the_branch(
    message: str, expected: str
) -> None:
    assert prose.Commit(hash="8250d2ea", message=message).message_to_lint() == expected


def test_lint_commit_reads_an_amend_replacement_and_names_the_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = prose.Commit(hash="8250d2ea", message=f"amend! Root\n\n{_REPLACEMENT_MESSAGE}")
    read_by_vale: list[str | None] = []

    def stub_vale(
        engine: prose.Engine, root: Path, args: Sequence[str], stdin: str | None = None
    ) -> str:
        read_by_vale.append(stdin)
        return _CONTRASTIVE_JSON

    monkeypatch.setattr(prose, "_run_vale", stub_vale)

    alerts = prose._lint_commit(_STUB_ENGINE, Path("."), commit)

    assert read_by_vale == [_REPLACEMENT_MESSAGE]
    assert [alert.annotation() for alert in alerts] == [
        "::error title=commit 8250d2ea::Hdf5Pure.Contrastive: "
        "Contrastive 'rather than': state the current behaviour without the alternative."
    ]


def test_lint_commit_runs_no_vale_on_an_amend_without_a_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_on_call(*args: object, **kwargs: object) -> str:
        raise AssertionError("vale ran on a commit with no body")

    monkeypatch.setattr(prose, "_run_vale", fail_on_call)

    commit = prose.Commit(hash="ad2e0f4", message="amend! Root\n")

    assert prose._lint_commit(_STUB_ENGINE, Path("."), commit) == []
