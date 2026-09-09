# Contributing

Thanks for considering helping this project. There are many ways you can help: using the crate and reporting bugs, reporting a file it cannot read or write, making additions and improvements to the crate and the documentation, and finding robustness bugs with crafted files.

hdf5-pure is a pure-Rust reader, writer and in-place editor for HDF5 files, maintained beside other work. A review may take a while when the maintainers are busy elsewhere.

## Bug reports and feature requests

File a GitHub [issue](https://github.com/CramBL/hdf5-pure/issues). Include as much information as possible. A format bug is easier debugged with the file, or with the script that wrote it:

- **A file this crate cannot read or edit.** Attach the file, or the script that wrote it: which library (the HDF5 C library, h5py, netCDF-4, MATLAB) and which version. `h5dump -pH file.h5` prints the structure without the data, which is likely all a report needs.
- **A file this crate wrote that another library rejects.** Attach the code that wrote it and the error the other library reported.
- **A panic or a wrong value.** A test or an example program that shows it.

Issues are also the place to get help or to put a question, with the `question` label. The repository has no discussion board.

The labels describe an issue along four axes: effect, code path, file format generation and cargo feature. The maintainers apply them. Pick them yourself if you like.

An issue labelled `decision` changes a public type or a documented contract, and the issue lists the options. Wait for the decision before starting on one.

When making a feature request, make it clear what problem you intend to solve with the feature, any ideas for how the crate could support solving that problem, any possible alternatives, and any disadvantages.

## AI policy

We **require all use of AI in contributions to follow our [AI policy](AI_POLICY.md)**.

If your contribution does not follow the policy, it will be closed.

## Submitting pull requests

The same rules apply here as for bug reports and feature requests. Plus:

- For a change over about 500 lines, file an issue prior to starting work, so that everyone can see what is in progress and the approach is agreed.
- A draft pull request is fine for feedback or to hand work over.
- Review the pull request yourself before requesting review.
- A pull request does one thing. Split a larger change into several.
- A pull request that changes what a user sees carries a changelog entry under `[Unreleased]` in `CHANGELOG.md`. [RELEASES.md](RELEASES.md) has the rules for the entry.
- The code follows [CODE_STYLE.md](CODE_STYLE.md).
- The comments and the documentation follow [DOC_STYLE.md](DOC_STYLE.md).
- Pull requests to update specific dependencies are welcome.

Before pushing, run the checks CI runs on every pull request:

```sh
just ci-essentials   # fmt, clippy, docs, tests and doctests with every feature
```

`just ci` runs the rest as well: portability, hygiene, prose, the API checks and the crosschecks against one release of the HDF5 C library. `just --list` shows every recipe, and [Testing](#testing) explains what each group covers.

## Crate features

Enabling a Cargo feature must not change the crate's default behavior. Cargo unifies enabled features across the entire dependency graph, so a library using hdf5-pure cannot control whether another library or the application enabled a feature.

A feature may expose additional APIs for opt-in functionality, but that functionality must be enabled through those APIs by the application. For example, a feature must not change what `FileBuilder::write` puts on disk for a file that does not use it. `fast-deflate` is the one feature that changes existing output: zlib-ng compresses to different bytes than the pure-Rust backend, and the data they decode to is the same.

## Commit history

Commits are atomic. Each commit does one thing, builds and passes the tests on its own, and its message describes the change it contains. The pull request description summarizes the whole.

- A refactoring is a separate commit from the functional change it prepares.
- A mechanical change (a rename, moving code, formatting) is a separate commit.
- A `Cargo.lock` update is a separate commit.

Our default workflow is to `rebase` a pull request onto `main`, so every commit lands as it is. When addressing review comments, fix the existing commits and force-push to your branch, which [`git absorb`](https://github.com/tummychow/git-absorb) and [`git revise`](https://github.com/mystor/git-revise) make easy. A pull request whose commits do not each stand on their own is squashed, at the maintainers' choice.

### Commit messages

The subject line is a sentence: a capital first letter, no trailing period, at most 72 characters. It may start with a lowercase tag for the part of the crate it touches, such as `edit: ` or `mat: `. It states what the change does, in the imperative: `Read`, not `Reads`, `Reading` or `Added`:

```
Read datatype message version 5, as HDF5 2.0 writes under latest bounds
Preserve a version 2 object header's optional fields across in-place rewrites
```

The body, when there is one, follows a blank line and explains why. Issue references and trailers go at the end, after a blank line: `Closes #NN`, `Signed-off-by:`, `Co-authored-by:` for a person, `Assisted-by:` for a tool. `just prose::commits` checks every commit since `origin/main`, as CI does.

## Testing

`just test` runs the suite under nextest with the default features, `just test-full` with every feature, and `just test-each-feature` each optional feature on its own. Doctests run through `just doctest` and `just doctest-full`.

- An addition to the public API has (at least) a test through that API, under `tests/`.
- A change to the reader or the writer has a test against a file the C library or h5py wrote, or a crosscheck, whichever shows the change.
- A test of a failure asserts the exact error: the `Error` or `FormatError` variant, or the major and minor codes the C library reports. [CODE_STYLE.md](CODE_STYLE.md#assert-the-exact-error) has the forms.

### Test data

The fixtures under `tests/data/` are grouped by the library that wrote them: `c` for the HDF5 C library, `h5py`, and `pure` for this crate. A fixture is regenerated by the tool that made it, never edited by hand. `just test-data::pure` rewrites the ones this crate writes, and `just test-data::c` the ones the C library writes, building it from source on first use.

### Crosschecks against the C library

`crates/crosscheck/` holds the tests that link the reference HDF5 C library and compare its reading of a file with ours. It is the only package that links C code. `just interop::test-bundled` builds the library the `hdf5-metno` binding bundles, so it doesn't need HDF5 on the host. `just interop::test-hdf5 1.14.6` links a specific release, built under `tmp/hdf5/` on first use, and `just interop::default` runs every release in the matrix.

### Portability, soundness and the API

- `just portability::default` checks the `no_std`, WASM and bare-metal builds, and runs Clippy on a 32-bit target with the truncating casts denied.
- `just soundness::miri` runs [Miri](https://github.com/rust-lang/miri) over the crate's `unsafe` code.
- `just api::semver` runs [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) against the last release, and `just api::msrv` checks that the library builds on the `rust-version` in `Cargo.toml`.
- `just fuzz::default` runs each fuzz target for thirty seconds on a nightly toolchain. The crate would benefit from more runtime, targets and corpora. A crafted file that makes it panic, allocate without bound or read a wrong value is a bug: file it.

### Hygiene and prose

`just hygiene::default` checks for unused dependencies, lints the workflows and spell-checks every tracked file. A word the format uses that the spell checker does not know goes in `typos.toml`. `just prose::added` lints the prose of every added line with [Vale](https://vale.sh) and the rules under `.vale/styles/`.

`just hygiene::lychee` checks every link in the Markdown files and the Rust sources, CI runs it weekly. `just hygiene::lychee-added` checks the links a change adds, CI runs it on every pull request.

## Documentation

The site under `docs/` is built with [MkDocs](https://www.mkdocs.org/) and published from a release tag. `just docs::serve` previews it locally at <http://127.0.0.1:8000>, and `just docs::build` fails on a broken link or a page missing from the nav. `CHANGELOG.md` is the source of the changelog page.

`just doc` builds the API documentation twice with warnings denied: as it is published, then again with the private items.

## Pull request review

It is very easy to add complexity to a project. Each line of code added is code that needs to be maintained in perpetuity, so the bar for merging is high.

When reviewing, we look for:

- The pull request title and description should be helpful.
- Each commit does one thing, and its message says what.
- A breaking change is marked `**Breaking:**` in its changelog entry, which decides the next release's version, and is documented in the pull request description.
- The code should be readable.
- The code should have helpful doc comments.
- The code should follow [CODE_STYLE.md](CODE_STYLE.md).
- The tests listed under [Testing](#testing) are there.

Each `0.x.0` release may break the public API, so a breaking change in a pull request is acceptable. We still avoid one when we can, and when we cannot we deprecate the old item with `#[deprecated]` first.

## Licensing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed under MIT or Apache-2.0 as described in the [README](README.md#license), without any additional terms or conditions.
