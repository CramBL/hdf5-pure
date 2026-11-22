# hdf5-pure-core

Provides the shared definitions used by [hdf5-pure][facade] and its implementation crates.
Core owns types re-exported by the file API. Binary parsers and encoders belong to [hdf5-pure-format][format].

## Shared requirements

- **Portability:** use `no_std` with `alloc`, importing from `core` or `alloc` where available.
  Keep core independent of implementation crates.
- **Checked state:** apply **parse, don't validate**. Construction and conversion establish
  invariants that subsequent operations use directly.
- **Reuse:** reuse shared definitions and expose workspace-only operations through
  [`__private`][private].

## Performance and errors

| Concern        | Requirement                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Code size      | Keep generic code small. Put type-independent work in concrete helpers.                                                  |
| Allocation     | Avoid unnecessary allocation, copying, and retained buffers.                                                             |
| Representation | Keep stored state compact without weakening its invariants.                                                              |
| Errors         | Handle invalid input through typed errors. Preserve useful context and causes. Build diagnostic-only context on failure. |

For workspace-wide guidance, see [ARCHITECTURE.md][architecture], [CODE_STYLE.md][style],
and [TESTING.md][testing].

[facade]: ../../README.md
[format]: ../hdf5-pure-format/README.md
[private]: src/__private.rs
[architecture]: ../../ARCHITECTURE.md
[style]: ../../CODE_STYLE.md
[testing]: ../../TESTING.md
