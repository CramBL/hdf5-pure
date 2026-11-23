# hdf5-pure-format

Implements HDF5 wire layouts, parsers, and encoders for [hdf5-pure][facade].
Shared definitions come from [hdf5-pure-core][core]. File access belongs to the facade.

## Shared foundations

Use these foundations when adding or extending parsers and encoders:

| Interface                                       | Purpose                                                                                                    |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [`ByteCursor` and bounded-array helpers][bytes] | Acquire complete byte regions and arrays before interpreting them.                                         |
| [`Decode<'input>`][codec]                       | Decode using supplied context. Results may borrow input. Each implementation defines framing.              |
| [`Encode`][codec]                               | Exact, infallible length and bounded writing. Errors leave output unchanged. Success preserves its suffix. |
| [`MessageBody`][message-body]                   | Associates a checked inline body with its object-header message type.                                      |
| [`MetadataSource`][source]                      | Read metadata through a caller-supplied byte source.                                                       |

`Decode` and `Encode` are crate-private. Workspace-facing operations use [`__private`][private].

- **Checked state:** apply **parse, don't validate**. Retain format invariants in checked values
  or immutable preparation so subsequent operations can use them directly.
- **Shared implementation:** concrete entry points and trait implementations use one codec path.
- **Portability:** keep implementations usable with `no_std` and `alloc`.
  Import from `core` or `alloc` where available.

## Performance and errors

- **Code size:** minimize generic code. Keep shared framing, diagnostics, and result construction
  in concrete helpers. Select widths outside element loops.
- **Allocation:** avoid unnecessary copies and intermediate buffers. Reuse prepared state and
  caller-provided output where useful.
- **Representation:** keep memory usage compact. Semantic interfaces need not dictate how
  their state is stored.
- **Errors:** return typed errors for malformed input and failed preparation.
  Preserve useful context and causes. Build diagnostic-only context on failure.

For workspace-wide guidance, see [ARCHITECTURE.md][architecture], [CODE_STYLE.md][style],
and [TESTING.md][testing].

[facade]: ../../README.md
[core]: ../hdf5-pure-core/README.md
[bytes]: src/bytes.rs
[codec]: src/codec.rs
[source]: src/metadata_source.rs
[private]: src/__private.rs
[message-body]: src/object_header/v2/record.rs#L30-L34
[architecture]: ../../ARCHITECTURE.md
[style]: ../../CODE_STYLE.md
[testing]: ../../TESTING.md
