# hdf5-pure-format

Implements HDF5 wire layouts, parsers, and encoders for [hdf5-pure][facade].
Shared definitions come from [hdf5-pure-core][core]. File access belongs to the facade.

## Choosing a codec interface

Choose the interface by the input or output boundary and the result the caller needs.

| Responsibility | Interface |
| --- | --- |
| Read one complete wire unit at the start of a slice | `DecodePrefix`, returning the value and its byte count in `Parsed<T>` |
| Interpret a complete region supplied by the caller | `DecodeFrame` |
| Read the same representation at either boundary | Both traits over one parser or shared framing step |
| Write a complete encoding into a bounded slice | `Encode` on a checked value or immutable prepared encoding |
| Write a typed object header message body | [`MessageBody`][message-body] on its checked encoder |
| Acquire a scalar or an array, or interpret a width or an address | [Byte helpers][bytes], [widths][widths], and [addresses][addresses] |
| Inspect fields, convert values, resolve references, traverse a file, place blocks, or edit bytes in place | Concrete operations, reading any file bytes through [`MetadataSource`][source] |

The [codec traits][codec] are crate-private. Expose concrete checked operations through
[`__private`][private] for workspace callers. Use `Decode` only for existing implementations
awaiting migration. Remove it with its last implementation. See [`src/codec.rs`][codec] for
the prefix, frame, and encoding contracts.

## Checked state and bounded output

Apply **parse, don't validate**. Establish versions, widths, counts, and geometry during
parsing or construction, and keep them in private checked state. Prepare every fallible
choice before writing. An encoder that returns an error leaves the entire output unchanged.

## Borrowing, performance, and errors

- **Lifetimes:** convert a borrowed result to an owned value before releasing its source buffer.
- **Portability:** keep implementations usable with `no_std` and `alloc`.
- **Code size:** keep generic adapters small. Select widths outside element loops.
- **Memory:** avoid unnecessary copies and intermediate buffers. Keep stored state compact.
- **Errors:** preserve typed causes. Build diagnostic-only context on failure.
- **Boundaries:** distinguish a parent failing to acquire a declared child, a child rejecting
  its input, and trailing bytes in an exact frame.

For workspace-wide guidance, see [ARCHITECTURE.md][architecture], [CODE_STYLE.md][style],
and [TESTING.md][testing].

[facade]: ../../README.md
[core]: ../hdf5-pure-core/README.md
[bytes]: src/bytes.rs
[widths]: src/width.rs
[addresses]: src/address.rs
[codec]: src/codec.rs
[source]: src/metadata_source.rs
[private]: src/__private.rs
[message-body]: src/object_header/v2/prepared_record.rs
[architecture]: ../../ARCHITECTURE.md
[style]: ../../CODE_STYLE.md
[testing]: ../../TESTING.md
