# Minimized malformed HDF5 regressions

These are minimized `cargo-fuzz` inputs used by ordinary tests. Retain the exact
submitted bytes as frozen regressions. Their original producer, producer version,
version of `cargo-fuzz`, pre-mutation files, and precise minimization commands are
unrecorded. No fixture-specific license or attribution is recorded.

## Origins

| File | Recorded import |
|---|---|
| `oom_chunked_string_huge_elem.h5` | `d25d0e286a30f14e17f27100b874ac44b537ec2d`, originally under `tests/fixtures/fuzz/` |
| `sohm_heap_id_length.h5` | `ee6458d05b8f2d7a1c198caf560ff382f7d87be3` |
| `zero_width_elem_chunked_string.h5` | `55dae3b06e30edecddad246c823c697d478789da`, originally under `tests/fixtures/fuzz/` |

The first and third files were relocated in `9c26c10a`. The source of the SOHM
input before its import and its relationship to the SOHM mutation-corpus seeds
are unrecorded.

## Consumers and invariants

`tests/main/fuzz_regressions.rs` exercises all three inputs:

| File | Retained malformed condition | Required observation |
|---|---|---|
| `oom_chunked_string_huge_elem.h5` | A chunked string dataset has shape `[50]`, element size `0xAAAA_AAAA`, and an empty link name | The chunk-geometry check rejects string reads with `InvalidChunkGeometry` before payload allocation |
| `sohm_heap_id_length.h5` | A shared-message heap ID has eight bytes where 96 are expected | Both shape and datatype reads on `d1` report `InvalidFractalHeapIdLength { expected: 96, actual: 8 }` through buffered and source readers |
| `zero_width_elem_chunked_string.h5` | A chunked string datatype has zero element width | Shape `[50]` and empty attributes remain readable. Element-size, datatype, raw, and string reads report `ZeroSizedDatatype { class: 3 }` |

The huge-element input is also consumed by `tests/main/element_size.rs` for its
declared width, `tests/main/object_path.rs` for unnamed-object lookup, and
`tests/main/repack.rs` for rejection before a destination is created.

For the huge-element and zero-width inputs, the consumers use
`fixture_with_recorded_extent` in `crates/test-util/src/fuzz.rs`. It zero-extends
an in-memory copy to the EOF address stored in the superblock, preserving the
datatype and link metadata. It never rewrites the committed file. The SOHM input
is read directly.

These malformed ordinary regressions are separate from the frozen mutation
starting points documented in [the corpus notice][corpus]. Regenerating a valid
file through a producer would discard their malformed conditions.

[corpus]: ../../../fuzz/corpus/README.md
