# Malformed HDF5 regression inputs

The three inputs in the following tables are minimized `cargo-fuzz` inputs used by
ordinary tests. Retain their exact imported bytes as regression fixtures. Their
original producer, producer version, version of `cargo-fuzz`, pre-mutation files,
and precise minimization commands are unrecorded. No fixture-specific license or attribution is recorded.

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

## Object-header version error precedence

`invalid_v1_version_without_padding.h5` is the exact 108-byte input retained from
[CI run 41208310457, job 124557928723][ci]. The run tested revision
`6e70fa70bbdc98ac76e78bc56266643cd0386f73` with `nightly-2026-08-13` and
`libfuzzer-sys` 0.4.13. Its artifact name was
`crash-7c17df84699b0065a107e88ffc70939cde74ca85`, and its SHA-1 is
`7c17df84699b0065a107e88ffc70939cde74ca85`. No local minimization or byte
transformation was applied. The producer, producer version, and license of the
input before fuzz mutation are unknown.

`tests/main/streaming_differential.rs` reads this input directly. Its root object
header has version byte 2 at physical offset 96 and ends after twelve prefix
bytes, before the alignment padding. Buffered and streaming attribute reads must
both report `InvalidObjectHeaderVersion(2)` before a truncation error. Retain the
exact input to preserve this error-precedence case independently of test builders.

These malformed ordinary regressions are separate from the retained mutation
starting points documented in [the corpus notice][corpus]. Regenerating a valid
file through a producer would discard their malformed conditions.

[corpus]: ../../../fuzz/corpus/README.md
[ci]: https://github.com/CramBL/hdf5-pure/actions/runs/41208310457/job/124557928723
