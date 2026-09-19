# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under Cargo's pre-1.0 conventions: a `0.x.0` bump may be breaking, `0.x.y` is not.

## [Unreleased]

### Changed

- **Breaking:** `DatatypeByteOrder` implements `Copy`.
- **Breaking:** `Datatype::FixedPoint` and `Datatype::FloatingPoint` group layout fields into new `FixedPointLayout` and `FloatingPointLayout` structs.
- Lossy reads of numeric datatypes (e.g. `i32` to `u32`) use soft clamping conversions, matching `libhdf5`.
- **Breaking:** `MessageType` uses a transparent `u16` representation and preserves unknown message type identifiers without an enum payload. The former CamelCase variants and `Unknown(u16)` constructor remain as deprecated compatibility APIs. `Unknown(id)` pattern matching must use `unknown_id()`.

### Fixed

- `Dataset::read` decodes partial edge chunks stored with skipped filters. Unfiltered edge chunks bypass the inverse filter pipeline, while full chunks process through the pipeline ([#589](https://github.com/CramBL/hdf5-pure/pull/589)).
- `Dataset::read` calculates chunk addresses for datasets whose implicit indices span expanded maximum chunk grids. Repacking these datasets preserves chunk data. Version 4 chunk sizes exceeding 32 bits remain intact until read or allocation operations require host-sized values ([#587](https://github.com/CramBL/hdf5-pure/pull/587)).
- `Dataset::read` converts numeric values when destination and stored datatypes differ ([#597](https://github.com/CramBL/hdf5-pure/pull/597)).

## [0.46.1] - 2026-09-15

### Fixed

- `DatasetBuilder::with_path_references` resolves a target whose path contains a `.` component against the identified object. Targets written or deleted in the same commit are rejected ([#573](https://github.com/CramBL/hdf5-pure/pull/573)).
- `Group::create_group_with` rejects group creation at the root group path from inside its closure and stages no edits. Group creations staged through group handles report paths lacking link names as invalid ([#573](https://github.com/CramBL/hdf5-pure/pull/573)).
- `FileBuilder::write`, group builders, and in-memory write methods report `FormatError::InvalidLinkName` when a dataset, group, or committed-datatype identifier does not form a valid object path component ([#580](https://github.com/CramBL/hdf5-pure/pull/580)).
- `DatasetBuilder::with_path_references` resolves reference targets regardless of equivalent path formatting when writing complete files ([#580](https://github.com/CramBL/hdf5-pure/pull/580)).
- `RepackOptions::drop_path` accepts target paths across equivalent path formatting. Specifying the root group as a drop path, or providing a source link that does not form a path component, returns `Error::RepackUnsupported` ([#580](https://github.com/CramBL/hdf5-pure/pull/580)).
- `File::persisted_free_space` reports free regions for files with address widths narrower than 8 bytes, enabling subsequent edits to reuse those space regions ([#581](https://github.com/CramBL/hdf5-pure/pull/581)).
- `File::commit` reads the superblock extension at its absolute address when persisting free-space managers ([#584](https://github.com/CramBL/hdf5-pure/pull/584)).

## [0.46.0] - 2026-09-14

### Added

- `RepackOptions::reject_unknown_messages_only_a_writer_must_understand` configures `repack` to reject a source whose object header holds an unknown message type requiring writer support ([#564](https://github.com/CramBL/hdf5-pure/pull/564)).

### Changed

- **Breaking:** A dataspace's unlimited maximum dimension evaluates as `MaxExtent::Unlimited`. The bytes stored in a file remain unchanged ([#562](https://github.com/CramBL/hdf5-pure/pull/562)).
- **Breaking:** The test-only feature checking allocation baselines is renamed to `__heap-baseline` ([#558](https://github.com/CramBL/hdf5-pure/pull/558)).
- Operations on malformed or missing inputs (reads, edits, appends, and copies) return errors describing the specific failure condition ([#566](https://github.com/CramBL/hdf5-pure/pull/566)).
- A chunked dataset whose layout contains an unknown chunk indexing type is rejected with `FormatError::InvalidChunkIndexType` ([#560](https://github.com/CramBL/hdf5-pure/pull/560)).
- The `provenance` feature builds on `sha2` `0.11` ([#559](https://github.com/CramBL/hdf5-pure/pull/559)).

### Fixed

- A lookup or staged write through a file or group handle accepts a complete object path. Repeated or trailing separators and `.` components normalize out, while a leading `/` resolves relative to the root group. Errors identify target objects by their path from the root group ([#567](https://github.com/CramBL/hdf5-pure/pull/567)).
- `Group::delete` and associated staged creation methods reject paths that do not specify a link, including `""`, `"."`, and `"/"` ([#567](https://github.com/CramBL/hdf5-pure/pull/567)).
- `File::open_rw` edits files whose addresses exceed `usize::MAX` on 32-bit target architectures using bounded backing. Address values exceeding platform pointer widths return `FormatError::ValueTooLargeForPlatform`. Files with pre-v2 superblocks or userblocks use whole-file mirroring bounded by platform pointer width ([#561](https://github.com/CramBL/hdf5-pure/pull/561), [#563](https://github.com/CramBL/hdf5-pure/pull/563)).
- An object header holding an unknown message type marked mandatory for readers is rejected with `FormatError::UnsupportedMessage` ([#555](https://github.com/CramBL/hdf5-pure/pull/555)).
- Read-only open operations (`File::open` and related read-only entry points) successfully read objects whose headers contain unknown messages marked mandatory for writers only. Read-write open operations continue to reject these objects ([#564](https://github.com/CramBL/hdf5-pure/pull/564)).

## [0.45.0] - 2026-09-11

### Changed

- **Breaking:** The published crate ships without a build script, mitigating supply-chain attack risks from `build.rs` execution during compilation ([#548](https://github.com/CramBL/hdf5-pure/pull/548)). The libmatio integration test and its linker directives move to the unpublished `hdf5-pure-crosscheck` package behind the private `__matio` feature, removing the internal `matio-crosscheck` feature.

### Fixed

- `Dataset::read` and `Group::get` read files whose superblock specifies unequal address and length widths. Raw data chunk B-tree v1 keys, symbol table entries, and group B-tree keys size correctly according to their respective widths ([#546](https://github.com/CramBL/hdf5-pure/issues/546)).
- Fletcher32 checksum calculation folds running sums to 16 bits, matching `libhdf5`'s `H5_checksum_fletcher32`. Chunks with running sums folding to non-zero multiples of 65,535 calculate valid checksums interoperable with `libhdf5` ([#428](https://github.com/CramBL/hdf5-pure/issues/428)).
- `Group::attrs` and `Dataset::attrs` read an attribute with a null dataspace in a version 1 object header as an empty array. Truncated attribute payloads are rejected with `FormatError::UnexpectedEof` ([#448](https://github.com/CramBL/hdf5-pure/issues/448)).

## [0.44.2] - 2026-09-10

### Changed

- Crate documentation is hosted on docs.rs, covering crate-level pages, task guides in the `_guide` module, and API references. Guide pages ship within the crate package, and <https://crambl.github.io/hdf5-pure/> redirects to docs.rs ([#531](https://github.com/CramBL/hdf5-pure/pull/531), [#532](https://github.com/CramBL/hdf5-pure/pull/532), [#535](https://github.com/CramBL/hdf5-pure/pull/535), [#536](https://github.com/CramBL/hdf5-pure/pull/536), [#537](https://github.com/CramBL/hdf5-pure/pull/537), [#538](https://github.com/CramBL/hdf5-pure/pull/538)).

## [0.44.1] - 2026-09-08

### Fixed

- HDF5 2.0 files written under latest library bounds read successfully, including compound and array datatypes.
- `Group::named_datatype_references` reports accurate reference counts stored in version 1 object headers.

### Changed

- Dual-licensed under MIT or Apache-2.0 at the user's option ([#511](https://github.com/CramBL/hdf5-pure/pull/511)).
- The repository location updated to `CramBL/hdf5-pure`, and documentation site redirects to <https://crambl.github.io/hdf5-pure/>.
- Releases publish to crates.io via Trusted Publishing from the repository Release workflow without long-lived tokens.
- The published crate package size decreases from 7.1 MiB to 5.2 MiB by including only the library, examples, licenses, README, and CHANGELOG, while retaining tests, test data, tooling, and documentation source in the repository.

## [0.44.0] - 2026-09-04

In-place editing supports files created by h5py and netCDF-4. `File::open_rw` edits objects tracking attribute creation order and adds datasets or groups to objects tracking link creation order (`track_order=True`). Rewrites preserve object header timestamps and `H5Pset_attr_phase_change` thresholds ([#416](https://github.com/CramBL/hdf5-pure/issues/416), [#422](https://github.com/CramBL/hdf5-pure/pull/422)). `File::open` and `File::open_streaming` read files with shared object header messages (`H5Pset_shared_mesg_index`), resolving datatypes, dataspaces, fill values, filter pipelines, and attributes from the shared-message heap, while `repack` converts shared messages to inline storage ([#417](https://github.com/CramBL/hdf5-pure/issues/417)). Object deletions truncate trailing file space to match active allocations ([#418](https://github.com/CramBL/hdf5-pure/issues/418)). `WriteMarkPolicy::AllowSnapshot` enables read-only path-based snapshots of files active under page-buffered writers ([#419](https://github.com/CramBL/hdf5-pure/issues/419)).

### Added

- `File::open_rw` edits objects tracking attribute creation order (h5py `track_order=True` and netCDF-4 files). New attributes receive the next creation index, overwrites retain existing indices, deletions leave index gaps, and dense attribute sets construct a creation-order B-tree alongside the name index ([#416](https://github.com/CramBL/hdf5-pure/issues/416)).
- `File::open_rw` creates datasets and groups inside groups tracking **link** creation order (h5py `track_order=True` groups and netCDF-4 groups). Additions receive the next link creation index. Additions exceeding the compact-storage threshold (8 links by default) are rejected ([#422](https://github.com/CramBL/hdf5-pure/pull/422)).
- `FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)` enables read-only path-based snapshots of files held by page-buffered writers without in-memory copying via `File::from_bytes`. Callers assert the writer has synchronized. SWMR pairs and read-write opens remain rejected ([#419](https://github.com/CramBL/hdf5-pure/issues/419)).
- `File::open` and `File::open_streaming` read files with shared object header messages (`H5Pset_shared_mesg_index`), resolving datatype, dataspace, fill value, filter pipeline, or attribute messages from the shared-message heap. The `repack` utility rewrites these files with inline messages. Edits that invalidate shared-message tables are rejected ([#417](https://github.com/CramBL/hdf5-pure/issues/417)).

### Fixed

- `Error::FileMarkedInUse` for files marked by page-buffered writers directs callers to `WriteMarkPolicy::AllowSnapshot`, `File::from_bytes`, and `File::clear_swmr_flag` ([#419](https://github.com/CramBL/hdf5-pure/issues/419)).
- Object deletions truncate trailing file space to the filesystem across files with persisted free-space managers and paged files. Commits shorten the file to the end of the last live allocation, aligned to page boundaries when required by the space strategy ([#418](https://github.com/CramBL/hdf5-pure/issues/418)).
- `File::open_rw` preserves object header timestamps and `H5Pset_attr_phase_change` thresholds during in-place edits on files created by `libhdf5`, h5py, or netCDF-4. Modification and change timestamps update to match the edit operation. Phase change thresholds remain preserved ([#422](https://github.com/CramBL/hdf5-pure/pull/422)).

## [0.43.1] - 2026-09-03

A file that deletes and re-appends objects smaller than a megabyte no longer grows without bound: `Dataset::append` and `BufferedAppender` now reuse a freed hole of any size on a paged file and on a file that persists its free-space managers ([#413](https://github.com/CramBL/hdf5-pure/issues/413)). Patch release, no API change.

### Fixed

- `Dataset::append` and `BufferedAppender` reuse a freed hole of any size on a paged file and on a file that persists its free-space managers, where holes under a megabyte were left alone and a file that deleted and re-appended objects smaller than that grew without bound. One manager rewrite now takes every hole the appended chunk fits in, up to a megabyte of them, and places its own tail inside that space rather than at end-of-file ([#413](https://github.com/CramBL/hdf5-pure/issues/413)).

## [0.43.0] - 2026-09-02

Staged edits are addressable while they are still staged. `Group::create_group`, `create_group_with` and `create_dataset` hand back a handle onto the object they stage, so a nested schema is built and its handles kept without a `commit` in between; such a handle stages further edits and answers a staged dataset's shape, maxshape, datatype and filters, reports the new `Error::NotCommitted` for anything that reads bytes, and reports the new `Error::StagingWithdrawn` once `Group::delete` withdraws the staging rather than answering for another object ([#392](https://github.com/CramBL/hdf5-pure/issues/392)). Appends improved on three fronts: `Dataset::append_staged` folds elements into a pending creation, which needs neither an unlimited dimension nor a chunked layout; a filtered dataset grows from a length that is not chunk-aligned by re-encoding its trailing chunk, except under a lossy pipeline where that would change committed values ([#393](https://github.com/CramBL/hdf5-pure/issues/393), [#407](https://github.com/CramBL/hdf5-pure/issues/407)); and `Dataset::append` reuses space an earlier commit freed instead of always extending end-of-file ([#387](https://github.com/CramBL/hdf5-pure/issues/387)). A `FileSpaceStrategy::Page` file also stops growing under delete-and-recreate churn ([#388](https://github.com/CramBL/hdf5-pure/issues/388)), and `with_libver_bounds` and `with_page_buffer_size` accept bounds and budgets they used to refuse ([#390](https://github.com/CramBL/hdf5-pure/issues/390), [#391](https://github.com/CramBL/hdf5-pure/issues/391)). **Breaking:** the three `create_*` methods return a handle instead of `()`, and a handle kept alive holds the file's exclusive lock like any other.

### Added

- A group or dataset staged by `Group::create_group`, `create_group_with` or `create_dataset` is addressable by name in the same session, so a nested schema is built and its handles cached without a `commit` in between. Such a handle stages further edits and answers a staged dataset's shape, maxshape, datatype and filters; anything that reads bytes reports the new `Error::NotCommitted` until the commit, and a handle whose staging `Group::delete` later withdraws reports the new `Error::StagingWithdrawn` rather than answering for another object ([#392](https://github.com/CramBL/hdf5-pure/issues/392)).
- `Dataset::append_staged` on a dataset staged in the same session folds the elements into the pending creation, so it needs neither an unlimited dimension nor a chunked layout, and `Group::delete` of an object staged in the same session withdraws the staging rather than staging a deletion the commit would refuse. Creating over a name the file already holds, or one this session already staged a creation at, is refused at the call — `delete` first to make it a replacement, or to withdraw the staging — so a returned handle never answers for another object. Staging the same group twice stays allowed and hands back another handle onto that one group ([#392](https://github.com/CramBL/hdf5-pure/issues/392)).

### Changed

- **Breaking:** `Group::create_group`, `create_group_with` and `create_dataset` return a handle to the object they stage instead of `()`; a handle kept alive holds the file's exclusive lock like any other, so drop it before reopening the file ([#392](https://github.com/CramBL/hdf5-pure/issues/392)).

### Fixed

- `FileBuilder::with_libver_bounds`, `FileCreateProperties::with_libver_bounds` and `FileAccessProperties::with_libver_bounds` accept a lower bound of `LibVer::V112`, `V114` or `LATEST` and write the 1.10 format, matching `H5Pset_libver_bounds`, where those bounds were refused as unsatisfiable ([#390](https://github.com/CramBL/hdf5-pure/issues/390)).
- `FileAccessProperties::with_page_buffer_size` accepts any budget of at least the file's page size, where it refused one below 1 MiB. A smaller buffer holds less resident memory in exchange for more writes on long contiguous runs; `0` still turns it off ([#391](https://github.com/CramBL/hdf5-pure/issues/391)).
- A `FileSpaceStrategy::Page` file stops growing under delete-and-recreate churn: deleting a group of empty resizable datasets, and repeatedly flushing `Dataset::append_staged`, now return the chunk indexes involved instead of stranding one per cycle. Space whose page type cannot be established is still held back until the whole page around it is free, so `space_accounting().reusable_free_bytes` can lag a delete by up to a page ([#388](https://github.com/CramBL/hdf5-pure/issues/388)).
- `Dataset::append` and `append_raw` grow a filtered dataset whose length is not a whole multiple of its chunk length, re-encoding the trailing chunk into a fresh allocation, so a `BufferedAppender` no longer pays a commit per unaligned flush or excludes staged edits while it owes one. The SWMR writer still refuses filtered appends ([#393](https://github.com/CramBL/hdf5-pure/issues/393)).
- `Dataset::append` and `Dataset::append_staged` refuse to grow a partial trailing chunk under a **lossy** filter pipeline (ZFP, or float D-scale scale-offset), where re-encoding it would change values that are already committed, and `BufferedAppender::flush` will not leave such a dataset off a chunk boundary; such a dataset takes whole-chunk appends from a chunk-aligned length ([#393](https://github.com/CramBL/hdf5-pure/issues/393), [#407](https://github.com/CramBL/hdf5-pure/issues/407)).
- `Dataset::append` and `BufferedAppender` reuse space an earlier commit freed on a paged file and on a file that persists its free-space managers, where they always extended end-of-file; the session takes that space out of the on-disk managers before writing into it, so a crash can strand it but never hand it out twice. Holes smaller than a megabyte are left alone, and the SWMR writer still appends at end-of-file ([#387](https://github.com/CramBL/hdf5-pure/issues/387)).

## [0.42.0] - 2026-08-29

Reading a file no longer needs a filesystem path. `File::from_source` and `File::from_source_with_options` open a file for streaming reads over anything implementing `Source`, exported now along with `ReadSeekSource` — an object store addressed by range request, a WebAssembly guest handed byte ranges by its host, a decrypting layer over a file — with the same on-demand metadata and chunk reads `File::open_streaming` gives a path, so peak memory tracks what a read touches rather than the size of the file ([#27](https://github.com/CramBL/hdf5-pure/issues/27)). A file the superblock marks as held by a writer is refused here as it is by the path opens, naming the recovery a caller without a path can actually reach, and a source that answers a read short is refused rather than followed into a parser. Additive minor bump.

### Added

- `File::from_source` and `File::from_source_with_options` open a file for streaming reads from any `Source` — an object store addressed by range request, a WebAssembly guest handed byte ranges by its host — with the same on-demand metadata and chunk reads `File::open_streaming` gives a path. `Source` and `ReadSeekSource` are exported to implement and to reuse; a file the superblock marks as held by a writer is refused here too, and a source that answers a read short is refused rather than followed into a parser ([#27](https://github.com/CramBL/hdf5-pure/issues/27)).

## [0.41.0] - 2026-08-27

Variable-length string attributes and oversized attribute sets are supported. `AttrValue::VarLenString` and its three variants write the standard variable-length string datatype (`H5T_STRING` with `STRSIZE = H5T_VARIABLE`), matching h5py and `libhdf5` ([#383](https://github.com/CramBL/hdf5-pure/issues/383)). `Group::set_attr` and `Dataset::set_attr` write attribute sets exceeding object header capacity into a fractal heap on `commit` and rebuild objects already storing dense attributes ([#102](https://github.com/CramBL/hdf5-pure/issues/102)). `Superblock`, `MessageType`, and `BaseAddress` are exported from the crate root, enabling direct access to `File::superblock().base_address` as a numeric value ([#323](https://github.com/CramBL/hdf5-pure/issues/323)). **Breaking:** Variable-length string attributes decode as `VarLenString` variants, `AttrValue::VarLenAsciiArray` is renamed `VarLenAsciiCharArray`, and `Superblock::serialize` is internal.

### Added

- `AttrValue::VarLenString`, `VarLenStringArray`, `VarLenAsciiString`, and `VarLenAsciiStringArray` write attributes in the standard variable-length string datatype (`H5T_STRING` with `STRSIZE = H5T_VARIABLE`), matching h5py and `libhdf5`. `VarLenAsciiCharArray` writes MATLAB's sequence-of-one-byte-strings shape ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).
- `Superblock`, `MessageType`, and `BaseAddress` are exported from the crate root, so types returned by `File::superblock` and `Error::MissingMessage` can be referenced in type signatures and stored. Both `Superblock` and `MessageType` are `#[non_exhaustive]` to accommodate future format extensions ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- `Group::set_attr` and `Dataset::set_attr` write attribute sets exceeding object header capacity - more than eight attributes or messages overflowing the 2-byte size field - into a fractal heap on `commit`, and rebuild objects already storing dense attributes. Datasets and groups created in place support dense attribute sets. Moving an attribute holding a repointable object reference out of the header is rejected to maintain reference repointing during commits. Replaced heaps are retained for `repack` reclamation, and dense (fractal-heap) _link_ storage is rejected ([#102](https://github.com/CramBL/hdf5-pure/issues/102)).

### Fixed

- `File::superblock().base_address` provides numeric value access via `BaseAddress::get` ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- `FileBuilder::set_attr_committed` and related `DatasetBuilder` and `GroupBuilder` methods stage variable-length string attributes into the file's global heap, writing valid string addresses readable by `libhdf5` ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).

### Changed

- **Breaking:** `Superblock::serialize` and associated `parse` constructors are crate-internal ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- **Breaking:** Variable-length string attributes decode as `VarLenString` variants, preserving the on-disk datatype across round-trip writes ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).
- **Breaking:** `AttrValue::VarLenAsciiArray` is renamed `VarLenAsciiCharArray` with `type_name` `"vlen_ascii_char[]"`. It writes MATLAB's VLEN sequence of one-byte strings, while `VarLenAsciiStringArray` writes variable-length ASCII strings ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).

## [0.40.0] - 2026-08-26

A read-write session gathers small writes into one per dirty page. A commit of eight staged dataset creations issues 4 write operations. `FileAccessProperties::with_page_buffer_size` adds an opt-in write-back page buffer for long sessions on paged or unpaged files, whereas `H5Pset_page_buffer_size` requires a paged file ([#288](https://github.com/CramBL/hdf5-pure/issues/288), [#308](https://github.com/CramBL/hdf5-pure/issues/308), [#357](https://github.com/CramBL/hdf5-pure/issues/357)). Free-space management reuses space freed by earlier commits across paged files, persisted free-space managers, `Dataset::append`, `BufferedAppender`, and variable-length overwrites ([#286](https://github.com/CramBL/hdf5-pure/issues/286), [#321](https://github.com/CramBL/hdf5-pure/issues/321), [#349](https://github.com/CramBL/hdf5-pure/issues/349), [#358](https://github.com/CramBL/hdf5-pure/issues/358)). Typed whole-dataset readers decode a single row window at a time ([#289](https://github.com/CramBL/hdf5-pure/issues/289)). Increasing a `MetadataCacheConfig` budget maintains read performance by indexing entries ([#367](https://github.com/CramBL/hdf5-pure/issues/367)). `File::metadata_cache_stats` and `Dataset::chunk_cache_stats` expose cache performance statistics ([#353](https://github.com/CramBL/hdf5-pure/issues/353), [#356](https://github.com/CramBL/hdf5-pure/issues/356)). A `File::open_rw` commit performs atomic object replacements in a single commit ([#305](https://github.com/CramBL/hdf5-pure/issues/305)). Datasets are validated when staged. A rejected commit preserves the staged set and restores pre-existing values ([#316](https://github.com/CramBL/hdf5-pure/issues/316), [#344](https://github.com/CramBL/hdf5-pure/issues/344)). Object references are screened against deleted or relocated objects ([#314](https://github.com/CramBL/hdf5-pure/issues/314), [#317](https://github.com/CramBL/hdf5-pure/issues/317), [#318](https://github.com/CramBL/hdf5-pure/issues/318), [#324](https://github.com/CramBL/hdf5-pure/issues/324)). `DatasetBuilder::with_ascii_strings` writes fixed-width string datasets ([#355](https://github.com/CramBL/hdf5-pure/issues/355)). `Dataset::write_staged` overwrites variable-length string datasets on any layout ([#321](https://github.com/CramBL/hdf5-pure/issues/321)). `MatError::from_source` carries embedded error types through MAT builder closures ([#378](https://github.com/CramBL/hdf5-pure/pull/378)). **Breaking:** A `Dataset` or `Group` handle remains valid across commits by re-resolving its path ([#351](https://github.com/CramBL/hdf5-pure/issues/351)). `File::group`, `Group::named_datatype`, and path resolution reject targets pointing to invalid object types with `Error::NotAGroup` and `Error::NotANamedDatatype`, matching `H5Gopen` and `H5Topen` ([#352](https://github.com/CramBL/hdf5-pure/issues/352), [#364](https://github.com/CramBL/hdf5-pure/issues/364), [#365](https://github.com/CramBL/hdf5-pure/issues/365)). Attributes maintain stored widths, adding 8-, 16-, and 32-bit integer variants, `F32`, and sized fixed-width string variants to `AttrValue` ([#350](https://github.com/CramBL/hdf5-pure/issues/350), [#354](https://github.com/CramBL/hdf5-pure/issues/354), [#359](https://github.com/CramBL/hdf5-pure/issues/359)). Invalid file structures are rejected: numeric elements wider than 8 bytes ([#361](https://github.com/CramBL/hdf5-pure/issues/361)), Extensible or Fixed Array indices with mismatched checksums ([#312](https://github.com/CramBL/hdf5-pure/issues/312)), and datasets with external storage ([#293](https://github.com/CramBL/hdf5-pure/issues/293), [#331](https://github.com/CramBL/hdf5-pure/issues/331), [#336](https://github.com/CramBL/hdf5-pure/issues/336)). Interrupted in-place appends publish values and checksums atomically in one write ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).

### Added

- A `File::open_rw` commit accepts a delete and a create at the same path, completing object replacement in a single commit and linearization point. A dataset can replace a group or vice versa. Replacing a group discards its subtree. Staged edits to an object undergoing replacement are rejected ([#305](https://github.com/CramBL/hdf5-pure/issues/305)).
- `Dataset::write_staged` overwrites variable-length string datasets configured with `with_vlen_strings` across contiguous, compact, and chunked (including filtered) layouts. Configurations using `with_path_references` are rejected ([#321](https://github.com/CramBL/hdf5-pure/issues/321)).
- Overwriting a variable-length dataset reclaims the global heap collection placed by the preceding overwrite in the current session. String rotations within a session reuse global heap allocations. Reclaim applies to collections placed by the active session ([#321](https://github.com/CramBL/hdf5-pure/issues/321)).
- `File::metadata_cache_stats` reports `MetadataCacheConfig` performance metrics including hit rate, evictions, and occupancy. `File::reset_metadata_cache_stats` resets these counters without evicting cached entries. The adaptive-resize policy of `H5AC_cache_config_t` is not modeled ([#353](https://github.com/CramBL/hdf5-pure/issues/353)).
- `DatasetBuilder::with_ascii_strings` and `with_strings` write fixed-width string datasets, sizing the datatype to the longest value and padding shorter entries. `with_ascii_strings_sized` and `with_strings_sized` set an explicit width to accommodate future values. Values exceeding the declared width trigger `FormatError::FixedStringTooLong` ([#355](https://github.com/CramBL/hdf5-pure/issues/355)).
- `Dataset::chunk_cache_stats` reports `ChunkCacheConfig` metrics including hit rate, rejections, evictions, and invalidations. `Dataset::reset_chunk_cache_stats` clears these counters without dropping cached chunks. Full dataset reads update `rejections()`, while row-window reads update `evictions()`. `H5Pset_cache`'s `rdcc_w0` parameter is not modeled ([#356](https://github.com/CramBL/hdf5-pure/issues/356)).
- `FileAccessProperties::with_page_buffer_size` configures a write-back page buffer for read-write sessions, operating as the `H5Pset_page_buffer_size` equivalent on paged or unpaged files. Executing 32 chunk appends and a commit performs 5 write operations. The session marks the superblock during execution. If interrupted during a flush, both this crate and `libhdf5` reject the resulting file ([#308](https://github.com/CramBL/hdf5-pure/issues/308), [#357](https://github.com/CramBL/hdf5-pure/issues/357)).
- `MatError::from_source` carries custom error types through builder closures and `DataProducer` as `MatError::Source`. Embedding crates recover the inner error via `Error::source` and `downcast_ref` ([#378](https://github.com/CramBL/hdf5-pure/pull/378)).

### Changed

- Read-write sessions gather write operations into one per dirty page. A commit of eight staged dataset creations on a paged file issues 4 writes. On-disk byte layouts and error recovery states remain identical. The SWMR writer does not gather writes ([#288](https://github.com/CramBL/hdf5-pure/issues/288)).
- Typed whole-dataset readers (`Dataset::read_f64` and related methods) decode data in row windows. Peak memory usage equals the returned dataset plus approximately 1 MiB. Reading an 8 MiB `f64` dataset allocates 10.1 MiB peak memory ([#289](https://github.com/CramBL/hdf5-pure/issues/289)).
- Steady-state in-place appends in unbuffered sessions and SWMR operations execute 5 write operations per append. Each checksummed structure publishes in a single write ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).
- Row-window reads extract target chunks from the cached chunk index. Windowed reads incur a constant memory allocation per chunk ([#289](https://github.com/CramBL/hdf5-pure/issues/289)).
- **Breaking:** Value overwrites specifying chunking, filters, extensible shapes, attributes, or fill values are rejected by `Dataset::write` and `Dataset::write_staged` at the time the write is staged ([#318](https://github.com/CramBL/hdf5-pure/issues/318)).
- **Breaking:** Datasets staged in `File::open_rw` sessions are validated upon staging. Invalid shapes, missing datatypes, or unsupported write options report errors immediately from `Group::create_dataset`, `Dataset::write_staged`, or `create_group_with` closures ([#316](https://github.com/CramBL/hdf5-pure/issues/316)).

### Fixed

- **Breaking:** Numeric elements wider than 8 bytes are rejected with `FormatError::NumericElementTooWide`. This rule applies to typed numeric readers for datasets and attributes. Rejected attributes are omitted from `attrs()` but remain visible via `attr_datatypes()`. `read_raw` remains unchanged ([#361](https://github.com/CramBL/hdf5-pure/issues/361)).
- Metadata cache lookups use index structures. Increasing `MetadataCacheConfig` memory limits maintains constant-time read performance regardless of cached entry counts ([#367](https://github.com/CramBL/hdf5-pure/issues/367)).
- **Breaking:** `Dataset` and `Group` handles remain valid following a `File::commit` by resolving their objects by path. Reads using handles for deleted objects fail as path lookup errors. Handles acquired via `Dataset::dereference` return `Error::StaleHandle` after commit operations execute. Both handle types implement `Clone` ([#351](https://github.com/CramBL/hdf5-pure/issues/351)).
- **Breaking:** `File::group` and `Group::group` reject paths targeting non-group objects with `Error::NotAGroup`, matching `H5Gopen`. Replacing a group with a dataset leaves handles reporting `Error::NotAGroup` ([#352](https://github.com/CramBL/hdf5-pure/issues/352)).
- **Breaking:** `Group::named_datatype` and `Group::named_datatype_references` reject paths targeting non-committed-datatype objects with `Error::NotANamedDatatype`, matching `H5Topen` ([#364](https://github.com/CramBL/hdf5-pure/issues/364)).
- **Breaking:** Path resolution through a non-group object returns `Error::NotAGroup` identifying the non-group path segment. For example, `File::group("a/b/c")` stopped by a dataset at `a/b` reports `a/b`. Unresolved components return `FormatError::PathNotFound` ([#365](https://github.com/CramBL/hdf5-pure/issues/365)).
- **Breaking:** Integer attributes preserve their stored bit widths. `AttrValue` adds 8-, 16-, and 32-bit integer and array variants. A 16-bit attribute decodes as `AttrValue::I16` and serializes in two bytes. Bit widths lacking direct Rust integer counterparts convert to 64-bit representations ([#350](https://github.com/CramBL/hdf5-pure/issues/350)).
- **Breaking:** Floating-point attributes preserve their stored bit widths via `AttrValue::F32` and `F32Array`. A 4-byte attribute decodes as `F32`. `as_f64` and `to_f64s` convert both 32-bit and 64-bit float attributes ([#354](https://github.com/CramBL/hdf5-pure/issues/354)).
- **Breaking:** Fixed-width string attributes preserve stored slot widths. Padded attribute slots decode as `AttrValue::AsciiStringSized`. Constructors like `AttrValue::ascii_string_sized` set explicit slot sizes (`H5T_C_S1` with `H5Tset_size(N)`). Slots sized exactly to string content decode as `AsciiString` ([#359](https://github.com/CramBL/hdf5-pure/issues/359)).
- Invalid link targets in files with userblocks are rejected. Address calculations relative to the superblock base address use checked arithmetic, reporting overflow or invalid bounds as `FormatError::OffsetOverflow` or `FormatError::AddressBelowBase` ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- Files persisting free space reuse space freed by earlier commits for extension and free-space-manager metadata blocks during commits. `FileSpaceStrategy::FsmAggr` files reuse freed allocations ([#358](https://github.com/CramBL/hdf5-pure/issues/358)).
- Repeated reads of chunked datasets exceeding cache capacity retain active chunks. Reads preserve both served and placed chunks in cache ([#356](https://github.com/CramBL/hdf5-pure/issues/356)).
- Sequential whole-dataset reads of memory-backed chunked datasets iterate chunks in file offset order ([#356](https://github.com/CramBL/hdf5-pure/issues/356)).
- `Dataset::append` and `BufferedAppender` allocate chunks and index blocks from free space freed by preceding commits. Files with persisted free-space managers append at end-of-file to maintain free-space record consistency ([#349](https://github.com/CramBL/hdf5-pure/issues/349)).
- A `File::commit` rejected prior to publication restores overwritten data values, preserving pre-commit dataset states. If restoration fails, the operation returns `Error::CommitPartiallyApplied` ([#344](https://github.com/CramBL/hdf5-pure/issues/344)).
- `File::create_with_options` rejects userblocks combined with `FileSpaceStrategy::Page`. Non-zero base address files do not persist free space, preventing paged files without free space from opening in read-write mode ([#308](https://github.com/CramBL/hdf5-pure/issues/308)).
- `DatasetBuilder` clears previous staging state when supplied with new element data, preventing address patching conflicts between sequential data calls ([#321](https://github.com/CramBL/hdf5-pure/issues/321)).
- `repack` preserves the filter pipeline order and filter optional flags from source datasets ([#333](https://github.com/CramBL/hdf5-pure/issues/333)).
- `File::copy` and `File::copy_from` copy datasets with unallocated storage by creating matching unallocated storage in the target file, aligning with `repack` ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).
- `File::copy` and `File::copy_from` reject datasets stored in external files (`H5Pset_external`) by name. Deleting an external dataset retains its object header as dead space ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).
- **Breaking:** `FileBuilder` rejects datasets staging element data with zero-element shapes. Staging zero data elements for zero-element shapes remains valid ([#332](https://github.com/CramBL/hdf5-pure/issues/332)).
- **Breaking:** Reading datasets stored in external files (`H5Pset_external`) returns `FormatError::UnsupportedExternalStorage`. Shape, datatype, and layout metadata remain readable ([#331](https://github.com/CramBL/hdf5-pure/issues/331)).
- **Breaking:** Overwriting or appending to datasets with external storage is rejected. Attribute edits remain supported. Replacing an external dataset via deletion and recreation in a single commit is supported ([#331](https://github.com/CramBL/hdf5-pure/issues/331)).
- **Breaking:** `repack` rejects datasets stored in external files (`H5Pset_external`) ([#293](https://github.com/CramBL/hdf5-pure/issues/293)).
- `repack` preserves unallocated datasets as unallocated storage in the target file. Resizable destination datasets maintain chunk index structures ([#293](https://github.com/CramBL/hdf5-pure/issues/293)).
- Editing a group referenced by multiple hard links is rejected, matching the rule for relocating dataset writes ([#327](https://github.com/CramBL/hdf5-pure/issues/327)).
- Object references update target addresses when a `File::open_rw` commit relocates the referenced object header. Unreachable addresses in chunked dataset data, dense attributes, or variable-length references remain unmodified ([#324](https://github.com/CramBL/hdf5-pure/issues/324)).
- `Dataset::write_staged` rejects builders configured with `with_path_references`. Overwriting with `with_reference_data` remains supported ([#318](https://github.com/CramBL/hdf5-pure/issues/318)).
- A rejected `File::commit` preserves the staged edit batch. Staging calls that fail stage no partial edits, including `create_group_with` closures ([#316](https://github.com/CramBL/hdf5-pure/issues/316)).
- Object references pointing to children of groups deleted in the same commit are rejected ([#314](https://github.com/CramBL/hdf5-pure/issues/314)).
- **Breaking:** Object references supplied as raw addresses (`with_reference_data`, `with_raw_data`, or `File::copy`) are screened at `commit` against deleted objects. Supplied addresses referring to headers relocated by the commit are rejected. Datatypes with unparseable references (widths over 8 bytes, dataset-region references, variable-length references, or complex compounds) are rejected ([#317](https://github.com/CramBL/hdf5-pure/issues/317)).
- `File::open_rw` commits adding object-reference datasets that target paths created in the same commit execute cleanly on files with userblocks ([#317](https://github.com/CramBL/hdf5-pure/issues/317)).
- **Breaking:** Extensible-Array and Fixed-Array chunk index checksums are validated, matching `libhdf5`. Indices with mismatched checksums are rejected during reads and free-space reclamation, returning `ChecksumMismatch` ([#312](https://github.com/CramBL/hdf5-pure/issues/312)).
- In-place appends to datasets with Extensible Array headers specifying element widths larger than contained fields are rejected ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).
- In-place appends publish data values and covering checksums in a single atomic write operation ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).
- Paged files (`FileSpaceStrategy::Page`) allocate updated free-space managers in existing free space. Unallocated pages can be reassigned between page types upon release ([#286](https://github.com/CramBL/hdf5-pure/issues/286)).

## [0.39.0] - 2026-08-17

Unallocated storage evaluates to the dataset's fill value, aligning with `libhdf5`. Unallocated chunks in partly written datasets also return the fill value ([#284](https://github.com/CramBL/hdf5-pure/issues/284)). A `File::open_rw` session can create empty chunked and extensible datasets, enabling schema-first writers to declare resizable datasets up front and grow them with `Dataset::append_staged`. Chunked datasets with `maxshape` exceeding shape in dimensions past the first write chunk-index slot numbering aligned with `libhdf5` layout requirements. An Extensible Array allocates only the data blocks containing chunks, so a 256-chunk dataset with `maxshape [unlimited, 65536]` writes 376 KB ([#299](https://github.com/CramBL/hdf5-pure/issues/299)). A chunk covering the dataset's edge pads with the fill value ([#296](https://github.com/CramBL/hdf5-pure/issues/296)). Scale-offset records the dataset's fill value in its filter parameters, matching `libhdf5`. `DatasetBuilder::with_fill_value` configures the encoder, `Dataset::append_staged` grows scale-offset datasets written by `libhdf5` or h5py, and `repack` re-encodes scale-offset data ([#287](https://github.com/CramBL/hdf5-pure/issues/287), [#297](https://github.com/CramBL/hdf5-pure/issues/297), [#300](https://github.com/CramBL/hdf5-pure/issues/300)). Opening a dataset or group by path isolates child lookups, with a lookup in a 1,024-child group allocating 23 KiB across 16 blocks ([#228](https://github.com/CramBL/hdf5-pure/issues/228)). **Breaking:** `FormatError::NoDataAllocated` is removed because unallocated storage evaluates successfully.

### Added

- A `File::open_rw` session can create an empty (zero-element) chunked or extensible dataset, so a schema-first writer declares its resizable datasets up front and grows them with `Dataset::append_staged`. Explicit `with_chunks` dimensions are required ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- `FormatError::UnreadableFillValue` reports a Fill Value message this parser cannot read on an unallocated dataset requiring that message. A dataset with fully allocated storage reads normally ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).

### Changed

- Opening a dataset or group by path (`File::dataset`, `File::group`) or by name (`Group::dataset`, `Group::group`) performs targeted lookups. A single lookup in a 1,024-child group allocates 23 KiB in 16 blocks. Opening each member of a large group incurs the group lookup cost once per open. A link with a malformed target does not affect lookups for different names. Listing the group continues to report the link ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- `DatasetBuilder::with_vlen_strings` stages strings directly without creating per-element copies. Writing 32,768 strings allocates 6.6 MiB across 104 blocks ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- `Dataset::append` reserves memory for its batch up front. Performing 512 appends of a 4 KiB chunk allocates 6.7 MiB across 14,469 blocks ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- A scale-offset dataset records its fill value in the filter's parameters, matching `libhdf5`. `DatasetBuilder::with_fill_value` configures the encoder to store elements equal to the fill value as a reserved code. A dataset without an explicit fill value records the default value of zero, matching `libhdf5`. In lossy float D-scale mode, elements within one decimal quantum of the recorded fill value decode as that fill value ([#297](https://github.com/CramBL/hdf5-pure/issues/297)).

### Removed

- **Breaking:** `FormatError::NoDataAllocated` is removed. Unallocated storage evaluates successfully ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).

### Fixed

- `repack` re-encodes scale-offset datasets whose filters record a fill value. This includes datasets created by `libhdf5` or h5py, as `libhdf5` records a fill value on all scale-offset datasets. Lossy float mode remains rejected ([#297](https://github.com/CramBL/hdf5-pure/issues/297)).
- Scale-offset rounds scaled residuals using the `llround` implementation from `libhdf5` ([#300](https://github.com/CramBL/hdf5-pure/issues/300)).
- A chunked dataset whose `maxshape` exceeds its shape in any dimension past the first writes with the chunk-index slot numbering required by `libhdf5`. Index slots are numbered over the dataset's maximum chunk grid, and Extensible Array unlimited dimensions rotate to the front ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A page of an Extensible Array data block holding no chunks is recorded as present, ensuring chunks in subsequent pages are read ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- Both chunked readers step over uninitialized pages in an Extensible Array data block. Files with pages written out of order by `libhdf5` read completely ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunked dataset specifying a maximum extent of zero alongside a non-zero current extent is rejected ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunked dataset storing one chunk with a `maxshape` permitting growth uses a Fixed Array layout ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A `maxshape` with multiple unlimited dimensions is rejected. Version-2 B-tree chunk indices are not supported for writing ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- An Extensible Array allocates only the data blocks containing chunks, matching `libhdf5`. A 256-chunk dataset with `maxshape [unlimited, 65536]` writes 376 KB ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A `maxshape` whose chunk index would allocate over 32 MiB for unallocated chunk descriptors is rejected. Memory limits evaluate in bytes so filtered indices with wider elements receive an identical memory allocation ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunk indexed beyond the 8,589,934,580 element slots addressable by an Extensible Array is rejected ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunk covering a dataset edge fills padding slots past that edge with the dataset fill value. A Fill Value message that cannot be parsed rejects the write ([#296](https://github.com/CramBL/hdf5-pure/issues/296)).
- Reading a dataset with unallocated storage returns its fill value for contiguous and chunked layouts across whole reads and row windows. Unallocated chunks in partly written datasets evaluate to the fill value. Datasets with Fill Value messages setting `H5D_FILL_TIME_NEVER` return zeros. A `repack` operation on an unwritten dataset writes fill values ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- An empty **filtered** chunked dataset configures its chunk-index element width to support compressed chunks ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- An empty **fixed-shape** chunked dataset writes with no chunk index, matching `libhdf5` ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- `Dataset::append_staged` grows scale-offset datasets written by `libhdf5` or h5py. Repacking a scale-offset dataset with a defined fill value remains rejected ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- A scale-offset chunk whose values span too much of the datatype to pack writes the header format matching `libhdf5` ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- `ScaleOffset::Integer(n)` with `n` equal to the datatype's bit width stores the chunk unfiltered, matching `libhdf5`. Specifying `n` larger than the datatype is rejected on read and write paths ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- A scale-offset chunk whose header declares zero bits per element while its filter declares a fill value reads as that fill value, matching `libhdf5` ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- Requesting an extensible zero-element dataset without `with_chunks` is rejected with `FormatError::InvalidChunkGeometry` ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).

## [0.38.0] - 2026-08-16

Reading and writing cost substantially less memory. A chunked read allocates about half as much, a variable-length read is roughly an order of magnitude faster over a large heap collection and allocates forty times less often, a deflated write allocates thirty-five times less, and writing a chunked dataset allocates a quarter as often for about 25% fewer bytes: the zlib codec is built once per call rather than once per chunk, sizing a structure no longer builds it once to measure and again to keep, and a whole read fills the chunk cache instead of evicting its own chunks ([#228](https://github.com/CramBL/hdf5-pure/issues/228), [#265](https://github.com/CramBL/hdf5-pure/issues/265), [#275](https://github.com/CramBL/hdf5-pure/issues/275)). Two test binaries hold those figures in place: `tests/allocation_bounds.rs` states the scaling rules and runs in every configuration, and `tests/allocation_baseline.rs` pins the exact numbers on one platform behind the new `heap-baseline` feature. On the writing side, `Dataset::buffered_appender` holds appended elements in memory and writes them a whole chunk at a time, so a filtered dataset takes an append of any length ([#262](https://github.com/CramBL/hdf5-pure/issues/262)); `FileAccessProperties::with_sync_policy(SyncPolicy::OnClose)` drops the `fsync` from every commit and append in favour of one at `close`, with the new `File::sync` for checkpoints in between ([#263](https://github.com/CramBL/hdf5-pure/issues/263)); and a `File::open_rw` commit reuses freed space for chunk data, chunk indexes and dense attribute heaps where it always appended, paged files included ([#261](https://github.com/CramBL/hdf5-pure/issues/261)). Four kinds of malformed file that used to panic or decode as valid are now refused: a datatype declaring a zero-byte element size ([#268](https://github.com/CramBL/hdf5-pure/issues/268)), an Extensible Array element too narrow to hold the address it must contain, a truncated Extensible Array index block ([#278](https://github.com/CramBL/hdf5-pure/pull/278)), and a deflated chunk whose zlib stream ends before its checksum, which read as valid data whenever it happened to decode to the expected length ([#228](https://github.com/CramBL/hdf5-pure/issues/228)). **Breaking:** the `parallel` cargo feature and its `rayon` dependency are gone — the feature gated a chunked-read path no public entry point reached, so no read changes behavior — along with seven names deprecated in 0.26.0 and 0.28.0 and five error variants that were never constructed ([#280](https://github.com/CramBL/hdf5-pure/pull/280)); `CompoundTypeBuilder::build` returns a `Result`, and `FormatError::ShapeDataMismatch` carries its element size as a `NonZeroUsize`. Files written by earlier versions still read.

### Added

- The `heap-baseline` cargo feature enables a maintainer-only test that checks this crate's recorded allocation figures; it is not a run-time dependency ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- `Dataset::buffered_appender` returns a `BufferedAppender` that holds appended elements in memory and writes them a whole chunk at a time, so a filtered dataset takes any append length; buffered elements reach the file only on `flush`, `finish`, or `discard`, and a SWMR session is refused ([#262](https://github.com/CramBL/hdf5-pure/issues/262)).
- `FileAccessProperties::with_sync_policy(SyncPolicy::OnClose)` drops the `fsync` from every commit and append, leaving one at `close`, with the new `File::sync` for checkpoints in between; writes still reach the operating system as they are made, so what moves to the caller is power-loss durability within the session ([#263](https://github.com/CramBL/hdf5-pure/issues/263)).
- A staged edit that would stop a live `BufferedAppender` from flushing — one naming its dataset or an ancestor, any edit at all while it still owes a realignment, or a second appender on the same dataset — is refused with `Error::EditUnsupported` at the call that makes it, rather than losing the buffered elements when the appender drops ([#262](https://github.com/CramBL/hdf5-pure/issues/262)).
- `Datatype::element_size` returns the element width as a `NonZeroU32`, refusing a zero-width type; prefer it to `type_size` wherever the width is about to be divided by ([#272](https://github.com/CramBL/hdf5-pure/pull/272)).

### Changed

- **Breaking:** `CompoundTypeBuilder::build` returns `Result<Datatype, FormatError>`, refusing a compound of no fields and one whose fields pack to zero bytes, the way `ExplicitCompoundTypeBuilder::build` already did ([#268](https://github.com/CramBL/hdf5-pure/issues/268)).
- **Breaking:** `FormatError::ShapeDataMismatch`'s `element_size` field is a `NonZeroUsize`, so the element counts its message reports are well defined by type rather than by convention ([#272](https://github.com/CramBL/hdf5-pure/pull/272)).
- `Dataset::append` accepts a filtered append of any length, where it required a whole number of chunks; the dataset's own length must still be chunk-aligned ([#262](https://github.com/CramBL/hdf5-pure/issues/262)).
- Writing a chunked dataset, appending to one, and writing a dense attribute heap no longer build a structure twice to measure it, so high chunk counts and large attribute sets cost less to write ([#265](https://github.com/CramBL/hdf5-pure/issues/265), [#275](https://github.com/CramBL/hdf5-pure/issues/275)).
- `Dataset::read_raw` and the typed whole-dataset reads allocate about half as much on a chunked dataset. A whole read now fills the chunk cache and stops rather than evicting its own chunks, so it retains the chunks it reached first where it used to retain the last ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- `Dataset::read_string_rows` and the other variable-length reads are roughly an order of magnitude faster on a large heap collection, and allocate about forty times less often; a collection of uniformly small objects moves more transient bytes in exchange, and one of mixed sizes moves no more than before ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- `FileBuilder::write` allocates about thirty-five times less on a deflated dataset, and `Dataset::read_raw` about ten times less reading one back: the zlib codec is built once per call rather than once per chunk. Output is byte-identical ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- Writing a chunked dataset allocates a quarter as often and about 25% fewer bytes: sizing a dataset's object header no longer builds the whole data region only to discard it, and the chunk splitter keeps its scratch across chunks ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).

### Removed

- **Breaking:** The `parallel` cargo feature and its `rayon` dependency are gone. The feature gated a chunked-read path no public entry point reached, so no read changes behavior; the `parallel_read` and `lane_partition` modules go with it ([#280](https://github.com/CramBL/hdf5-pure/pull/280)).
- **Breaking:** `FileAccessOptions`, `DatasetAccessOptions`, `FileCreateOptions`, `FileBuilder::with_create_options`, and `File::access_options` are gone; use the `Properties` spellings they were deprecated for in 0.26.0 ([#280](https://github.com/CramBL/hdf5-pure/pull/280)).
- **Breaking:** `File::open_rw_bounded` and `File::open_rw_bounded_with_options` are gone; pass `MemoryStrategy::Bounded` to `File::open_rw_with_options` for the same strict refusal ([#280](https://github.com/CramBL/hdf5-pure/pull/280)).
- **Breaking:** `FormatError::CompressionError`, `FormatError::ChunkAssemblyError`, `FormatError::DuplicateDatasetName`, `Error::AlignmentError`, and `MatError::RaggedMatrix` are gone. The first was constructed at two sites that now report `FormatError::FilterError`, the variant every other filter already used; the rest were never constructed ([#280](https://github.com/CramBL/hdf5-pure/pull/280)).

### Fixed

- An Extensible Array header naming a filtered element too narrow to hold the address and filter mask it must contain is refused; the size arithmetic previously underflowed, panicking under the overflow checks tests and fuzz targets build with ([#278](https://github.com/CramBL/hdf5-pure/pull/278)).
- A truncated Extensible Array index block is refused when the file is read whole, where the reader returned the chunks it had and reported success; reading the same file through `File::open_streaming` already refused it ([#278](https://github.com/CramBL/hdf5-pure/pull/278)).
- A deflated chunk whose zlib stream ends without reaching its checksum is refused rather than decoded. Such a chunk read as valid data whenever it happened to decode to the expected length, since the adler32 that would have caught it was never reached ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- A datatype declaring a zero-byte element size is refused with the new `FormatError::ZeroSizedDatatype` when its message is parsed; reading such a dataset previously panicked on a division by that size ([#268](https://github.com/CramBL/hdf5-pure/issues/268)).
- Writing a dataset or committed datatype whose element size is zero is refused with the same error, on both the whole-file and `File::open_rw` paths; a chunked write of one previously panicked, and a contiguous one produced a file this crate refuses to read ([#268](https://github.com/CramBL/hdf5-pure/issues/268)).
- A `File::open_rw` commit reuses freed space for a chunked dataset's chunk data and index, and for a dense attribute heap, where both were always appended at the end of the file; a replacement needs one free region large enough to hold it whole ([#261](https://github.com/CramBL/hdf5-pure/issues/261)).
- A paged file (`FileSpaceStrategy::Page`) reuses its freed space too, drawing only from the page type being written so metadata and raw data cannot come to share a page; it previously appended for every allocation. Free space another writer recorded whose page type cannot be established is kept but never reused ([#261](https://github.com/CramBL/hdf5-pure/issues/261)).
- A commit that fails before its superblock repoint returns the free regions it had drawn from, instead of leaking them for the rest of the session ([#261](https://github.com/CramBL/hdf5-pure/issues/261)).

## [0.36.0] - 2026-08-13

Writing complex data to a `.mat` file gets substantially faster. `mat::complex::i16_array`, and one helper per component class, write a large complex array in bulk from a `#[serde(serialize_with = ...)]` field — roughly twenty-five times faster than the per-element path for the same bytes — and an ordinary complex write, `MatBuilder`'s writers included, is about five times faster than before ([#260](https://github.com/CramBL/hdf5-pure/pull/260)). The bulk helpers accept anything implementing `mat::ComplexElement`, a new unsafe layout trait implemented for this crate's `Complex*` types and, under the new `num-complex` feature, for `num_complex::Complex<T>`. Reading a group's members now costs one walk rather than one per member: `Group::iter_datasets` and `Group::iter_groups` yield opened handles paired with their names, where opening each name from `datasets()` re-walks the group every time ([#259](https://github.com/CramBL/hdf5-pure/pull/259)). Additive minor bump.

### Added

- `mat::complex::i16_array`, and one helper per component class, write a large complex array in bulk from a `#[serde(serialize_with = ...)]` field, roughly twenty-five times faster than the per-element path for the same bytes; an empty slice keeps its component class, where a plain `Vec` writes an empty `double` ([#260](https://github.com/CramBL/hdf5-pure/pull/260)).
- `mat::ComplexElement`, the unsafe layout trait those helpers accept, is implemented for the `Complex*` types and — under the new `num-complex` feature — for `num_complex::Complex<T>` ([#260](https://github.com/CramBL/hdf5-pure/pull/260)).
- `Group::iter_datasets` and `Group::iter_groups` yield a group's members as opened handles paired with their names, walking the group once where opening each name from `datasets()` re-walks it per member ([#259](https://github.com/CramBL/hdf5-pure/pull/259)).

### Changed

- Writing a complex array is about five times faster, `MatBuilder`'s writers included ([#260](https://github.com/CramBL/hdf5-pure/pull/260)).

## [0.35.0] - 2026-08-10

A MAT cell array takes its shape and its metadata from the same rules as every other value this crate writes. An empty cell array is `0x0`, MATLAB's own `{}`, where it was `0x1`, and a cell array follows `mat::Options::one_dimensional_mode` like every other 1-D value, so `RowVector` writes `1xN` where a cell used to be a column whatever the option asked for; both are reachable only under non-default options, and `isempty` held under either empty shape, so a reader that only tested emptiness is unaffected. Every object a MAT write interns under `#refs#` — cell elements, struct elements, the MCOS subsystem's helpers — now carries the `H5PATH` attribute MATLAB writes on all but one of its own, which this crate wrote on none ([#258](https://github.com/CramBL/hdf5-pure/pull/258)). Files written by earlier versions still read.

### Changed

- **Breaking:** an empty cell array is written `0x0` rather than `0x1`, matching MATLAB's own `{}` and every other empty this crate writes. Reachable only under `EmptySequencePolicy::Cell` ([#258](https://github.com/CramBL/hdf5-pure/pull/258)).
- **Breaking:** a cell array takes the orientation `mat::Options::one_dimensional_mode` asks for, as every other 1-D value already did, so `RowVector` writes `1xN` where it used to write `Nx1` ([#258](https://github.com/CramBL/hdf5-pure/pull/258)).

### Fixed

- Every object a `.mat` write interns under `#refs#` — cell elements, struct elements, the MCOS subsystem's helpers — carries the `H5PATH` attribute MATLAB writes on all but one of its own; this crate wrote it on none ([#258](https://github.com/CramBL/hdf5-pure/pull/258)).

## [0.34.0] - 2026-08-08

A `.mat` file written by this crate opens under MATLAB's `load`. MATLAB's MAT reader rejects a version-3 superblock, so `mat::Options` defaults to the HDF5 1.8 format. Set `mat::Options::libver` to `LibVer::V110` for the 1.10 format required by compression. Format selection is available across the API. `FileBuilder::with_libver_bounds`, `FileAccessProperties::with_libver_bounds`, and `RepackOptions::with_libver_bounds` configure output formats for builds, edit sessions, and repack operations. `FormatError::LibverTooOldForContent` reports content that a `LibVer` bound cannot express. When specified without bounds, `repack` preserves the source file's format. Two smaller breaking changes are included. `LibVer::WRITER_OUTPUT` is renamed to `LibVer::WRITER_DEFAULT`. An empty MAT value is written with `EmptyMarkerEncoding::DataAsDims`, matching MATLAB and `matio`, so `isempty` evaluates to true on read ([#247](https://github.com/CramBL/hdf5-pure/pull/247)). Committed (`H5Tcommit`) datatypes are supported for reading and writing. `FileBuilder::commit_datatype` writes named type objects. `DatasetBuilder::with_committed_datatype` and the `set_attr_committed` methods reference them. `Group::named_datatypes` lists committed types. Datasets and attributes referencing a committed datatype resolve directly to that type, matching the layout netCDF-4 and h5py write for user-defined types ([#254](https://github.com/CramBL/hdf5-pure/issues/254)). Attribute handling preserves on-disk representations. `repack` preserves each attribute's datatype and shape ([#241](https://github.com/CramBL/hdf5-pure/issues/241)). Enumeration attributes decode for the caller, preserving h5py `np.bool_` attributes ([#248](https://github.com/CramBL/hdf5-pure/pull/248)). `Group::attr_datatypes` and `Dataset::attr_datatypes` expose the on-disk `Datatype` ([#253](https://github.com/CramBL/hdf5-pure/pull/253)). `File::open_streaming` fetches adjacent chunks in a single read, optimizing row-by-row chunked access ([#250](https://github.com/CramBL/hdf5-pure/pull/250)). Files written by earlier versions read successfully.

### Added

- `FileBuilder::with_libver_bounds` configures the output on-disk format. An upper bound of `LibVer::V18` writes the HDF5 1.8 format, while bounds reaching 1.10 write the HDF5 1.10 format ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FormatError::LibverTooOldForContent` reports content that a `LibVer` bound cannot express, including chunked, filtered, or resizable datasets and file-space settings ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileAccessProperties::with_libver_bounds` constrains an editing session to a format. `File::open_rw` rejects modifications that require a newer library version ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `RepackOptions::with_libver_bounds` explicitly sets the repack output format. Without this option, `repack` preserves the source file's format, upgrading only when required by content ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `LibVer::WRITER_OLDEST` names the oldest format the writer produces ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `mat::Options::libver` sets the newest HDF5 format a `.mat` file may use, defaulting to `LibVer::V18`. `mat::MatError::CompressionNeedsNewerFormat` reports incompatible settings ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileBuilder::commit_datatype` and `GroupBuilder::commit_datatype` write a committed (`H5Tcommit`) datatype, creating a named type object that multiple objects can share. `DatasetBuilder::with_committed_datatype` and the `set_attr_committed` methods reference named types. An uncommitted name or a type mismatch fails the operation ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `Group::named_datatypes`, `Group::named_datatype`, and `Group::named_datatype_references` read committed datatype objects. These objects do not appear in `datasets()` or `groups()` ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `Group::attr_datatypes` and `Dataset::attr_datatypes` return an attribute's on-disk `Datatype`. This includes the stored width and the `enum[FALSE, TRUE]` that designates an h5py `np.bool_` as boolean. Every attribute is reported, including attributes lacking an `AttrValue` ([#253](https://github.com/CramBL/hdf5-pure/pull/253)).

### Changed

- **Breaking:** MAT files write in the HDF5 1.8 format by default to allow MATLAB `load` compatibility. MATLAB releases before R2021b used HDF5 1.8.12 and do not open version-3 superblocks. Set `mat::Options::libver` to `LibVer::V110` for compressed files ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- **Breaking:** `mat::Options::default` uses `EmptyMarkerEncoding::DataAsDims`, matching what MATLAB and `matio` write. Empty values read back as empty under `isempty` ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- **Breaking:** `LibVer::WRITER_OUTPUT` is renamed `LibVer::WRITER_DEFAULT`. Its value remains unchanged ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileBuilder::with_create_properties` resets properties omitted from its argument. A bound set prior to the call does not affect a file whose property list specifies no version ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileBuilder::write` creates the destination file after data serialization finishes. A rejected build leaves any existing file at that path untouched ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- An edit session writes a contiguous dataset's data-layout message matching the format of the opened file. A `.mat` file edited through `File::open_rw` remains readable by MATLAB ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `File::open_streaming` fetches adjacent dataset chunks in a single read. This optimizes read performance for files written row-by-row with small chunk sizes. A read fetches target chunks and caps buffer usage at 256 KiB ([#250](https://github.com/CramBL/hdf5-pure/pull/250)).

### Fixed

- An object header holding compact attributes declares its attribute count. `H5Oget_info().num_attrs` matches attribute iteration, preserving `MATLAB_*` attributes during `h5repack` round trips ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- Files with userblocks containing object-header continuation blocks or dense link storage read successfully ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- An empty MAT value encodes `MATLAB_class` and `MATLAB_empty`. Both emitters output consistent dimension metadata, including for empty `Matrix` instances ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `repack` preserves each attribute's datatype and shape. Enumeration, compound, bit-field, and opaque attributes are copied during repack operations. Reference attributes are rejected ([#241](https://github.com/CramBL/hdf5-pure/issues/241)).
- An enumeration attribute decodes through its integer base type and returns to the caller, preserving `np.bool_` attributes from h5py files ([#248](https://github.com/CramBL/hdf5-pure/pull/248)).
- A committed (`H5Tcommit`) datatype resolves to the named type. `Dataset::datatype`, `Dataset::read_*`, `attrs()`, and `attr_datatypes()` return the target named type ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `repack` recreates committed datatypes and maintains references from datasets and attributes to the shared named object. Dropping a referenced committed datatype is rejected ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `FormatError::UnsupportedSohmReference` rejects messages stored in the shared-message (SOHM) heap ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).

## [0.33.0] - 2026-08-02

A file whose superblock marks it as held by a writer is rejected. `File::open`, `open_streaming`, `open_rw`, `open_swmr_writer`, and `repack` report the new `Error::FileMarkedInUse`. This matches the check `H5Fopen` makes of the same byte. `File::open_swmr` successfully follows a flagged file. `File::from_bytes` ignores the status byte, allowing a caller holding the bytes to read a flagged file on a read-only mount where the `File::clear_swmr_flag` recovery command lacks write access. The check applies to version-3 superblocks, matching `libhdf5`. `open_swmr_writer` requires a version-3 superblock. A read-write open validates the superblock before building its backing. Rejecting a mirrored file requires a few bounded reads. A version-1 superblock's status flags and chunk B-tree K load from the exact offsets written by `libhdf5`. Files written by earlier versions read successfully.

### Added

- `Error::FileMarkedInUse` reports an open rejected by the superblock's status-flags byte. This status flag outlives the process that set it. It indicates an active writer or a writer that exited without closing the file ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).

### Changed

- **Breaking:** `File::open`, `File::open_streaming`, `File::open_rw`, `File::open_swmr_writer`, and `repack` reject a file whose superblock marks it as held by a writer, matching `H5Fopen`. `File::open_swmr` successfully follows such a file. `File::from_bytes` skips the check. `File::clear_swmr_flag` recovers a stale flag ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).
- **Breaking:** `File::open_swmr_writer` requires a version-3 superblock, aligning with `libhdf5` ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).
- A read-write open validates the superblock before allocating the file backing. `File::open_rw` checks the superblock through the handle and builds its backing only when the file passes validation. Rejecting a mirrored file requires only a few bounded reads ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).

### Fixed

- A version-1 superblock's status flags and chunk B-tree K map to the exact offsets written by `libhdf5`. `File::superblock()` accurately assigns a v1 file's `indexed_storage_internal_node_k` and `consistency_flags` ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).

## [0.32.0] - 2026-07-31

`Display` now covers the types that describe what a file holds — `AttrValue`, `Datatype` and its component enums, `MessageType`, `Layout`, `ChunkIndex` and `Filter` — so a message quoting one reads as HDF5 rather than as a Rust value ([#242](https://github.com/CramBL/hdf5-pure/pull/242)). A name the file records is escaped and truncated wherever it is written, and a member list elided past sixteen. `DType::Other` carries the `Datatype` itself, which is the only view a caller gets of a type nested in a compound field or an array base ([#244](https://github.com/CramBL/hdf5-pure/pull/244)).

### Added

- `Display` for `AttrValue`, `Datatype`, `DatatypeByteOrder`, `StringPadding`, `CharacterSet`, `ReferenceType`, `MessageType`, `Layout`, `ChunkIndex` and `Filter`. `AttrValue` writes the value — `1.5`, `"metres"`, `[1, 2, 3]` — and elides an array past eight elements, reporting how many it dropped ([#242](https://github.com/CramBL/hdf5-pure/pull/242)).
- `AttrValue::type_name` gives the name of the type a value holds, such as `f64` or `ascii_string[]`. It names every variant, so a caller that matched on this `#[non_exhaustive]` enum and reached its `_` arm can still report what it received ([#242](https://github.com/CramBL/hdf5-pure/pull/242)).

### Changed

- **Breaking:** `DType::Other` carries the `Datatype` rather than a string describing it, so a type nested in a compound field or an array base can be matched on. It writes as `other(opaque[3] "rgb")` ([#244](https://github.com/CramBL/hdf5-pure/pull/244)).
- **Breaking:** `DType::Array` writes its shape as `array<f32, 2x3>` rather than `array<f32, [2, 3]>` ([#242](https://github.com/CramBL/hdf5-pure/pull/242)).
- **Breaking:** a name a file records — a compound member's, an enum label's, a filter's — is escaped and truncated wherever it is written, and a member list is elided past sixteen ([#242](https://github.com/CramBL/hdf5-pure/pull/242)).
- **Breaking:** `Error::MissingMessage` names the message — `missing required message: data layout` — instead of its Rust variant, and the unrecognized-chunk-index error reports the raw index-type byte ([#242](https://github.com/CramBL/hdf5-pure/pull/242)).

## [0.31.0] - 2026-07-30

An attribute now reads back as the `AttrValue` variant it was written from: the dataspace kind decides scalar against array, so a one-element array stays an array, and the charset selects the `Ascii` variants, so `MATLAB_class` reads as `AsciiString` and `MATLAB_fields` as `VarLenAsciiArray` ([#239](https://github.com/CramBL/hdf5-pure/pull/239)). That fidelity means several variants can carry one logical value, so read through the new accessors — `AttrValue::as_str`, `as_strings`, `as_i64`, `as_u64`, `as_f64`, `to_i64s`, `to_u64s`, `to_f64s` — each of which spans every variant that can hold the shape it names and applies its range rule per element ([#238](https://github.com/CramBL/hdf5-pure/pull/238)). Two data-correctness fixes come with it: an unsigned array reads as the new `AttrValue::U64Array` rather than an `I64Array` of reinterpreted bits, so a value above `i64::MAX` no longer reads back negative, and `repack` stops re-encoding the attributes it copies — a fixed-width ASCII string used to come out UTF-8 and a variable-length array fixed-width, which is the encoding MATLAB and matio require. Separately, an attribute holding an empty string is written with a one-byte-wide string datatype instead of a zero-size one, which libhdf5 rejects while iterating an object's attributes: a single empty-string attribute made every attribute on that object unreadable to the C library ([#240](https://github.com/CramBL/hdf5-pure/pull/240)). Widths, true variable-length strings, dataspace rank and fixed-string padding are still not recovered on read, and are tracked in [#241](https://github.com/CramBL/hdf5-pure/issues/241). Files written by earlier versions still read.

### Added

- `AttrValue::as_str`, `as_strings`, `as_i64`, `as_u64`, `as_f64`, `to_i64s`, `to_u64s` and `to_f64s` read an attribute value without matching on its variant. Each spans every variant that can carry the shape asked for — both string charsets and all four integer widths, scalar or one-element array — and applies its range rule per element, so a value that does not fit reports `None` rather than a wrapped number. The prefix states the cost: `as_*` borrows or copies, `to_*` allocates ([#238](https://github.com/CramBL/hdf5-pure/pull/238)).
- `AttrValue::U64Array` writes an unsigned 64-bit array attribute, which `I64Array` could not represent above `i64::MAX` ([#238](https://github.com/CramBL/hdf5-pure/pull/238)).

### Changed

- **Breaking:** an attribute reads back as the `AttrValue` variant it was written from: the dataspace kind decides scalar against array, so a one-element array stays an array, and the charset selects the `Ascii` variants. `MATLAB_class` reads as `AsciiString` rather than `String` and `MATLAB_fields` as `VarLenAsciiArray` rather than `StringArray`. Read values through `AttrValue::as_str`/`as_strings`/`as_i64`/`to_i64s`, which span every shape. Widths are still widened, and a true variable-length string reads as the fixed-width variant of its charset ([#239](https://github.com/CramBL/hdf5-pure/pull/239)).
- **Breaking:** an unsigned integer array reads as `AttrValue::U64Array` instead of an `I64Array` holding reinterpreted bits, so a value above `i64::MAX` no longer reads back negative ([#239](https://github.com/CramBL/hdf5-pure/pull/239)).

### Fixed

- An attribute holding an empty string is written with a one-byte-wide string datatype rather than a zero-size one, which libhdf5 rejects while iterating an object's attributes — a single empty-string attribute made every attribute on that object unreadable to the C library ([#240](https://github.com/CramBL/hdf5-pure/pull/240)).
- `repack` no longer re-encodes an attribute it copies: a variable-length ASCII array stays variable-length, and a fixed-width ASCII string keeps its charset, where both previously came out as UTF-8 fixed-width ([#239](https://github.com/CramBL/hdf5-pure/pull/239)).
- A MAT file honors `MATLAB_class` and `MATLAB_empty` whichever integer width, charset, or one-element shape its writer chose, rather than reporting an unexpected attribute type or reading the flag as absent ([#238](https://github.com/CramBL/hdf5-pure/pull/238), [#239](https://github.com/CramBL/hdf5-pure/pull/239)).

## [0.30.0] - 2026-07-30

`DatasetBuilder::with_lzf` writes h5py's LZF filter (id 32000), a fast lossless compressor h5py reads without any plugin installed. LZF datasets - including h5py-written ones - can be read, edited in place, and repacked ([#231](https://github.com/CramBL/hdf5-pure/pull/231)). The MAT serde writer honors `Options::null_policy`. `None`, `()`, a unit struct, and `Value::Null` write MATLAB `struct([])`. MATLAB code can reference it unconditionally. A Rust reader relying on `#[serde(default)]` for a non-`Option` field finds the field present. `NullPolicy::Omit` omits the field entirely. Two new options are added. `Options::unit_variant_encoding` writes a fieldless enum variant as its name or as its declaration index. `Options::empty_sequence_policy` selects `[]` or `{}` for a sequence that evaluates to empty ([#232](https://github.com/CramBL/hdf5-pure/pull/232)). The writer collects a flat numeric or complex sequence packed, one element wide. Serializing a `Vec<f64>` or `Vec<ComplexI16>` allocates approximately 1x its own size in memory. The empty-value paths for both emitters produce identical byte output. Requesting a filter that another would displace is rejected. Failing streams return `FilterError`. Decoding a chunk limits memory reservation to the stream's theoretical maximum expansion bounds ([#233](https://github.com/CramBL/hdf5-pure/issues/233)). Files written by earlier versions read successfully.

### Added

- `DatasetBuilder::with_lzf` writes h5py's LZF filter (id 32000), a fast lossless compressor h5py reads without any plugin. LZF datasets - including h5py-written ones - can be read, edited in place, and repacked. Combining LZF with deflate is rejected ([#231](https://github.com/CramBL/hdf5-pure/pull/231)).
- `Options::unit_variant_encoding` selects whether a fieldless enum variant is written as its name (`UnitVariantEncoding::Name`, the default) or as its declaration index in a `uint32` (`UnitVariantEncoding::Index`). Serde provides both to the serializer, making either reachable. `Index` outputs a numeric value for a reader configured to decode an integer. An index cannot be interpreted without the schema that fixes the ordering. `Name` serves as the safe default. The index originates from serde, counted from zero. An explicit discriminant (`enum E { A = 5 }`) does not dictate the serialized value.
- `Options::empty_sequence_policy` selects the MATLAB class of an empty sequence. `EmptySequencePolicy::DoubleArray` (the default) writes `[]`. `Cell` writes `{}`. An empty `serialize_seq` carries no element type. `Cell` applies correctly when the sequence would have held structs.
- `NullPolicy::Omit` omits the field entirely.

### Changed

- **Breaking:** Requesting a filter that another would displace is rejected. Calling `with_zfp` alongside `with_shuffle`, `with_deflate`, or `with_lzf`, or calling `with_scale_offset` alongside `with_shuffle`, fails with a filter error naming both conflicting filters ([#233](https://github.com/CramBL/hdf5-pure/issues/233)).
- **Breaking:** `FormatError::DecompressionError` is removed. A deflate stream that fails to decode reports `FormatError::FilterError`. Shuffle, scale-offset, and LZF share this error variant, allowing a single match arm for all decode failures ([#233](https://github.com/CramBL/hdf5-pure/issues/233)).
- The serde writer collects a flat numeric or complex sequence packed, one element wide. Serializing a `Vec<f64>` or `Vec<ComplexI16>` allocates approximately 1x its own size in memory. A sequence whose elements do not all agree falls back to the per-element form at the point of divergence. Cell arrays and matrices built from equal-length rows serialize reliably. This optimization produces the identical byte output.
- The serde writer assigns an empty cell array the MATLAB shape `[0, 1]`. This applies the `[n, 1]` rule consistently for both empty and non-empty arrays, matching the conversion of an empty BEVE array. The `isempty` condition evaluates to true. The result of `size(x, 2)` remains stable.
- **Breaking:** The serde writer honors `Options::null_policy`. `None`, `()`, a unit struct, and `serde_json::Value::Null` write MATLAB `struct([])` by default. The `isfield` function reports true. MATLAB code can reference the field unconditionally with `isempty(fieldnames(x))`. Set `NullPolicy::Omit` to drop the field entirely.

    For the reader, a populated field does not require `#[serde(default)]`. A reader configured with `#[serde(default)]` for a non-`Option` field fails on these files. The field contains a struct value, preventing the default from triggering. `struct([])` deserializes into `Option<T>`, `Vec<T>`, `serde_json::Value`, and `()`. It yields a type error for a bare scalar, `String`, struct, or map. Define such a field as `Option<T>`, or serialize using `NullPolicy::Omit`.

### Fixed

- Decoding a chunk bounds its memory reservation to the stream's theoretical maximum expansion. Deflate and LZF reserve at most what their own stream can expand to. Scale-offset rejects a chunk claiming more bytes than the chunk physically holds ([#233](https://github.com/CramBL/hdf5-pure/issues/233)).
- An empty `Matrix<T>` serializes properly without triggering `EmptySequencePolicy::Cell` internally. The policy application is isolated from the `Matrix` wrapper type's internal `data` field lowering. This prevents an empty numeric or complex matrix from erroneously converting to a cell array and failing type recovery. A caller's explicitly defined empty sequence continues to honor the policy.
- An `Options` struct persisted by an earlier version deserializes correctly. The two new fields define a serde default directly on the struct. This design ensures missing fields map safely to their defaults during load.
- `NullPolicy::Error` rejects a null at the file root. The root serializer routes through the standard lowering pipeline and reports the designated error. The root namespace lacks a slot name. A null there represents an empty variable _namespace_. The `Omit` and `DoubleArray` policies write a valid file with no variables, producing identical bytes to an empty root map or a fieldless struct. Such a file reads back as a struct, not as `None`, because the deserializer treats the root as a struct.
- A fieldless enum variant reads back from either encoding. The deserializer resolves a name or an index. It processes indices from any numeric class, supporting an index typed natively in MATLAB (a `double`).
- An empty marker produces a consistent element type across serde emitters. Both `to_bytes` and `to_bytes_with_options` write a `uint64` zero-element dataset for `struct([])`, matching the empty marker convention of `libhdf5`. The encoding logic evaluates in a single shared path. An empty cell array written through `MatBuilder` under `EmptyMarkerEncoding::ZeroElement` outputs `uint64`. The class attributes identifying the array remain standard, and `libhdf5` reads the result successfully.

## [0.29.0] - 2026-07-28

Dense attributes take on the reference library's geometry: both indexes are multi-level B-trees of 512-byte nodes, and the heap is a doubling table of direct blocks reached through indirect blocks, so a large attribute set grows by adding blocks rather than rounding one up to a power of two. The two attribute-count ceilings go with it, along with the errors that reported them, and the remaining heap-size error now bounds the heap's address space rather than one direct block — the release's only breaking changes ([#195](https://github.com/CramBL/hdf5-pure/issues/195)). MAT v7.3 files no longer have to be held in memory to be written: `MatBuilder::finish_to` assembles onto any `io::Write`, `MatBuilder::write_blocks` stages a numeric array whose bytes a `DataProducer` supplies one block at a time, `mat::to_file` streams rather than buffering, and `FileBuilder::with_userblock_content` keeps a wrapper format's header reachable on those paths ([#226](https://github.com/CramBL/hdf5-pure/pull/226)). Chunked datasets are written back to back instead of padded to the host's cache line, which aligned nothing measurable and made the same dataset larger on `aarch64` than on `x86_64` ([#227](https://github.com/CramBL/hdf5-pure/issues/227)). One soundness fix: a MATLAB matrix shape whose `rows * cols` wraps `usize` is refused at every entry point, where the wrapped product could previously match a short data vector and the writer's transpose then wrote past its allocation ([#230](https://github.com/CramBL/hdf5-pure/pull/230)). Two dense attributes whose names hash alike are also now indexed in the order the reference library searches, which every earlier version got wrong ([#225](https://github.com/CramBL/hdf5-pure/issues/225)). Files written by earlier versions still read.

### Added

- An object carries any number of dense (fractal-heap) attributes: both the name index and the huge-object index are now multi-level B-trees of fixed 512-byte nodes, matching what the reference C library emits, instead of one leaf grown to fit ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- Dense attributes are held in a doubling table of direct blocks reached through indirect blocks, the same heap geometry the reference C library uses, so a large attribute set no longer rounds its storage up to a power of two ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- `MatBuilder::finish_to` assembles a `.mat` onto any `io::Write` (`MatBuilder::write` onto a path), as do `mat::to_writer` / `mat::to_writer_with_options` for the serde entry points. Byte-for-byte what the buffered calls produce, on a sink that need not be seekable ([#226](https://github.com/CramBL/hdf5-pure/pull/226)).
- `MatBuilder::write_blocks` stages a numeric array whose bytes a `DataProducer` supplies one `Block` at a time during the write, so a dataset larger than memory can be written. Uncompressed only, since the layout needs the region's exact size before it writes anything ([#226](https://github.com/CramBL/hdf5-pure/pull/226)).
- `FileBuilder::with_userblock_content` makes the userblock part of the file the writer emits, so a wrapper format's header survives the streaming output paths that leave nothing to patch afterwards ([#226](https://github.com/CramBL/hdf5-pure/pull/226)).

### Changed

- `FileBuilder::with_userblock` now refuses a size the format does not define — it must be zero or a power of two of at least 512 — with the new `FormatError::InvalidUserblockSize`, instead of writing a file whose superblock nothing can find ([#226](https://github.com/CramBL/hdf5-pure/pull/226)).
- **Breaking:** `FormatError::TooManyDenseAttributes` and `FormatError::TooManyHugeDenseAttributes` are removed along with the 61,680- and 43,690-attribute limits that produced them ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- **Breaking:** `FormatError::DenseAttributeHeapTooLarge` now carries only `limit`, and bounds the heap's 40-bit address space rather than a single 2 GiB direct block ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- Every dense attribute set has different bytes: its name index is a tree of 512-byte nodes, and its attributes sit in a doubling table whose blocks start at 1 KiB and grow by adding blocks rather than by rounding one up to a power of two. Files written by earlier versions still read ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- `mat::to_file` and `mat::to_file_with_options` stream to disk rather than building the whole file in memory first. Same bytes ([#226](https://github.com/CramBL/hdf5-pure/pull/226)).
- Chunks are written back to back instead of padded to the host's cache line, so a chunked dataset no longer occupies more space on `aarch64` than on `x86_64`, and chunk placement no longer varies by target. Files written by earlier versions still read ([#227](https://github.com/CramBL/hdf5-pure/issues/227)).

### Fixed

- A MATLAB matrix shape whose `rows * cols` wraps `usize` is refused everywhere it can enter: `Matrix::from_row_major` and `Matrix::zeros` panic, while the serde and file-reading paths return an error. Previously the wrapped product could match a short data vector, and the writer's transpose then wrote past its allocation ([#230](https://github.com/CramBL/hdf5-pure/pull/230)).
- Two dense attributes whose names hash alike are indexed in name order rather than insertion order, so the reference C library can open both by name; written the other way round one of the pair was unfindable by name, while iteration still reported both. A file written by 0.28.0 or earlier is corrected by `repack` ([#225](https://github.com/CramBL/hdf5-pure/issues/225)).

## [0.28.0] - 2026-07-28

`File::open_rw` picks its own backing. A latest-format file with no userblock is edited in bounded memory rather than through a whole-file mirror, and the mirror is now the fallback for the files the bounded engine cannot edit rather than the default for everything. Nothing about a file's space strategy decides which open a caller reaches for any more, so `File::open_rw_bounded` is deprecated: it survives only as the strict default, now expressible as `MemoryStrategy::Bounded` on `FileAccessProperties`, and `File::edit_backing` reports which backend an open actually resolved to. Two guarantees that used to be silently ignored are refused before any work happens: the SWMR writer will not accept a `Bounded` it cannot honor, and `File::create_with_options` checks a creation/access pair up front rather than leaving a file on disk and returning the reopen's error.

### Added

- `FileAccessProperties::with_memory_strategy` and `MemoryStrategy` say how much memory a read-write open may spend holding the file: `Bounded` refuses a file the bounded engine cannot edit rather than mirroring it, `Auto` falls back to the mirror, and `Mirrored` always mirrors. `File::edit_backing` reports which backend an open resolved to, as an `EditBacking` ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).

### Changed

- `File::open_rw` now edits a latest-format file with no userblock in **bounded memory** instead of building a whole-file mirror, and falls back to the mirror only for a file the bounded engine cannot edit. Nothing about a file's space strategy decides which open a caller reaches for any more. A large `Dataset::append` is applied in whole-chunk batches on the bounded backing, so a crash mid-call leaves a valid shorter dataset; pass `MemoryStrategy::Mirrored` for the previous unconditional mirror ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- `File::open_rw` refuses a paged file without persisted free space at open rather than at commit, since neither backing can edit one. This includes a paged file with a userblock, whose free-space managers go unseeded for the same reason ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- `File::open_rw` now applies `FileAccessProperties::with_metadata_cache`, which the whole-file mirror ignored, and its reads are served from the file rather than from a snapshot taken at open — visible only to a session sharing a file with a lock-free writer ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- `File::open_swmr_writer_with_options` refuses an explicit `MemoryStrategy::Bounded` instead of silently mirroring; this writer always mirrors, and `Auto` or unset is satisfied by that ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- `File::create_with_options` refuses a creation/access pair it could not then reopen — a paged file with `persist = false`, or a userblock under `MemoryStrategy::Bounded` — before writing anything, instead of leaving a file on disk and returning an error ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- A userblock that is not a whole number of file-space pages now reports `FormatError::UserblockNotPageAligned` naming both sizes, rather than an `InvalidFileSpacePageSize` that called a valid page size invalid ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).

### Deprecated

- `File::open_rw_bounded` and `open_rw_bounded_with_options`: `File::open_rw` now picks the bounded engine on its own, so these survive only as the strict `MemoryStrategy::Bounded` default. Pass that strategy to `File::open_rw_with_options` to keep the refusal ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).

## [0.27.0] - 2026-07-27

The two read-write engines converge. `File::open_rw` now commits staged edits to a genuine paged file, and `File::open_rw_bounded` offers the full staged edit surface — `Dataset::write`, attribute edits, `create_*`/`delete`, `copy`, `space_accounting` — while holding only what a commit is building rather than a whole-file mirror. Neither the file's internal space strategy nor the kind of edit being made decides which open a caller reaches for, and `Dataset::append` grows a free-space-persisting file from either one. `Error::BoundedStagedUnsupported` is gone along with the refusals that returned it, the single breaking change here. Separately, MAT v7.3 complex arrays gain integer components across the serde, `Matrix<T>`, and `MatBuilder` surfaces, so a capture that samples as 16-bit integer pairs stores four bytes per sample instead of eight; three defects on the complex path are fixed with it, one of which changes the stored shape of a 1-D complex array written through `to_bytes_with_options`.

### Added

- `File::open_rw` commits staged edits to a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`), through a commit that keeps each page homogeneous and rewrites the per-page-type free-space managers, so the full edit surface is no longer limited to `File::open_rw_bounded`'s appends. A paged file is still refused unless it persists its free space and has no userblock ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- `File::open_rw_bounded` offers the full staged edit surface — `Dataset::write`, attribute edits, `create_*`/`delete`, `copy`, `commit`, `space_accounting` — at bounded memory: a commit holds only what it is building rather than a whole-file mirror. It still requires a latest-format file with 8-byte offsets and no userblock ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- `Dataset::append` grows a file that persists its free space, including a paged one, from `File::open_rw` as well as `File::open_rw_bounded`; the on-disk free-space managers are re-homed when the file is closed ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- A `Dataset` reached by object reference can append on either read-write open, not only `File::open_rw_bounded`. It is refused once the session stages or commits an edit, because a commit can move the object header the handle names ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- MAT v7.3 complex arrays with integer components: `mat::ComplexI8`/`I16`/`I32`/`I64` and the `ComplexU*` counterparts join `Complex64`/`Complex32` across the serde, `Matrix<T>`, and `MatBuilder::write_complex_*` surfaces, so a capture that samples as 16-bit integer pairs stores four bytes per sample instead of eight. Components are never converted between widths: an `int16` complex dataset deserializes into `ComplexI16` and nothing else, in either direction.

### Changed

- Dropping a read-write `File` without `close` now re-homes the on-disk free-space managers of a persisting file and flushes, matching what `close` does; previously only `File::open_rw_bounded` handles did this. Staged edits are still discarded on drop ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- **Breaking:** `Error::BoundedStagedUnsupported` is removed, along with the refusals that returned it ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).
- A MAT complex vector written through `to_bytes_with_options` now takes the configured `OneDimensionalMode` like every other 1-D array; it was always a MATLAB row vector before, so existing callers of that path get columns under the default and their stored shape changes.
- A MAT complex dataset whose `MATLAB_class` disagrees with its `{real, imag}` compound is refused instead of decoded, including a `{imag, real}` compound that used to read back swapped.

### Fixed

- A one-element MAT complex array deserializes into a `Vec<Complex*>`, matching the allowance the real numeric path already makes for a one-element numeric array.
- An empty MAT complex array of an integer class reads back as an empty complex array of that class rather than as an untyped empty vector.

## [0.26.0] - 2026-07-27

Attributes have no size ceiling. One too large for an object-header message selects fractal-heap storage on its own. One too large even for a managed heap object becomes a _huge_ object. `FormatError::DenseAttributeTooLarge` is removed, allowing shapes that `libhdf5` writes ([#195](https://github.com/CramBL/hdf5-pure/issues/195)). This release fixes three defects on that path involving variable-length attributes in a fractal heap, dense attributes in a file with a userblock, and quadratic read times for large attribute sets ([#214](https://github.com/CramBL/hdf5-pure/pull/214), [#195](https://github.com/CramBL/hdf5-pure/issues/195)). The property-list types are renamed to reflect the HDF5 configuration objects they represent - `FileAccessProperties`, `FileCreateProperties`, and `DatasetAccessProperties`. Checking each one's settings against the official group pages caught two properties documented under the wrong class ([#198](https://github.com/CramBL/hdf5-pure/issues/198)). The release includes breaking changes where every break is a one-line call-site edit. Deprecated aliases keep 0.25.0 code compiling for this cycle.

### Added

- An attribute of any size is written successfully. One too large for an object-header message selects fractal-heap storage on its own. One too large for a managed heap object becomes a _huge_ object. A name, datatype, or dataspace longer than the 2-byte field describing it is rejected with the new `FormatError::AttributeFieldTooLong` ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- `FormatError::TooManyHugeDenseAttributes` defines the bound on how many huge attributes a single object may carry ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- `FormatError::UnexpectedHugeObjectBTree` rejects a fractal heap whose huge-objects B-tree is not the record layout this reader decodes ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).

### Changed

- **Breaking:** `FileAccessOptions`, `FileCreateOptions`, and `DatasetAccessOptions` are renamed `FileAccessProperties`, `FileCreateProperties`, and `DatasetAccessProperties`. A type standing in for a whole HDF5 property list says so in its name. `FileBuilder::with_create_properties` and `File::access_properties` follow suit. Deprecated aliases under the old names keep 0.25.0 code compiling for this cycle. `H5Pset_libver_bounds` is documented as the file-access property it is ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).

### Fixed

- Reading an object with many huge dense attributes parses the heap's huge-object index once per walk. The read time scales linearly with their number (1,600 such attributes load in ~75 ms) ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- A variable-length attribute stored in a fractal heap retains its values across writes and reads ([#214](https://github.com/CramBL/hdf5-pure/pull/214)).
- Dense (fractal-heap) attributes are readable in a file with a userblock through `File::open`, `File::open_streaming`, `repack`, and `copy`. The heap address computes as relative to the base address. This successfully reads files generated by `libhdf5` ([#214](https://github.com/CramBL/hdf5-pure/pull/214)).

### Removed

- **Breaking:** `FormatError::DenseAttributeTooLarge` is removed. The library supports all valid attribute sizes ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).

## [0.25.0] - 2026-07-26

An API-consolidation release. File properties are now reusable values. One `FileAccessOptions` (the `fapl` analogue) carries cache budgets and the locking policy to every open. The new `FileCreateOptions` (the `fcpl` analogue) lets a file layout be defined once and applied to any write, including through `File::create_with_options`. The read-write mirror backend now honors access options. The bounded backend honors the locking policy. Public types the HDF5 format will keep growing are now `#[non_exhaustive]`. A future datatype class, reference kind, or MATLAB class is an additive change. A test guards each seal against silent removal. Enumerations can be built over any integer base type. This release fixes `repack` corruption on datasets with variable-length or reference members. It also fixes a hang when reading a file from inside a builder closure. The release includes a number of breaking changes listed below. Most are one-line call-site edits.

### Added

- `FileCreateOptions` collects the file-creation properties (userblock, file-space strategy and page size, library-version bounds) into one reusable value - the `fcpl` analogue. This is applied with `FileBuilder::with_create_options` or the new `File::create_with_options`. This maps to `H5Fcreate(name, flags, fcpl, fapl)` and provides the first way to reach creation properties from the owned-handle path ([#205](https://github.com/CramBL/hdf5-pure/issues/205)).
- `FileAccessOptions::with_locking` carries the file-locking policy. `File::open_rw_with_options` and `open_swmr_writer_with_options` accept the options, so one `fapl` value serves every open. The mirror backend honors the configured chunk cache. `open_rw_bounded*` honors the locking policy ([#204](https://github.com/CramBL/hdf5-pure/issues/204)).
- `EnumTypeBuilder::with_base` builds an enumeration over any integer base type. The `raw_value` for a member is given as its raw little-endian bytes and `i64_value` is used for wider integers ([#208](https://github.com/CramBL/hdf5-pure/issues/208)).
- Chunked, filtered, and resizable variable-length datasets now write. `DatasetBuilder::with_vlen_strings` accepts `with_chunks`, `with_deflate`, and `with_maxshape`. The `repack` utility reproduces such a dataset with its chunk geometry, filters, and unlimited dimension intact. Adding one to an existing file through the in-place edit engine is rejected ([#109](https://github.com/CramBL/hdf5-pure/issues/109)).
- `Dataset::write_staged` overwrites a dataset through its full `DatasetBuilder`. This is the builder-level counterpart of `Dataset::write` for element kinds that are not `H5Element` ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- `Group::create_group_with` stages a new group configured through a `StagedGroup` closure. A group's attributes and children land with its creation. `set_attr` cannot reach it, since it requires a group that already resolves ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).

### Fixed

- Reading the same `File` from inside a builder closure (`Group::create_dataset`, `create_group_with`, `Dataset::write_staged`, `append_staged`) completes successfully. The closures configure a builder off the session lock, allowing a staged dataset to depend on data already in the file. It sees the file as it was before the call, since staged edits resolve only on `commit` ([#200](https://github.com/CramBL/hdf5-pure/issues/200)).
- `repack` safely copies a dataset whose datatype _contains_ a variable-length member, an object-reference member, or both - such as a compound with a variable-length string field. The process rewrites embedded addresses like top-level ones. `libhdf5` reads the destination file successfully ([#201](https://github.com/CramBL/hdf5-pure/issues/201)).

### Changed

- **Breaking:** `EnumTypeBuilder::build` returns `Result<Datatype, FormatError>`. It rejects a non-integer base type, a member value too wide for the base, and raw bytes whose length disagrees with it ([#208](https://github.com/CramBL/hdf5-pure/issues/208)).
- **Breaking:** `EnumMember` is now `#[non_exhaustive]`. Build enumerations through the builder ([#208](https://github.com/CramBL/hdf5-pure/issues/208)).
- **Breaking:** Three rejections on the owned write path report specific errors. An unaligned SWMR append returns `SwmrAppendUnsupported`. An ineligible immediate append returns `AppendInPlaceUnsupported`. A missing edit target returns `PathNotFound` when the handle is resolved ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- **Breaking:** `AttrValue`, `DType`, `Datatype`, `ReferenceType`, `LibVer`, `Object`, `CompoundMember`, `FileSpaceInfo`, `VerifyResult`, `mat::MatClass`, and the four `mat::opaque` decode structs are now `#[non_exhaustive]`. A new datatype, reference kind, library-version bound, or MATLAB class is an additive change. Add a `_` arm when matching a read-back value. Constructing the existing variants is unaffected, including `Datatype` literals for types this crate has no constructor for ([#206](https://github.com/CramBL/hdf5-pure/pull/206)).
- **Breaking:** `RepackOptions` is built only through `new` and `drop_path`, matching `FileAccessOptions`. The `drop` field is private and readable with `RepackOptions::drop_paths` ([#206](https://github.com/CramBL/hdf5-pure/pull/206)).

### Removed

- **Breaking:** `File::open_rw_with_locking` is removed. Pass the policy on the open via `File::open_rw_with_options(path, FileAccessOptions::new().with_locking(..))` ([#204](https://github.com/CramBL/hdf5-pure/issues/204)).
- **Breaking:** `Datatype::parse` and `Datatype::serialize` are now crate-internal. Read a dataset's type with `Dataset::datatype` and pass a `Datatype` to `DatasetBuilder::with_dtype`, which encodes it. `Datatype::type_size` stays public ([#206](https://github.com/CramBL/hdf5-pure/pull/206)).
- **Breaking:** `FormatError::ChunkedVlenStringUnsupported` is removed. The library supports these datasets ([#109](https://github.com/CramBL/hdf5-pure/issues/109)).
- **Breaking:** `AppendWriter`, `SwmrWriter`, and `EditSession` are removed. Use `File::open_rw` (or `File::open_swmr_writer`) with owned `Dataset` and `Group` handles. `File::open_rw_with_options` combined with `FileAccessOptions::with_locking` replaces `AppendWriter::open_with_locking`. `File::clear_swmr_flag` replaces `SwmrWriter::clear_swmr_flag`. The former `EditSession` methods map to `Dataset::append`/`append_staged`/`write`/`write_staged`/`set_attr`/`remove_attr`, `Group::create_group`/`create_dataset`/`delete`/`set_attr`, and `File::copy`/`copy_from`/`space_accounting`. An object staged in an uncommitted batch is reachable only through `create_group_with` ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).

## [0.24.0] - 2026-07-24

Variable-length writes lose their 65,535-element cap: `DatasetBuilder::with_vlen_strings` and `repack` now split across as many global heap collections as they need, and resolving an element is no longer quadratic ([#189](https://github.com/CramBL/hdf5-pure/issues/189)). Attributes too large for compact or dense storage are now refused by name instead of silently dropped or written unreadable ([#190](https://github.com/CramBL/hdf5-pure/issues/190), [#191](https://github.com/CramBL/hdf5-pure/issues/191)). Reads gain bounds: a chunked dataset declaring an impossible per-chunk size is refused rather than allocated, with the new `Dataset::element_size` to size a read up front ([#185](https://github.com/CramBL/hdf5-pure/issues/185)), and row windows of inner-chunked and variable-length string datasets stream instead of falling back to a whole read ([#183](https://github.com/CramBL/hdf5-pure/pull/183), [#186](https://github.com/CramBL/hdf5-pure/pull/186)). Additive minor bump.

### Added

- `OBJECT_HEADER_MESSAGE_MAX` is the largest message a version 2 object header can describe (65,535 bytes), the bound behind the new oversized-message refusals ([#190](https://github.com/CramBL/hdf5-pure/issues/190)).
- `Dataset::element_size` returns the on-disk byte width of one element (HDF5's `H5Tget_size`), so a caller reading an untrusted file can multiply it by the element count from `shape` to bound a read's allocation before requesting it, rather than trusting the file's declared extent ([#185](https://github.com/CramBL/hdf5-pure/issues/185)).

### Fixed

- Reading a chunked dataset whose datatype or chunk extent declares an impossible per-chunk logical size (over the 4 GiB format limit, e.g. a fixed-length string element of billions of bytes) is now refused with an `InvalidChunkGeometry` error instead of eagerly allocating the whole declared extent, so a crafted file can no longer drive a multi-gigabyte out-of-memory allocation from a few kilobytes ([#185](https://github.com/CramBL/hdf5-pure/issues/185)).
- Variable-length datasets and attributes with more than 65,535 elements now write correctly, split across as many global heap collections as they need, so `DatasetBuilder::with_vlen_strings` and `repack` are no longer capped there ([#189](https://github.com/CramBL/hdf5-pure/issues/189)).
- Resolving a variable-length element now binary-searches its heap collection's directory instead of scanning it, so reading a large variable-length string dataset is no longer quadratic in its element count ([#189](https://github.com/CramBL/hdf5-pure/issues/189)).
- `Dataset::read_raw_rows` and the typed `read_*_rows` now stream a row window of an inner-chunked dataset by decoding only the chunks the window overlaps, instead of falling back to a whole read, so peak memory scales with the window plus one chunk rather than the dataset ([#183](https://github.com/CramBL/hdf5-pure/pull/183)).
- `Dataset::read_string_rows` on variable-length strings now resolves only the window's heap references instead of reading and resolving the whole dataset before slicing, so the row-window memory bound holds for every windowed read: peak allocation is the window's references, its text, and the metadata of the heap collections it touches ([#186](https://github.com/CramBL/hdf5-pure/pull/186)).
- `FileBuilder::write`/`finish` and `repack` now refuse a compact attribute whose object-header message exceeds `OBJECT_HEADER_MESSAGE_MAX` (the new `FormatError::AttributeMessageTooLarge`, naming the attribute) instead of truncating its size field, which silently dropped the attribute or left the file unreadable ([#190](https://github.com/CramBL/hdf5-pure/issues/190)).
- An attribute too large for dense (fractal-heap) storage is now refused with the new `FormatError::DenseAttributeTooLarge` naming it, instead of being written into a heap that read back empty and aborted an assertion-enabled reference C library; more than 61,680 attributes on one object is likewise refused with `FormatError::TooManyDenseAttributes` ([#191](https://github.com/CramBL/hdf5-pure/issues/191)).
- Dense attribute sets whose total passes 64 KiB are no longer refused by `EditSession` copies: the bound now tracks each attribute's size, which is what the emitter is actually limited by, so multi-megabyte sets of individually small attributes are written and copied normally. The dense-copy refusal now reports `FormatError` rather than `Error::EditUnsupported`, so it can name the attribute ([#191](https://github.com/CramBL/hdf5-pure/issues/191)).
- A dense attribute heap larger than 64 KiB now records a maximum direct block size covering the block it actually contains, instead of a fixed 65,536 that its own block exceeded. Such files already read correctly in both libraries, so this changes their bytes without changing their meaning; heaps at or below 64 KiB are byte-for-byte unchanged ([#191](https://github.com/CramBL/hdf5-pure/issues/191)).

## [0.23.2] - 2026-07-23

Two fixes to the windowed row-read API introduced in 0.23.0: a full-range `Dataset::read_raw_rows` / `read_*_rows` window now delegates to the whole read instead of paying a full-size copy on top of it on layouts whose windowed reads fall back to one (inner-chunked storage, variable-length strings), and `Dataset::read_string_rows` now slices a multi-dimensional variable-length string dataset by row rather than by first-dimension index. Non-breaking patch.

### Fixed

- `Dataset::read_raw_rows` and the typed `read_*_rows` now delegate to a whole read when the window covers every row, so full-range windows on layouts whose windowed reads fall back to a whole read (inner-chunked storage, variable-length strings) no longer pay a full-size copy on top of it ([#181](https://github.com/CramBL/hdf5-pure/pull/181)).
- `Dataset::read_string_rows` on a multi-dimensional variable-length string dataset now slices by row — each row spanning its inner dimensions — instead of treating the flat element array as one string per row, so a windowed read returns the same rows as `read_raw_rows` ([#182](https://github.com/CramBL/hdf5-pure/pull/182)).

## [0.23.1] - 2026-07-23

Two file-space fixes result from documenting and fuzz-testing the paged and persisted surface ([#178](https://github.com/CramBL/hdf5-pure/issues/178)). A fresh `persist = true` file with a non-paged strategy now records a defined end-of-allocation. An assertion-enabled build of `libhdf5` opens it successfully. The `File::open_rw_bounded` rejection for a non-persisting paged file now advises the correct recovery. Non-breaking patch.

### Fixed

- A fresh file written with `persist = true` and a non-paged file-space strategy (`FsmAggr`/`Aggr`/`None`) now records a defined end-of-allocation in its File Space Info message. An assertion-enabled build of `libhdf5` opens it successfully. Release builds already tolerated this state ([#178](https://github.com/CramBL/hdf5-pure/issues/178)).
- The rejection when opening a _non-persisting_ paged file with `File::open_rw_bounded` advises recreating the file with `persist = true`, the correct method to grow a paged file in place. `File::open_rw` also rejects paged files ([#178](https://github.com/CramBL/hdf5-pure/issues/178)).

## [0.23.0] - 2026-07-22

Paged file-space support lands ([#173](https://github.com/CramBL/hdf5-pure/issues/173)): `FileBuilder::with_file_space_strategy(FileSpaceStrategy::Page, …)` now writes a **genuine paged file** — page-aligned allocations with metadata and raw data in separate pages and per-page-type free-space managers — and `File::open_rw_bounded` grows a file that persists its free space, including a paged one, with bounded memory, rewriting its managers at `File::close` so the reference C library reads the result. Also new: `Dataset::read_raw_rows` and the typed `read_*_rows` stream a `[start, start + count)` leading-dimension row window without materializing the whole dataset ([#170](https://github.com/CramBL/hdf5-pure/pull/170)). Additive minor bump.

### Added

- `FileBuilder::with_file_space_strategy(FileSpaceStrategy::Page, …)` now writes a **genuine paged file** instead of only recording the label: allocations are aligned to `with_file_space_page_size`, metadata and raw data occupy separate pages, and each page's free tail is tracked in a per-page-type free-space manager, so the reference C library reads it as a paged file, parses the managers (`H5Fget_freespace`), and re-paginates it on write ([#173](https://github.com/CramBL/hdf5-pure/issues/173)).
- `File::open_rw_bounded` now grows a file that persists its free space (`H5Pset_file_space_strategy(persist = true)`): its on-disk free-space managers are seeded on open and rewritten at `File::close`, so bounded-memory appends round-trip through the reference C library. This includes a genuine **paged** file (`H5F_FSPACE_STRATEGY_PAGE`), whose appends are kept page-homogeneous (raw and metadata in separate pages) and whose per-page-type managers are rewritten at close; a paged file without persisted free space is refused ([#173](https://github.com/CramBL/hdf5-pure/issues/173)).
- `Dataset::read_raw_rows` and the typed `read_f64_rows`/`read_f32_rows`/`read_i8_rows`/`read_i16_rows`/`read_i32_rows`/`read_i64_rows`/`read_u8_rows`/`read_u16_rows`/`read_u32_rows`/`read_u64_rows`/`read_string_rows` read a leading-dimension row window `[start, start + count)` without materializing the whole dataset, so a large dataset can be streamed a fixed number of rows at a time; inner-chunked and variable-length string windows fall back to a whole read sliced to the window ([#170](https://github.com/CramBL/hdf5-pure/pull/170)).

## [0.22.0] - 2026-07-22

The owned-handle API lands ([#148](https://github.com/CramBL/hdf5-pure/issues/148)): `Dataset`, `Group`, and `Object` are now owned handles with no `<'f>` lifetime and `File` is cheaply cloneable, so a handle can be stored, cached, sent across threads, and outlive its `File`. A file opened with `File::open_rw` or `File::create` reads, appends, edits, and commits through those handles (`Dataset::append` is immediate and crash-atomic), with `File::open_swmr_writer` for lock-free SWMR appends and `File::open_rw_bounded` for reading and appending with memory bounded independent of file size. The legacy `EditSession`, `SwmrWriter`, and `AppendWriter` are deprecated in favor of it. Also new: filtered in-place append ([#144](https://github.com/CramBL/hdf5-pure/issues/144)), layout/filter and live-space introspection ([#149](https://github.com/CramBL/hdf5-pure/issues/149), [#150](https://github.com/CramBL/hdf5-pure/issues/150)), and configurable fill values ([#151](https://github.com/CramBL/hdf5-pure/issues/151)). **Breaking:** the handle lifetime is gone (drop `Dataset<'_>`), and `File::refresh` now reports outstanding handles at runtime with `Error::HandlesOutstanding`.

### Breaking

- **Breaking:** `Dataset`, `Group`, and `Object` are now owned handles with no `<'f>` lifetime — `File::dataset`/`group`/`root` hand back handles that share ownership of the open file (internally `Arc`), so a handle can be stored in a struct, cached, sent across threads, and outlive the `File` value it came from, and `File` is now cheaply cloneable. Code that never named the handle lifetime is unaffected; code that wrote `Dataset<'_>` should drop the lifetime, and `File::refresh` now returns `Error::HandlesOutstanding` when a handle or `File` clone is still alive instead of enforcing it at compile time ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).

### Added

- `File::open_rw_bounded` (and `_with_options`) opens a file for reading and appending with **bounded memory** — no whole-file mirror: streaming-grade reads plus the same immediate, crash-atomic `Dataset::append` as `open_rw`, with large appends applied in whole-chunk batches so peak memory stays at the configured caches plus a few chunks regardless of file or call size. The staged edit surface returns the new `Error::BoundedStagedUnsupported` ([#147](https://github.com/CramBL/hdf5-pure/issues/147)).
- `File::open_rw` opens a file for reading **and** writing through owned handles, and `File::create` builds a new file the same way: `Dataset::append` grows a chunked, unlimited, Extensible-Array-indexed dataset in place (immediate and crash-atomic, reading back through the same handle), while `Dataset::write`/`set_attr`/`remove_attr`, `Group::create_dataset`/`create_group`/`delete`, and `File::copy` stage edits that `File::commit` applies as one transaction. A write on a read-only file returns `Error::ReadOnly` ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- The owned-handle write surface reaches parity with `EditSession`: `Dataset::append_staged` grows a dataset with a rebuilt index staged until `commit` — including the **filtered** and non-chunk-aligned appends the immediate `Dataset::append` refuses; `File::copy_from` stages a cross-file `H5Ocopy` from a buffered read-only file; `Group::set_attr`/`remove_attr` edit a group's (or the root's) compact attributes; `File::space_accounting` and `File::has_staged_edits` report live space use and whether a commit is pending; and `File::open_rw_with_locking` opens with an explicit `FileLocking` policy. `File::close` now seals the file, so a write through a surviving handle returns the new `Error::FileClosed` ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- `File::open_swmr_writer` opens a file for SWMR (single-writer/multiple-reader) appending through owned handles: it takes **no** OS lock (so concurrent readers, and Windows' mandatory locks, are never blocked) and raises the superblock's SWMR-write flag, cleared on `File::close`. Only immediate `Dataset::append` is allowed, over the unfiltered, chunk-aligned SWMR subset; the staged edit surface returns the new `Error::SwmrStagedUnsupported`, and `File::clear_swmr_flag` recovers a flag left set by a crashed writer ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- `EditSession::append_inplace` grows an existing **chunked, unlimited, Extensible-Array dataset** in place at amortized `O(1)` cost — immediate and crash-atomic, needing no `commit` — and can be interleaved with the session's staged group/dataset/attribute/delete edits on one open file, with no reopening between the fast appends and the tree edits. Unfiltered datasets accept any-length appends, filtered datasets whole chunks only; a userblock or pre-v2 file, an unallocated or non-Extensible-Array index, or a multi-hard-link dataset is refused with `Error::AppendInPlaceUnsupported` (use `append_dataset` instead) ([#146](https://github.com/CramBL/hdf5-pure/issues/146)).
- `EditSession::set_dataset_attr` / `remove_dataset_attr` add, update, or remove a compact **dataset** attribute — fixed-size or variable-length string — staged until `commit`; a dense (fractal-heap) attribute store or a multi-hard-link dataset is refused ([#146](https://github.com/CramBL/hdf5-pure/issues/146)).
- `EditSession::append_dataset` grows an existing **chunked, unlimited dataset** in place along its first dimension — **filtered** (deflate/shuffle/fletcher32/scale-offset, and ZFP with the `zfp` feature) or not, and of any length (a trailing partial chunk is rewritten) — without requiring SWMR; existing chunk data stays put while the appended chunks and a rebuilt Extensible-Array index are added, and the result reads back in the reference C library and h5py. Datasets that are not Extensible-Array-indexed (a version-1 B-tree, fixed-array, or single-chunk index), higher than rank 1, use a filter this engine cannot re-encode, or have more than one hard link are refused ([#144](https://github.com/CramBL/hdf5-pure/issues/144)).
- `Dataset` gains read-only introspection — `is_chunked`, `maxshape`, `chunk_shape`, and `filters` — so callers can check a dataset's storage, extensibility, and filter pipeline (for example append eligibility) without decoding any data ([#144](https://github.com/CramBL/hdf5-pure/issues/144)).
- `Dataset::layout`, `chunk_index`, `chunks`, and `filter_pipeline` expose the full storage layout and filter pipeline through the curated `Layout`, `ChunkIndex`, `Chunk`, and `Filter` types — the storage class, chunk-index kind (with `ChunkIndex::supports_inplace_append`), and each chunk's absolute file address, on-disk size, and filter mask, plus each filter's id, name, optional flag, and client data — so a caller can locate and read one chunk at a time without materializing the dataset. Enumerating a version-2 B-tree index's chunks is not yet supported ([#149](https://github.com/CramBL/hdf5-pure/issues/149)).
- `EditSession::space_accounting` reports a mutating session's live space usage as a `SpaceAccounting` — the current logical file size, the total reusable free bytes, and the reusable free regions as absolute `(offset, length)` pairs — the active-editor counterpart of `File::file_size` and `persisted_free_space`; it reflects committed state plus immediate in-place appends, not edits still staged for `commit` ([#150](https://github.com/CramBL/hdf5-pure/issues/150)).
- `DatasetBuilder::with_fill_value` records a dataset's fill value — the value HDF5 reports for never-written elements — and `Dataset::fill_value` reads one back, from this crate's files as well as the reference C library's and h5py's; the fill value's type must match the dataset datatype ([#151](https://github.com/CramBL/hdf5-pure/issues/151)).

### Deprecated

- `AppendWriter` is deprecated in favor of `File::open_rw` plus `Dataset::append`, which offers the same amortized `O(1)` in-place append through one open file that also reads and edits; it still works and will be removed in a later release ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- `SwmrWriter` and `EditSession` are deprecated in favor of the owned-handle API and will be removed in a later release: open with `File::open_swmr_writer` or `File::open_rw` and mutate through owned `Dataset`/`Group` handles that read and write one file by name (`Dataset::append`/`append_staged`/`write`, `Group::create_dataset`/`create_group`/`delete`, `File::copy_from`/`commit`/`clear_swmr_flag`) ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).

### Fixed

- Reading through a `Dataset` handle after appending through that same handle no longer returns stale data: the append now invalidates the handle's cached chunk index, which previously still pointed at the relocated trailing chunk ([#147](https://github.com/CramBL/hdf5-pure/issues/147)).
- Variable-length string/sequence reads and `Dataset::chunks` introspection now work on read-write files (`File::open_rw` and the new bounded mode): these paths previously read the global heap through an empty byte view on the mirror backend and failed with an EOF error ([#147](https://github.com/CramBL/hdf5-pure/issues/147)).
- Reading an attribute or dataset whose dataspace declares dimensions whose product overflows `u64` no longer panics: the element count now saturates so the size and limit checks reject the file as a format error ([#142](https://github.com/CramBL/hdf5-pure/issues/142)).
- docs.rs now documents the full public API — the `ndarray`, `serde` (`mat`), `zfp`, `provenance`, and `parallel` surfaces, previously hidden by a default-features-only build — and repairs the broken rustdoc intra-doc links across the public API ([#154](https://github.com/CramBL/hdf5-pure/pull/154)).

## [0.21.2] - 2026-07-14

The `.mat` serializer now drops a struct field that serializes as a Rust unit `()` — most commonly a `serde_json::Value::Null` — like `Option::None` instead of aborting the encode. Parser hardening: the buffered and streaming readers agree on a malformed v1 object header, and crafted files return a format error instead of panicking on an arithmetic overflow across the metadata parsers. Non-breaking patch.

### Fixed

- `mat::to_bytes` no longer aborts the whole encode when a struct field serializes as a Rust unit `()` — most commonly a `serde_json::Value::Null` field: the field is now dropped like `Option::None` (read it back with `#[serde(default)]`) instead of failing with `UnsupportedType("() / unit")` ([#141](https://github.com/CramBL/hdf5-pure/pull/141)).
- The buffered and streaming readers now agree on a malformed v1 object header: the buffered path stops at the declared object-header size instead of reading (and following) a chunk-0 message that overruns it ([#140](https://github.com/CramBL/hdf5-pure/pull/140)).
- Parsing a crafted file now returns a format error instead of panicking on an arithmetic overflow, hardening address and size computations across the metadata parsers (local heap, symbol table, datatype sizing, and the chunk/fixed-array/extensible-array indexes) ([#140](https://github.com/CramBL/hdf5-pure/pull/140)).

## [0.21.1] - 2026-07-08

Base-address normalization now rejects a `u64` overflow with an `OffsetOverflow` error instead of panicking or silently wrapping, hardening the parser against a crafted superblock base address. The check covers the superblock root-group address on both the read and edit paths and group-child object-header addresses. Non-breaking patch.

### Fixed

- Reject base-address normalization that overflows `u64` instead of panicking or wrapping, covering the superblock root-group address (read and edit paths) and group-child object-header addresses.

## [0.21.0] - 2026-07-02

`EditSession` gains three in-place additions: an **empty (zero-element) contiguous dataset** and a **provenance-tagged dataset** (`DatasetBuilder::with_provenance`, behind the `provenance` feature); a **variable-length attribute value** (`AttrValue::VarLenAsciiArray`) and a **variable-length-string dataset** (`DatasetBuilder::with_vlen_strings`); and an **object-reference dataset** (`DatasetBuilder::with_path_references`). Chunked/extensible variants of each stay refused. Additive minor bump.

### Added

- `EditSession` now adds, in place, an **empty (zero-element) contiguous dataset** and a **provenance-tagged dataset** (`DatasetBuilder::with_provenance`, behind the `provenance` feature); a chunked/extensible empty dataset stays refused ([#105](https://github.com/CramBL/hdf5-pure/issues/105)).
- `EditSession` now adds, in place, a dataset, group, or root attribute with a **variable-length value** (`AttrValue::VarLenAsciiArray`) and a **variable-length-string dataset** (`DatasetBuilder::with_vlen_strings`); dense-attribute storage and a chunked/extensible variable-length-string dataset stay refused ([#105](https://github.com/CramBL/hdf5-pure/issues/105)).
- `EditSession` now adds, in place, an **object-reference dataset** (`DatasetBuilder::with_path_references`); a target the same commit is still writing is refused rather than resolved to a stale address, and a chunked/extensible reference dataset stays refused ([#105](https://github.com/CramBL/hdf5-pure/issues/105)).

### Fixed

- `EditSession::create_dataset(...).with_vlen_strings(...)` no longer silently corrupts the added dataset: `commit()` now writes and patches its global heap collection, so the dataset reads back instead of failing with `InvalidGlobalHeapSignature` ([#105](https://github.com/CramBL/hdf5-pure/issues/105)).

## [0.20.1] - 2026-07-01

HDF5 **enumeration datasets** now read back through the typed integer/float readers via their integer base type, so an enum dataset written with `EnumTypeBuilder` / `DatasetBuilder::with_enum_i32_data` reads its codes instead of failing with a `TypeMismatch`. Non-breaking patch.

### Fixed

- Typed integer and float readers (`Dataset::read_i32`, `read_u8`, …) now decode an **HDF5 enumeration dataset** as its integer base type, so an enum dataset written with `EnumTypeBuilder` / `DatasetBuilder::with_enum_i32_data` reads its codes back instead of failing with a `TypeMismatch`; member names stay available via `DType::Enum`, and no name-based enum-to-enum conversion is performed ([#129](https://github.com/CramBL/hdf5-pure/issues/129)).

## [0.20.0] - 2026-06-24

MATLAB **struct arrays** now read: a `MATLAB_class="struct"` group whose fields are datasets of per-element object references is transposed into an array-of-structs, so `mat::from_file` / `mat::from_bytes` read a `1×N` / `N×1` struct array into `Vec<T>` and an `M×N` array into `Vec<Vec<T>>`. Additive minor bump.

### Added

- MATLAB **struct arrays** now deserialize: a `MATLAB_class="struct"` group whose fields are datasets of per-element object references is transposed into an array-of-structs, so `mat::from_bytes` / `mat::from_file` read a `1×N` / `N×1` struct array into `Vec<T>` and an `M×N` array into `Vec<Vec<T>>` — previously refused with a `Reference` type mismatch. A scalar struct still reads as a single struct ([#127](https://github.com/CramBL/hdf5-pure/issues/127)).

## [0.19.0] - 2026-06-22

`EditSession` now edits files that carry a **userblock** (non-zero base address), such as MATLAB v7.3 `.mat` files. It reads and writes addresses relative to the base and preserves the userblock bytes, so every edit works - value overwrites, additions, relocating overwrites of every layout with old storage reclaimed, object deletion, in-file and cross-file copy, group creation, and compact attributes. Cross-file copy from a userblock _source_ is rejected. This release also fixes reading and repacking a chunked dataset from such a file. Additive minor bump.

### Added

- `EditSession` now opens and edits files that carry a **userblock** (non-zero base address), such as MATLAB v7.3 `.mat` files. It reads and writes addresses relative to the base and preserves the userblock bytes verbatim. Every edit is supported - value overwrites, additions, relocating overwrites of every layout (with old storage reclaimed), object deletion, in-file and cross-file copy, group creation, and compact attributes. Cross-file copy from a userblock _source_ is rejected ([#104](https://github.com/CramBL/hdf5-pure/issues/104)).

### Fixed

- Reading and repacking a **chunked dataset from a file with a userblock** (non-zero base address) now works. The base address applies correctly to chunked data during reads ([#104](https://github.com/CramBL/hdf5-pure/issues/104)).

## [0.18.0] - 2026-06-20

Broad MATLAB v7.3 read support for MCOS opaque types — cell arrays, the modern `string` class, `datetime` / `duration` / `categorical`, `table` / `timetable`, enumeration arrays, and `containers.Map`, including objects nested inside structs, cells, and table columns, all resolved through the file's `#subsystem#`/MCOS store. Also adds in-place overwrite and copy of chunked & filtered datasets in `EditSession`, a faster MAT write path, and two compound-datatype read fixes. **Breaking:** `MatError` is now `#[non_exhaustive]`; minor bump.

### Added

- `EditSession::write_dataset` now overwrites **chunked and filtered** datasets in place: unfiltered chunks (and filtered chunks that re-encode to the same size or smaller) are written into their existing slots — a shrinking filtered overwrite rebuilds the fixed-/extensible-array index in place to record the new sizes — while one whose re-encoded chunks no longer fit is rebuilt and relocated with the old storage reclaimed. A version-2 B-tree chunk index is still refused ([#101](https://github.com/CramBL/hdf5-pure/issues/101)).
- `EditSession::copy` / `copy_from` now copy a **chunked or filtered** dataset, preserving its chunk payloads and filter pipeline byte-for-byte (the chunk index is rebuilt at the new location, so a B-tree-v1 or implicit-indexed source becomes an equivalent v4 index); a version-2 B-tree index or a sparse chunk grid is still refused ([#101](https://github.com/CramBL/hdf5-pure/issues/101)).
- MATLAB **cell arrays** now deserialize: `mat::from_bytes` / `mat::from_file` resolve each element's `#refs#` object reference and rebuild the sequence, so `Vec<Struct>`, ragged `Vec<Vec<T>>`, `Vec<Option<T>>` (with `None` slots restored), and nested cells round-trip — previously refused with `UnsupportedType("cell array")`. New public `Dataset::dereference` and `Object` resolve an HDF5 object reference (`H5R_OBJECT`) to the group or dataset it names, and MATLAB's reserved `#refs#` / `#subsystem#` groups are skipped on read ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).
- The modern MATLAB **`string`** class now deserializes: an opaque (`MATLAB_object_decode=3`) `string` dataset's object id is resolved against the `#subsystem#/MCOS` store and its UTF-16 saveobj payload decoded, so values written with `Options::with_modern_strings()` round-trip and a scalar `string` reads back as a Rust `String` ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).
- MATLAB **`datetime`**, **`duration`**, and **`categorical`** now deserialize into the new public `MatDatetime` / `MatDuration` / `MatCategorical` types (Unix-epoch millisecond instants, durations in milliseconds, and category codes plus names — lossless, with `nanoseconds()` / `seconds()` / `labels()` helpers). Any other MCOS opaque class (`table`, `containers.Map`, `dictionary`, user `classdef`s, …) is surfaced losslessly as its raw property map rather than refused, so unknown opaque variables still read; function handles and legacy objects (`MATLAB_object_decode` 1/2) remain refused by name. **Breaking:** `MatError` is now `#[non_exhaustive]` ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).
- Nested MATLAB **MCOS objects now decode**: a `string` / `datetime` / `duration` / `categorical` / struct / user-class value embedded inside another opaque object resolves to its real value instead of the raw `uint32` reference metadata, so a nested `datetime` (in a struct, a cell, or a table column) reads back decoded ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).
- MATLAB **`table`** and **`timetable`** variables now read. Each column is addressable by its variable name, so a table deserializes straight into your own struct (field name = column name) — `string` / `datetime` / `duration` / `categorical` / struct / user-class columns included — or into the new public `MatTable` / `MatTimetable` for schema-agnostic access through the `MatColumn` enum, with row names and timetable row-times exposed. Numeric columns surface as `f64` through `MatColumn` (read the typed-struct path for exact integer width); a table's `Properties` (units, descriptions, …) is not yet surfaced ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).
- MATLAB **enumeration** arrays now deserialize into the new public `MatEnum` (the class name plus each element's member name, row-major), wherever they appear — a top-level variable, a user-class property, a cell, or a struct field. The underlying value backing each member is not surfaced ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).
- MATLAB **`containers.Map`** variables now deserialize as a `key -> value` map: a string/char-keyed map reads straight into a `HashMap<String, V>` / `BTreeMap<String, V>` or a struct keyed by the map's keys, and numeric keys are presented as strings (`1.0` -> `"1"`). The `dictionary` type still reads losslessly as its raw property map; a typed `MatMap` introspection view is not yet provided ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).

### Fixed

- Read HDF5 **version-1 and version-2 compound datatypes** correctly: the member layout was misparsed (the v1 dimension block skipped one 4-byte reserved field, and v2 names were left unpadded), so complex data written by MATLAB and older HDF5 writers — including real-MATLAB `datetime` arrays — now decodes instead of failing with a type mismatch ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).
- An empty `datetime` or `duration` object stored with no `data` / `millis` property (e.g. a zero-row timetable's row-times) now decodes as empty instead of aborting the whole-file read ([#114](https://github.com/CramBL/hdf5-pure/issues/114)).

### Performance

- Serializing a MATLAB v7.3 file is faster: the default `mat::to_bytes` write path now shares the cache-tiled column-major transpose (≈8% faster on a 512×512 `f64` matrix) instead of a strided copy, and numeric/field buffers across the read and write paths are pre-sized or filled in a single pass. Reading a numeric array no longer materializes an intermediate boxed-scalar buffer, and a `uint32` array nested under an MCOS object is decoded once instead of twice ([#122](https://github.com/CramBL/hdf5-pure/pull/122)).

## [0.17.0] - 2026-06-18

Repack now reproduces three more datatype classes faithfully — non-string variable-length sequences, object-reference datasets, and time datatypes — and the dataset read and write hot paths are several times faster (bulk numeric decode, contiguous-row chunk scatter, compress-once filtered writes). **Breaking:** `Datatype::Time` gained a `byte_order` field, so code matching that variant must account for it; minor bump.

### Added

- Repack now reproduces three more datatype classes faithfully: non-string variable-length sequences (re-staged through a fresh global heap), object-reference datasets (each address rewritten to its target's new location in the compacted file), and time datatypes (byte order preserved). Chunked/filtered/resizable VL and reference datasets, region or non-8-byte object references, and an object reference to a dropped or out-of-hierarchy target are still refused by name ([#107](https://github.com/CramBL/hdf5-pure/issues/107)).
- **Breaking:** `Datatype::Time` gained a `byte_order` field so a time type's byte order survives a read/serialize round-trip (it was previously dropped on read and forced little-endian); code matching the `Time` variant must account for the new field ([#107](https://github.com/CramBL/hdf5-pure/issues/107)).

### Fixed

- A null or empty variable-length element now writes a zero heap address (HDF5's null-reference convention) instead of an all-ones undefined-address sentinel, which the reference C library rejected as a bad heap index when reading such an element back ([#107](https://github.com/CramBL/hdf5-pure/issues/107)).

### Performance

- Decoding a numeric dataset into a typed `Vec` (`Dataset::read_i32`/`read_u16`/`read_f64` and siblings) now bulk-decodes native-/big-endian standard-layout values instead of going element by element, making integer reads several times faster (≈15× for `read_i32`, ≈9× for `read_u16` on a 1M-element array); sub-byte-precision and unusual layouts keep the exact same results ([#113](https://github.com/CramBL/hdf5-pure/pull/113)).
- Reading a chunked dataset now scatters each chunk into the output one contiguous row at a time rather than element by element, ≈3× faster chunk assembly (a 1024×1024 uncompressed read drops from ~7.6 ms to ~2.2 ms) ([#113](https://github.com/CramBL/hdf5-pure/pull/113)).
- Writing a chunked, filtered dataset now compresses each chunk once instead of twice (the object-header sizing pass no longer recompresses), ≈2–3× faster compressed writes (a 1024×1024 shuffle+deflate write drops from ~45 ms to ~16 ms) ([#113](https://github.com/CramBL/hdf5-pure/pull/113)).
- The byte-shuffle filter is specialized for the common element widths, the chunk cache no longer copies decompressed chunks in and out on the hot path, and the deflate decoder pre-sizes its output buffer ([#113](https://github.com/CramBL/hdf5-pure/pull/113)).

## [0.16.0] - 2026-06-18

Centers on `repack`: it now copies compressed chunks **verbatim** (so lossy filters survive byte-exact) and runs **fully out-of-core**, and gains variable-length-string support. Also adds in-place dataset-value overwrite, dense-attribute and cross-file object copy, in-place addition of chunked/filtered/extensible datasets, and free-space reclaim for chunked deletes. It includes reader hardening (a multi-filter chunk-mask corruption fix, sub-byte integer precision, decompression-bomb bounds, and safer B-tree/heap rejections). Additive minor bump.

### Added

- Repack now copies a chunked dataset's compressed chunks **verbatim**, eliminating the per-dataset decompression blowup and the decompress→recompress round-trip. Lossy filters now survive byte-exact - float D-scale scale-offset, ZFP, SZIP, and even filters this crate cannot itself apply ([#82](https://github.com/CramBL/hdf5-pure/issues/82), [#84](https://github.com/CramBL/hdf5-pure/issues/84), [#85](https://github.com/CramBL/hdf5-pure/issues/85)). The verbatim path covers a fully-allocated chunk grid. A sparse chunked or a contiguous/compact filtered dataset still re-encodes and rejects a lossy filter by name.
- Repack is now **fully out-of-core**, closing [#82](https://github.com/CramBL/hdf5-pure/issues/82): it streams the source (`File::open_streaming`) and the output (`FileBuilder::finish_to`, a `std::io::Write` sink), so peak memory is bounded by one chunk plus the file's metadata regardless of dataset size. This extended the streaming reader to also read attributes (compact, shared, dense, and VL-string) and traverse v1 symbol-table groups ([#27](https://github.com/CramBL/hdf5-pure/issues/27)).
- Variable-length string dataset writing and repack: `DatasetBuilder::with_vlen_strings(&[&str])` writes a contiguous VL UTF-8 string dataset (1D, or ND via `with_shape`), matching the C library's `H5Tvlen_create(H5T_C_S1)` layout so the C library and h5py read it back. Repack now round-trips contiguous/compact VL-string datasets, preserving charset, padding, the null-vs-empty distinction, embedded NULs, and non-UTF-8 bytes. Chunked, filtered, or resizable VL-string datasets and non-string VL datatypes are still rejected by name ([#83](https://github.com/CramBL/hdf5-pure/issues/83)).
- In-place overwrite of dataset values: `EditSession::write_dataset(path)` replaces an existing contiguous or compact dataset's values (HDF5's `H5Dwrite` whole-dataset write), returning the same `DatasetBuilder` as `create_dataset`. The replacement must match the on-disk datatype and shape. A same-length contiguous overwrite writes straight into the existing data block, while a length change or a compact dataset relocates the header like an addition. Chunked and filtered datasets, and a relocating overwrite of a multiply-hard-linked dataset, are rejected by name ([#79](https://github.com/CramBL/hdf5-pure/issues/79)).
- Object copy now reproduces dense (fractal-heap) attribute storage: above the compact threshold of 8 attributes HDF5 stores attributes in a fractal heap indexed by a B-tree v2. `EditSession::copy` and `copy_from` now read the source attributes and re-emit them into a fresh destination-local heap, same-file and cross-file ([#87](https://github.com/CramBL/hdf5-pure/issues/87)). For now a single direct block is emitted: a set too large for one direct block is rejected by name, as is a cross-file dense set whose values are variable-length or reference data.
- Cross-file object copy: `EditSession::copy_from` copies a dataset or whole group subtree out of a _separate_ open `File` into the file being edited - the cross-file form of HDF5's `H5Ocopy`, alongside the same-file `EditSession::copy` ([#78](https://github.com/CramBL/hdf5-pure/issues/78)). The source is read and validated eagerly, so it returns a `Result`. Because the copy is verbatim, it rejects by name anything whose stored bytes embed a source-file address - variable-length and reference data or attributes, and any shared header message. The source must be a buffered file (`File::open` / `File::from_bytes`, not `open_streaming`) with 8-byte offsets and no userblock.
- Free-space reclaim for chunked datasets on in-place delete: deleting a chunked dataset (or a group whose subtree contains one) now returns its chunk data blocks and chunk-index structure to the free list, reused by a later commit and truncated away when the freed run reaches end-of-file. This covers single-chunk, implicit, fixed array, extensible array, and v1 B-tree indexes. A v2 B-tree index, an out-of-bounds or overlapping span, or VL global-heap data is left in place to avoid freeing live bytes ([#77](https://github.com/CramBL/hdf5-pure/issues/77)).
- In-place add of chunked, filtered, and extensible datasets: `EditSession::create_dataset` now accepts `with_chunks`, the writer's filters (`with_deflate`, `with_shuffle`, `with_fletcher32`, `with_scale_offset`, `with_zfp`), and `with_maxshape` (optionally unlimited). The added object header is byte-identical to a freshly written one, and the prior root stays intact until the superblock is repointed last ([#76](https://github.com/CramBL/hdf5-pure/issues/76)).

### Fixed

- Reading a virtual (VDS) dataset now fails with a clear `FormatError::UnsupportedVirtualLayout`. VDS reading is tracked as a planned feature ([#111](https://github.com/CramBL/hdf5-pure/issues/111)).
- Multi-filter chunks where only _some_ filters were skipped for a chunk (the per-chunk `filter_mask`, e.g. shuffle+gzip on an incompressible chunk that the C library stores shuffled but not deflated) now have the surviving filters reversed, fixing silent value corruption on spec-valid files ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- Integers with sub-byte precision or a non-zero bit offset (`H5Tset_precision` / `H5Tset_offset`) now decode correctly in the dataset and attribute readers - masked to the significant bits and sign-extended at the precision boundary. Compound fields with such layouts are still rejected by name ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- A malformed v1 B-tree with a cyclic or pathologically deep internal node - in either the chunk index or a group's symbol table - now errors. Traversal is bounded by a depth cap ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- Deflate-compressed chunks are now bounded to their expected decompressed size: a chunk that inflates past it (a decompression bomb) or decodes to the wrong length is rejected with `FormatError::DecompressionError` or `FormatError::DataSizeMismatch` ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- A truncated or corrupt fixed-rate ZFP chunk now decodes without panicking (`zfp` feature) ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- Reading an object from a _filtered_ fractal managed heap is now rejected cleanly with `FormatError::UnsupportedFilteredHeapObject`. The indirect-block child-pointer walk used the wrong stride for filter-encoded blocks ([#80](https://github.com/CramBL/hdf5-pure/issues/80)).
- Object copy (`EditSession::copy` and `copy_from`) no longer rejects an object whose Attribute Info message carries an _undefined_ fractal-heap address - the reference C library and h5py emit that message (to record attribute creation order) alongside compact, inline attributes, and the editor mistook its mere presence for dense storage. It now inspects the heap address and rejects only genuine dense storage, on both the same-file and cross-file paths ([#78](https://github.com/CramBL/hdf5-pure/issues/78)).
- In-place delete now reclaims an object's storage only when the link being removed is its _last_ hard link. The editor counts every hard link before reclaiming and leaves a multiply-linked object's storage in place (a safe leak the repack path still compacts) ([#77](https://github.com/CramBL/hdf5-pure/issues/77)).
- Malformed chunk geometry is now rejected up front by both `FileBuilder` and `EditSession` (`FormatError::InvalidChunkGeometry` or `Error::EditUnsupported`): a chunk rank that disagrees with the shape, a zero chunk dimension, a max shape of the wrong rank or smaller than the current shape, chunking a scalar, and an element count that overflows `u64` ([#76](https://github.com/CramBL/hdf5-pure/issues/76)). Zero-element extensible datasets remain valid.

## [0.15.0] - 2026-06-16

Adds generic element-typed dataset I/O, file- and dataset-level cache tuning, in-place group attribute editing, OS advisory file locking for the editor, and a gallery of runnable examples; also hardens the 32-bit/WASM readers against silent truncation. Additive minor bump, with two intended behavior changes (editor file locking and the new truncation guards) noted below.

### Added

- Generic, type-parameterized dataset I/O: `DatasetBuilder::with_data(&[T])` writes any supported scalar and `Dataset::read::<T>()` reads one back, so you can write code generic over the element type instead of reaching for `with_i64_data` / `read_i64` and friends. Backed by the now feature-independent `H5Element` bound (previously available only with the `ndarray` feature). Both delegate to the existing typed methods, so behavior is unchanged ([#53](https://github.com/CramBL/hdf5-pure/issues/53)).
- File-access options applied at open time via `FileAccessOptions` and the matching `*_with_options` constructors (`File::open_with_options`, `open_streaming_with_options`, `open_swmr_with_options`, `from_bytes_with_options`): `MetadataCacheConfig` bounds the streaming reader's metadata cache and `ChunkCacheConfig` tunes the chunk cache ([#65](https://github.com/CramBL/hdf5-pure/pull/65)).
- Per-dataset chunk-cache control: `File::dataset_with_options` / `Group::dataset_with_options` take a `DatasetAccessOptions` that overrides the file-wide chunk-cache default for a single dataset, mirroring HDF5's `H5Pset_chunk_cache` access property list. `Dataset::chunk_cache_config()` reports the effective setting ([#48](https://github.com/CramBL/hdf5-pure/issues/48)).
- `ChunkCacheConfig::from_h5p_cache(rdcc_nslots, rdcc_nbytes)` builds a chunk-cache config straight from HDF5's `H5Pset_cache` raw-data parameters ([#66](https://github.com/CramBL/hdf5-pure/pull/66)).
- `Dataset::chunk_cache_stats()` reports a read-only snapshot of a dataset's chunk-cache occupancy (index loaded, retained chunks, retained bytes), so callers can confirm their chunk-cache tuning is taking effect ([#68](https://github.com/CramBL/hdf5-pure/pull/68)).
- In-place group attribute editing: `EditSession::set_group_attr` adds or replaces a compact group attribute and `EditSession::remove_group_attr` removes one, without rewriting the file ([#64](https://github.com/CramBL/hdf5-pure/pull/64)).
- OS advisory file locking for the in-place editor, the crash-safe half of HDF5's concurrency model and the analogue of `H5Pset_file_locking`. `EditSession::open` takes an exclusive lock, so a second editor (or any concurrent writer) gets the new `Error::FileLocked`; the kernel releases it on any process exit, including a crash, so a crashed editor never leaves a stale lock. Control it with the new `FileLocking` policy (`EditSession::open_with_locking`) or `HDF5_USE_FILE_LOCKING=FALSE` for filesystems where locking is unavailable. `SwmrWriter` and the readers intentionally take no lock: SWMR is single-writer-by-contract and built for concurrent reads, and `std`'s whole-file lock would block readers (fatally on Windows, where locks are mandatory) ([#73](https://github.com/CramBL/hdf5-pure/issues/73)).
- A gallery of runnable, self-checking examples in `examples/` covering the core API: write/read, generic element I/O, groups & attributes, compression, compound & complex types, ndarray, in-place editing, repack, SWMR, and file-space strategy. Run any with `cargo run --example <name>` ([#54](https://github.com/CramBL/hdf5-pure/issues/54)).

### Changed

- 32-bit / WASM hardening: the chunked-data and MATLAB matrix readers now return an error instead of silently truncating when a file-derived dimension or element count exceeds the platform's pointer width. Every remaining narrowing `as` cast in the library is now either a checked conversion or carries an `#[expect(…, reason = "…")]` justifying why it is bounded, enforced by a hard deny of the narrowing-cast lints on a 32-bit CI target — replacing the previous count-based ratchet, which a new cast could slip past by removing an unrelated one ([#72](https://github.com/CramBL/hdf5-pure/issues/72)).

### Fixed

- Read dense groups and dense attributes whose link/attribute names are very long (stored as fractal-heap "huge" objects); previously failed with `InvalidObjectHeaderVersion` ([#63](https://github.com/CramBL/hdf5-pure/pull/63)).
- `EditSession` now clears the superblock's write/SWMR consistency flag on commit instead of preserving whatever the source file carried, so editing a file an interrupted SWMR writer left flagged produces a cleanly-closed file the reference C library can reopen ([#73](https://github.com/CramBL/hdf5-pure/issues/73)).

## [0.14.0] - 2026-06-15

Completes free-space management ([#21](https://github.com/CramBL/hdf5-pure/issues/21)) and closes several interoperability gaps with the reference HDF5 C library. Additive minor bump.

### Added

- File-space strategy on the file-creation property list: `FileBuilder::with_file_space_strategy` and `with_file_space_page_size`, read back with `File::file_space_strategy()` / `File::file_space_info()` ([#55](https://github.com/CramBL/hdf5-pure/pull/55)). Mirrors `H5Pset_file_space_strategy` / `H5Pset_file_space_page_size`.
- `File::persisted_free_space()` reads the on-disk free-space managers of a file written with `persist = true` ([#56](https://github.com/CramBL/hdf5-pure/pull/56)).
- `EditSession` persists free space across reopen: it seeds its free list from the on-disk managers and writes it back on commit, so freed space is reused by later sessions instead of leaking ([#58](https://github.com/CramBL/hdf5-pure/pull/58)).

### Fixed

- The reference C library can now add objects to files this crate writes (group headers were missing a Group Info message, which the C library requires before inserting a link) ([#59](https://github.com/CramBL/hdf5-pure/pull/59)).
- Read large dense groups whose fractal heap grows a multi-row root indirect block (~150+ links) ([#60](https://github.com/CramBL/hdf5-pure/pull/60)).
- Read large dense groups whose name index is a 3-or-more-level v2 B-tree (~26k+ links) ([#62](https://github.com/CramBL/hdf5-pure/pull/62)).

## [0.13.0] - 2026-06-15

Free-space management ([#21](https://github.com/CramBL/hdf5-pure/issues/21), [#45](https://github.com/CramBL/hdf5-pure/pull/45)).

### Added

- `EditSession` now reuses space freed by earlier commits and truncates the file when free space reaches the end, so add/delete churn stays bounded instead of growing the file every commit.
- Whole-file `repack(src, dst, &RepackOptions)` rewrites a file with no dead space, optionally dropping objects (`RepackOptions::new().drop_path("grp/old")`). It refuses with `Error::RepackUnsupported` rather than silently degrade anything it cannot reproduce exactly (e.g. variable-length, reference, or lossy-filtered data).

### Fixed

- `Datatype::serialize` produced empty bytes for the time, bit-field, and opaque datatype classes, corrupting any datatype message that used one of them ([#45](https://github.com/CramBL/hdf5-pure/pull/45)).

## [0.12.1] - 2026-06-10

Internal robustness and tests ([#26](https://github.com/CramBL/hdf5-pure/issues/26)); no public API or on-disk-format change.

### Added

- Property-based tests for the write/read roundtrip and parser robustness ([#44](https://github.com/CramBL/hdf5-pure/pull/44)).
- A Miri CI job covering the crate's only non-trivial `unsafe` (the aligned chunk buffer) ([#43](https://github.com/CramBL/hdf5-pure/pull/43)).

### Changed

- Internal cleanup of B-tree v1 size arithmetic into named helpers ([#42](https://github.com/CramBL/hdf5-pure/pull/42)).

## [0.12.0] - 2026-06-10

### Added

- `EditSession` edits object headers that span multiple chunks (e.g. objects carrying several attributes) ([#32](https://github.com/CramBL/hdf5-pure/issues/32)).
- `EditSession` edits version 0/1 (symbol-table) files in place — the default format from the C library and h5py ([#32](https://github.com/CramBL/hdf5-pure/issues/32)). Adding and deleting is supported; copying a version-1 object is not.

### Fixed

- `EditSession::commit` now `fsync`s appended data before repointing the root, making its "repoint last" crash-safety guarantee real ([#32](https://github.com/CramBL/hdf5-pure/issues/32)).

## [0.11.0] - 2026-06-09

### Added

- In-place file editing via `EditSession` ([#32](https://github.com/CramBL/hdf5-pure/issues/32)): `open(path)`, then `create_dataset` / `create_group` / `delete` / `copy`, applied by `commit()`. Changes are appended and the superblock repointed last, so cost scales with the edit, not the file size, and a failed commit leaves the file valid. It refuses with `Error::EditUnsupported` cases it cannot reproduce faithfully (userblocks, pre-1.10 formats, dense storage, chunked/compressed new datasets). Freed space is not reclaimed (see [#21](https://github.com/CramBL/hdf5-pure/issues/21)).
- File inspection: `is_hdf5(path)` / `is_hdf5_bytes(&[u8])`, `File::file_size()`, and `File::libver_bound()` (new `LibVer` enum) ([#32](https://github.com/CramBL/hdf5-pure/issues/32)).
- `FileBuilder::with_libver_bounds(low, high)`, mirroring `H5Pset_libver_bounds` ([#32](https://github.com/CramBL/hdf5-pure/issues/32)). This crate writes one format (the 1.10+ version-3 superblock), so it acts as a compatibility guard: `finish()` fails with `FormatError::LibverBoundsUnsatisfiable` if the bounds exclude that format.

## [0.10.0] - 2026-06-09

### Changed

- **Breaking:** the public API is now a curated surface; internal format modules are `pub(crate)` ([#33](https://github.com/CramBL/hdf5-pure/issues/33)). Code using the documented reader/writer/builder API is unaffected; code reaching into internal module paths (e.g. `hdf5_pure::object_header::…`) must stop.

### Added

- `Dataset::verify_provenance` (feature `provenance`) checks a dataset against the `_provenance_sha256` hash written by `with_provenance`.

### Removed

- The `fast-checksum` feature and its `crc32fast` dependency — it gated unused CRC32 code (HDF5 uses lookup3). Drop it from any feature list that named it.
- Several internal subsystems that were never wired into the reader or writer.

## [0.9.0] - 2026-06-08

### Removed

- **Breaking:** `parallel_read::decompress_chunks_parallel` and `decompress_chunks_sequential` — public but unused ([#33](https://github.com/CramBL/hdf5-pure/issues/33)). Reader/writer code is unaffected. CI now runs `cargo-semver-checks` to catch unintended API changes.

## [0.8.0] - 2026-06-05

### Added

- Streaming reads for files too large to buffer: `File::open_streaming(path)` reads metadata and chunks on demand instead of loading the whole file ([#27](https://github.com/CramBL/hdf5-pure/issues/27)). Streams contiguous, compact, and all chunk-index layouts; limited to latest-format groups, and attribute reading is not yet supported. The buffered `File::open` path is unchanged.
- 32-bit and bare-metal robustness ([#27](https://github.com/CramBL/hdf5-pure/issues/27)): file offsets/lengths that do not fit the platform now error (`ValueTooLargeForPlatform` / `OffsetOverflow`) instead of truncating. CI runs the suite on 32-bit (i686) and builds for `thumbv7em-none-eabi` `no_std`.
- N-dimensional array I/O via the optional `ndarray` feature ([#24](https://github.com/CramBL/hdf5-pure/issues/24)): `DatasetBuilder::with_ndarray` and `Dataset::read_array` / `read_array_dyn`. Off by default; implies `std`.

### Changed

- Writing a dataset whose shape disagrees with the data now fails with `FormatError::ShapeDataMismatch` instead of producing an unreadable file.

### Removed

- The `mmap` feature and its `memmap2` dependency — declared but never implemented ([#24](https://github.com/CramBL/hdf5-pure/issues/24)). Drop it if you named it.

## [0.7.0] - 2026-06-03

### Added

- SWMR (single-writer / multiple-reader) support for 1-D, unlimited, Extensible-Array-indexed datasets ([#17](https://github.com/CramBL/hdf5-pure/issues/17)):
    - `File::open_swmr(path)` and `File::refresh()` re-read data appended by a concurrent writer.
    - `SwmrWriter::open(path)` appends chunks in place (`append_i32`, `append_f64`, `append_raw`). The writer orders operations so concurrent readers only see a consistent prefix.
    - `close()` clears the SWMR flag. The standalone `clear_swmr_flag(path)` function recovers files left flagged by a crash.
    - Append operations require `std` and are limited to unfiltered, chunk-aligned, single-dimension datasets. The library rejects unsupported targets with `Error::SwmrAppendUnsupported`.

### Changed

- **Breaking:** `Error` and `FormatError` are now `#[non_exhaustive]`; `match` over them needs a wildcard arm. Future variant additions are now non-breaking.

### Fixed

- Extensible Array chunk index: reading more than 20 chunks returned wrong data and writing more than 244 silently dropped the excess ([#17](https://github.com/CramBL/hdf5-pure/issues/17)).

## [0.6.0]

### Added

- Scale-offset filter (HDF5 filter id 6), read and write, via `.with_scale_offset(mode)` ([#13](https://github.com/CramBL/hdf5-pure/issues/13)). Integer mode is lossless; float decimal-scaling is lossy. Datasets compressed with it by other tools now decode instead of failing with `UnsupportedFilter(6)`.

## [0.5.1]

### Fixed

- Chunked datasets indexed by a Fixed Array now use the paged data block layout above the page size (>1024 chunks at the default), and the reader decodes them; previously such files were written corrupt and rejected on read ([#14](https://github.com/CramBL/hdf5-pure/issues/14)).

## [0.5.0]

### Added

- serde roundtrip for `Matrix<Complex64>` / `Matrix<Complex32>`, including empty matrices (which previously lost their complex class).
- Sealed `mat::MatElement` trait, so an unsupported element type is a compile error rather than a silent class loss.

### Changed

- **Breaking:** `Matrix<T>` serde now requires `T: MatElement` instead of `T: 'static`. Such uses previously produced malformed MAT files at runtime.
- The MAT deserializer flattens 1×N and N×1 values to a 1-D sequence in `deserialize_any` (matching `deserialize_seq`).
- Numeric/complex readers preserve 1×N / N×1 shape at the value layer; any flattening happens at the serde level.

[Unreleased]: https://github.com/CramBL/hdf5-pure/compare/v0.46.1...HEAD
[0.46.1]: https://github.com/CramBL/hdf5-pure/compare/v0.46.0...v0.46.1
[0.46.0]: https://github.com/CramBL/hdf5-pure/compare/v0.45.0...v0.46.0
[0.45.0]: https://github.com/CramBL/hdf5-pure/compare/v0.44.2...v0.45.0
[0.44.2]: https://github.com/CramBL/hdf5-pure/compare/v0.44.1...v0.44.2
[0.44.1]: https://github.com/CramBL/hdf5-pure/compare/v0.44.0...v0.44.1
[0.44.0]: https://github.com/CramBL/hdf5-pure/compare/v0.43.1...v0.44.0
[0.43.1]: https://github.com/CramBL/hdf5-pure/compare/v0.43.0...v0.43.1
[0.43.0]: https://github.com/CramBL/hdf5-pure/compare/v0.42.0...v0.43.0
[0.42.0]: https://github.com/CramBL/hdf5-pure/compare/v0.41.0...v0.42.0
[0.41.0]: https://github.com/CramBL/hdf5-pure/compare/v0.40.0...v0.41.0
[0.40.0]: https://github.com/CramBL/hdf5-pure/compare/v0.39.0...v0.40.0
[0.39.0]: https://github.com/CramBL/hdf5-pure/compare/v0.38.0...v0.39.0
[0.38.0]: https://github.com/CramBL/hdf5-pure/compare/v0.36.0...v0.38.0
[0.36.0]: https://github.com/CramBL/hdf5-pure/compare/v0.35.0...v0.36.0
[0.35.0]: https://github.com/CramBL/hdf5-pure/compare/v0.34.0...v0.35.0
[0.34.0]: https://github.com/CramBL/hdf5-pure/compare/v0.33.0...v0.34.0
[0.33.0]: https://github.com/CramBL/hdf5-pure/compare/v0.32.0...v0.33.0
[0.32.0]: https://github.com/CramBL/hdf5-pure/compare/v0.31.0...v0.32.0
[0.31.0]: https://github.com/CramBL/hdf5-pure/compare/v0.30.0...v0.31.0
[0.30.0]: https://github.com/CramBL/hdf5-pure/compare/v0.29.0...v0.30.0
[0.29.0]: https://github.com/CramBL/hdf5-pure/compare/v0.28.0...v0.29.0
[0.28.0]: https://github.com/CramBL/hdf5-pure/compare/v0.27.0...v0.28.0
[0.27.0]: https://github.com/CramBL/hdf5-pure/compare/v0.26.0...v0.27.0
[0.26.0]: https://github.com/CramBL/hdf5-pure/compare/v0.25.0...v0.26.0
[0.25.0]: https://github.com/CramBL/hdf5-pure/compare/v0.24.0...v0.25.0
[0.24.0]: https://github.com/CramBL/hdf5-pure/compare/v0.23.2...v0.24.0
[0.23.2]: https://github.com/CramBL/hdf5-pure/compare/v0.23.1...v0.23.2
[0.23.1]: https://github.com/CramBL/hdf5-pure/compare/v0.23.0...v0.23.1
[0.23.0]: https://github.com/CramBL/hdf5-pure/compare/v0.22.0...v0.23.0
[0.22.0]: https://github.com/CramBL/hdf5-pure/compare/v0.21.2...v0.22.0
[0.21.2]: https://github.com/CramBL/hdf5-pure/compare/v0.21.1...v0.21.2
[0.21.1]: https://github.com/CramBL/hdf5-pure/compare/v0.21.0...v0.21.1
[0.21.0]: https://github.com/CramBL/hdf5-pure/compare/v0.20.1...v0.21.0
[0.20.1]: https://github.com/CramBL/hdf5-pure/compare/v0.20.0...v0.20.1
[0.20.0]: https://github.com/CramBL/hdf5-pure/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/CramBL/hdf5-pure/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/CramBL/hdf5-pure/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/CramBL/hdf5-pure/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/CramBL/hdf5-pure/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/CramBL/hdf5-pure/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/CramBL/hdf5-pure/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/CramBL/hdf5-pure/compare/v0.12.1...v0.13.0
[0.12.1]: https://github.com/CramBL/hdf5-pure/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/CramBL/hdf5-pure/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/CramBL/hdf5-pure/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/CramBL/hdf5-pure/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/CramBL/hdf5-pure/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/CramBL/hdf5-pure/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/CramBL/hdf5-pure/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/CramBL/hdf5-pure/releases/tag/v0.6.0
[0.5.1]: https://github.com/CramBL/hdf5-pure/releases/tag/v0.5.1
[0.5.0]: https://github.com/CramBL/hdf5-pure/commit/1afca3c
