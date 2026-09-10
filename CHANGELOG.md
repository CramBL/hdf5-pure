# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) under Cargo's pre-1.0 conventions: a `0.x.0` bump may be breaking, `0.x.y` is not.

## [Unreleased]

### Changed

- The documentation is on docs.rs: the crate-level page, the task guides in the `_guide` module, and the API reference. The guide pages ship in the crate, and the documentation site at <https://crambl.github.io/hdf5-pure/> redirects to docs.rs ([#531](https://github.com/CramBL/hdf5-pure/pull/531), [#532](https://github.com/CramBL/hdf5-pure/pull/532), [#535](https://github.com/CramBL/hdf5-pure/pull/535), [#536](https://github.com/CramBL/hdf5-pure/pull/536), [#537](https://github.com/CramBL/hdf5-pure/pull/537), [#538](https://github.com/CramBL/hdf5-pure/pull/538)).

## [0.44.1] - 2026-09-08

### Fixed

- Files HDF5 2.0 writes under its latest library bounds are read. Their compound and array datatypes were rejected with `FormatError::InvalidDatatypeVersion`.
- `Group::named_datatype_references` reports the count a version 1 object header stores, where it reported 1 for every committed datatype in such a header.

### Changed

- Licensed under MIT or Apache-2.0, at your option, where it was MIT alone ([#511](https://github.com/CramBL/hdf5-pure/pull/511)).
- The repository moved to `CramBL/hdf5-pure`, and the documentation site to <https://crambl.github.io/hdf5-pure/>.
- Releases are published to crates.io through Trusted Publishing from the repository's Release workflow, with no long-lived token.
- The published crate holds the library, the examples, the licenses, the README and the changelog. The tests, the test data, the tooling and the documentation site stay in the repository, which takes the package from 7.1 MiB to 5.2 MiB.

## [0.44.0] - 2026-09-04

Files that h5py and netCDF-4 write are editable in place now. `File::open_rw` edits an object that tracks attribute creation order and adds a dataset or group to one that tracks link creation order (h5py's `track_order=True`, and everything netCDF-4 writes), where every such object was refused, and it keeps the object header's timestamps and `H5Pset_attr_phase_change` thresholds across the rewrite instead of dropping both ([#416](https://github.com/CramBL/hdf5-pure/issues/416), [#422](https://github.com/CramBL/hdf5-pure/pull/422)). `File::open` and `File::open_streaming` read a file that shares object header messages (`H5Pset_shared_mesg_index`), resolving a datatype, dataspace, fill value, filter pipeline or attribute out of the shared-message heap, and `repack` rewrites such a file with every message stored inline ([#417](https://github.com/CramBL/hdf5-pure/issues/417)). Deleting objects gives trailing space back to the filesystem rather than leaving `File::file_size()` at a high-water mark ([#418](https://github.com/CramBL/hdf5-pure/issues/418)), and `WriteMarkPolicy::AllowSnapshot` lets a read-only open take a path-based snapshot of a file a page-buffered writer holds ([#419](https://github.com/CramBL/hdf5-pure/issues/419)). Additive minor bump.

### Added

- `File::open_rw` edits an object that tracks attribute creation order (h5py's `track_order=True`, and everything netCDF-4 writes), where every such object was refused: a new attribute takes the next creation index, an overwrite keeps the one it had, a deletion leaves a gap, and a set that goes dense gains a creation-order B-tree beside its name index ([#416](https://github.com/CramBL/hdf5-pure/issues/416)).
- `File::open_rw` creates a dataset or group inside a group that tracks **link** creation order (h5py's `track_order=True` groups, and every netCDF-4 group), each addition taking the group's next link creation index; an addition that would push the group's links past its compact-storage threshold — 8 by default — is still refused ([#422](https://github.com/CramBL/hdf5-pure/pull/422)).
- `FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)` lets a read-only open take a path-based snapshot of a file a page-buffered writer holds, without copying it through `File::from_bytes`; the caller asserts the writer has synced, and a SWMR pair and every read-write open are still refused ([#419](https://github.com/CramBL/hdf5-pure/issues/419)).
- `File::open` and `File::open_streaming` read a file that shares object header messages (`H5Pset_shared_mesg_index`), resolving a datatype, dataspace, fill value, filter pipeline or attribute message out of the shared-message heap where one was refused with `FormatError::UnsupportedSohmReference`, and `repack` rewrites such a file with every message stored inline. An edit that would leave the shared-message table stale is still refused ([#417](https://github.com/CramBL/hdf5-pure/issues/417)).

### Fixed

- `Error::FileMarkedInUse` no longer points a reader at `File::open_swmr` for a file marked by a page-buffered writer, which that reader refuses in its turn; the message now names the snapshot opt-in above, `File::from_bytes`, and `File::clear_swmr_flag` ([#419](https://github.com/CramBL/hdf5-pure/issues/419)).
- Deleting objects now gives the space at the end of the file back to the filesystem instead of leaving `File::file_size()` at a high-water mark, on a file that persists its free-space managers and on a paged file alike. A commit shortens the file to just above its last live allocation, page-aligned where the strategy requires it ([#418](https://github.com/CramBL/hdf5-pure/issues/418)).
- `File::open_rw` keeps an object header's stored timestamps and `H5Pset_attr_phase_change` thresholds across an in-place edit, where every rewrite dropped both and left `H5Oget_info` reporting zero times on any file the C library, h5py or netCDF-4 wrote. The modification and change times are moved to the edit; the thresholds are preserved but do not yet steer the compact-to-dense switch ([#422](https://github.com/CramBL/hdf5-pure/pull/422)).

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

Variable-length string attributes and oversized attribute sets both work now. `AttrValue::VarLenString` and its three siblings write the standard variable-length string datatype (`H5T_STRING` with `STRSIZE = H5T_VARIABLE`), the one h5py and the reference C library write, where the only variable-length attribute this crate produced before was MATLAB's sequence-of-one-byte-strings shape ([#383](https://github.com/CramBL/hdf5-pure/issues/383)). `Group::set_attr` and `Dataset::set_attr` write an attribute set the object header cannot hold into a fractal heap on `commit`, and rebuild an object that already stores its attributes in one, where both were refused ([#102](https://github.com/CramBL/hdf5-pure/issues/102)). Three types the API hands back without exporting are now exported, `Superblock`, `MessageType` and `BaseAddress`, which makes `File::superblock().base_address` usable as a number again after 0.40.0 changed it to a type with no public accessor ([#323](https://github.com/CramBL/hdf5-pure/issues/323)). **Breaking:** a variable-length string attribute reads back as a `VarLenString` variant rather than as the fixed-width variant of its charset and arity, `AttrValue::VarLenAsciiArray` is renamed `VarLenAsciiCharArray`, and `Superblock::serialize` is internal.

### Added

- `AttrValue::VarLenString`, `VarLenStringArray`, `VarLenAsciiString` and `VarLenAsciiStringArray` write an attribute in the standard variable-length string datatype (`H5T_STRING` with `STRSIZE = H5T_VARIABLE`), the one h5py and the reference C library write; `VarLenAsciiCharArray` still writes MATLAB's sequence-of-one-byte-strings shape ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).
- `Superblock`, `MessageType` and `BaseAddress` are exported from the crate root, so the types `File::superblock` and `Error::MissingMessage` hand back can be named in a signature and stored, not only read in place. Both `Superblock` and `MessageType` are `#[non_exhaustive]`, since the format they describe can grow ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- `Group::set_attr` and `Dataset::set_attr` write an attribute set the object header cannot hold — more than eight attributes, or one whose message overflows its 2-byte size field — into a fractal heap on `commit`, and rebuild an object that already stores its attributes in one, where both were refused before. A dataset or group created in place may carry such a set too. Moving an attribute that holds a repointable object reference out of the header is refused rather than putting it beyond the reference repointing a commit does, the heap a rebuild replaces is left for `repack` to reclaim, and dense (fractal-heap) *link* storage is still refused ([#102](https://github.com/CramBL/hdf5-pure/issues/102)).

### Fixed

- `File::superblock().base_address` reads as a number again, through `BaseAddress::get`. 0.40.0 changed that field from a `u64` to a type whose every accessor was crate-internal, so a consumer could see the value and do nothing with it ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- `FileBuilder::set_attr_committed` and its `DatasetBuilder`/`GroupBuilder` siblings stage a variable-length string attribute's strings into the file's global heap. Such an attribute was written with every element pointing at address 0, so the file was accepted, the attribute vanished from `attrs`, and the C library read it as empty ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).

### Changed

- **Breaking:** `Superblock::serialize` is internal: it was callable on the value `File::superblock` returns, and writes the v2/v3 format whatever superblock it is given. The `parse` constructors beside it are internal too ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- **Breaking:** A variable-length string attribute reads back as a `VarLenString` variant rather than as the fixed-width variant of its charset and arity, so writing the value back keeps the datatype it was found in ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).
- **Breaking:** `AttrValue::VarLenAsciiArray` is renamed `VarLenAsciiCharArray`, and its `type_name` is now `"vlen_ascii_char[]"`. It writes MATLAB's VLEN sequence of one-byte strings, not the variable-length ASCII string the old name implied — that is `VarLenAsciiStringArray` ([#383](https://github.com/CramBL/hdf5-pure/issues/383)).

## [0.40.0] - 2026-08-26

A read-write session writes far less and grows the file far less. The many small writes a commit or an in-place append makes are gathered into one per dirty page — a commit of eight staged dataset creations that issued 24 writes now issues 4 — and `FileAccessProperties::with_page_buffer_size` adds an opt-in write-back page buffer on top of that for long sessions, on a paged or unpaged file alike where `H5Pset_page_buffer_size` requires a paged one ([#288](https://github.com/CramBL/hdf5-pure/issues/288), [#308](https://github.com/CramBL/hdf5-pure/issues/308), [#357](https://github.com/CramBL/hdf5-pure/issues/357)). Files stop growing under delete-and-recreate churn on every space strategy: a paged file, a file that persists its free-space managers, `Dataset::append` and `BufferedAppender`, and repeated variable-length overwrites all draw on space an earlier commit freed instead of extending end-of-file ([#286](https://github.com/CramBL/hdf5-pure/issues/286), [#321](https://github.com/CramBL/hdf5-pure/issues/321), [#349](https://github.com/CramBL/hdf5-pure/issues/349), [#358](https://github.com/CramBL/hdf5-pure/issues/358)). Reads cost less as well — the typed whole-dataset readers decode a row window at a time rather than the whole dataset at once ([#289](https://github.com/CramBL/hdf5-pure/issues/289)) — and raising a `MetadataCacheConfig` budget no longer slows reads down, since the cache indexes its entries rather than scanning them ([#367](https://github.com/CramBL/hdf5-pure/issues/367)). What a cache budget bought can now be measured rather than guessed, through `File::metadata_cache_stats` and `Dataset::chunk_cache_stats` and their resets ([#353](https://github.com/CramBL/hdf5-pure/issues/353), [#356](https://github.com/CramBL/hdf5-pure/issues/356)). A `File::open_rw` commit is a stronger transaction: a delete and a create at the same path replace an object in one commit rather than two ([#305](https://github.com/CramBL/hdf5-pure/issues/305)), a dataset is validated as it is staged rather than at commit, a refused commit leaves the staged set whole and puts back any values it had already written over ([#316](https://github.com/CramBL/hdf5-pure/issues/316), [#344](https://github.com/CramBL/hdf5-pure/issues/344)), and object references are screened against the objects the same commit deletes or relocates instead of being left pointing into freed space ([#314](https://github.com/CramBL/hdf5-pure/issues/314), [#317](https://github.com/CramBL/hdf5-pure/issues/317), [#318](https://github.com/CramBL/hdf5-pure/issues/318), [#324](https://github.com/CramBL/hdf5-pure/issues/324)). On the writing side, `DatasetBuilder::with_ascii_strings` and its siblings write fixed-width string datasets ([#355](https://github.com/CramBL/hdf5-pure/issues/355)), `Dataset::write_staged` overwrites a variable-length-string dataset on any layout ([#321](https://github.com/CramBL/hdf5-pure/issues/321)), and `MatError::from_source` carries an embedding crate's own error type through the MAT builder's closures ([#378](https://github.com/CramBL/hdf5-pure/pull/378)). **Breaking:** a `Dataset` or `Group` handle stays usable across a commit, looking its object up again by path ([#351](https://github.com/CramBL/hdf5-pure/issues/351)); `File::group`, `Group::named_datatype` and path resolution refuse a name that reaches the wrong kind of object with the new `Error::NotAGroup` and `Error::NotANamedDatatype`, the way `H5Gopen` and `H5Topen` do ([#352](https://github.com/CramBL/hdf5-pure/issues/352), [#364](https://github.com/CramBL/hdf5-pure/issues/364), [#365](https://github.com/CramBL/hdf5-pure/issues/365)); and an attribute keeps the width it is stored at, so `AttrValue` gains 8-, 16- and 32-bit integer variants, `F32`, and sized fixed-width string variants ([#350](https://github.com/CramBL/hdf5-pure/issues/350), [#354](https://github.com/CramBL/hdf5-pure/issues/354), [#359](https://github.com/CramBL/hdf5-pure/issues/359)). Several classes of file that used to decode as sound are now refused: a numeric element wider than 8 bytes, which read back as its low 8 ([#361](https://github.com/CramBL/hdf5-pure/issues/361)); an Extensible or Fixed Array index whose checksum does not match ([#312](https://github.com/CramBL/hdf5-pure/issues/312)); and a dataset whose data lives in external files, which read, wrote and repacked as though it did not ([#293](https://github.com/CramBL/hdf5-pure/issues/293), [#331](https://github.com/CramBL/hdf5-pure/issues/331), [#336](https://github.com/CramBL/hdf5-pure/issues/336)). And an interrupted in-place append no longer leaves a dataset unreadable, where a value and the checksum covering it were published in two writes ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).

### Added

- A `File::open_rw` commit accepts a delete and a create at the same path, so replacing an object is one commit and one linearization point instead of two commits with the object missing in between. A dataset may replace a group or the reverse, replacing a group discards its subtree, and a staged edit to the object being replaced is still refused ([#305](https://github.com/CramBL/hdf5-pure/issues/305)).
- `Dataset::write_staged` overwrites a variable-length-string dataset with `with_vlen_strings`, on a contiguous, compact, or chunked (including filtered) layout alike. `with_path_references` is still refused ([#321](https://github.com/CramBL/hdf5-pure/issues/321)).
- Overwriting the same variable-length dataset again reclaims the global heap collection the previous overwrite placed, so rotating its strings in a session no longer grows the file by the whole payload every commit. Only collections the session itself placed are reclaimed ([#321](https://github.com/CramBL/hdf5-pure/issues/321)).
- `File::metadata_cache_stats` reports what a `MetadataCacheConfig` budget bought (hit rate, evictions, occupancy) so it can be measured rather than guessed, and `File::reset_metadata_cache_stats` clears the counters without evicting. The adaptive-resize policy of `H5AC_cache_config_t` is still not modeled ([#353](https://github.com/CramBL/hdf5-pure/issues/353)).
- `DatasetBuilder::with_ascii_strings` and `with_strings` write a fixed-width string dataset, sizing the datatype to the longest value and padding the rest, and `with_ascii_strings_sized`/`with_strings_sized` declare the width instead so a later, longer value still fits. A value the declared width cannot hold is refused with the new `FormatError::FixedStringTooLong` rather than truncated ([#355](https://github.com/CramBL/hdf5-pure/issues/355)).
- `Dataset::chunk_cache_stats` reports what a `ChunkCacheConfig` budget bought — hit rate, rejections, evictions, invalidations — so it can be measured rather than guessed, and `Dataset::reset_chunk_cache_stats` clears the counters without dropping the chunks. Read `rejections()` and `evictions()` together: a whole read reports the first and never the second, a row window the reverse. `H5Pset_cache`'s `rdcc_w0` is still not modeled ([#356](https://github.com/CramBL/hdf5-pure/issues/356)).
- `FileAccessProperties::with_page_buffer_size` gives a read-write session a write-back page buffer, the `H5Pset_page_buffer_size` analogue: 32 chunk appends and a commit issued 5 writes where the default gathering issued 188. It works on a paged or unpaged file alike, where `H5Pset_page_buffer_size` requires a paged one. Such a session marks the file in the superblock for its lifetime, so one that dies mid-flush leaves a file this crate and the C library both refuse rather than one that reads clean, and it repays that mark's two `fsync`s only over a long session ([#308](https://github.com/CramBL/hdf5-pure/issues/308), [#357](https://github.com/CramBL/hdf5-pure/issues/357)).
- `MatError::from_source` carries a caller's own error type through the builder's closures and `DataProducer` as the new `MatError::Source`, so an embedding crate recovers it with `Error::source` and `downcast_ref` instead of flattening it to text through `MatError::Custom` ([#378](https://github.com/CramBL/hdf5-pure/pull/378)).

### Changed

- A read-write session issues far fewer writes for the same file: one per dirty page, gathered across the many small writes a commit or in-place append makes. A commit of eight staged dataset creations that made 24 separate writes on a paged file now issues 4, with no change to the bytes on disk or to what a failed write leaves behind, and the SWMR writer gathers nothing ([#288](https://github.com/CramBL/hdf5-pure/issues/288)).
- The typed whole-dataset readers (`Dataset::read_f64` and its eight siblings) decode a row window at a time instead of the whole dataset at once, so peak memory is the values they return plus about a mebibyte rather than twice the dataset: an 8 MiB `f64` dataset peaked at 16.8 MB and now peaks at 10.1 MB ([#289](https://github.com/CramBL/hdf5-pure/issues/289)).
- A steady-state in-place append that is not write-gathered — an unbuffered session and every SWMR append — costs 5 writes where it cost 8, since each checksummed structure is now published in a single write ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).
- A row window takes only its own chunks out of the cached chunk index rather than the whole index, so reading a dataset window by window costs a constant per chunk instead of one allocation per chunk of the dataset per window ([#289](https://github.com/CramBL/hdf5-pure/issues/289)).
- **Breaking:** A value overwrite asking for more than a value overwrite — chunking or filters, an extensible shape, an attribute, or a fill value — is refused by `Dataset::write`/`Dataset::write_staged` as the write is staged rather than by the later `commit` ([#318](https://github.com/CramBL/hdf5-pure/issues/318)).
- **Breaking:** A dataset staged into a `File::open_rw` session is validated as it is staged, so a bad shape, a missing datatype, or a combination the engine cannot write is reported by the call that stages it — `Group::create_dataset`, `Dataset::write_staged`, or a `create_group_with` closure — rather than by the later `commit` ([#316](https://github.com/CramBL/hdf5-pure/issues/316)).

### Fixed

- **Breaking:** A numeric element wider than 8 bytes is refused with the new `FormatError::NumericElementTooWide` instead of decoding to its low 8 bytes, where a 9-byte value holding 2^64 read back as `0`. It covers the typed numeric readers on datasets and attributes alike: an attribute this refuses is omitted from `attrs()` and still reported in full by `attr_datatypes()`, and `read_raw` is unaffected ([#361](https://github.com/CramBL/hdf5-pure/issues/361)).
- A `MetadataCacheConfig` budget can be raised without slowing reads down: the metadata cache indexes its entries rather than scanning them, so a cached read costs the same at any budget instead of growing with the number of entries held. Reading 6,000 datasets from a streaming file at the 8 MiB budget the guide recommends was 26% slower than leaving the cache off, and is now about twice as fast as leaving it off ([#367](https://github.com/CramBL/hdf5-pure/issues/367)).
- **Breaking:** A `Dataset` or `Group` handle stays usable after a `File::commit`, looking its object up again by path rather than answering from the header the commit moved away from — which covers an edit made through another handle to the same object too. Reading through a handle onto a deleted object now fails the way opening it would, and one reached by `Dataset::dereference` reports the new `Error::StaleHandle` once anything but an immediate append has run, where it previously returned `Error::AppendInPlaceUnsupported` on an append and stale data on a read. Both types are now `Clone` ([#351](https://github.com/CramBL/hdf5-pure/issues/351)).
- **Breaking:** `File::group` and `Group::group` refuse a name that reaches something other than a group with the new `Error::NotAGroup`, the way `H5Gopen` does, instead of handing back a handle whose `attrs()` answered with that object's attributes. A commit that replaces a group with a dataset at the same path leaves a live handle reporting the same error ([#352](https://github.com/CramBL/hdf5-pure/issues/352)).
- **Breaking:** `Group::named_datatype` and `Group::named_datatype_references` refuse a name that reaches something other than a committed datatype with the new `Error::NotANamedDatatype`, the way `H5Topen` does, instead of answering with a dataset's own element type and reference count ([#364](https://github.com/CramBL/hdf5-pure/issues/364)).
- **Breaking:** A path resolved *through* something that is not a group reports `Error::NotAGroup` naming that object, the way a final component already did, instead of a `PathNotFound` reading "object header is not a group" for every such path: `File::group("a/b/c")` stopped by a dataset at `a/b` names `a/b`. A component that names nothing at all is still `FormatError::PathNotFound` ([#365](https://github.com/CramBL/hdf5-pure/issues/365)).
- **Breaking:** An integer attribute keeps the width it is stored at. `AttrValue` gains 8-, 16- and 32-bit variants with their arrays, so a 16-bit attribute reads back as `AttrValue::I16` rather than `I64` and one written from it takes two bytes on disk; a width with no Rust integer of its own still widens to 64-bit ([#350](https://github.com/CramBL/hdf5-pure/issues/350)).
- **Breaking:** A floating-point attribute keeps its width too, through the new `AttrValue::F32` and `F32Array`: a 4-byte attribute reads back as `F32` rather than `F64`, and `as_f64`/`to_f64s` read either width ([#354](https://github.com/CramBL/hdf5-pure/issues/354)).
- **Breaking:** A fixed-width string attribute keeps the width it is stored at. A padded slot read out of a file arrives as the new `AttrValue::AsciiStringSized` rather than `AsciiString`, and `AttrValue::ascii_string_sized` and its three siblings declare a width outright — the `H5T_C_S1` plus `H5Tset_size(N)` slot — so rewriting a 64-byte slot from a 2-byte value no longer shrinks the datatype. A slot sized to its own content still reads back as `AsciiString` ([#359](https://github.com/CramBL/hdf5-pure/issues/359)).
- A crafted link target in a file that has a userblock is refused rather than wrapping to a near-zero address (or panicking in a debug build): resolving a path through it added the superblock base address with an unchecked `+`. Adding or removing that base is now one checked operation, reported as `FormatError::OffsetOverflow` or the new `FormatError::AddressBelowBase` ([#323](https://github.com/CramBL/hdf5-pure/issues/323)).
- A file that persists its free space stops growing under delete-and-recreate churn: the extension and free-space-manager blocks each commit rewrites go into space an earlier commit freed rather than at end-of-file, where a `FileSpaceStrategy::FsmAggr` file gained about 590 bytes per commit forever. `FileSpaceStrategy::Page` already placed its own ([#358](https://github.com/CramBL/hdf5-pure/issues/358)).
- Reading a chunked dataset larger than its cache over and over keeps hitting the chunks it retained, where every second read used to hit nothing: a read now keeps the chunks it was served as well as those it placed, rather than handing them back one per miss to build a set the next read discarded. A read wanting chunks the cache does not hold still takes the slots ([#356](https://github.com/CramBL/hdf5-pure/issues/356)).
- A repeated whole read of a memory-backed chunked dataset walks its chunks in file order rather than the chunk index map's, so the chunks one read retains are the ones the next read reaches first instead of an arbitrary subset. Only the two streaming read loops were sorting ([#356](https://github.com/CramBL/hdf5-pure/issues/356)).
- `Dataset::append` and `BufferedAppender` allocate their chunks and index blocks out of the free space a prior commit left, instead of always extending end-of-file, so replacing an unlimited dataset no longer doubles the file. A file that persists its free-space managers still appends at end-of-file, since a reuse that does not grow the file would leave those records stale ([#349](https://github.com/CramBL/hdf5-pure/issues/349)).
- A `File::commit` refused before it publishes puts back the values it had already written over, so it leaves every dataset reading what it read before rather than applying a same-length `Dataset::write` while discarding the rest of the batch. The new `Error::CommitPartiallyApplied` reports the one case where that restore itself failed ([#344](https://github.com/CramBL/hdf5-pure/issues/344)).
- `File::create_with_options` refuses a userblock combined with `FileSpaceStrategy::Page` instead of writing the file and then failing the open it promised: free space is never persisted for a file with a non-zero base address, and a paged file without it cannot be opened read-write ([#308](https://github.com/CramBL/hdf5-pure/issues/308)).
- A `DatasetBuilder` given element data twice — `with_vlen_strings` or `with_path_references` followed by `with_raw_data`, say — no longer keeps the first call's staging, which patched heap addresses or resolved object addresses over part of the second call's bytes ([#321](https://github.com/CramBL/hdf5-pure/issues/321)).
- `repack` reproduces a dataset's filter pipeline in the order the source stored it, and keeps each filter's optional flag, instead of rebuilding it in this crate's own order with every filter but LZF marked mandatory. A source that ordered `fletcher32` before deflate checksummed the uncompressed bytes and arrived checksumming the compressed ones; the verbatim chunk-copy path was never affected ([#333](https://github.com/CramBL/hdf5-pure/issues/333)).
- `File::copy` and `File::copy_from` reproduce a dataset whose storage was never allocated, instead of refusing it as out of bounds. The copy declares the same empty storage rather than materializing the fill value it reads as, matching `repack` ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).
- `File::copy` and `File::copy_from` refuse a dataset whose data lives in external files (`H5Pset_external`) by name, instead of by an out-of-bounds error a valid never-written dataset also got. Deleting such a dataset now leaves its object header as dead space rather than reclaiming it ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).
- **Breaking:** `FileBuilder` refuses a dataset that stages element data under a zero-element shape, instead of writing a file the reference library refuses to open as corrupt (or, chunked, silently dropping the data). Staging no data for such a shape is unchanged ([#332](https://github.com/CramBL/hdf5-pure/issues/332)).
- **Breaking:** Reading a dataset whose data lives in external files (`H5Pset_external`) is refused with `FormatError::UnsupportedExternalStorage` instead of answering the fill value for every element of data this crate never looked at. Its shape, datatype, and layout still read ([#331](https://github.com/CramBL/hdf5-pure/issues/331)).
- **Breaking:** Overwriting or appending to such a dataset is refused too, instead of writing the new elements into the HDF5 file and leaving it disagreeing with the external files about where the data is — a write that reported `Ok` and was then visible to neither library. Attribute edits are unaffected, and deleting and recreating the dataset in one commit replaces it ([#331](https://github.com/CramBL/hdf5-pure/issues/331)).
- **Breaking:** `repack` refuses a dataset whose data lives in external files (`H5Pset_external`) instead of returning `Ok` having written a copy with none of it. Such a dataset carries the same undefined data address a never-written one does, and this crate does not follow the external files ([#293](https://github.com/CramBL/hdf5-pure/issues/293)).
- `repack` preserves a dataset whose storage was never allocated instead of writing its fill value out for every element, so a schema-only file stays one rather than arriving materialized at the size its shape declares. A resizable destination keeps the eagerly built chunk index every empty resizable dataset here gets ([#293](https://github.com/CramBL/hdf5-pure/issues/293)).
- Editing a group reachable through more than one hard link is refused rather than silently diverging its other links, which were left naming an object header the same commit freed — showing the pre-commit group until something reused the span, and unreadable to both libraries after. The same rule already governed a relocating dataset write ([#327](https://github.com/CramBL/hdf5-pure/issues/327)).
- An object reference already stored in a file keeps naming its target when a `File::open_rw` commit rewrites that target's header elsewhere, instead of being left pointing into the space the same commit freed. Addresses this crate cannot reach — inside a chunked dataset's chunks, a dense attribute, or a variable-length reference — are left as they were ([#324](https://github.com/CramBL/hdf5-pure/issues/324)).
- `Dataset::write_staged` refuses a builder carrying `with_path_references` rather than writing its unresolved placeholder zeros as final, which destroyed a working object-reference dataset and reported success. `with_reference_data`, whose addresses are resolved already, overwrites as before ([#318](https://github.com/CramBL/hdf5-pure/issues/318)).
- A refused `File::commit` leaves the staged set whole instead of discarding the edits it never objected to, so committing again repeats the same refusal rather than silently applying the rest of the batch. A staging call that refuses now stages nothing, including a `create_group_with` that had already recorded part of its subtree ([#316](https://github.com/CramBL/hdf5-pure/issues/316)).
- An object reference to a child of a group the same commit deletes is refused rather than written pointing into the space that delete reclaims; the deleted path itself was already refused ([#314](https://github.com/CramBL/hdf5-pure/issues/314)).
- **Breaking:** An object reference supplied as an *address* — `with_reference_data`, `with_raw_data`, or an in-file `File::copy` re-emitting a source's references — is screened at `commit` against the objects the same commit deletes, as one named by path already was; it was written as-is and left naming freed storage. A supplied address is refused as well when it names a header the commit rewrites elsewhere, and a datatype whose references this screen cannot read out of the element bytes — one wider than 8 bytes, a dataset-region reference, a variable length of them, or a compound holding any of those beside a reference it can read — is refused rather than written unscreened ([#317](https://github.com/CramBL/hdf5-pure/issues/317)).
- A `File::open_rw` commit that adds an object-reference dataset naming a path the *same commit* places no longer panics in a debug build on a file with a userblock. The commit preflight stood a zero in for an address not yet known, which underflowed the conversion to the stored, base-relative form ([#317](https://github.com/CramBL/hdf5-pure/issues/317)).
- **Breaking:** Extensible-Array and Fixed-Array chunk indexes have their checksums validated, matching the reference C library, so a torn or bit-rotted index is refused rather than decoded as sound — on reading it and on reclaiming its space when the dataset is deleted. A file with an already-corrupt index that this crate used to read now reports `ChecksumMismatch` ([#312](https://github.com/CramBL/hdf5-pure/issues/312)).
- Appending in place to a dataset whose Extensible Array header names an element wider than the fields it holds is refused rather than panicking. Only a corrupt or crafted file reaches it; the whole-file read path already refused the same header ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).
- An interrupted in-place append no longer leaves a dataset unreadable. A value and the checksum covering it are published in one write, where they were two that only became atomic when they shared a file-space page — so a dataset whose object header exceeded one page, such as a compound type with many members or a few string attributes, tore on every append ([#307](https://github.com/CramBL/hdf5-pure/issues/307)).
- A paged file (`FileSpaceStrategy::Page`) stops growing under delete-and-recreate churn: the free-space managers a commit rewrites are placed in free space rather than in a fresh page whose remainder nothing recorded, which cost a page per commit. A wholly free page can also be reopened for the other page type, so space a delete released is no longer stranded for the kind of data that released it ([#286](https://github.com/CramBL/hdf5-pure/issues/286)).

## [0.39.0] - 2026-08-17

Unallocated storage now reads as the dataset's fill value, matching the reference C library: a dataset created and never written was unreadable here, and the never-written chunks of a partly written one read as zeros whatever its fill value said ([#284](https://github.com/CramBL/hdf5-pure/issues/284)). A `File::open_rw` session can now create the empty chunked and extensible datasets that reading them required, so a schema-first writer declares its resizable datasets up front and grows them with `Dataset::append_staged`. Chunked writing is correct in more shapes: a dataset whose `maxshape` exceeds its shape in any dimension past the first is written with the chunk-index slot numbering the reference library expects, where it read the last rows as garbage in every rank above 1 — and this crate's own reader had the matching defect on files other libraries wrote — while an Extensible Array allocates only the data blocks its chunks land in, so a 256-chunk dataset with `maxshape [unlimited, 65536]` writes 376 KB rather than 7.9 MB ([#299](https://github.com/CramBL/hdf5-pure/issues/299)). A chunk covering the dataset's edge pads with the fill value rather than zeros, so a dataset later extended into that tail no longer reads zeros in every reader ([#296](https://github.com/CramBL/hdf5-pure/issues/296)). Scale-offset records the dataset's fill value in its filter parameters as the reference does: `DatasetBuilder::with_fill_value` reaches the encoder, `Dataset::append_staged` grows a scale-offset dataset written by libhdf5 or h5py where it was refused outright, and `repack` re-encodes one ([#287](https://github.com/CramBL/hdf5-pure/issues/287), [#297](https://github.com/CramBL/hdf5-pure/issues/297), [#300](https://github.com/CramBL/hdf5-pure/issues/300)). Opening a dataset or group by path no longer reads every other child of the group on the way: one lookup in a 1,024-child group allocates 23 KiB in 16 blocks where it took 310 KiB in 2,084 ([#228](https://github.com/CramBL/hdf5-pure/issues/228)). **Breaking:** `FormatError::NoDataAllocated` is gone, since unallocated storage is no longer an error.

### Added

- A `File::open_rw` session can create an empty (zero-element) chunked or extensible dataset, so a schema-first writer declares its resizable datasets up front and grows them with `Dataset::append_staged`; explicit `with_chunks` dimensions are required ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- `FormatError::UnreadableFillValue` reports a Fill Value message this parser cannot read *on a dataset that needs it* — one with storage that was never allocated. A dataset whose storage is fully allocated reads normally regardless of that message ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).

### Changed

- Opening a dataset or group by path (`File::dataset`, `File::group`) or by name (`Group::dataset`, `Group::group`) no longer reads every other child of the group on the way: one lookup in a 1,024-child group allocates 23 KiB in 16 blocks where it took 310 KiB in 2,084, so opening each member of a large group in turn costs the group once per open rather than once per member per open. A link whose *target* is malformed no longer fails a lookup of a different name, which parsing every link made it do; listing the group still reports it ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- `DatasetBuilder::with_vlen_strings` stages the caller's strings without copying each one first, so writing 32,768 of them allocates 6.6 MiB in 104 blocks where it took 12.8 MiB in 32,876 ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- `Dataset::append` reserves its batch instead of growing into it, so an append loop allocates about a quarter less: 512 appends of a 4 KiB chunk cost 6.7 MiB in 14,469 blocks where they took 8.7 MiB in 19,077 ([#228](https://github.com/CramBL/hdf5-pure/issues/228)).
- A scale-offset dataset records its fill value in the filter's parameters, matching the reference C library: `DatasetBuilder::with_fill_value` now reaches the encoder, which stores elements equal to it as a reserved code instead of widening each chunk's range to cover it. A dataset with no fill value records the library default of zero, as the reference does, so its chunks may pack one bit per element wider than before — and in the lossy float D-scale mode, an element within one decimal quantum of the recorded fill value now decodes as that value, where it decoded as itself ([#297](https://github.com/CramBL/hdf5-pure/issues/297)).

### Removed

- **Breaking:** `FormatError::NoDataAllocated` is gone. Unallocated storage is no longer an error, so nothing constructs it ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).

### Fixed

- `repack` re-encodes a scale-offset dataset whose filter records a fill value, where it refused one — which was essentially every such dataset written by libhdf5 or h5py, since the reference records a fill value on all of them. Only the lossy float mode is still refused ([#297](https://github.com/CramBL/hdf5-pure/issues/297)).
- Scale-offset rounds a scaled residual the way the reference's `llround` does. Rounding `x + 0.5` instead of `x` sent the largest value below one half to 1 rather than 0, which changed that element and, through the chunk's span, could widen every element beside it ([#300](https://github.com/CramBL/hdf5-pure/issues/300)).
- A chunked dataset whose `maxshape` exceeds its shape in any dimension past the first is written with the chunk-index slot numbering the reference C library expects, so the reference library reads it correctly. It read the last rows as garbage before, in every rank above 1: the index slots were numbered over the dataset's current chunk grid rather than its maximum one, and an Extensible Array's unlimited dimension was not rotated to the front. This crate's own reader had the matching defect on files written by other libraries ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A page of an Extensible Array data block that holds no chunk is recorded as present rather than absent, so the chunks in the pages after it are read. A `maxshape` wide enough to page a data block silently lost them ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- Both chunked readers step over an uninitialized page of an Extensible Array data block instead of stopping at it, so a file whose pages the reference C library filled out of order reads completely ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunked dataset naming a maximum extent of zero beside a non-zero current one is refused instead of panicking with a division by zero ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunked dataset of one stored chunk whose `maxshape` allows more gets a Fixed Array rather than the single-chunk layout, which the reference library asserts on (`H5D__single_idx_get_addr`) and aborts the reading process over ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A `maxshape` with more than one unlimited dimension is refused instead of written, where it produced a dataset the reference library cannot open at all ("already found unlimited dimension"). Its chunk index is a version-2 B-tree, which this crate does not write ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- An Extensible Array allocates only the data blocks its chunks land in, matching the reference C library, so a `maxshape` far wider than the shape costs what the chunks cost rather than what the gap between them does. A 256-chunk dataset with `maxshape [unlimited, 65536]` wrote 7.9 MB against libhdf5's 376 KB, and `repack` of such a file inflated it eleven-fold; the array's own header statistics disagreed with libhdf5 by a factor of forty ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A `maxshape` whose chunk index would spend more than 32 MiB on elements describing no chunk is refused. Stated in bytes rather than element slots, so a filtered index — whose elements are twice the width — gets the same budget as an unfiltered one ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunk numbered past the 8,589,934,580 element slots an Extensible Array can address is refused. Such a chunk landed in no block and was dropped, leaving a dataset this crate read as zeros and the reference library refused to read at all ([#299](https://github.com/CramBL/hdf5-pure/issues/299)).
- A chunk covering the dataset's edge holds the dataset's fill value in the slots past that edge, where it held zeros — so a `with_fill_value` dataset later extended into that tail read zeros instead of the fill value, in every reader. A Fill Value message this parser cannot read now refuses the write rather than padding with a guess ([#296](https://github.com/CramBL/hdf5-pure/issues/296)).
- Reading a dataset whose storage was never allocated returns its fill value instead of failing, for contiguous and chunked layouts alike, through whole reads and row windows. The never-allocated chunks of a partly written chunked dataset now read as the fill value too, where they read as zeros regardless of it. A dataset whose Fill Value message sets `H5D_FILL_TIME_NEVER` still reads as zeros, and `repack` of a never-written dataset writes the fill value out rather than preserving the unallocated storage ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- An empty **filtered** chunked dataset declared a chunk-index element too narrow for its own chunks, so filling it through the reference C library produced a file this crate read as truncated compressed data, and filling it through `Dataset::append` was refused outright. Datasets with at least one chunk are byte-identical to before ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- An empty **fixed-shape** chunked dataset is written with no chunk index, matching the C library, where the zero-entry Fixed Array it used to write made `H5Dget_num_chunks` fail on the dataset ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).
- `Dataset::append_staged` grows a scale-offset dataset written by the reference C library or h5py, which was refused with "scaleoffset: encoding with a defined fill value is not supported" — a mode those libraries record on *every* scale-offset dataset, not only ones given an explicit fill value. `repack` of such a dataset is still refused ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- A scale-offset chunk whose values span too much of the datatype to pack now writes the same header the C library writes for it. Decoders ignore the differing field on that path, so existing files read the same either way ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- `ScaleOffset::Integer(n)` with `n` equal to the datatype's bit width stores the chunk unfiltered, as the reference library does in both directions. It was applied on read only, so such a dataset was written packed and read back raw — a file this crate reported as truncated and the C library read as garbage. An `n` larger than the datatype is refused on both paths, where reading one ignored it ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- A scale-offset chunk whose header declares zero bits per element while its filter declares a fill value reads as that fill value, matching the C library, where it read as the header's minimum. No encoder emits that combination, so only a damaged chunk reaches it ([#287](https://github.com/CramBL/hdf5-pure/issues/287)).
- Requesting an extensible zero-element dataset without `with_chunks` is refused with `FormatError::InvalidChunkGeometry` instead of panicking ([#284](https://github.com/CramBL/hdf5-pure/issues/284)).

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

A `.mat` file this crate writes now opens under MATLAB's `load`. MATLAB's MAT reader refuses a version 3 superblock even on releases whose own libhdf5 reads one without difficulty, so `mat::Options` defaults to the HDF5 1.8 format; set `mat::Options::libver` to `LibVer::V110` for the previous one, which compression requires. Selecting the format is a general capability rather than a MAT one: `FileBuilder::with_libver_bounds`, `FileAccessProperties::with_libver_bounds` and `RepackOptions::with_libver_bounds` decide what a build, an edit session and a repack produce, `FormatError::LibverTooOldForContent` reports content a bound cannot express instead of silently upgrading the file, and a repack given no bound carries the source file's format forward rather than rewriting every file in the 1.10 format. Two smaller breaks come with it: `LibVer::WRITER_OUTPUT` is now `LibVer::WRITER_DEFAULT`, and an empty MAT value is written with `EmptyMarkerEncoding::DataAsDims`, matching MATLAB and matio, so it reads back empty under a plain `isempty` ([#247](https://github.com/CramBL/hdf5-pure/pull/247)). Committed (`H5Tcommit`) datatypes are read and written: `FileBuilder::commit_datatype` writes the named type object, `DatasetBuilder::with_committed_datatype` and the `set_attr_committed` methods name one, `Group::named_datatypes` lists them, and a dataset or attribute that uses one resolves to the type it names rather than to the zero-width time type it used to decode as, which is the shape netCDF-4 and h5py write for a user-defined type ([#254](https://github.com/CramBL/hdf5-pure/issues/254)). Attributes keep more of what the file holds: `repack` carries each attribute's own datatype and shape across instead of rebuilding it from an `AttrValue` ([#241](https://github.com/CramBL/hdf5-pure/issues/241)), an enumeration attribute reaches the caller instead of being dropped, which is where every h5py `np.bool_` attribute went ([#248](https://github.com/CramBL/hdf5-pure/pull/248)), and `Group::attr_datatypes` and `Dataset::attr_datatypes` give the on-disk `Datatype` that `attrs()` normalizes away ([#253](https://github.com/CramBL/hdf5-pure/pull/253)). `File::open_streaming` fetches a run of adjacent chunks in one read, which is what makes a file written a row at a time, one small chunk per row, read at a sensible speed ([#250](https://github.com/CramBL/hdf5-pure/pull/250)). Files written by earlier versions still read.

### Added

- `FileBuilder::with_libver_bounds` selects the on-disk format rather than only validating it: an upper bound of `LibVer::V18` writes the HDF5 1.8 format, and anything reaching 1.10 writes the 1.10 one ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FormatError::LibverTooOldForContent` reports content the requested bound cannot express, rather than silently upgrading the file — a chunked, filtered, or resizable dataset, or any file-space setting ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileAccessProperties::with_libver_bounds` holds an editing session to a format, so `File::open_rw` refuses an addition that would make the file need a newer library instead of making it silently ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `RepackOptions::with_libver_bounds` makes a repack's output format a guarantee. Without it, `repack` now carries the source file's format forward, upgrading only where the content leaves no choice — it used to rewrite every file in the 1.10 format ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `LibVer::WRITER_OLDEST` names the oldest format the writer produces ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `mat::Options::libver` sets the newest HDF5 format a `.mat` file may use, defaulting to `LibVer::V18`. `mat::MatError::CompressionNeedsNewerFormat` reports the one combination that cannot hold ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileBuilder::commit_datatype` and `GroupBuilder::commit_datatype` write a committed (`H5Tcommit`) datatype — a named type object several objects can share. `DatasetBuilder::with_committed_datatype` and the `set_attr_committed` methods name one; a name the file does not commit, or one whose type disagrees with the naming object's, fails the write ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `Group::named_datatypes`, `Group::named_datatype` and `Group::named_datatype_references` read committed datatype objects, which appear in neither `datasets()` nor `groups()` ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `Group::attr_datatypes` and `Dataset::attr_datatypes` give an attribute's on-disk `Datatype`, which `attrs()` normalizes away — the stored width, and the `enum[FALSE, TRUE]` that marks an h5py `np.bool_` as boolean rather than a one-byte integer. Every attribute is reported, including the ones `attrs()` omits for having no `AttrValue`, though an attribute's rank stays unexposed ([#253](https://github.com/CramBL/hdf5-pure/pull/253)).

### Changed

- **Breaking:** MAT files are written in the HDF5 1.8 format by default, so MATLAB can `load` them; MATLAB used HDF5 1.8.12 before R2021b and cannot open a version 3 superblock. Set `mat::Options::libver` to `LibVer::V110` for the previous format, which compression requires ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- **Breaking:** `mat::Options::default` uses `EmptyMarkerEncoding::DataAsDims`, matching what MATLAB and `matio` write, so an empty value reads back as empty under a plain `isempty` ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- **Breaking:** `LibVer::WRITER_OUTPUT` is now `LibVer::WRITER_DEFAULT`, since the writer no longer emits a single format. Its value is unchanged ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileBuilder::with_create_properties` resets the properties its argument does not carry, so a bound set before the call no longer decides the format of a file whose property list names no version ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `FileBuilder::write` creates the destination only once the writer has bytes for it, so a refused build leaves an existing file at that path untouched ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- An edit session writes a contiguous dataset's data-layout message in the format of the file it opened, so a `.mat` file edited through `File::open_rw` stays readable by MATLAB ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `File::open_streaming` fetches a dataset's adjacent chunks in one read instead of one read each, which is what makes a file written a row at a time — thousands of chunks of a few dozen bytes — read at a sensible speed. A read still fetches only the chunks it needs, and holds at most 256 KiB of them at a time ([#250](https://github.com/CramBL/hdf5-pure/pull/250)).

### Fixed

- An object header holding compact attributes declares how many it has, so `H5Oget_info().num_attrs` agrees with iteration instead of reporting zero — an `h5repack` round trip used to strip every `MATLAB_*` attribute from a `.mat` file without warning ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- A file with a userblock reads whole when it holds an object-header continuation block or dense link storage, which the C library writes and this crate does not; `File::open` used to fail outright on such a file ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- An empty MAT value carries `MATLAB_class` and `MATLAB_empty` and nothing else, matching MATLAB, and both emitters agree on its dimensions — including for an empty `Matrix`, which one of them wrote as a plain zero-element dataset ([#247](https://github.com/CramBL/hdf5-pure/pull/247)).
- `repack` keeps each attribute's own datatype and shape instead of rebuilding it from an `AttrValue`, which widened every integer and float to 64 bits, turned a variable-length string into a fixed-width one, and flattened a rank-2 attribute to rank 1. Enumeration, compound, bit-field and opaque attributes are now carried across rather than refused; a reference attribute still is ([#241](https://github.com/CramBL/hdf5-pure/issues/241)).
- An enumeration attribute reaches the caller, decoded through its integer base type as enum dataset data already is; `attrs()` skipped it before, so every `np.bool_` attribute in an h5py-written file — stored as `enum[FALSE, TRUE]` — went missing without a trace. The member names have no `AttrValue` to live in, so the codes are what survives ([#248](https://github.com/CramBL/hdf5-pure/pull/248)).
- A committed (`H5Tcommit`) datatype resolves to the type it names, so `Dataset::datatype`, `Dataset::read_*`, `attrs()` and `attr_datatypes()` report a named type instead of the zero-width time type its stored reference used to decode as — the shape netCDF-4 and h5py write for a user-defined type ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `repack` reproduces a committed datatype: the named object is recreated and every dataset and attribute that used it names the same object, instead of producing an output the C library could not read *any* attributes from. Dropping a committed type something still names is refused ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).
- `FormatError::UnsupportedSohmReference` refuses a message stored in the shared-message (SOHM) heap instead of following its heap id as an address ([#254](https://github.com/CramBL/hdf5-pure/issues/254)).

## [0.33.0] - 2026-08-02

A file whose superblock marks it as held by a writer is refused rather than opened: `File::open`, `open_streaming`, `open_rw`, `open_swmr_writer` and `repack` report the new `Error::FileMarkedInUse`, which is the check `H5Fopen` makes of the same byte — a file a crashed SWMR writer left flagged used to open, and `open_rw` used to edit it in place under a writer the file still recorded ([#245](https://github.com/CramBL/hdf5-pure/issues/245)). `File::open_swmr` follows such a file instead of refusing it, since that pairing is what the flag exists for, and `File::from_bytes` does not consult the byte at all, so a caller holding the bytes can still read a flagged file on a read-only mount, where the `File::clear_swmr_flag` recovery (the `h5clear -s` equivalent) cannot get the write access it needs. The check applies to version-3 superblocks, which is where the C library applies it, and `open_swmr_writer` now requires one for the same reason libhdf5 does. Two smaller changes come with it: a read-write open validates the superblock before it builds its backing, so refusing a mirrored file no longer reads the whole file first, and a version-1 superblock's status flags and chunk B-tree K are read from the offsets the C library writes them to rather than swapped. Files written by earlier versions still read.

### Added

- `Error::FileMarkedInUse` reports an open refused by the superblock's status-flags byte. Unlike `Error::FileLocked` it outlives the process that set it, so it means a writer is active *or* one exited without closing the file ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).

### Changed

- **Breaking:** `File::open`, `File::open_streaming`, `File::open_rw`, `File::open_swmr_writer` and `repack` refuse a file whose superblock marks it as held by a writer, matching `H5Fopen`; a file left flagged by a crashed SWMR writer used to open, and `open_rw` used to edit it in place. `File::open_swmr` follows such a file as before, `File::from_bytes` does not check, and `File::clear_swmr_flag` recovers a stale flag ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).
- **Breaking:** `File::open_swmr_writer` requires a version-3 superblock, as the C library does, rather than version 2 or 3 ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).
- A read-write open that refuses a file no longer reads the file first: `File::open_rw` validates the superblock through the handle and builds its backing only once nothing can refuse it, so refusing a mirrored file costs a few bounded reads rather than a whole-file copy ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).

### Fixed

- A version-1 superblock's status flags and chunk B-tree K are read from the offsets the C library writes them to; the two were swapped, so `File::superblock()` reported a v1 file's `indexed_storage_internal_node_k` as its `consistency_flags` and vice versa ([#245](https://github.com/CramBL/hdf5-pure/issues/245)).

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

`DatasetBuilder::with_lzf` writes h5py's LZF filter (id 32000), a fast lossless compressor h5py reads without any plugin installed, and LZF datasets — including h5py-written ones — can be read, edited in place, and repacked ([#231](https://github.com/CramBL/hdf5-pure/pull/231)). The MAT serde writer honors `Options::null_policy`, which it previously ignored: `None`, `()`, a unit struct, and `Value::Null` now write MATLAB `struct([])` rather than dropping the field, so MATLAB code can reference it unconditionally — at the cost that a Rust reader relying on `#[serde(default)]` for a non-`Option` field now finds the field present. `NullPolicy::Omit` writes the previous output. Two new options join it: `Options::unit_variant_encoding` writes a fieldless enum variant as its name or as its declaration index, and `Options::empty_sequence_policy` picks `[]` or `{}` for a sequence that turned out to be empty ([#232](https://github.com/CramBL/hdf5-pure/pull/232)). That writer also collects a flat numeric or complex sequence packed, one element wide, instead of one 56-byte value per element — serializing a `Vec<f64>` cost 7x its own size and a `Vec<ComplexI16>` 14x, both now about 1x, with the same bytes out — and the empty-value paths its two emitters take now agree with each other. On the filter side, requesting a filter that another would displace is refused instead of silently dropped, `FormatError::DecompressionError` is removed in favor of the `FilterError` that shuffle, scale-offset and LZF already reported, and decoding a chunk reserves memory against what its own stream could expand to rather than against a size the file merely declares ([#233](https://github.com/CramBL/hdf5-pure/issues/233)). Files written by earlier versions still read.

### Added

- `DatasetBuilder::with_lzf` writes h5py's LZF filter (id 32000), a fast lossless compressor h5py reads without any plugin, and LZF datasets — including h5py-written ones — can be read, edited in place, and repacked; combining LZF with deflate is refused ([#231](https://github.com/CramBL/hdf5-pure/pull/231)).
- `Options::unit_variant_encoding` picks whether a fieldless enum variant is written as its name (`UnitVariantEncoding::Name`, the default and previous behavior) or as its declaration index in a `uint32` (`UnitVariantEncoding::Index`). Serde hands the serializer both, so either is reachable; `Index` suits a reader that already expects the integer, but an index cannot be interpreted without the schema that fixes the ordering, so `Name` is the better default. The index is serde's, counted from zero, so an explicit discriminant (`enum E { A = 5 }`) is not the number that reaches the file.
- `Options::empty_sequence_policy` picks the MATLAB class of a sequence that turned out to be empty: `EmptySequencePolicy::DoubleArray` (the default and previous behavior) writes `[]`, `Cell` writes `{}`. An empty `serialize_seq` carries no element type, so there is nothing to infer the class from; `Cell` is right when the sequence would have held structs.
- `NullPolicy::Omit` selects the pre-0.30 serde behavior of dropping the field.

### Changed

- **Breaking:** requesting a filter that another would displace is refused instead of silently dropped. `with_zfp` alongside `with_shuffle`, `with_deflate` or `with_lzf`, and `with_scale_offset` alongside `with_shuffle`, now fail with a filter error naming both; only the scale-offset/ZFP clash did before ([#233](https://github.com/CramBL/hdf5-pure/issues/233)).
- **Breaking:** `FormatError::DecompressionError` is removed. A deflate stream that fails to decode reports `FormatError::FilterError`, which shuffle, scale-offset and LZF already used, so "this chunk did not decode" is one match arm rather than two ([#233](https://github.com/CramBL/hdf5-pure/issues/233)).
- The serde writer collects a flat numeric or complex sequence packed, one element wide, instead of one 56-byte `MatValue` per element. A `Vec<f64>` cost 7x its own size while being serialized and a `Vec<ComplexI16>` cost 14x; both now cost about 1x. A sequence whose elements do not all agree spills to the previous per-element form at the point they diverge, so cell arrays and matrices built from equal-length rows are unaffected. Same bytes out.
- The serde writer gives an empty cell array MATLAB shape `[0, 1]` rather than `[0, 0]`, which is the `[n, 1]` rule it already applies to a non-empty one, and matches what an empty BEVE array converts to. `isempty` holds either way; `size(x, 2)` no longer changes as a list empties out.
- **Breaking:** the serde writer honors `Options::null_policy`, which it previously ignored. `None`, `()`, a unit struct, and `serde_json::Value::Null` now write MATLAB `struct([])` by default rather than dropping the field, so `isfield` reports true and MATLAB code can reference the field unconditionally with `isempty(fieldnames(x))`. Set `NullPolicy::Omit` for the old output.

  Note the reader-side consequence, which cuts both ways. A field that is present no longer needs `#[serde(default)]`, but a reader that *relies* on `#[serde(default)]` for a non-`Option` field now fails on newly written files: the field is present with a struct value rather than missing, so the default is never consulted. `struct([])` reads into `Option<T>`, `Vec<T>`, `serde_json::Value` and `()`, and reports a type error for a bare scalar, `String`, struct or map. Give such a field type `Option<T>`, or write with `NullPolicy::Omit`.

### Fixed

- Decoding a chunk no longer reserves memory against a size the file merely declares: deflate and LZF reserve at most what their own stream could expand to, and scale-offset refuses a chunk claiming more bytes than the chunk holds. A small file declaring an enormous chunk previously drove the allocation before anything had checked the claim ([#233](https://github.com/CramBL/hdf5-pure/issues/233)).
- An empty `Matrix<T>` serializes under `EmptySequencePolicy::Cell`. The policy was applied where the `Matrix` sentinel's own `data` field is lowered, so an empty matrix became a cell array and the sentinel handler had no vector left to recover the element class from, failing with "Matrix::data must be a Vec<T>, got cell array" for every empty numeric and complex matrix. The policy is now pinned for that field, since it is internal plumbing rather than a sequence the caller wrote; a caller's own empty sequence still follows it.
- An `Options` persisted by an earlier version still deserializes. The two fields added this release carried no serde default, so an older serialized `Options` failed with a missing-field error. The default is now declared on the struct, so future additions stay loadable without anyone having to remember.
- `NullPolicy::Error` refuses a null at the file root, which it previously did not. The root serializer never consulted `null_policy` at all, so a root `None`, `()` or `Value::Null` wrote a zero-variable file even under the one policy whose whole purpose is to report nulls. It now routes through the same lowering as every other slot, and reports the same error. The other two policies are unchanged and now documented rather than incidental: the root names no slot, so a null there is an empty variable *namespace*, and both still write a valid file with no variables, byte-identical to what an empty root map or a fieldless struct writes. Note such a file does not read back as `None`, since the deserializer presents the root as a struct.
- A fieldless enum variant reads back from either encoding. The deserializer accepted only a name, so a file written under the new `UnitVariantEncoding::Index` could not be read by this crate at all; it now resolves an index too, from whatever numeric class carries it, which also lets an index typed in MATLAB (a `double`) be read.
- An empty marker has the same element type from both serde emitters. `to_bytes` wrote a `uint64` zero-element dataset for `struct([])` where `to_bytes_with_options` under the same default options wrote `uint8`, so the two produced different files for one value. Both now write `uint64`, matching the reference library's own empty marker, and the choice is made in one place rather than duplicated. This also changes an empty cell array written through `MatBuilder` under `EmptyMarkerEncoding::ZeroElement` from `uint8` to `uint64`; the class attributes that identify it are unchanged, and the reference library reads either.

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

Attributes lose their size ceiling: one too large for an object-header message selects fractal-heap storage on its own, and one too large even for a managed heap object becomes a *huge* object, so `FormatError::DenseAttributeTooLarge` is gone rather than refusing a shape the reference library writes ([#195](https://github.com/CramBL/hdf5-pure/issues/195)). Three defects on that path are fixed with it: a variable-length attribute stored in a fractal heap silently lost its values, dense attributes were unreadable in a file with a userblock, and reading many of them was quadratic in their number ([#214](https://github.com/CramBL/hdf5-pure/pull/214), [#195](https://github.com/CramBL/hdf5-pure/issues/195)). The property-list types are renamed to say what they stand in for — `FileAccessProperties`, `FileCreateProperties`, `DatasetAccessProperties` — and checking each one's settings against the official group pages caught two properties documented under the wrong class ([#198](https://github.com/CramBL/hdf5-pure/issues/198)). Breaking, but every break is a one-line call-site edit, and deprecated aliases keep 0.25.0 code compiling for this cycle.

### Added

- An attribute of any size is written rather than refused: one too large for an object-header message selects fractal-heap storage on its own, and one too large for a managed heap object becomes a *huge* object. A name, datatype, or dataspace longer than the 2-byte field describing it is still refused, as the new `FormatError::AttributeFieldTooLong` ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- `FormatError::TooManyHugeDenseAttributes` names the one bound the new huge-object path adds, on how many such attributes a single object may carry ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- `FormatError::UnexpectedHugeObjectBTree` refuses a fractal heap whose huge-objects B-tree is not the record layout this reader decodes, instead of reading an object ID out of another field's bytes ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).

### Changed

- **Breaking:** `FileAccessOptions`, `FileCreateOptions`, and `DatasetAccessOptions` are renamed `FileAccessProperties`, `FileCreateProperties`, and `DatasetAccessProperties`, so that a type standing in for a whole HDF5 property list says so in its name; `FileBuilder::with_create_options` and `File::access_options` follow suit. Deprecated aliases under the old names keep 0.25.0 code compiling for this cycle, and `H5Pset_libver_bounds` is now documented as the file-access property it is ([#198](https://github.com/CramBL/hdf5-pure/issues/198)).

### Fixed

- Reading an object with many huge dense attributes now parses the heap's huge-object index once per walk instead of once per attribute, so the read is no longer quadratic in their number (1,600 such attributes drop from ~164 ms to ~75 ms) ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).
- A variable-length attribute stored in a fractal heap keeps its values. Written into the heap before its global-heap references had addresses, it read back with its values lost, silently ([#214](https://github.com/CramBL/hdf5-pure/pull/214)).
- Dense (fractal-heap) attributes are readable in a file with a userblock, through `File::open`, `File::open_streaming`, `repack` and `copy` alike. The heap address was taken as an absolute file offset rather than one relative to the base address, so the read failed on a file the reference C library reads correctly ([#214](https://github.com/CramBL/hdf5-pure/pull/214)).

### Removed

- **Breaking:** `FormatError::DenseAttributeTooLarge` is gone, as no attribute size is refused any more ([#195](https://github.com/CramBL/hdf5-pure/issues/195)).

## [0.25.0] - 2026-07-26

An API-consolidation release. File properties are now reusable values rather than scattered function variants: one `FileAccessOptions` (the `fapl` analogue) carries cache budgets and the locking policy to every open, and the new `FileCreateOptions` (the `fcpl` analogue) lets a file layout be defined once and applied to any write, including through `File::create_with_options`. Two long-standing gaps fell out of that work — the read-write mirror backend was silently discarding access options, and the bounded backend always locked regardless of policy. Public types the HDF5 format will keep growing are now `#[non_exhaustive]`, so a future datatype class, reference kind, or MATLAB class is an additive change instead of a breaking one, and a test guards each seal against silent removal. Enumerations can finally be built over any integer base type. Two defects from 0.24.0 are fixed: `repack` corrupting a dataset whose datatype contains a variable-length or reference member, and a hang when reading a file from inside a builder closure. The release carries a number of breaking changes, all listed below; most are one-line call-site edits.

### Added

- `FileCreateOptions` collects the file-creation properties (userblock, file-space strategy and page size, library-version bounds) into one reusable value — the `fcpl` analogue — applied with `FileBuilder::with_create_options` or the new `File::create_with_options`, which mirrors `H5Fcreate(name, flags, fcpl, fapl)` and is the first way to reach creation properties from the owned-handle path ([#205](https://github.com/CramBL/hdf5-pure/issues/205)).
- `FileAccessOptions::with_locking` carries the file-locking policy, and `File::open_rw_with_options` / `open_swmr_writer_with_options` accept the options, so one `fapl` value now serves every open. The mirror backend honors the configured chunk cache (it previously discarded access options), and `open_rw_bounded*` honors the locking policy instead of always locking ([#204](https://github.com/CramBL/hdf5-pure/issues/204)).
- `EnumTypeBuilder::with_base` builds an enumeration over any integer base type (not just `i32`/`u8`), with `raw_value` for a member given as its raw little-endian bytes and `i64_value` for wider integers ([#208](https://github.com/CramBL/hdf5-pure/issues/208)).
- Chunked, filtered, and resizable variable-length datasets now write: `DatasetBuilder::with_vlen_strings` accepts `with_chunks`, `with_deflate`, and `with_maxshape`, and `repack` reproduces such a dataset with its chunk geometry, filters, and unlimited dimension intact. Adding one to an existing file through the in-place edit engine is still refused ([#109](https://github.com/CramBL/hdf5-pure/issues/109)).
- `Dataset::write_staged` overwrites a dataset through its full `DatasetBuilder`, the builder-level counterpart of `Dataset::write` for element kinds that are not `H5Element` ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- `Group::create_group_with` stages a new group configured through a `StagedGroup` closure, so a group's attributes and children land with its creation (`set_attr` cannot reach it, since it needs a group that already resolves). `create_group(name)` is unchanged ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).

### Fixed

- Reading the same `File` from inside a builder closure (`Group::create_dataset`, `create_group_with`, `Dataset::write_staged`, `append_staged`) no longer hangs the process. The closures now configure a builder off the session lock, so a staged dataset may depend on data already in the file; it sees the file as it was before the call, since staged edits resolve only on `commit` ([#200](https://github.com/CramBL/hdf5-pure/issues/200)).
- `repack` no longer corrupts a dataset whose datatype *contains* a variable-length member, an object-reference member, or both — such as a compound with a variable-length string field. The embedded addresses were copied verbatim and left pointing into the source file, producing a destination this crate read back without complaint and the reference C library could not read at all; they are now rewritten like top-level ones ([#201](https://github.com/CramBL/hdf5-pure/issues/201)).

### Changed

- **Breaking:** `EnumTypeBuilder::build` returns `Result<Datatype, FormatError>`; it now refuses a non-integer base type, a member value too wide for the base, and raw bytes whose length disagrees with it, instead of writing a malformed datatype ([#208](https://github.com/CramBL/hdf5-pure/issues/208)).
- **Breaking:** `EnumMember` is now `#[non_exhaustive]`, which `EnumTypeBuilder`'s arbitrary-base support makes free — build enumerations through the builder rather than a literal ([#208](https://github.com/CramBL/hdf5-pure/issues/208)).
- **Breaking:** three refusals on the owned write path now report a more specific error: an unaligned SWMR append gives `SwmrAppendUnsupported` instead of a `ChunkedReadError`, an ineligible immediate append gives `AppendInPlaceUnsupported` instead of `AppendUnsupported`, and a missing edit target gives `PathNotFound` when the handle is resolved instead of `AppendUnsupported`/`EditUnsupported` at commit ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).
- **Breaking:** `AttrValue`, `DType`, `Datatype`, `ReferenceType`, `LibVer`, `Object`, `CompoundMember`, `FileSpaceInfo`, `VerifyResult`, `mat::MatClass`, and the four `mat::opaque` decode structs are now `#[non_exhaustive]`, so a new datatype, reference kind, library-version bound, or MATLAB class no longer breaks callers. Add a `_` arm when matching a read-back value; constructing the existing variants is unaffected, including `Datatype` literals for types this crate has no constructor for ([#206](https://github.com/CramBL/hdf5-pure/pull/206)).
- **Breaking:** `RepackOptions` is now built only through `new` and `drop_path`, matching `FileAccessOptions`; the `drop` field is private and readable with `RepackOptions::drop_paths` ([#206](https://github.com/CramBL/hdf5-pure/pull/206)).

### Removed

- **Breaking:** `File::open_rw_with_locking` is gone; the policy is an option on the open — `File::open_rw_with_options(path, FileAccessOptions::new().with_locking(..))` ([#204](https://github.com/CramBL/hdf5-pure/issues/204)).
- **Breaking:** `Datatype::parse` and `Datatype::serialize` are now crate-internal. Read a dataset's type with `Dataset::datatype` and pass a `Datatype` to `DatasetBuilder::with_dtype`, which encodes it; `Datatype::type_size` stays public ([#206](https://github.com/CramBL/hdf5-pure/pull/206)).
- **Breaking:** `FormatError::ChunkedVlenStringUnsupported` is gone, as nothing refuses those datasets any more ([#109](https://github.com/CramBL/hdf5-pure/issues/109)).
- **Breaking:** `AppendWriter`, `SwmrWriter`, and `EditSession`, deprecated since 0.22.0, are gone. Use `File::open_rw` (or `File::open_swmr_writer`) with owned `Dataset` and `Group` handles; `File::open_rw_with_options` with `FileAccessOptions::with_locking` replaces `AppendWriter::open_with_locking` and `File::clear_swmr_flag` replaces `SwmrWriter::clear_swmr_flag`. The former `EditSession` methods map to `Dataset::append`/`append_staged`/`write`/`write_staged`/`set_attr`/`remove_attr`, `Group::create_group`/`create_dataset`/`delete`/`set_attr`, and `File::copy`/`copy_from`/`space_accounting`, with an object staged in an uncommitted batch reachable only through `create_group_with` ([#148](https://github.com/CramBL/hdf5-pure/issues/148)).

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

Two file-space fixes from documenting and fuzz-testing the paged and persisted surface ([#178](https://github.com/CramBL/hdf5-pure/issues/178)): a fresh `persist = true` file with a non-paged strategy now records a defined end-of-allocation, so an assertion-enabled build of the reference C library opens it instead of aborting, and the `File::open_rw_bounded` refusal for a non-persisting paged file now advises the right recovery. Non-breaking patch.

### Fixed

- A fresh file written with `persist = true` and a non-paged file-space strategy (`FsmAggr`/`Aggr`/`None`) now records a defined end-of-allocation in its File Space Info message instead of the undefined sentinel, so an assertion-enabled build of the reference C library opens it instead of aborting; release builds already tolerated it ([#178](https://github.com/CramBL/hdf5-pure/issues/178)).
- The refusal when opening a *non-persisting* paged file with `File::open_rw_bounded` no longer points at `File::open_rw` (which also refuses a paged file); it now advises recreating the file with `persist = true`, the way to grow a paged file in place ([#178](https://github.com/CramBL/hdf5-pure/issues/178)).

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

`EditSession` now edits files that carry a **userblock** (non-zero base address), such as MATLAB v7.3 `.mat` files: it reads and writes addresses relative to the base and preserves the userblock bytes, so every edit works — value overwrites, additions, relocating overwrites of every layout with old storage reclaimed, object deletion, in-file and cross-file copy, group creation, and compact attributes — with only cross-file copy from a userblock *source* still refused. Also fixes reading and repacking a chunked dataset from such a file. Additive minor bump.

### Added

- `EditSession` now opens and edits files that carry a **userblock** (non-zero base address), such as MATLAB v7.3 `.mat` files: it reads and writes addresses relative to the base and preserves the userblock bytes verbatim. Every edit is supported — value overwrites, additions, relocating overwrites of every layout (with old storage reclaimed), object deletion, in-file and cross-file copy, group creation, and compact attributes; only cross-file copy from a userblock *source* is still refused ([#104](https://github.com/CramBL/hdf5-pure/issues/104)).

### Fixed

- Reading and repacking a **chunked dataset from a file with a userblock** (non-zero base address) now works; previously the base address was applied only to contiguous data, so chunked reads from such a file failed ([#104](https://github.com/CramBL/hdf5-pure/issues/104)).

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

Centers on `repack`: it now copies compressed chunks **verbatim** (so lossy filters survive byte-exact) and runs **fully out-of-core**, and gains variable-length-string support. Also adds in-place dataset-value overwrite, dense-attribute and cross-file object copy, in-place addition of chunked/filtered/extensible datasets, and free-space reclaim for chunked deletes; plus reader hardening (a multi-filter chunk-mask corruption fix, sub-byte integer precision, decompression-bomb bounds, and safer B-tree/heap refusals). Additive minor bump.

### Added

- Repack now copies a chunked dataset's compressed chunks **verbatim** instead of decompressing and re-compressing them, eliminating the per-dataset decompression blowup and the decompress→recompress round-trip, so **lossy** filters now survive byte-exact — float D-scale scale-offset, ZFP, SZIP, and even filters this crate cannot itself apply ([#82](https://github.com/CramBL/hdf5-pure/issues/82), [#84](https://github.com/CramBL/hdf5-pure/issues/84), [#85](https://github.com/CramBL/hdf5-pure/issues/85)). The verbatim path covers a fully-allocated chunk grid; a sparse chunked or a contiguous/compact filtered dataset still re-encodes and refuses a lossy filter by name.
- Repack is now **fully out-of-core**, closing [#82](https://github.com/CramBL/hdf5-pure/issues/82): it streams the source (`File::open_streaming`) and the output (`FileBuilder::finish_to`, a `std::io::Write` sink), so peak memory is bounded by one chunk plus the file's metadata regardless of dataset size. This extended the streaming reader to also read attributes (compact, shared, dense, and VL-string) and traverse v1 symbol-table groups ([#27](https://github.com/CramBL/hdf5-pure/issues/27)).
- Variable-length string dataset writing and repack: `DatasetBuilder::with_vlen_strings(&[&str])` writes a contiguous VL UTF-8 string dataset (1D, or ND via `with_shape`), matching the C library's `H5Tvlen_create(H5T_C_S1)` layout so the C library and h5py read it back. Repack now round-trips contiguous/compact VL-string datasets, preserving charset, padding, the null-vs-empty distinction, embedded NULs, and non-UTF-8 bytes; chunked, filtered, or resizable VL-string datasets and non-string VL datatypes are still refused by name ([#83](https://github.com/CramBL/hdf5-pure/issues/83)).
- In-place overwrite of dataset values: `EditSession::write_dataset(path)` replaces an existing contiguous or compact dataset's values (HDF5's `H5Dwrite` whole-dataset write), returning the same `DatasetBuilder` as `create_dataset`. The replacement must match the on-disk datatype and shape; a same-length contiguous overwrite writes straight into the existing data block, while a length change or a compact dataset relocates the header like an addition. Chunked and filtered datasets, and a relocating overwrite of a multiply-hard-linked dataset, are refused by name ([#79](https://github.com/CramBL/hdf5-pure/issues/79)).
- Object copy now reproduces dense (fractal-heap) attribute storage: above the compact threshold of 8 attributes HDF5 stores attributes in a fractal heap indexed by a B-tree v2, and `EditSession::copy` and `copy_from` previously refused such objects. They now read the source attributes and re-emit them into a fresh destination-local heap, same-file and cross-file ([#87](https://github.com/CramBL/hdf5-pure/issues/87)). For now a single direct block is emitted: a set too large for one direct block is refused by name, as is a cross-file dense set whose values are variable-length or reference data.
- Cross-file object copy: `EditSession::copy_from` copies a dataset or whole group subtree out of a *separate* open `File` into the file being edited — the cross-file form of HDF5's `H5Ocopy`, alongside the same-file `EditSession::copy` ([#78](https://github.com/CramBL/hdf5-pure/issues/78)). The source is read and validated eagerly, so it returns a `Result`. Because the copy is verbatim, it refuses by name anything whose stored bytes embed a source-file address — variable-length and reference data or attributes, and any shared header message. The source must be a buffered file (`File::open` / `File::from_bytes`, not `open_streaming`) with 8-byte offsets and no userblock.
- Free-space reclaim for chunked datasets on in-place delete: deleting a chunked dataset (or a group whose subtree contains one) now returns its chunk data blocks and chunk-index structure to the free list, reused by a later commit and truncated away when the freed run reaches end-of-file, where previously a chunked dataset's storage was left as dead bytes ([#77](https://github.com/CramBL/hdf5-pure/issues/77)). Covers single-chunk, implicit, fixed array, extensible array, and v1 B-tree indexes; a v2 B-tree index, an out-of-bounds or overlapping span, or VL global-heap data is left in place rather than risk freeing live bytes.
- In-place add of chunked, filtered, and extensible datasets: `EditSession::create_dataset` now accepts `with_chunks`, the writer's filters (`with_deflate`, `with_shuffle`, `with_fletcher32`, `with_scale_offset`, `with_zfp`), and `with_maxshape` (optionally unlimited) — previously only contiguous, unfiltered datasets ([#76](https://github.com/CramBL/hdf5-pure/issues/76)). The added object header is byte-identical to a freshly written one, and the prior root stays intact until the superblock is repointed last.

### Fixed

- Reading a virtual (VDS) dataset now fails with a clear `FormatError::UnsupportedVirtualLayout` instead of a misleading `UnsupportedVersion(0)` (which rendered as "unsupported superblock version: 0"); VDS reading is tracked as a planned feature ([#111](https://github.com/CramBL/hdf5-pure/issues/111)).
- Multi-filter chunks where only *some* filters were skipped for a chunk (the per-chunk `filter_mask`, e.g. shuffle+gzip on an incompressible chunk that the C library stores shuffled but not deflated) now have the surviving filters reversed instead of being returned raw, fixing silent value corruption on spec-valid files ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- Integers with sub-byte precision or a non-zero bit offset (`H5Tset_precision` / `H5Tset_offset`) now decode correctly in the dataset and attribute readers — masked to the significant bits and sign-extended at the precision boundary — instead of returning the raw stored word with its padding bits; compound fields with such layouts are still refused by name ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- A malformed v1 B-tree with a cyclic or pathologically deep internal node — in either the chunk index or a group's symbol table — now errors instead of recursing until the stack overflows and aborts the process; traversal is bounded by a depth cap ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- Deflate-compressed chunks are now bounded to their expected decompressed size: a chunk that inflates past it (a decompression bomb) or decodes to the wrong length is refused with `FormatError::DecompressionError` / `DataSizeMismatch` instead of allocating unbounded memory or silently zero-filling the result ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- A truncated or corrupt fixed-rate ZFP chunk now decodes without panicking (`zfp` feature) instead of aborting on an out-of-range slice past the end of the buffer ([#97](https://github.com/CramBL/hdf5-pure/issues/97)).
- Reading an object from a *filtered* fractal managed heap is now refused cleanly with `FormatError::UnsupportedFilteredHeapObject` instead of silently misparsing it (the indirect-block child-pointer walk used the wrong stride for filter-encoded blocks) ([#80](https://github.com/CramBL/hdf5-pure/issues/80)).
- Object copy (`EditSession::copy` and `copy_from`) no longer refuses an object whose Attribute Info message carries an *undefined* fractal-heap address — the reference C library and h5py emit that message (to record attribute creation order) alongside compact, inline attributes, and the editor mistook its mere presence for dense storage. It now inspects the heap address and refuses only genuine dense storage, on both the same-file and cross-file paths ([#78](https://github.com/CramBL/hdf5-pure/issues/78)).
- In-place delete now reclaims an object's storage only when the link being removed is its *last* hard link; previously it freed the blocks unconditionally, so deleting one of several hard links returned still-referenced storage and silently corrupted the surviving link once those bytes were reused ([#77](https://github.com/CramBL/hdf5-pure/issues/77)). The editor now counts every hard link before reclaiming and leaves a multiply-linked object's storage in place (a safe leak the repack path still compacts).
- Malformed chunk geometry is now refused up front by both `FileBuilder` and `EditSession` (`FormatError::InvalidChunkGeometry` / `Error::EditUnsupported`) instead of panicking in the chunk splitter: a chunk rank that disagrees with the shape, a zero chunk dimension, a max shape of the wrong rank or smaller than the current shape, chunking a scalar, and an element count that overflows `u64` ([#76](https://github.com/CramBL/hdf5-pure/issues/76)). Zero-element extensible datasets remain valid.

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
  - `File::open_swmr(path)` plus `File::refresh()` re-read data appended by a concurrent writer.
  - `SwmrWriter::open(path)` appends chunks in place (`append_i32` / `append_f64` / `append_raw`), ordered so a reader or a crashed writer only ever sees a consistent prefix. `close()` clears the SWMR flag; `clear_swmr_flag(path)` recovers a file left flagged by a crash.
  - Limited to unfiltered, chunk-aligned, single-unlimited-dimension datasets; unsupported targets are rejected with `Error::SwmrAppendUnsupported`. Requires `std`.

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

[Unreleased]: https://github.com/CramBL/hdf5-pure/compare/v0.44.1...HEAD
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
