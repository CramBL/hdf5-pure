[`FileSpaceSettings`](crate::FileSpaceSettings) configures a file's allocation strategy,
persistence, free-section threshold, and page size. Apply the value through
[`FileBuilder::with_file_space_settings`](crate::FileBuilder::with_file_space_settings) or
[`FileCreateProperties::with_file_space_settings`](crate::FileCreateProperties::with_file_space_settings).
The settings are separate from the raw File Space Info message and from observations of free
sections. This page also describes how the configuration affects [editing](crate::_guide::editing)
and [repacking](crate::_guide::repack).

A runnable example lives at [`examples/file_space.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/file_space.rs). Run it with:

```console
$ cargo run --example file_space
```

## Configuring file-space management

[`FileSpaceSettings::new`](crate::FileSpaceSettings::new) selects the strategy with persistence
disabled, threshold 1, and [`FileSpacePageSize::DEFAULT`](crate::FileSpacePageSize::DEFAULT).
[`FileSpaceSettings::default`](crate::FileSpaceSettings::default) selects
[`FsmAggr`](crate::FileSpaceStrategy::FsmAggr). The C library configures these properties with
`H5Pset_file_space_strategy` and `H5Pset_file_space_page_size`.

[`Aggr`](crate::FileSpaceStrategy::Aggr) and [`None`](crate::FileSpaceStrategy::None) have no
free-space managers. Their effective settings always have persistence disabled and threshold 1.
The setters ignore inputs for those two fields, as `H5Pset_file_space_strategy` does for strategies
without managers (`H5Pfcpl.c`, HDF5 2.2.0). Page size is retained under every strategy and affects
allocation only under [`Page`](crate::FileSpaceStrategy::Page).

[`FileCreateProperties`](crate::FileCreateProperties) stores one optional settings value.
[`FileCreateProperties::file_space_settings`](crate::FileCreateProperties::file_space_settings)
returns that value, including normalized fields and defaults. New creation properties leave it
unset, so the writer omits the File Space Info message. Passing an explicit default requests a
message. Any explicit settings require the HDF5 1.10 writer format.

Apply the properties with [`FileBuilder::with_create_properties`](crate::FileBuilder::with_create_properties)
or [`File::create_with_options`](crate::File::create_with_options).
Applying creation properties replaces all earlier creation options, including clearing file-space
settings when the supplied properties leave them unset.

[`FileSpaceSettings::with_page_size`](crate::FileSpaceSettings::with_page_size) takes the checked
[`FileSpacePageSize`](crate::FileSpacePageSize) value. Construct it with
[`FileSpacePageSize::try_from`](crate::FileSpacePageSize::try_from), and read its byte count with
[`FileSpacePageSize::get`](crate::FileSpacePageSize::get).

```rust
use hdf5_pure::{FileCreateProperties, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy};

fn paged_layout(page_size: FileSpacePageSize) -> FileCreateProperties {
    let settings = FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_persistence(true)
        .with_page_size(page_size);
    FileCreateProperties::new().with_file_space_settings(settings)
}
let page_size = FileSpacePageSize::try_from(8192)?;
assert_eq!(
    paged_layout(page_size).file_space_settings().map(|settings| settings.page_size()),
    Some(page_size)
);
# Ok::<(), hdf5_pure::FormatError>(())
```

```rust
use hdf5_pure::FileSpacePageSize;
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("out.h5");
use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};

let mut b = FileBuilder::new();
b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
b.with_file_space_settings(
    FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_page_size(FileSpacePageSize::try_from(8192)?),
);
b.write(&path)?;

assert_eq!(
    File::open(&path)?.file_space_settings()?.map(|settings| settings.strategy()),
    Some(FileSpaceStrategy::Page)
);
# Ok::<(), hdf5_pure::Error>(())
```

## Strategy variants

The [`FileSpaceStrategy`](crate::FileSpaceStrategy) enum mirrors HDF5's `H5F_fspace_strategy_t`:

| Variant | HDF5 constant | Behavior |
|---|---|---|
| [`FileSpaceStrategy::FsmAggr`](crate::FileSpaceStrategy::FsmAggr) | `H5F_FSPACE_STRATEGY_FSM_AGGR` | The HDF5 default: free-space managers, aggregators, and the virtual file driver. |
| [`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page) | `H5F_FSPACE_STRATEGY_PAGE` | Paged aggregation backed by free-space managers. |
| [`FileSpaceStrategy::Aggr`](crate::FileSpaceStrategy::Aggr) | `H5F_FSPACE_STRATEGY_AGGR` | Aggregators and the virtual file driver only, no free-space managers. |
| [`FileSpaceStrategy::None`](crate::FileSpaceStrategy::None) | `H5F_FSPACE_STRATEGY_NONE` | No free-space tracking: allocation only ever appends. |

Under [`Aggr`](crate::FileSpaceStrategy::Aggr), staged edits and immediate appends use separate metadata and raw aggregates with 2048-byte refill blocks. The session retains unused aggregate reservations across commits, and [`File::close`](crate::File::close) or dropping the last handle releases unused aggregate space at the end of allocation (EOA) after publishing the shorter EOA with the current root. An uncertain publication prevents this cleanup. Aggregate state is forgotten on reopen.

Whole-file writes under `Aggr` or `FsmAggr` use separate metadata and raw aggregates with 2048-byte refill blocks. This also applies when the strategy is unset. The writer releases unused tails at EOA before output. Global heap collections use raw allocations. Each dense-attribute blob is one metadata allocation, including any embedded huge-object payloads. Each combined chunk-data and array-index region is one raw allocation. Buffered and streaming output use the same final addresses, and streaming providers supply one chunk or block at a time.

Under `FsmAggr`, the writer reuses abandoned aggregate spare subject to the free-section threshold before reserving more aggregate space. With persistence enabled, it releases the aggregate reservations after placement and records the remaining reusable spare in a free-space manager. The manager's own allocations are excluded from its free sections. If no sections remain after tail release and threshold admission, the writer records the persist flag with undefined manager addresses. With persistence disabled, the construction free list is discarded after output, so its unused interior regions are unavailable after reopening. `Aggr` also leaves unused interior reservations allocated without a manager.

Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), ordinary read-write sessions try reusable
freed space before the metadata or raw aggregate, with persistence enabled or disabled. This also
applies to a file without a recorded strategy. Abandoned aggregate spare is tracked subject to the
free-section threshold. Active reservations are excluded from persisted free sections. A commit may
surrender both windows when their spare permits releasing a larger dead suffix, or when recycling
sub-threshold persistent manager storage. Clean teardown returns unused spare through the persistent
managers, or forgets the session free list when persistence is disabled. Reopening starts with empty
aggregate windows. SWMR sessions under `FsmAggr` use no classic aggregates.

Under `FsmAggr` and persistent [`Page`](crate::FileSpaceStrategy::Page), edits reuse freed space through free-space managers. Non-persistent PAGE sessions append in fresh pages and retain freed extents without reuse. Under `Aggr` and [`None`](crate::FileSpaceStrategy::None), deletion discards the freed extents. `None` allocates at the end of the file and retains its length after deletion.

## The threshold parameter

[`FileSpaceSettings::with_threshold`](crate::FileSpaceSettings::with_threshold) sets the
free-section admission threshold in bytes. The default is 1, and zero is accepted. Under
[`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) and [`Page`](crate::FileSpaceStrategy::Page), the writer
records the threshold in the File Space Info message.

Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) and persistent [`Page`](crate::FileSpaceStrategy::Page), a read-write session tracks a freed region of at least the threshold. It tracks a shorter one only where the region adjoins tracked free space, which it merges into, and drops it otherwise, as the C library does. A dropped region is not reused. Under [`Page`](crate::FileSpaceStrategy::Page) a region shorter than a page merges only with free space in its own page. The writer and the editors track page tails whatever the threshold, and so does the C library.

## Inspecting recorded settings

[`File::file_space_settings`](crate::File::file_space_settings) returns the recorded strategy,
persistence flag, threshold, and page size as a [`FileSpaceSettings`](crate::FileSpaceSettings).
These settings are stored in the File Space Info message of the superblock extension. They
describe the recorded configuration. A writable session's reusable-byte total and free ranges are
available through [`File::reusable_space_stats`](crate::File::reusable_space_stats) and
[`File::reusable_free_sections`](crate::File::reusable_free_sections).

```rust
use hdf5_pure::FileSpacePageSize;
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("out.h5");
# use hdf5_pure::{FileBuilder, FileSpaceSettings, FileSpaceStrategy};
# let mut b = FileBuilder::new();
# b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
# b.with_file_space_settings(
#     FileSpaceSettings::new(FileSpaceStrategy::Page)
#         .with_page_size(FileSpacePageSize::try_from(8192)?),
# );
# b.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
if let Some(settings) = file.file_space_settings()? {
    println!("strategy: {:?}", settings.strategy());
    println!("persist:  {}", settings.persist());
    println!("threshold: {}", settings.threshold());
    println!("page size: {}", settings.page_size().get());
}
assert_eq!(
    file.file_space_settings()?.map(|settings| settings.page_size().get()),
    Some(8192)
);
# Ok::<(), hdf5_pure::Error>(())
```

Both [`File::file_space_settings`](crate::File::file_space_settings) and
[`File::file_space_info`](crate::File::file_space_info) return `Ok(None)` when the file has no File
Space Info message. A malformed message or unreadable superblock extension returns
[`Error::Format`](crate::Error::Format). Opening the file and reading unrelated datasets remain
possible when their metadata can be read independently. Callers that need an effective fallback
can apply [`Option::unwrap_or_default`] after propagating the inspection error.

[`File::file_space_info`](crate::File::file_space_info) provides an advanced, owned
[`FileSpaceInfo`](crate::FileSpaceInfo) view of the message, including the manager addresses,
page-end metadata threshold, and pre-manager end of allocation. A version 0 message has no
pre-manager end-of-allocation field and, when persistent, has six manager slots. Its page size is
reported as [`FileSpacePageSize::DEFAULT`](crate::FileSpacePageSize::DEFAULT). Version 1 has twelve
manager slots when persistent. Both accessors work with buffered and streaming readers and read
the current published metadata on editing handles.

### API migration

Creation properties and builders use one [`FileSpaceSettings`](crate::FileSpaceSettings) value.
The split creation methods are removed:

| Removed creation API | Replacement |
|---|---|
| `FileCreateProperties::with_file_space_strategy` | `FileCreateProperties::with_file_space_settings` |
| `FileCreateProperties::with_file_space_page_size` | `FileSpaceSettings::with_page_size`, then apply the value |
| `FileCreateProperties::file_space_strategy` | `FileCreateProperties::file_space_settings`, then inspect the value |
| `FileCreateProperties::file_space_page_size` | `FileCreateProperties::file_space_settings`, then `FileSpaceSettings::page_size` |
| `FileBuilder::with_file_space_strategy` | `FileBuilder::with_file_space_settings` |
| `FileBuilder::with_file_space_page_size` | `FileSpaceSettings::with_page_size`, then apply the value |

The page-size-only configuration is an explicit
[`FileSpaceSettings::new`](crate::FileSpaceSettings::new) with
[`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), followed by
[`with_page_size`](crate::FileSpaceSettings::with_page_size). It records the same configuration.
The configuration getter returns effective values, including defaults. The old getters exposed
separately requested values.

The recorded-settings and raw-message accessors distinguish absence from parse failure. The raw
accessor was public before version 0.47.0. Its return type changes from an optional borrowed value
to a fallible optional owned value.

| Previous API | Current API |
|---|---|
| `File::file_space_strategy() -> Option<FileSpaceStrategy>` | `file.file_space_settings()?` |
| `File::file_space_info() -> Option<&FileSpaceInfo>` | `File::file_space_info() -> Result<Option<FileSpaceInfo>, Error>` |
| `File::persisted_free_space()` | `File::persisted_free_sections()` |
| `File::space_accounting() -> Result<SpaceAccounting, Error>` | `File::reusable_space_stats()` for scalars or `File::reusable_free_sections()` for ranges |
| `SpaceAccounting` | `ReusableSpaceStats` and typed section reports |

The strategy-only getter is removed. Read a strategy with
[`FileSpaceSettings::strategy`](crate::FileSpaceSettings::strategy).
[`FileSpaceSettings`](crate::FileSpaceSettings) normalizes
[`Aggr`](crate::FileSpaceStrategy::Aggr) and [`None`](crate::FileSpaceStrategy::None) to persistence
disabled and threshold 1. The recorded page size is retained under every strategy, and allocation
uses pages only under [`Page`](crate::FileSpaceStrategy::Page).

The tuple-based persisted-space accessor is removed. Inspect section addresses and nonzero
lengths through [`FreeSections::sections`](crate::FreeSections::sections), or read the total with
[`FreeSections::free_bytes`](crate::FreeSections::free_bytes). The report distinguishes absent
persistence from configured persistence with zero sections.

The session-accounting method and report type are removed. Read logical size and reusable bytes
through [`File::reusable_space_stats`](crate::File::reusable_space_stats), which uses cached
counters without enumerating sections. Use
[`File::reusable_free_sections`](crate::File::reusable_free_sections) for physical ranges,
including allocator-held reservations. Both inspectors observe committed changes and immediate
appends. Staged changes appear after commit. The reservation rules are described by
[`ReusableSpaceStats::reusable_free_bytes`](crate::ReusableSpaceStats::reusable_free_bytes).

## Paged files (`FileSpaceStrategy::Page`)

The writer uses the configured page size for [`Page`](crate::FileSpaceStrategy::Page) allocation.
[`FileSpacePageSize`](crate::FileSpacePageSize) supports sizes from 512 bytes through 1 GiB and
defaults to 4096 bytes. A userblock must occupy a whole number of pages. With persistence enabled,
the writer records page tails in per-page-type free-space managers. The page allocation categories
and editing restrictions are described below.

```rust
use hdf5_pure::FileSpacePageSize;
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("paged.h5");
use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy, MaxExtent};

let mut b = FileBuilder::new();
b.create_dataset("samples")
    .with_i32_data(&(0..1000).collect::<Vec<i32>>())
    .with_shape(&[1000])
    .with_maxshape(&[MaxExtent::Unlimited])
    .with_chunks(&[256]);
b.with_file_space_settings(
    FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_persistence(true)
        .with_threshold(0)
        .with_page_size(FileSpacePageSize::DEFAULT),
);
b.write(&path)?;
assert_eq!(
    File::open(&path)?.file_space_settings()?.map(|settings| settings.strategy()),
    Some(FileSpaceStrategy::Page)
);
# Ok::<(), hdf5_pure::Error>(())
```

With persistence enabled, [`File::open_rw`](crate::File::open_rw) supports staged commits and immediate appends on eligible PAGE files. Commits and clean close rewrite the per-page-type managers, including the known free remainder of newly allocated manager pages. Reopening recovers the final persisted sections. Held append reserves remain excluded until released. Both write paths keep metadata and raw pages separate and page-align the end of allocation.

With persistence disabled, staged edits require the whole-file mirror. [`MemoryStrategy::Auto`](crate::MemoryStrategy::Auto) selects it, and explicit bounded editing is rejected. New allocations begin at or beyond the first page boundary covering both the original stored EOA and the physical extent converted relative to the effective base. The editor leaves old allocations, apparent holes, and page tails unavailable. Freed extents are discarded without reuse or truncation. Reopening repeats this policy with an empty reusable-space state. The editor preserves `PAGE, persist = false` in the File Space Info message and keeps the manager address list empty.

A non-persistent PAGE commit requires supported attribute storage and a complete tree proved free of object references. Dense attributes and other storage the editor cannot inspect for references are rejected before writing or consuming staged edits. Immediate append, SWMR writing, and page buffering are unsupported. The editor writes new content before publishing the root and EOA through the superblock, with barriers governed by the selected synchronization policy. The editor retains the existing superblock extension and all other pre-existing allocations.

PAGE allocations use metadata or raw pages, with the explicit array-index exception: array chunk indexes written beside chunk data share its raw pages. Version 2 B-tree chunk indexes, object headers, fractal-heap structures, and free-space managers use metadata pages. Global heap collections use raw pages. A dense-attribute blob is one metadata reservation, including any embedded huge-object payloads. Separately allocated huge-object payloads use raw allocations.

Files with a userblock use stored-address page alignment relative to the effective base. [`File::open_rw`](crate::File::open_rw) selects the mirror, and staged commits preserve the userblock. Immediate [`Dataset::append`](crate::Dataset::append) returns [`Error::AppendInPlaceUnsupported`](crate::Error::AppendInPlaceUnsupported) on these files.

Deleting from a persistent paged file returns the space to the per-page-type free lists, so a create/delete cycle settles at a steady size, and whole free pages at the end of the file are given back to the filesystem. Deleting a version 2 B-tree chunk-indexed dataset returns its `BTHD`, `BTIN`, and `BTLF` allocations as metadata and its recorded chunk allocations as raw data. Deleting the last hard link of an object with dense attribute or link storage returns the name and creation-order version 2 B-tree allocations and the header and blocks of each fractal heap as metadata. A heap's huge objects return as raw data where they sit apart from the heap's metadata, and as metadata where they adjoin it inside a page. A heap allocation of a page or more that begins on a page boundary is held as dead, since libhdf5 allocates it as whole pages of neither type. A freed region whose page type cannot be established is held as dead, and the page around it is returned once every other byte of it is free too. That still covers an array chunk index this crate did not lay down beside its chunk data, most often the index of an empty resizable dataset, which has no chunk data at all. The total from [`ReusableSpaceStats::reusable_free_bytes`](crate::ReusableSpaceStats::reusable_free_bytes) therefore lags a delete by up to a page on such a file, and it does not accumulate.

## Persisting free space across sessions

Passing `persist = true` under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) or [`Page`](crate::FileSpaceStrategy::Page) records that freed space should be tracked on disk across closes. Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) the writer records unused aggregate spare from creation, subject to the threshold. For a file with no reusable sections, the writer records the persist flag with undefined manager addresses. A genuine paged file (above) already tracks its page-tail free space from creation. When a later edit frees a region, by deleting a dataset for example, the freed region is recorded in on-disk free-space-manager blocks (`FSHD`/`FSSE`), and a later session, this crate's or libhdf5's, recovers and reuses it.

[`File::persisted_free_sections`](crate::File::persisted_free_sections) returns a typed report of
the tracked free regions, sorted by address, with combined and per-manager totals. A read-write
session recovers these regions into its free list, so reuse spans sessions.

The editor places replacement manager blocks in reusable free space where they fit. Under
[`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), it reserves a contiguous allocation for its extension
and managers that is smaller than the threshold. It first publishes replacement metadata elsewhere,
then rewrites the reserved allocation and publishes it again if the new metadata fits.
The reserved allocation is excluded from ordinary free-section admission. Unused aggregate spare
is released through ordinary threshold admission during recycling. Close or drop also returns
unused spare through a manager rewrite. An uncertain publication prevents this cleanup.

Repeated fixed-width root attribute updates reach a stable file length across commits and reopen
cycles when the [conditions for reusing the root header](crate::_guide::editing#how-it-works) hold
and replacement manager metadata fits its reserved allocation. Other relocated metadata follows
ordinary threshold admission and may leave sub-threshold gaps.

Freed space that reaches the end of the file is given back to the filesystem: the commit shortens the file to where the run starts, to a page boundary on a paged file, and the managers it writes cover nothing above the new length. A few blocks' worth is kept for the tails the following commits have to write, and a run that would return less than that is left alone, so what this guarantees is that the file ends just above its last live allocation, not that a delete never makes a file longer. For a guaranteed shrink that also moves live objects down, use [repack](crate::_guide::repack).

An in-place [`Dataset::append`](crate::Dataset::append) reuses persisted free space as well, under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) and [`Page`](crate::FileSpaceStrategy::Page) alike. Before spending a free region, the editor removes it from the on-disk managers in a separate crash-safe publication. One draw takes every hole the appended chunk fits in, up to a megabyte of them. Unspent reserve returns at the next commit or close. A crash between reservation and return strands those bytes without advertising them as reusable. Under `FsmAggr`, appends try this reserve before the metadata or raw aggregate. Unused aggregate reservations are already excluded from the persisted free sections.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("managed.h5");
use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};

let mut builder = FileBuilder::new();
builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
builder.with_file_space_settings(
    FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
);
builder.write(&path)?;

// Create then delete a dataset through the owned-handle editor; the freed
// region is recorded on disk.
let file = File::open_rw(&path)?;
file.root().create_dataset("scratch", |b| {
    b.with_f64_data(&vec![0.0; 4096]);
})?;
file.commit()?;
file.root().delete("scratch")?;
file.commit()?;
file.close()?;

// Later opens can read the persisted free regions.
let file = File::open(&path)?;
let report = file.persisted_free_sections()?.expect("persistence is configured");
let free = report.free_sections();
let total_free = free.free_bytes();
println!("persisted free regions: {}", free.section_count());
println!("total persisted free bytes: {total_free}");
assert!(total_free > 0);
# Ok::<(), hdf5_pure::Error>(())
```

[`File::persisted_free_sections`](crate::File::persisted_free_sections) returns `Ok(None)` when
the file does not record persistence and `Ok(Some(report))` when persistence is configured,
including when the report has zero sections. Buffered and streaming readers support this
inspection. Editing handles return
[`Error::FreeSectionInspectionUnsupported`](crate::Error::FreeSectionInspectionUnsupported).
Malformed free-space metadata returns [`Error::Format`](crate::Error::Format). A read-write
session skips free regions from every persisted manager if any manager cannot be read or parsed.
Section addresses use
[`FreeSectionCoordinates::Stored`](crate::FreeSectionCoordinates::Stored), relative to the HDF5
base address. [Reading data](crate::_guide::reading) is unaffected by the presence or absence of
these managers.

When free space is not persisted under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), an ordinary read-write session reuses freed space before aggregate reservations, with both kept in memory for that session alone. Under [`Aggr`](crate::FileSpaceStrategy::Aggr), staged edits and immediate appends use unused aggregate reservations within the session. Under `Aggr` and [`None`](crate::FileSpaceStrategy::None), freed extents are discarded. If churn has left a file with unused gaps and you want to reclaim them outright, [repacking](crate::_guide::repack) rewrites the file compactly.
