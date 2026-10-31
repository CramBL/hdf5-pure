[`FileSpaceSettings`](crate::FileSpaceSettings) controls where a file allocates new storage and
whether it remembers free regions after closing. The strategy matters when
[editing](crate::_guide::editing) or [repacking](crate::_guide::repack) a file.

## Free space and file size

Deleting a dataset can leave a hole between live allocations. Reusing that hole avoids growth,
but does not move the allocations after it.

```text
Before deletion:  [ live A ][ dataset B ][ live C ]
After deletion:   [ live A ][   free    ][ live C ]
After reuse:      [ live A ][ new data  ][ live C ]
```

A free tail is different: an eligible editor can shorten the file without moving live data.
The strategy, ownership checks, and allocation boundaries determine which regions it can release.
[Repack](crate::_guide::repack) can also remove interior holes by rewriting the surviving objects.

### Allocations, tracking, and reuse

File-space management involves more than counting free bytes. Reuse depends on what owns each
region, what kind of allocation can use it, and what information survives closing the file.

Datasets need storage for both their values (*raw data*) and the structures that describe them
(*metadata*). An *allocation* sets aside a range of bytes for either purpose. A contiguous byte
range is also called an *extent* here.

| Term | Meaning |
|---|---|
| Free section | A free byte range tracked by the allocator or a free-space manager |
| Free-space manager | Bookkeeping that records free sections |
| Persistence | Recording tracked sections in the file so later sessions can recover them |
| Aggregate | A block reserved in advance to serve several allocations |
| Page | A fixed-size unit used by PAGE to separate metadata from raw data |
| Threshold | Minimum size of a freed region to track without merging it with an existing section |
| Publication | Updating the metadata that readers use to find data and free space |

The strategy determines how allocations are placed and free space is tracked. Persisted sections
describe recorded free space. Session reports describe what the current editor can reuse,
including unused space in aggregate blocks.
Enough reusable bytes in total do not guarantee that a particular write will fit.

## Configuring file-space management

Start with a strategy, then set persistence, threshold, and page size:

```rust
use hdf5_pure::{FileCreateProperties, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy};

let page_size = FileSpacePageSize::try_from(8192)?;
let settings = FileSpaceSettings::new(FileSpaceStrategy::Page)
    .with_persistence(true)
    .with_page_size(page_size);
let properties = FileCreateProperties::new().with_file_space_settings(settings);
assert_eq!(
    properties.file_space_settings().map(|settings| settings.page_size()),
    Some(page_size),
);
# Ok::<(), hdf5_pure::FormatError>(())
```

Parsing the byte count produces a checked [`FileSpacePageSize`](crate::FileSpacePageSize).
Supported sizes range from 512 bytes through 1 GiB, inclusive.
You can also use sizes such as 6144 bytes, not just powers of two.

| Setting | Default |
|---|---|
| Persistence | Disabled |
| Free-section threshold | 1 byte |
| Page size | [`FileSpacePageSize::DEFAULT`](crate::FileSpacePageSize::DEFAULT), 4096 bytes |

[`FileSpaceSettings::default`](crate::FileSpaceSettings::default) selects
[`FsmAggr`](crate::FileSpaceStrategy::FsmAggr). libhdf5 exposes strategy, persistence, threshold,
and page size through `H5Pset_file_space_strategy` and `H5Pset_file_space_page_size`.

### Applying the settings

Pass the settings directly to a [`FileBuilder`](crate::FileBuilder):

```rust
use hdf5_pure::{File, FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy};
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("out.h5");

let page_size = FileSpacePageSize::try_from(6144)?;
let settings = FileSpaceSettings::new(FileSpaceStrategy::Page).with_page_size(page_size);

let mut builder = FileBuilder::new();
builder.with_file_space_settings(settings);
builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
builder.write(&path)?;

assert_eq!(File::open(&path)?.file_space_settings()?, Some(settings));
# Ok::<(), hdf5_pure::Error>(())
```

Alternatively, use [`FileCreateProperties`](crate::FileCreateProperties) with
[`FileBuilder::with_create_properties`](crate::FileBuilder::with_create_properties) or
[`File::create_with_options`](crate::File::create_with_options).

New creation properties leave file-space settings unset, so the writer omits the File Space Info
message. Setting them explicitly makes the writer include the message, even when the values match
the defaults. Any explicit settings require the HDF5 1.10 writer format.

Applying creation properties replaces earlier creation options. If the supplied properties leave
file-space settings unset, applying them clears an earlier file-space setting too.

[`Aggr`](crate::FileSpaceStrategy::Aggr) and [`None`](crate::FileSpaceStrategy::None) have no
free-space managers. Their effective settings always have persistence disabled and threshold 1.
Page size is retained under every strategy, but affects allocation only under
[`Page`](crate::FileSpaceStrategy::Page).

`H5Pset_file_space_strategy` likewise ignores persistence and threshold for strategies without
managers (`H5Pfcpl.c`, HDF5 2.2.0).

A complete example is available as [file_space.rs][example]:

```console
$ cargo run --example file_space
```

## Strategy variants

[`FileSpaceStrategy`](crate::FileSpaceStrategy) mirrors HDF5's `H5F_fspace_strategy_t`.
All four strategies can be configured.

| Strategy | HDF5 constant | Allocation |
|---|---|---|
| [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) | `H5F_FSPACE_STRATEGY_FSM_AGGR` | Free-space managers and separate metadata/raw aggregates |
| [`Page`](crate::FileSpaceStrategy::Page) | `H5F_FSPACE_STRATEGY_PAGE` | Page-aware allocation with free-space managers |
| [`Aggr`](crate::FileSpaceStrategy::Aggr) | `H5F_FSPACE_STRATEGY_AGGR` | Separate metadata/raw aggregates, without managers |
| [`None`](crate::FileSpaceStrategy::None) | `H5F_FSPACE_STRATEGY_NONE` | Allocation at the end of the file, without free-space tracking |

The difference is visible after a committed deletion:

| Strategy | Reuse of deleted storage | After reopening |
|---|---|---|
| `FsmAggr`, no persistence | Within the session | Earlier holes are not recovered |
| `FsmAggr`, persistent | Through free-space managers | Recorded sections are recovered |
| `Page`, persistent | Through page-type free lists | Recorded sections are recovered |
| `Page`, no persistence | No reuse or truncation | A fresh, empty reuse state |
| `Aggr` | Deleted extents are discarded | Aggregate state is forgotten |
| `None` | Deleted extents are discarded | File length is retained |

A file without recorded settings uses the default allocation behavior.
[Nonpersistent PAGE editing](#nonpersistent-page-editing) has additional restrictions.

### Aggregates

An aggregate reserves a block and serves allocations from it. The session can use the remaining
bytes for later allocations of the same type. These bytes are unused aggregate space, not
persisted free sections.

```text
Metadata aggregate:  [ allocated metadata ][ unused space ]
Raw aggregate:       [ allocated data     ][ unused space ]
```

Whole-file writes and ordinary editing sessions use separate metadata and raw aggregates with
2048-byte refill blocks under `Aggr` and `FsmAggr`. This also applies when the strategy is unset.
SWMR disables classic aggregation under `FsmAggr`, but not `Aggr`.

Under `FsmAggr`, staged edits first try tracked free regions, then unused aggregate space.
Immediate appends follow the same order when persistence is disabled. With persistence enabled,
they instead release unused aggregate space through manager publication. Each requested extent
is allocated directly, without a new aggregate reservation.

Under `Aggr`, commits do not discard the unused space in the current aggregate blocks.
Once an aggregate moves to a new block, any unused bytes left in an interior block are no longer
available for allocation.

The *end of allocation* (EOA) is the file's recorded allocation limit. On a clean close or
last-handle drop, the editor releases unused aggregate space at that limit after publishing the
shorter EOA with the current root.

An uncertain publication prevents that cleanup. Reopening starts with no active aggregate blocks.

## The threshold parameter

[`FileSpaceSettings::with_threshold`](crate::FileSpaceSettings::with_threshold) sets the
free-section admission threshold in bytes. The default is 1. Zero is accepted.
The writer records the threshold under `FsmAggr` and `Page`.

Under `FsmAggr` and persistent `Page`, a freed region at least as large as the threshold is tracked.
A shorter region is tracked only if it joins existing tracked free space. Otherwise it is dropped
and cannot be reused.

For example, with a 64-byte threshold and an eligible adjacent region:

```text
Threshold: 64 B

Isolated: [ live ][ free: 32 B ][ live ] -> not tracked

Adjacent, before: [ tracked: 128 B ][ free: 32 B ]
Adjacent, after:  [        tracked: 160 B        ]
```

PAGE adds a boundary rule: a region smaller than a page merges only with compatible free space in
that same page. This crate tracks page tails regardless of the threshold.

libhdf5 adds page-tail fragments through `H5MF_alloc` without applying the freed-region threshold
(`H5MF.c`, HDF5 2.2.0).

## Inspecting file space

Choose an accessor according to what you need to observe:

| Purpose | API | Handle |
|---|---|---|
| Recorded configuration | [`file_space_settings()`](crate::File::file_space_settings) | Buffered, streaming, or editing |
| Raw File Space Info | [`file_space_info()`](crate::File::file_space_info) | Buffered, streaming, or editing |
| Persisted sections | [`persisted_free_sections()`](crate::File::persisted_free_sections) | Read-only |
| Current reusable totals | [`reusable_space_stats()`](crate::File::reusable_space_stats) | Editing |
| Current reusable ranges | [`reusable_free_sections()`](crate::File::reusable_free_sections) | Editing |

Editing handles read published metadata through the first two accessors. Session inspectors
include committed changes and immediate appends, but not edits still waiting for a commit.

### Inspecting a live editing session

The following example stages a deletion before inspecting its committed effect:

```rust
use hdf5_pure::File;
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("live_session.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.with_file_space_settings(
#     hdf5_pure::FileSpaceSettings::new(hdf5_pure::FileSpaceStrategy::FsmAggr)
#         .with_persistence(true),
# );
# builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
# builder.create_dataset("scratch").with_i32_data(&[10, 20, 30]);
# builder.write(&path)?;

let file = File::open_rw(&path)?;
let before = file.reusable_space_stats()?;

file.root().delete("scratch")?;
assert_eq!(file.reusable_space_stats()?, before);
file.commit()?;

let stats = file.reusable_space_stats()?;
println!("logical size:  {}", stats.logical_size());
println!("reusable free: {}", stats.reusable_free_bytes());

let free = file.reusable_free_sections()?;
for section in free.sections() {
    println!(
        "reusable section: {}..{}",
        section.address().get(),
        section.end_address().get(),
    );
}
# Ok::<(), hdf5_pure::Error>(())
```

[`ReusableSpaceStats`](crate::ReusableSpaceStats) uses cached session counters.
Its reusable total includes unused aggregate reservations, even below the free-section threshold.
It excludes PAGE space that is no longer used but cannot safely be reused, for example because
its page type is unknown. [Page reclamation](#reclaiming-pages) explains this dead or unclassified space.

The total need not equal a persisted report: unused space in active aggregate blocks is not
persisted as free sections.
It also does not predict growth, because allocation depends on each range's size and, under PAGE,
its page type.

Contiguity alone can prevent an allocation from fitting:

```text
Request: one contiguous 64-byte extent

[ free: 32 B ][ live ][ free: 32 B ] -> 64 B total, but no fit
[             free: 64 B          ] -> 64 B total, fits
```

This drawing isolates the contiguity requirement. PAGE also requires a compatible page type.

### Address coordinates

Persisted sections have [`StoredAddress`](crate::StoredAddress) addresses relative to the opened
image's effective HDF5 base. Session-reusable sections have
[`PhysicalAddress`](crate::PhysicalAddress) offsets from the beginning of the source, including
any userblock.

For a file whose effective base is 4096:

```text
Physical offset:  0                        4096                   4224
                  |        userblock       |   HDF5 allocations   | free section
Stored address:                            0                      128

physical offset = 4096 + stored address  (not to scale)
```

[`FreeSection::address`](crate::FreeSection::address) and
[`FreeSection::end_address`](crate::FreeSection::end_address) return the section's address type.
Use that type's `get()` method when you need the numeric offset.

For a relocated image, the effective base is the physical signature position. It can differ from
the base encoded in the superblock.

### Handle errors

| Call on the wrong handle | Error |
|---|---|
| Session totals or ranges on a read-only file | [`Error::ReadOnly`](crate::Error::ReadOnly) |
| Persisted sections on an editing file | [`Error::FreeSectionInspectionUnsupported`](crate::Error::FreeSectionInspectionUnsupported) |

## Inspecting recorded settings

[`File::file_space_settings`](crate::File::file_space_settings) returns the recorded strategy,
persistence flag, threshold, and page size. The value comes from the File Space Info message in
the superblock extension.

```rust
use hdf5_pure::File;
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("out.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.with_file_space_settings(
#     hdf5_pure::FileSpaceSettings::new(hdf5_pure::FileSpaceStrategy::Page)
#         .with_page_size(hdf5_pure::FileSpacePageSize::try_from(8192)?),
# );
# builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
# builder.write(&path)?;

let file = File::open(&path)?;
if let Some(settings) = file.file_space_settings()? {
    println!("strategy:  {:?}", settings.strategy());
    println!("persist:   {}", settings.persist());
    println!("threshold: {}", settings.threshold());
    println!("page size: {}", settings.page_size().get());
}
assert_eq!(
    file.file_space_settings()?.map(|settings| settings.page_size().get()),
    Some(8192),
);
# Ok::<(), hdf5_pure::Error>(())
```

Absence is not a parse failure:

| Recorded metadata | Result from either settings or raw-message inspection |
|---|---|
| No File Space Info message | `Ok(None)` |
| Valid message | `Ok(Some(...))` |
| Malformed message or unreadable extension | [`Error::Format`](crate::Error::Format) |

Reading unrelated datasets can still succeed when their metadata is independently readable.
To use effective defaults for absent settings, propagate the error first, then apply
[`Option::unwrap_or_default`].

The raw [`FileSpaceInfo`](crate::FileSpaceInfo) view also contains manager addresses, a page-end
metadata threshold, and the pre-manager EOA. These fields describe recorded metadata, not current
session reservations.

| Message version | Persistent manager slots | Pre-manager EOA | Page size |
|---|---|---|---|
| 0 | Six | Absent | Reported as `FileSpacePageSize::DEFAULT` |
| 1 | Twelve | Present | Recorded value |

## Paged files (`FileSpaceStrategy::Page`)

PAGE separates metadata and raw-data allocations into pages. The end of allocation is page-aligned.
A userblock must occupy a whole number of pages.

```text
Page:     | metadata          | raw               | wholly free       |
Contents: | metadata + tail   | raw data + tail   | entirely free     |
Reuse:    | metadata only     | raw only          | metadata or raw   |
```

A free tail inside a typed page can serve only that page's allocation type. A wholly free page
has no remaining type to conflict with.

The allocation categories are:

| Storage | Page type |
|---|---|
| Object headers, fractal-heap structures, free-space managers | Metadata |
| Version 2 B-tree chunk indexes | Metadata |
| Global-heap collections | Raw |
| Chunk data and array indexes written beside that data | Raw |
| A dense-attribute blob, including embedded huge-object payloads | Metadata |
| Separately allocated huge-object payloads | Raw |

Array chunk indexes written beside their data are the explicit index exception. The index and
chunk data form one raw allocation.

### Persistent PAGE editing

Enable persistence when creating a paged file:

```rust
use hdf5_pure::{File, FileBuilder, FileSpacePageSize, FileSpaceSettings, FileSpaceStrategy, MaxExtent};
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("paged.h5");

let mut builder = FileBuilder::new();
builder.with_file_space_settings(
    FileSpaceSettings::new(FileSpaceStrategy::Page)
        .with_persistence(true)
        .with_threshold(0)
        .with_page_size(FileSpacePageSize::DEFAULT),
);
builder.create_dataset("samples")
    .with_i32_data(&(0..1000).collect::<Vec<i32>>())
    .with_shape(&[1000])
    .with_maxshape(&[MaxExtent::Unlimited])
    .with_chunks(&[256]);
builder.write(&path)?;

assert_eq!(
    File::open(&path)?.file_space_settings()?.map(|settings| settings.strategy()),
    Some(FileSpaceStrategy::Page),
);
# Ok::<(), hdf5_pure::Error>(())
```

[`File::open_rw`](crate::File::open_rw) supports staged commits and immediate appends on eligible
persistent PAGE files. The editor publishes per-page-type managers and recovers their sections
on reopening.

Manager sections exclude complete metadata allocations. Immediate appends publish allocation
ownership before new references, as described under [publication ordering](#allocation-publication).

### Nonpersistent PAGE editing

Without persistence, staged edits require a whole-file mirror.
[`MemoryStrategy::Auto`](crate::MemoryStrategy::Auto) selects it. Explicit bounded editing is rejected.

| Operation or feature | Nonpersistent PAGE |
|---|---|
| Staged edits | Mirrored only |
| Immediate append | Unsupported |
| SWMR writing | Unsupported |
| Page buffering | Unsupported |
| Reuse of old allocations or tails | Unsupported |
| Reclamation and truncation | Unsupported |

A commit requires supported attribute storage and a complete tree proved free of object references.
Dense attributes and storage the editor cannot inspect for references are rejected before writing
or consuming staged edits.

New allocations start at or beyond the first page boundary covering both the stored EOA and the
physical extent, converted relative to the effective base. Old allocations, apparent holes, and
page tails remain unavailable.

A staged commit uses fresh pages even when an old hole exists:

```text
Before: [ old meta ][ dead hole ][ old raw ] EOA
After:  [ old meta ][ dead hole ][ old raw ][ new meta ][ new raw ] EOA
                          ^ remains unavailable for reuse
```

This schematic groups pages by allocation type. Actual placement follows the boundary rule above.

The editor preserves the File Space Info message with persistence disabled and no manager addresses.
It retains the existing superblock extension and other old allocations. Each reopened session starts
with an empty reuse state.

The commit writes new content before publishing the root and EOA through the superblock.
Barriers follow the selected [synchronization policy](crate::_guide::editing#choosing-the-fsync-cadence).

### Userblocks

PAGE alignment uses stored addresses relative to the effective base. Files with a userblock require
the mirror, and staged commits preserve the prefix.

Immediate [`Dataset::append`](crate::Dataset::append) returns
[`Error::AppendInPlaceUnsupported`](crate::Error::AppendInPlaceUnsupported) on these files.

### Reclaiming pages

Deleting an object releases its allocations only when complete ownership is established.
For example, deleting the last hard link can release a dataset's raw chunks and version 2 B-tree
index, or an object's dense link and attribute heaps.

| Freed storage | Reclamation |
|---|---|
| `BTHD`, `BTIN`, and `BTLF` chunk-index allocations | Metadata |
| Recorded chunk allocations | Raw |
| Dense name/creation-order indexes and fractal-heap headers/blocks | Metadata |
| Huge objects separate from heap metadata | Raw |
| Huge objects adjoining heap metadata inside a page | Metadata |
| Storage whose page type cannot be established | Held as dead until the whole page is free |

Page-aligned heap allocations of a page or more are held as dead. With contiguous address-space
drivers, libhdf5 classifies allocations at least a page large as generic-large
(`H5MF__alloc_to_fs_type` in `H5MF.c`, HDF5 2.2.0).

Recognized generic-large PAGE free sections have a separate recovery rule: complete interior pages
are seeded as raw pages, while partial edges remain unclassified.

```text
Pages:   | page A         | page B         | page C         | page D         |
Free:           [=========|================|================|======]
Recover: | unclassified   | reusable raw   | reusable raw   | unclassified   |
```

The recovery labels apply to the free bytes shown in each page.
Pages B and C are wholly covered by the section. The edge fragments in pages A and D remain unclassified.

An array index allocated separately from its chunk data can also have an unclassified page type.
This occurs, for example, for an empty resizable dataset with no chunk data.
Its page is released only after every other byte is free too.

In that specific index case, the reusable-byte count can lag a deletion by up to one page without
accumulating across deletions. This is not a general bound on fragmentation.

Shared global-heap collections require a complete analysis of references from surviving objects.
They become reclaimable only when no surviving object references them.

A repeated create/delete cycle can settle at a steady size when ownership, threshold admission,
allocation sizes, and page types allow reuse. Whole free pages at the tail can be returned to the
filesystem. Deleting an arbitrary object does not guarantee that the file shrinks.

## Persisting free space across sessions

Enable persistence under `FsmAggr` or `Page` to record eligible tracked sections for later sessions.
Both this crate and libhdf5 can recover those regions.

```text
Session A             On disk                         Session B
tracked free  ----->  FSHD headers + FSSE lists ----> recovered sections
```

The version 0 manager header has the `FSHD` signature. Its version 0 serialized section list has `FSSE`.
These are the header and section-list structures described in "Free-space Index" of the
[HDF5 format specification, version 4.0][free-space-format].

The writer records unused aggregate space under persistent `FsmAggr`, subject to the threshold.
Persistent PAGE files already record page-tail free space at creation.
If no reusable sections remain, the persistence flag is still recorded, with undefined manager addresses.

### Inspecting persisted sections

Create a temporary dataset, delete it, and inspect the persisted free regions after closing:

```rust
use hdf5_pure::{File, FileBuilder, FileSpaceSettings, FileSpaceStrategy};
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("managed.h5");

let mut builder = FileBuilder::new();
builder.with_file_space_settings(
    FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
);
builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
builder.write(&path)?;

let file = File::open_rw(&path)?;
file.root().create_dataset("scratch", |builder| {
    builder.with_f64_data(&vec![0.0; 4096]);
})?;
file.commit()?;
file.root().delete("scratch")?;
file.commit()?;
file.close()?;

let file = File::open(&path)?;
let report = file.persisted_free_sections()?.expect("persistence is configured");
println!("free regions: {}", report.free_sections().section_count());
println!("free bytes:   {}", report.free_sections().free_bytes());
println!("manager bytes: {}", report.manager_metadata_bytes());
assert!(report.free_sections().free_bytes() > 0);
# Ok::<(), hdf5_pure::Error>(())
```

The report combines all tracked sections, sorted by address, and supplies per-manager totals.
Persistence and the presence of free sections are separate:

| File configuration | Result |
|---|---|
| Persistence absent or disabled | `Ok(None)` |
| Persistence configured | `Ok(Some(report))`, even with zero sections |

Buffered and streaming read-only handles support inspection. Editing handles must use the session
inspectors instead.

### Manager metadata bytes

[`PersistedFreeSections::manager_metadata_bytes`](crate::PersistedFreeSections::manager_metadata_bytes)
measures persisted bookkeeping, not the regions available for reuse:

```text
free_sections().free_bytes()    = tracked free regions
manager_metadata_bytes()       = FSHD headers + allocated FSSE storage
```

The metadata count includes unused section-list capacity. It equals the sum of
[`FreeSpaceManagerSections::metadata_bytes`](crate::FreeSpaceManagerSections::metadata_bytes)
for the defined managers.

A header-only manager has zero free-section bytes but positive metadata bytes.
Undefined slots contribute zero.

The count excludes the File Space Info message, its object header, userblocks, dataset storage,
other metadata, aggregate reservations, dead or unclassified storage, and trailing slack.
Relocating the image or adding a userblock does not change it.

libhdf5's `H5Fget_info2` can include managers active only in memory (`H5F__get_info` and
`H5MF_get_freespace` in `H5Fint.c` and `H5MF.c`, HDF5 1.14.6).

`H5FS_size` can report a loaded section list's in-memory size
(`H5FS.c`, HDF5 1.14.6), so it can differ from this persisted allocation count.

## Recovering persisted free space

Writable opening has two recovery policies:

| Policy | On manager parsing failure | On an invalid individual extent |
|---|---|---|
| [`Tolerant`](crate::PersistedFreeSpacePolicy::Tolerant), the default | Recover nothing from the manager set | Skip the extent, retaining valid sections |
| [`Strict`](crate::PersistedFreeSpacePolicy::Strict) | Fail recovery | Fail recovery |

Select strict recovery through [`FileAccessProperties`](crate::FileAccessProperties):

```rust
use hdf5_pure::{File, FileAccessProperties, PersistedFreeSpacePolicy};
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("strict_recovery.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
# builder.write(&path)?;

let options = FileAccessProperties::new()
    .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict);
let file = File::open_rw_with_options(&path, options)?;
assert_eq!(
    file.access_properties().persisted_free_space_policy(),
    PersistedFreeSpacePolicy::Strict,
);
# Ok::<(), hdf5_pure::Error>(())
```

Strict manager recovery reports contextual [`Error::PersistedFreeSpace`](crate::Error::PersistedFreeSpace).
It fails before changing file bytes, length, or status flags.
Read-only opening ignores the policy.

Persisted inspection reports invalid metadata.
Both persisted inspection and strict recovery verify manager checksums, even without the
`checksum` feature.

| Persisted inspection failure | Error |
|---|---|
| Overlapping or duplicate free extents, within or across managers | [`Error::OverlappingFreeSections`](crate::Error::OverlappingFreeSections) |
| Malformed metadata, overlapping manager allocations, duplicate manager-header addresses | [`Error::Format`](crate::Error::Format) |

[Reading datasets](crate::_guide::reading) does not depend on whether the file persists free space.

## Allocation publication

An immediate append can allocate from persisted free regions. It must remove those allocations
from the published free-space managers before linking new references to them.
Otherwise a later session could reuse storage that a live object owns.

The ordering is:

```text
[1] Allocate provisional extents, then write payloads and index blocks
       |
       v
[2] Publish replacement managers (exclude owned extents and manager storage)
       |
       v
[3] Record extension and file length; complete the selected sync barrier
       |
       v
[4] Link references to the new allocations
```

Until publication, the preceding managers remain authoritative. Replacement managers exclude each
complete owned allocation and their own metadata storage, including unused section-list capacity.
The editor retains the current root and leaves unrelated staged edits pending.

Persistent appends allocate exact extents.
Unused aggregate space is released through the threshold policy. PAGE publication also preserves
page tails and classes.

The editor drains earlier pending writes before reusing storage retired by a staged commit.
It checks manager checksums regardless of the `checksum` feature and rejects free sections
overlapping published manager or extension allocations.

### Interruptions and errors

The publication point changes what a later session can recover:

| Interruption | Possible allocation state |
|---|---|
| Before ownership publication | Preceding managers still describe free space |
| After publication, before linking the owner | An actual allocation can remain unreferenced |

Earlier speculative reservations cannot be reconstructed from persisted managers.
Publishing exact ownership rewrites managers more frequently, and loading/serialization costs depend
on the number of sections.

Before attempted publication, a known failure restores allocator state, discards queued writes,
and truncates a grown image to its saved length. If rollback fails, the session becomes uncertain.

After uncertain publication, the editor returns the triggering error and discards queued storage
writes. Subsequent guarded reference and reuse requests return
[`Error::AllocationPublicationUncertain`](crate::Error::AllocationPublicationUncertain).
Discarded writes are excluded from teardown.

This is an ordering guarantee, not a guarantee against torn physical metadata writes.
Durability also depends on the
[synchronization policy](crate::_guide::editing#choosing-the-fsync-cadence) and
[page buffering](crate::_guide::editing#interrupted-page-buffered-sessions).

An opt-in [lossy tail rewrite](crate::_guide::editing#eligibility) can change previously published
values through repeated re-encoding. Allocation publication does not make that transformation lossless.

## Reuse, truncation, and repack

A commit can release a free tail under `FsmAggr` or persistent PAGE.
It shortens the file to an allocation boundary, or a page boundary for PAGE, and publishes managers
that cover nothing above the new length.

The editor can retain space needed for later commit metadata. A short free suffix can remain allocated.
A deletion can therefore leave the file unchanged or even grow it.
Releasing a tail does not compact an interior hole.

```text
Free tail (same file, when eligible):
  Before: [ live ][ live ][ free tail ]
  After:  [ live ][ live ]

Interior hole (new file, after repack):
  Source: [ live ][  free ][ live ]
  Output: [ live ][ live ]
```

Default repacking preserves recorded effective settings, including persistence.
[`RepackOptions::with_file_space_settings`](crate::RepackOptions::with_file_space_settings) selects
an output policy and rebuilds destination managers.
See [repack file-space management](crate::_guide::repack#file-space-management) for an example.

## Allocation details

The following rules explain differences between session totals, persisted sections, and file length.

### Whole-file placement

Whole-file writers release unused aggregate tails at EOA before producing output.
They use these allocation units:

| Storage | Allocation unit |
|---|---|
| Global-heap collection | One raw allocation |
| Dense-attribute blob, including embedded huge objects | One metadata allocation |
| Combined chunk data and array index | One raw allocation |

Buffered and streaming output use the same final addresses. Streaming providers supply one chunk
or block at a time.

Under `FsmAggr`, placement tries unused space left in earlier aggregate blocks before reserving
another block, subject to the free-section threshold. With persistence enabled, remaining reusable
aggregate space is recorded after the reservations are released.

Manager allocations are excluded from their own sections. If the section list is empty after tail
release and threshold admission, the writer records persistence with undefined manager addresses.

Without persistence, the construction free list is discarded after output.
Its unused interior regions are unavailable on reopening.
`Aggr` also leaves abandoned interior reservations allocated without a manager.

### Manager recycling

The editor places replacement manager blocks in reusable space where they fit.
Under persistent `FsmAggr`, it can reserve a contiguous sub-threshold allocation for the extension
and managers.

It publishes replacements elsewhere before rewriting the reserved allocation.
If the new metadata fits, it publishes that allocation again.
The reservation is excluded from ordinary free-section admission.

Unused aggregate space released during recycling follows ordinary threshold admission.
A commit may release unused space in both active aggregate blocks to free a larger unused suffix
or recycle sub-threshold manager storage.

Clean close or last-handle drop returns unused aggregate space through a manager rewrite when
persistence is enabled. Without persistence, it forgets the session free list.
Uncertain publication prevents this cleanup.

Fixed-width root attribute updates can reach a stable length across commits and reopen cycles when
the [root-header reuse conditions](crate::_guide::editing#how-it-works) hold and replacement managers
fit their reservation. Other relocated metadata follows ordinary threshold admission and can leave
sub-threshold gaps.

## API migration

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

For page-size-only configuration, use [`FileSpaceSettings::new`](crate::FileSpaceSettings::new)
with [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), then
[`with_page_size`](crate::FileSpaceSettings::with_page_size).
The settings getter returns normalized effective values.
The removed getters returned separately requested values.

Recorded-settings and raw-message inspection distinguish absence from parse failure.
The raw accessor was public before 0.47.0. Its return type changes from an optional borrowed value
to a fallible optional owned value.

| Previous API | Current API |
|---|---|
| `File::file_space_strategy() -> Option<FileSpaceStrategy>` | `file.file_space_settings()?`, then `FileSpaceSettings::strategy` |
| `File::file_space_info() -> Option<&FileSpaceInfo>` | `File::file_space_info() -> Result<Option<FileSpaceInfo>, Error>` |
| `File::persisted_free_space()` | `File::persisted_free_sections()` |
| `File::space_accounting() -> Result<SpaceAccounting, Error>` | `File::reusable_space_stats()` for scalars or `File::reusable_free_sections()` for ranges |
| `SpaceAccounting` | `ReusableSpaceStats` and typed section reports |

The persisted report distinguishes absent persistence from an empty report.
Inspect addresses and nonzero lengths through [`FreeSections::sections`](crate::FreeSections::sections),
or use [`FreeSections::free_bytes`](crate::FreeSections::free_bytes) for the total.

[example]: https://github.com/CramBL/hdf5-pure/blob/main/examples/file_space.rs
[free-space-format]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
