A written HDF5 file can record how it manages free space: which allocation strategy it uses, whether freed regions are tracked across closes, and the page size used for paged allocation. This page covers [`FileBuilder::with_file_space_strategy`](crate::FileBuilder::with_file_space_strategy) and [`with_file_space_page_size`](crate::FileBuilder::with_file_space_page_size), which mirror HDF5's `H5Pset_file_space_strategy` and `H5Pset_file_space_page_size`, and how the choice interacts with [editing](crate::_guide::editing) and [repacking](crate::_guide::repack).

A runnable example lives at [`examples/file_space.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/file_space.rs). Run it with:

```console
$ cargo run --example file_space
```

## Recording a strategy

[`FileBuilder::with_file_space_strategy(strategy, persist, threshold)`](crate::FileBuilder::with_file_space_strategy) records the file-space management strategy, whether free space is persisted across closes, and the smallest free section the managers will track. [`with_file_space_page_size(size)`](crate::FileBuilder::with_file_space_page_size) sets the file-space page size used for paged allocation. Both are recorded in the file's superblock extension, so the reference HDF5 C library and a later reopen observe the choice.

[`Aggr`](crate::FileSpaceStrategy::Aggr) and [`None`](crate::FileSpaceStrategy::None) have no free-space managers, so for either the writer records a clear persist flag and the default threshold of 1, whatever the call passes for `persist` and `threshold`. The C library's `H5Pset_file_space_strategy` ignores both for those strategies too.

Both are also fields on [`FileCreateProperties`](crate::FileCreateProperties), the reusable file-creation-property value (the `fcpl` analogue). Build one in a helper, then apply it with [`FileBuilder::with_create_properties`](crate::FileBuilder::with_create_properties) or [`File::create_with_options`](crate::File::create_with_options) wherever a file is written, so the call chain is written once:

```rust
use hdf5_pure::{FileCreateProperties, FileSpaceStrategy};

fn paged_layout() -> FileCreateProperties {
    FileCreateProperties::new()
        .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(8192)
}
# assert_eq!(paged_layout().file_space_page_size(), Some(8192));
```

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("out.h5");
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

let mut b = FileBuilder::new();
b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
b.with_file_space_strategy(FileSpaceStrategy::Page, false, 1) // strategy, persist, threshold
    .with_file_space_page_size(8192);
b.write(&path)?;

assert_eq!(
    File::open(&path)?.file_space_strategy(),
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

Whole-file writes under `Aggr` or `FsmAggr` with persistence disabled use separate metadata and raw aggregates with 2048-byte refill blocks. This also applies when the strategy is unset. The writer releases unused tails at EOA before output and leaves unused interior reservations allocated without a free-space manager. Global heap collections use raw allocations. Each dense-attribute blob is one metadata allocation, including any embedded huge-object payloads. Each combined chunk-data and array-index region is one raw allocation. Buffered and streaming output use the same final addresses, and streaming providers supply one chunk or block at a time. Under `FsmAggr`, the writer reuses abandoned aggregate spare subject to the free-section threshold before reserving more aggregate space. The construction free list is discarded after output, so its unused interior regions are unavailable after reopening. Whole-file writes under `FsmAggr` with persistence enabled use no classic aggregates.

Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), ordinary read-write sessions try reusable freed space before the metadata or raw aggregate, with persistence enabled or disabled. This also applies to a file without a recorded strategy. Abandoned aggregate spare is tracked subject to the free-section threshold. Active reservations remain exclusive to the session across commits and are excluded from persisted free sections. A commit may surrender both windows when their spare permits releasing a larger dead suffix. Clean teardown returns unused spare through the persistent managers, or forgets the session free list when persistence is disabled. Reopening starts with empty aggregate windows. SWMR sessions under `FsmAggr` use no classic aggregates.

Under `FsmAggr` and [`Page`](crate::FileSpaceStrategy::Page), edits reuse freed space through free-space managers. Under `Aggr` and [`None`](crate::FileSpaceStrategy::None), deletion discards the freed extents. `None` allocates at the end of the file and retains its length after deletion.

## The threshold parameter

The `threshold` argument is the smallest free-space section, in bytes, that the free-space managers track, and the C library's default is `1` (every freed section is eligible). Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) and [`Page`](crate::FileSpaceStrategy::Page) it is recorded in the file and round-trips through the reference C library.

Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) and [`Page`](crate::FileSpaceStrategy::Page) a read-write session tracks a freed region of at least the threshold. It tracks a shorter one only where the region adjoins tracked free space, which it merges into, and drops it otherwise, as the C library does. A dropped region is not reused. Under [`Page`](crate::FileSpaceStrategy::Page) a region shorter than a page merges only with free space in its own page. The writer and the editors track page tails whatever the threshold, and so does the C library.

## Reading the strategy back

The strategy lives in a superblock-extension message (a standalone object header the superblock points at), so it survives a reopen. [`File::file_space_strategy()`](crate::File::file_space_strategy) returns the recorded [`FileSpaceStrategy`](crate::FileSpaceStrategy), and [`File::file_space_info()`](crate::File::file_space_info) returns the full [`FileSpaceInfo`](crate::FileSpaceInfo) record (persist flag, threshold, page size, and the free-space manager addresses).

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("out.h5");
# let mut b = hdf5_pure::FileBuilder::new();
# b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
# b.with_file_space_strategy(hdf5_pure::FileSpaceStrategy::Page, false, 1)
#     .with_file_space_page_size(8192);
# b.write(&path)?;
use hdf5_pure::File;

let file = File::open(&path)?;
if let Some(info) = file.file_space_info() {
    println!("strategy: {:?}", info.strategy);
    println!("persist:  {}", info.persist);
    println!("threshold: {}", info.threshold);
    println!("page size: {}", info.page_size.get());
}
# assert_eq!(file.file_space_info().map(|i| i.page_size.get()), Some(8192));
# Ok::<(), hdf5_pure::Error>(())
```

[`file_space_strategy()`](crate::File::file_space_strategy) (and [`file_space_info()`](crate::File::file_space_info)) return `None` for a file without a recorded strategy, which is what the C library writes when the default is left in place.

## Paged files (`FileSpaceStrategy::Page`)

[`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page) writes a **genuine paged file**, not just a recorded label. Every allocation is aligned to the page size set by [`with_file_space_page_size`](crate::FileBuilder::with_file_space_page_size), which defaults to 4096 and must be from 512 bytes to 1 GiB, with any userblock a whole number of pages. Each page is kept homogeneous (defined precisely below), and each page's free tail is tracked in a per-page-type free-space manager. The reference HDF5 C library reads the result as a paged file, parses the managers (`H5Fget_freespace` matches the tracked total), reads every dataset, and re-paginates the file when it writes to it.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("paged.h5");
use hdf5_pure::{FileBuilder, FileSpaceStrategy, MaxExtent};

let mut b = FileBuilder::new();
b.create_dataset("samples")
    .with_i32_data(&(0..1000).collect::<Vec<i32>>())
    .with_shape(&[1000])
    .with_maxshape(&[MaxExtent::Unlimited])
    .with_chunks(&[256]);
b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0) // persist so it can be grown later
    .with_file_space_page_size(4096);
b.write(&path)?;
# assert_eq!(
#     hdf5_pure::File::open(&path)?.file_space_strategy(),
#     Some(FileSpaceStrategy::Page)
# );
# Ok::<(), hdf5_pure::Error>(())
```

A paged file is grown in place through [`File::open_rw`](crate::File::open_rw), whichever backing it picks: it appends in bounded memory and rewrites the per-page-type managers at [`File::close`](crate::File::close), and commits the full staged edit surface through a page-aware tail that rewrites the same managers. Both keep each page homogeneous and page-align the end of allocation, so the reference C library reopens the result as a paged file. Homogeneous here means a page holds either raw allocations or metadata allocations, with one deliberate exception: the array chunk indexes this crate writes share the raw pages of the chunk data beside them and are tracked as raw space to match. A version 2 B-tree chunk index remains metadata, including when the reference C library wrote it. Object headers, fractal-heap structures, and free-space managers use metadata pages. Global heap collections use raw pages. A dense-attribute blob is one metadata reservation, including any embedded huge-object payloads. Separately allocated huge-object payloads use raw allocations. A paged file created **without** `persist = true` has no on-disk record of which pages hold metadata versus raw data, so neither editor can keep the two segregated and both reject it ([`Error::EditUnsupported`](crate::Error::EditUnsupported)) before any byte changes. Recreate it with `persist = true` to append to it later. [`File::open_rw`](crate::File::open_rw) opens a paged file with a userblock in the whole-file mirror, as it opens every file with a userblock, and [`File::commit`](crate::File::commit) applies the same staged edits to it, with its pages counted from the end of the userblock. [`Dataset::append`](crate::Dataset::append) returns [`Error::AppendInPlaceUnsupported`](crate::Error::AppendInPlaceUnsupported) there, as on any file with a userblock.

Deleting from a paged file returns the space to the per-page-type free lists, so a create/delete cycle settles at a steady size, and whole free pages at the end of the file are given back to the filesystem. Deleting a version 2 B-tree chunk-indexed dataset returns its `BTHD`, `BTIN`, and `BTLF` allocations as metadata and its recorded chunk allocations as raw data. Deleting the last hard link of an object with dense attribute or link storage returns the name and creation-order version 2 B-tree allocations and the header and blocks of each fractal heap as metadata. A heap's huge objects return as raw data where they sit apart from the heap's metadata, and as metadata where they adjoin it inside a page. A heap allocation of a page or more that begins on a page boundary is held as dead, since the reference C library allocates it as whole pages of neither type. A freed region whose page type cannot be established is held as dead, and the page around it is returned once every other byte of it is free too. That still covers an array chunk index this crate did not lay down beside its chunk data, most often the index of an empty resizable dataset, which has no chunk data at all. [`space_accounting().reusable_free_bytes`](crate::SpaceAccounting) therefore lags a delete by up to a page on such a file, and it does not accumulate.

## Persisting free space across sessions

Passing `persist = true` under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) or [`Page`](crate::FileSpaceStrategy::Page) records that freed space should be tracked on disk across closes. Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) a brand-new file has no free space, so this initially records the intent alone. A genuine paged file (above) already tracks its page-tail free space from creation. When a later edit frees a region, by deleting a dataset for example, the freed region is recorded in on-disk free-space-manager blocks (`FSHD`/`FSSE`), and a later session, this crate's or the reference C library's, recovers and reuses it.

[`File::persisted_free_space()`](crate::File::persisted_free_space) returns the tracked free regions as `(address, length)` pairs sorted by address, and a read-write session seeds its free list from them, so reuse spans sessions.

Every commit rewrites those manager blocks, and they are themselves placed in free space where any fits, so a file under delete-and-recreate churn settles at a steady size. Active aggregate spare remains reserved and is excluded from those blocks. Close or drop returns unused spare through a manager rewrite, even if allocation within a window left the file length unchanged. An uncertain publication prevents this cleanup. Under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) a threshold above the length of the blocks is the exception: the file grows by a set of blocks on each commit that appends the new set at its end, since the session drops each superseded set that is isolated from tracked free space.

Freed space that reaches the end of the file is given back to the filesystem: the commit shortens the file to where the run starts, to a page boundary on a paged file, and the managers it writes cover nothing above the new length. A few blocks' worth is kept for the tails the following commits have to write, and a run that would return less than that is left alone, so what this guarantees is that the file ends just above its last live allocation, not that a delete never makes a file longer. For a guaranteed shrink that also moves live objects down, use [repack](crate::_guide::repack).

An in-place [`Dataset::append`](crate::Dataset::append) reuses persisted free space as well, under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) and [`Page`](crate::FileSpaceStrategy::Page) alike. Before spending a free region, the editor removes it from the on-disk managers in a separate crash-safe publication. One draw takes every hole the appended chunk fits in, up to a megabyte of them. Unspent reserve returns at the next commit or close. A crash between reservation and return strands those bytes without advertising them as reusable. Under `FsmAggr`, appends try this reserve before the metadata or raw aggregate. Unused aggregate reservations are already excluded from the persisted free sections.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("managed.h5");
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

let mut builder = FileBuilder::new();
builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1); // persist = true
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
let free = file.persisted_free_space()?;
let total_free: u64 = free.iter().map(|&(_, len)| len).sum();
println!("persisted free regions: {}", free.len());
println!("total persisted free bytes: {total_free}");
# assert!(total_free > 0);
# Ok::<(), hdf5_pure::Error>(())
```

[`persisted_free_space()`](crate::File::persisted_free_space) returns an empty list when the file does not persist free space, and on a handle that reads the file on demand or edits it. It returns an error for a malformed free-space manager. A read-write session opens such a file and skips the free regions of that manager, and in a file that is not paged those of every other manager too. The addresses are file offsets relative to the base address, and [reading data](crate::_guide::reading) is unaffected by the presence or absence of these managers.

When free space is not persisted under [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), an ordinary read-write session reuses freed space before aggregate reservations, with both kept in memory for that session alone. Under [`Aggr`](crate::FileSpaceStrategy::Aggr), staged edits and immediate appends use unused aggregate reservations within the session. Under `Aggr` and [`None`](crate::FileSpaceStrategy::None), freed extents are discarded. If churn has left a file with unused gaps and you want to reclaim them outright, [repacking](crate::_guide::repack) rewrites the file compactly.
