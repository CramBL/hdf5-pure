# File-Space Strategy

A written HDF5 file can record how it manages free space: which allocation strategy it uses, whether freed regions are tracked across closes, and the page size used for paged allocation. This page covers `FileBuilder::with_file_space_strategy` and `with_file_space_page_size`, which mirror HDF5's `H5Pset_file_space_strategy` and `H5Pset_file_space_page_size`, and how the choice interacts with [editing](editing.md) and [repacking](repack.md).

!!! tip
    A runnable example lives at [examples/file_space.rs](https://github.com/CramBL/hdf5-pure/blob/main/examples/file_space.rs). Run it with:

    ```bash
    cargo run --example file_space
    ```

## Recording a strategy

`FileBuilder::with_file_space_strategy(strategy, persist, threshold)` records the file-space management strategy, whether free space is persisted across closes, and the smallest free section the managers will track. `with_file_space_page_size(size)` sets the file-space page size used for paged allocation. Both are recorded in the file's superblock extension, so the reference HDF5 C library and a later reopen observe the choice.

Both are also fields on `FileCreateProperties`, the reusable file-creation-property value (the `fcpl` analogue). Build one in a helper, then apply it with `FileBuilder::with_create_properties` or `File::create_with_options` wherever a file is written, instead of repeating the call chain:

```rust
use hdf5_pure::{FileCreateProperties, FileSpaceStrategy};

fn paged_layout() -> FileCreateProperties {
    FileCreateProperties::new()
        .with_file_space_strategy(FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(8192)
}
```

```rust
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

let mut b = FileBuilder::new();
b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
b.with_file_space_strategy(FileSpaceStrategy::Page, false, 1) // strategy, persist, threshold
    .with_file_space_page_size(8192);
b.write("out.h5").unwrap();

assert_eq!(
    File::open("out.h5").unwrap().file_space_strategy(),
    Some(FileSpaceStrategy::Page)
);
```

## Strategy variants

The `FileSpaceStrategy` enum mirrors HDF5's `H5F_fspace_strategy_t`:

| Variant | HDF5 constant | Behavior |
|---|---|---|
| `FileSpaceStrategy::FsmAggr` | `H5F_FSPACE_STRATEGY_FSM_AGGR` | Free-space managers, aggregators, and the virtual file driver — the HDF5 default. |
| `FileSpaceStrategy::Page` | `H5F_FSPACE_STRATEGY_PAGE` | Paged aggregation backed by free-space managers. |
| `FileSpaceStrategy::Aggr` | `H5F_FSPACE_STRATEGY_AGGR` | Aggregators and the virtual file driver only, no free-space managers. |
| `FileSpaceStrategy::None` | `H5F_FSPACE_STRATEGY_NONE` | No free-space tracking; allocation only ever appends. |

## The threshold parameter

The `threshold` argument is the smallest free-space section, in bytes, that the free-space managers are asked to track; the C library's default is `1` (every freed section is eligible). It is recorded in the file and round-trips through the reference C library. In this crate the value is currently **advisory**: the paged writer and the bounded editor track every page tail and freed section regardless of the recorded threshold, so a `threshold > 1` is preserved on disk but does not change which sections this crate records.

## Reading the strategy back

The strategy lives in a superblock-extension message (a standalone object header the superblock points at), so it survives a reopen. `File::file_space_strategy()` returns the recorded `FileSpaceStrategy`, and `File::file_space_info()` returns the full `FileSpaceInfo` record (persist flag, threshold, page size, and the free-space manager addresses).

```rust
use hdf5_pure::File;

let file = File::open("out.h5").unwrap();
if let Some(info) = file.file_space_info() {
    println!("strategy: {:?}", info.strategy);
    println!("persist:  {}", info.persist);
    println!("threshold: {}", info.threshold);
    println!("page size: {}", info.page_size);
}
```

`file_space_strategy()` (and `file_space_info()`) return `None` when the file records no strategy, which is what the C library also writes when the default is left in place.

## Paged files (`FileSpaceStrategy::Page`)

`FileSpaceStrategy::Page` writes a **genuine paged file**, not just a recorded label. Every allocation is aligned to the page size set by `with_file_space_page_size` (default 4096; it must be a power of two `>= 512`, and any userblock must be a whole number of pages), each page is kept homogeneous (defined precisely below), and each page's free tail is tracked in a per-page-type free-space manager. The reference HDF5 C library reads the result as a paged file, parses the managers (`H5Fget_freespace` matches the tracked total), reads every dataset, and re-paginates the file when it writes to it.

```rust
use hdf5_pure::{FileBuilder, FileSpaceStrategy};

let mut b = FileBuilder::new();
b.create_dataset("samples")
    .with_i32_data(&(0..1000).collect::<Vec<i32>>())
    .with_shape(&[1000])
    .with_maxshape(&[u64::MAX])
    .with_chunks(&[256]);
b.with_file_space_strategy(FileSpaceStrategy::Page, true, 0) // persist so it can be grown later
    .with_file_space_page_size(4096);
b.write("paged.h5").unwrap();
```

A paged file is grown in place through [`File::open_rw`](editing.md), whichever backing it picks: it appends in bounded memory and rewrites the per-page-type managers at `File::close`, and commits the full staged edit surface through a page-aware tail that rewrites the same managers. Both keep each page homogeneous and page-align the end of allocation, so the reference C library reopens the result as a paged file. Homogeneous here means a page holds either raw dataset bytes or file metadata, never both, with one deliberate exception: a chunked dataset's index structure is written in the same run as the chunk data it indexes, so it shares those raw pages and is tracked as raw space to match. Object headers, heaps, and the free-space managers themselves always sit in metadata pages. A paged file created **without** `persist = true` has no on-disk record of which pages hold metadata versus raw data, so neither editor can keep the two segregated and both refuse it (`Error::EditUnsupported`) before any byte changes — recreate it with `persist = true` if you need to append to it later. A paged file with a userblock is refused for the same reason: the free-space bookkeeping is not yet base-address aware, so its managers cannot be seeded and the page bookkeeping has nothing to work from.

Deleting from a paged file returns the space to the per-page-type free lists, so a create/delete cycle settles at a steady size rather than growing, and whole free pages at the end of the file are given back to the filesystem. Where a freed region's page type cannot be established — a chunk index this crate did not lay down beside its chunk data, most often the index of an empty resizable dataset, which has no chunk data at all — the region is held as dead rather than offered for reuse, and the page around it is returned once every other byte of it is free too. `space_accounting().reusable_free_bytes` therefore lags a delete by up to a page on such a file; it does not accumulate.

## Persisting free space across sessions

Passing `persist = true` records that freed space should be tracked on disk across closes. For the non-paged strategies (`FsmAggr`, `Aggr`, `None`) a brand-new file has no free space, so this initially only records the intent; a genuine paged file (above) already tracks its page-tail free space from creation. When a later edit frees a region — for example by deleting a dataset — the freed region is recorded in on-disk free-space-manager blocks (`FSHD`/`FSSE`) rather than discarded, and a later session, this crate's or the reference C library's, recovers and reuses it.

`File::persisted_free_space()` returns the tracked free regions as `(address, length)` pairs sorted by address, and a read-write session seeds its free list from them so reuse spans sessions rather than just the open session.

Every commit rewrites those manager blocks, and they are themselves placed in free space where any fits, so a file under delete-and-recreate churn settles at a steady size instead of gaining a set of managers per commit.

Freed space that reaches the end of the file is given back to the filesystem rather than recorded: the commit shortens the file to where the run starts — to a page boundary on a paged file — and the managers it writes name nothing above the new length. A few blocks' worth is kept for the tails the following commits have to write, and a run that would return less than that is left alone, so what this guarantees is that the file ends just above its last live allocation, not that a delete never makes a file longer. For a guaranteed shrink that also moves live objects down, use [repack](repack.md).

An in-place `Dataset::append` reuses persisted free space as well, on the flat strategies and on a paged file alike, but it has no superblock repoint of its own to update the managers with. It therefore takes space out of them first — every hole the appended chunk fits in, up to a megabyte of them per draw, rewriting the managers without that space under the same crash-atomic repoint a commit uses — and spends only from there, so a hole the managers still advertise is never written into. What a session does not spend goes back at its next commit and at close; a crash in between strands it, which wastes those bytes and never hands them out twice. The file grows only when no hole can hold the chunk.

```rust
use hdf5_pure::{File, FileBuilder, FileSpaceStrategy};

let mut builder = FileBuilder::new();
builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 1); // persist = true
builder.write("managed.h5").unwrap();

// Create then delete a dataset through the owned-handle editor; the freed
// region is recorded on disk.
let file = File::open_rw("managed.h5").unwrap();
file.root()
    .create_dataset("scratch", |b| {
        b.with_f64_data(&vec![0.0; 4096]);
    })
    .unwrap();
file.commit().unwrap();
file.root().delete("scratch").unwrap();
file.commit().unwrap();
file.close().unwrap();

// Later opens can read the persisted free regions.
let file = File::open("managed.h5").unwrap();
let free = file.persisted_free_space();
let total_free: u64 = free.iter().map(|&(_, len)| len).sum();
println!("persisted free regions: {}", free.len());
println!("total persisted free bytes: {total_free}");
```

!!! note
    `persisted_free_space()` is empty when the file does not persist free space, and for the streaming backend (which does not load the manager blocks). The addresses are file offsets relative to the base address; [reading data](reading.md) is unaffected by the presence or absence of these managers.

When free space is not persisted, a read-write session still reuses space within a single session but does not carry a free list across closes. If churn has left a file with unused gaps and you want to reclaim them outright, [repacking](repack.md) rewrites the file compactly.
