# SWMR (Single Writer, Multiple Readers)

SWMR lets a single process append to an unlimited dataset in place while other processes read it concurrently, and it interoperates with the reference HDF5 C library and h5py in both directions. This page covers how to lay out a SWMR-capable dataset, append to it durably, follow it from a reader, and recover a file left flagged by a writer that exited uncleanly.

!!! tip "Runnable example"
    A complete single-process demonstration lives in [`examples/swmr.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/swmr.rs). Run it with:

    ```bash
    cargo run --example swmr
    ```

    It writes and then reads in one process to show the mechanics; in practice the writer and reader are separate processes.

## How it works

The writer appends chunks and flushes them in dependency order, child structures before the parent metadata that references them, with a durability barrier after each step. The dataset's authoritative size (its dataspace dimension) is published last, as the single commit point: before it a reader sees the old length, after it the new one, and never a torn view. A reader therefore only ever observes a consistent prefix of the data. To pick up newly appended data, a reader re-reads with [`refresh()`](#following-a-growing-file).

Because the on-disk format is standard HDF5, a reader opened by this crate can follow a file being written by the reference C library or h5py in SWMR mode, and vice versa.

A SWMR writer honors [`SyncPolicy`](editing.md#choosing-the-fsync-cadence) like any other read-write session. A reader **on the same machine** is unaffected: what it sees is the operating system's view of the file, which every write reaches as it is made, in the order it was made — the `fsync` barriers carry that order across power loss, not across processes. `SyncPolicy::OnClose` therefore drops the per-append barriers, as the reference library's SWMR path does, while keeping the one at teardown that clears the flag below. One caveat comes with it: a reader on *another host* over NFS may not see writes a client is holding until a flush.

## Laying out the dataset

A SWMR-capable dataset must have one unlimited dimension and be chunked. The latest format indexes such a dataset with an Extensible Array, which is selected automatically. Create it with the usual [writing](writing.md) builder: set the initial extent with `with_shape`, mark the dimension unlimited with `with_maxshape(&[u64::MAX])`, and pick a chunk shape with `with_chunks`.

```rust
use hdf5_pure::FileBuilder;

let mut builder = FileBuilder::new();
builder
    .create_dataset("log")
    .with_i32_data(&[0, 1, 2])   // initial rows
    .with_shape(&[3])
    .with_maxshape(&[u64::MAX])  // one unlimited dimension
    .with_chunks(&[1]);
builder.write("stream.h5").unwrap();
```

## Appending in place

Open the existing file with `File::open_swmr_writer` and append through a `Dataset` handle. Each append call flushes durably, leaving the file valid for any concurrent reader throughout.

```rust
use hdf5_pure::File;

let writer = File::open_swmr_writer("stream.h5").unwrap();
let mut log = writer.dataset("log").unwrap();
log.append(&[3i32, 4, 5]).unwrap();
log.append(&[6i32, 7]).unwrap();
drop(log);
writer.close().unwrap(); // clears the SWMR flag; or just drop the file
```

`close()` clears the file's SWMR-write flag and flushes, marking the file cleanly closed. Prefer calling it over relying on `Drop`, so the rare flush error surfaces; dropping the writer also clears the flag.

The append helpers are:

| Method | Appends |
| --- | --- |
| `Dataset::append(&[T])` | values of any `H5Element` type (`i32`, `f64`, …) |
| `Dataset::append_raw(&[u8])` | raw little-endian element bytes |

`append` encodes its values as little-endian bytes and forwards to `append_raw`. With any of them, the appended length must be a whole number of chunks and the dataset's current length must already be chunk-aligned.

!!! warning "Appends must be chunk-aligned"
    Both the dataset's current length and each appended length must be multiples of the chunk length. An append that is not chunk-aligned, or whose byte length is not a whole number of elements, returns an error and publishes nothing, so a reader still sees the prior consistent prefix. After such an error the file should be dropped rather than reused, because its in-memory mirror may have advanced past what reached disk.

## Following a growing file

Open the file for reading with `File::open_swmr`, which retains a live filesystem handle so the reader can re-read appended data. The initial view is a consistent snapshot; call `refresh()` to advance to a newer one after the writer appends.

```rust
use hdf5_pure::File;

let mut file = File::open_swmr("stream.h5").unwrap();
let n = file.dataset("log").unwrap().shape().unwrap()[0];
// ... later, after the writer appends ...
file.refresh().unwrap();                 // re-read appended data
let ds = file.dataset("log").unwrap();
println!("now {} rows", ds.shape().unwrap()[0]);
```

`refresh()` is the SWMR reader's refresh primitive, analogous to the C library's `H5Drefresh` and h5py's `Dataset.refresh()`. After it returns, newly fetched `Dataset` and `Group` handles observe the appended chunks and extended dimensions. Handles are owned and keep the file open, so `refresh()` needs exclusive access: drop any outstanding `Dataset`/`Group` handle (and any `File` clone) before calling it — otherwise it returns `Error::HandlesOutstanding` — then re-fetch the handles afterward, as shown above. See [reading](reading.md) for the dataset access APIs used here.

!!! note "Refresh cost"
    Each `refresh()` re-reads the entire file from disk (`O(file size)`) and re-validates the superblock checksum; a transient parse failure from catching a writer mid-flush is retried a bounded number of times. When following a large, steadily growing log, budget refresh frequency accordingly. `refresh()` returns an error if the file was not opened with `File::open_swmr` (the in-memory `File::from_bytes` path cannot refresh).

## Recovering a flagged file

While a `File::open_swmr_writer` is open, the file's superblock carries an active-SWMR-writer flag (matching the reference C library and h5py) so concurrent readers may open it accordingly. `close()` or dropping the writer clears it. If a writer process exits without a clean close, the file is left flagged.

That flag is durable under every `SyncPolicy` — clearing it is one of the writes `close` and `drop` force a barrier for, since a lost clear would refuse every subsequent open until `File::clear_swmr_flag` ran. It is what every other open consults: while it stands, `File::open`, `File::open_streaming`, `File::from_source`, `File::open_rw` and a second `File::open_swmr_writer` all fail with `Error::FileMarkedInUse`, exactly as `H5Fopen` fails with *"file is already open for write"*. `File::open_swmr` is the one open that follows it rather than refusing — that pairing is what the flag is for. `File::from_bytes` does not consult it, since its caller already holds a snapshot of the bytes.

A page-buffered writer raises bit 0 of the same byte *without* the SWMR bit, and no SWMR reader can follow half a pair. That mark has a reader of its own: `FileAccessProperties::with_write_mark_policy(WriteMarkPolicy::AllowSnapshot)`, which admits a snapshot read of it and never of a SWMR pair. See [Reading](reading.md).

The flag cannot distinguish a live writer from a crashed one, so recover a file you know has no writer with the h5clear equivalent:

```rust
use hdf5_pure::File;

File::clear_swmr_flag("stream.h5").unwrap();
```

`clear_swmr_flag` is safe to call on a file whose flag is already clear. It takes the exclusive OS lock first, so it cannot clear the flag out from under a live `File::open_rw` writer — but a SWMR writer holds no lock, so check that one is really gone before clearing.

The check applies to version-3 superblocks, which is where the C library applies it and the only version this crate's SWMR writer accepts.

## Supported subset and requirements

SWMR append supports the following subset, distinct from the general [editing](editing.md) and writing paths:

| Requirement | Detail |
| --- | --- |
| Dimensionality | exactly one unlimited dimension |
| Storage | chunked (Extensible Array index, latest format) |
| Filters | unfiltered (no compression on the appended dataset); `File::open_rw` takes filtered ones |
| Append granularity | chunk-aligned appends |
| File layout | no userblock (zero base address); latest-format v3 superblock |
| Growth | unbounded |
| Build | requires `std` (the default); the in-memory/WASM path cannot refresh |

`File::open_swmr_writer` rejects files that fall outside this subset, returning `Error::SwmrAppendUnsupported` for a non-latest-format superblock or a userblock file before performing any mutating write.

!!! tip "Appending without SWMR"
    If you don't need concurrent readers, two paths append to an unlimited dataset in place with fewer restrictions than the SWMR writer — both handle **filtered** (compressed) datasets, from any length and by any length. [`Dataset::append_staged`](editing.md#appending-to-an-unlimited-dataset) is the general, composable one-off path. [`File::open_rw` + `Dataset::append`](editing.md#streaming-appends) is the throughput path that stays open across many appends and grows the index in place at amortized `O(1)` cost. Reach for SWMR only when a separate process must read the dataset while it grows.
