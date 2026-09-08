//! What the read paths are allowed to allocate, as rules rather than as numbers
//! (issue #228).
//!
//! Every bound here is a statement about how the work *scales* — a windowed read
//! allocates on the order of its window, a chunked read allocates a fixed number
//! of blocks per chunk — so it holds on every platform, where an exact count does
//! not. Those are pinned separately, on one platform, in
//! `tests/allocation_baseline.rs`.
//!
//! Scaling is not the same as unbreakable, and the margins differ: the byte and
//! peak bounds have several times' headroom and will not notice a `Vec` growth
//! step anywhere, while `BLOCKS_PER_CHUNK` sits about 3% above a measured
//! 3.00-per-chunk rate on purpose — one more allocation *inside the per-chunk
//! loop* is exactly what it exists to catch, and a growth step there would fail
//! it too. That is the trade it is meant to make.
//!
//! Measurement is per region and per thread (`tests/common/allocation.rs`), so
//! these tests share a process without serializing against each other: what a
//! region records is what the calling thread allocated inside it, and nothing
//! else. That last part is also how a test here could go quiet, which is why each
//! one asserts a floor as well as a ceiling — see [`allocation::Measured`].

use hdf5_pure::{
    EditBacking, File, FileAccessProperties, FileBuilder, FileSpaceStrategy, MemoryStrategy,
    SyncPolicy,
};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

use test_util::allocation;

use allocation::measure;

#[test]
fn windowed_row_read_allocates_on_the_order_of_the_window() {
    // 4 MiB of f64 rows with inner-split storage chunks: [2048, 32, 8] in
    // [64, 16, 4] chunks — a 2x2 inner chunk grid per 64-row band, deflated so
    // chunks really decode (32 KiB decompressed each).
    const N0: usize = 2048;
    const ROW_ELEMS: usize = 32 * 8;
    const DATASET_BYTES: usize = N0 * ROW_ELEMS * 8;

    let data: Vec<f64> = (0..N0 * ROW_ELEMS).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64, 32, 8])
        .with_chunks(&[64, 16, 4])
        .with_deflate(3);
    builder.write(&path).unwrap();
    drop(data);

    // The streaming backend reads from disk on demand; the buffered backend
    // would hold the whole file in memory by design and drown the signal.
    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    // A 64-row window straddling a chunk-band boundary mid-file, so it decodes
    // chunks from two leading bands.
    let (window, measured) = measure("windowed_row_read", || ds.read_f64_rows(992, 64).unwrap());

    // Window + raw/typed conversion + a few decompressed 32 KiB chunks lands
    // well under 1 MiB; a whole-read fallback peaks above the full dataset
    // (the assembled whole read plus cached chunks and the sliced window).
    assert!(
        measured.peak_bytes < (DATASET_BYTES / 4) as u64,
        "peak allocation during the windowed read must be bounded by the window, \
         not the {DATASET_BYTES}-byte dataset; measured {measured}"
    );

    // The read's own output is in this measurement, so its size is a floor a
    // measurement of nothing cannot reach. Without it the ceiling above passes
    // on a read whose work moved to a thread this region does not see.
    const WINDOW_BYTES: usize = 64 * ROW_ELEMS * 8;
    assert!(
        measured.live_bytes >= WINDOW_BYTES as u64,
        "the window this read returned is not in its own measurement, so the \
         ceiling above is bounding something other than the read: {measured}"
    );

    // The window must still be the right bytes.
    let expected: Vec<f64> = (992 * ROW_ELEMS..(992 + 64) * ROW_ELEMS)
        .map(|i| i as f64)
        .collect();
    assert_eq!(window, expected);
}

#[test]
fn windowed_vlen_string_read_allocates_on_the_order_of_the_window() {
    // ~4 MiB of variable-length string payload: 32k rows of 128-byte strings,
    // plus ~512 KiB of heap references in the dataset itself. The writer packs
    // the strings into one giant heap collection — the degenerate case for a
    // windowed read, whose directory alone rivals the window if parsed whole.
    const N0: usize = 32 * 1024;
    const STR_LEN: usize = 128;
    const PAYLOAD_BYTES: usize = N0 * STR_LEN;

    let strings: Vec<String> = (0..N0)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    builder.write(&path).unwrap();
    drop(refs);
    drop(strings);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("labels").unwrap();

    // 256 mid-file rows — ~32 KiB of text plus ~4 KiB of references, resolved
    // against a window-filtered slice of the collection's directory.
    let (window, measured) = measure("windowed_vlen_read", || {
        ds.read_string_rows(15_000, 256).unwrap()
    });

    // Window references + text + touched heap-collection directories land far
    // under 1 MiB; resolving every reference first peaks above the full payload
    // (all references plus a Vec<String> of every row). This also catches a
    // collection parse that starts buffering whole collections instead of
    // walking their metadata.
    assert!(
        measured.peak_bytes < (PAYLOAD_BYTES / 4) as u64,
        "peak allocation during the windowed vlen read must be bounded by the window, \
         not the {PAYLOAD_BYTES}-byte payload; measured {measured}"
    );

    // The other half of the same story, which the peak cannot see. A global heap
    // collection chains each object's position in the previous object's header,
    // so reaching the window's objects means passing every object before them —
    // the traversal is the format's, not this crate's, and no bound here can
    // remove it. What it must not cost is an *allocation* apiece: reading this
    // window once took 65,536 of them, two per object of a collection it kept 256
    // entries from (issue #228). One per eight objects is far above what a
    // windowed walk needs and far below what a per-object one takes, which is the
    // gap this bound is placed in.
    assert!(
        measured.blocks < (N0 / 8) as u64,
        "a windowed vlen read must not allocate per object of the collection it \
         walks: measured {measured} over a {N0}-object collection"
    );

    // The third axis, and the one this test was blind to: bytes *moved*. The
    // peak cannot see it, because each directory window replaces the last rather
    // than accumulating, and the block count *falls* when refills get larger —
    // so a change that reads a megabyte per refill instead of a page passes both
    // bounds above while reading thirty times the file. What holds is that the
    // walk crosses the collection about once: it must pass every object before
    // the window, since the format chains each object's position in the previous
    // one, but it has no reason to pass any of them twice.
    assert!(
        measured.bytes < (PAYLOAD_BYTES * 2) as u64,
        "a windowed vlen read must cross its collection about once, not \
         repeatedly: measured {measured} against a {PAYLOAD_BYTES}-byte payload"
    );

    // The strings this read returned are in the measurement, so their size is a
    // floor a measurement of nothing cannot reach.
    const WINDOW_TEXT_BYTES: usize = 256 * STR_LEN;
    assert!(
        measured.live_bytes >= WINDOW_TEXT_BYTES as u64,
        "the strings this read returned are not in its own measurement, so the \
         bounds above are bounding something other than the read: {measured}"
    );

    // The window must still be the right strings.
    assert_eq!(window.len(), 256);
    let expected: Vec<String> = (15_000..15_000 + 256)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    assert_eq!(window, expected);
}

/// A dataset whose chunks lie end to end for its whole length is what the
/// coalescing reader (`src/chunk_span.rs`) exists for, and the worst case for
/// its memory bound: a coalescer with no budget would merge all 2,048 chunks
/// into a single 8 MiB span and hold the dataset a second time to serve them.
/// The 256 KiB budget is what keeps the span a constant beside the read rather
/// than a second copy of it.
///
/// The read here is [`Dataset::read_raw`] rather than a typed one on purpose:
/// a typed read allocates its own copy of the dataset, and a term that large
/// hides the one being measured.
///
/// The three bounds are one story told three ways — peak, bytes, blocks — and
/// each catches a defect the others do not: a second buffer of dataset size, a
/// throwaway copy of every chunk, and a per-chunk `Vec` in the scatter loop.
#[test]
fn whole_read_of_adjacent_chunks_costs_a_constant_per_chunk() {
    // 8 MiB of f64 in 4 KiB chunks: 2,048 of them, unfiltered so a chunk's
    // stored bytes are its data and a span holds no less than the file does.
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: usize = 512;
    const CHUNKS: u64 = (N0 / CHUNK_ELEMS) as u64;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS as u64]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    let (all, measured) = measure("whole_read", || ds.read_raw().unwrap());

    // The read's own output is the dataset, and the parsed chunk index is most
    // of what is left; an unbudgeted span would add the dataset again.
    assert!(
        measured.peak_bytes < (DATASET_BYTES + DATASET_BYTES / 2) as u64,
        "peak allocation during the whole read must not grow with the run of \
         adjacent chunks; measured {measured} against a {DATASET_BYTES}-byte dataset"
    );

    // Bytes *ever* allocated, which the peak cannot see: a copy of each chunk
    // that is made and dropped costs the dataset over again without moving the
    // high-water mark at all. The read's own output is one dataset; the chunk
    // cache is entitled to its configured budget on top, and nothing else here
    // scales with the data.
    assert!(
        measured.bytes < (DATASET_BYTES + DATASET_BYTES / 4) as u64,
        "a whole read must allocate the dataset about once — its output — not a \
         second time in throwaway per-chunk copies; measured {measured} against a \
         {DATASET_BYTES}-byte dataset"
    );

    // Per-chunk *count*, which neither byte figure can see: a `Vec` of the
    // chunk's coordinates costs 32 bytes and is invisible next to 8 MiB, but it
    // is an allocator round trip per chunk of every dataset this crate reads.
    const BLOCKS_PER_CHUNK: u64 = 3;
    assert!(
        measured.blocks <= BLOCKS_PER_CHUNK * CHUNKS + 256,
        "a whole read must cost a small constant number of allocations per chunk, \
         not a growing one: measured {measured} over {CHUNKS} chunks, above the \
         {BLOCKS_PER_CHUNK} per chunk this bound allows"
    );

    // The read's own output is the dataset, so its size is a floor a measurement
    // of nothing cannot reach — which is what all three ceilings above become if
    // the per-chunk work moves to a thread this region does not see.
    assert!(
        measured.live_bytes >= DATASET_BYTES as u64,
        "the dataset this read returned is not in its own measurement, so the \
         three bounds above are bounding something other than the read: {measured}"
    );

    assert_eq!(all.len(), DATASET_BYTES);
    assert_eq!(
        &all[DATASET_BYTES - 8..],
        &((N0 - 1) as f64).to_le_bytes()[..]
    );
}

/// A typed whole-dataset read costs its output, not its output *and* a whole
/// copy of the stored bytes beside it.
///
/// Decoding used to be a `read_raw` followed by a conversion of the entire
/// buffer, so both were live at once and the peak was twice the dataset: a
/// caller reading a 4 GiB array needed 8 GiB (issue #289). What replaced it
/// sweeps row windows, and both figures below say so — the peak because only one
/// window of stored bytes stands beside the output, and the byte total because
/// each window is decoded straight into that output rather than into a buffer
/// that is then copied into it.
///
/// The fixture is the one [`whole_read_of_adjacent_chunks_costs_a_constant_per_chunk`]
/// reads raw, at the same stored width as the requested type, so "the dataset"
/// is one figure for both the stored bytes and the decoded values and the two
/// tests are directly comparable. The margin is a quarter of the dataset against
/// a window of one mebibyte — an eighth of it — so a window budget that grew
/// with the dataset instead of staying constant would fail this, and so would a
/// return to reading whole.
#[test]
fn typed_whole_read_costs_its_output_and_a_window() {
    // 8 MiB of f64 in 4 KiB chunks, read as `f64`: output and stored bytes are
    // the same size, so one constant names both.
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: usize = 512;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS as u64]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    let (values, measured) = measure("typed_whole_read", || ds.read_f64().unwrap());

    assert!(
        measured.peak_bytes < (DATASET_BYTES + DATASET_BYTES / 4) as u64,
        "peak allocation during a typed whole read must be its output plus a \
         window, not a second copy of the {DATASET_BYTES}-byte dataset; \
         measured {measured}"
    );

    // Bytes *ever* allocated, which the peak cannot see: a window decoded into a
    // fresh `Vec` and then appended to the output, or a chunk copied into a
    // cache that has no room for it, costs the whole dataset over again without
    // moving the high-water mark at all. Two copies are what this read is
    // entitled to — its output, and the stored bytes it sweeps past one window
    // at a time — plus the span buffer each window refills, so the bound is
    // placed where a *third* would put it.
    assert!(
        measured.bytes < (3 * DATASET_BYTES) as u64,
        "a typed whole read must allocate its output and one sweep of the \
         stored bytes, not a third copy of either; measured {measured} against \
         a {DATASET_BYTES}-byte dataset"
    );

    // Per-chunk *count*, which neither byte figure can see, and the axis a sweep
    // is most easily made quadratic on: a window that takes the whole chunk
    // index out of the cache to find the chunks it overlaps allocates per chunk
    // of the *dataset* per window, and this read visits each chunk once through
    // eight windows.
    const BLOCKS_PER_CHUNK: u64 = 5;
    const CHUNKS: u64 = (N0 / CHUNK_ELEMS) as u64;
    assert!(
        measured.blocks <= BLOCKS_PER_CHUNK * CHUNKS + 256,
        "a typed whole read must cost a small constant number of allocations \
         per chunk, not one per chunk per window: measured {measured} over \
         {CHUNKS} chunks, above the {BLOCKS_PER_CHUNK} per chunk this bound allows"
    );

    // The values this read returned are in the measurement, so their size is a
    // floor a measurement of nothing cannot reach.
    assert!(
        measured.live_bytes >= DATASET_BYTES as u64,
        "the values this read returned are not in its own measurement, so the \
         bounds above are bounding something other than the read: {measured}"
    );

    // The values must still be the right ones, in the right order — a sweep that
    // lost or repeated a window would pass every bound above.
    assert_eq!(values.len(), N0);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[N0 / 2], (N0 / 2) as f64);
    assert_eq!(values[N0 - 1], (N0 - 1) as f64);
}

/// Writing a chunked dataset costs a constant per chunk, and about one copy of
/// the data beyond the one the caller handed over.
///
/// The two bounds catch different things. The count catches a scratch `Vec` in
/// the per-chunk loop — the splitter kept three of them, one of which was a
/// coordinate vector no caller ever read. The byte bound catches a whole second
/// copy of the dataset, which is what the object-header sizing pass was making:
/// it built the entire data region to learn how long it would be, then dropped it
/// and built it again at the real address (issue #228).
#[test]
fn chunked_write_costs_a_constant_per_chunk() {
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: u64 = 512;
    const CHUNKS: u64 = (N0 as u64) / CHUNK_ELEMS;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("write.h5");

    let ((), measured) = measure("chunked_write", || {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[N0 as u64])
            .with_chunks(&[CHUNK_ELEMS]);
        builder.write(&path).unwrap();
    });

    // The writer's own copy of the caller's data is in this measurement, so the
    // dataset's size is a floor a measurement of nothing cannot reach.
    assert!(
        measured.bytes >= DATASET_BYTES as u64,
        "the write's own copy of the data is not in this measurement, so the \
         bounds below are measuring something other than the write: {measured}"
    );

    // Measured at 3.05 copies: the caller's bytes staged in the builder, the
    // split into chunk buffers, and the assembled data region. A fourth is a
    // regression; anything that removes one of these three lowers it.
    assert!(
        measured.bytes < (DATASET_BYTES * 7 / 2) as u64,
        "a chunked write may hold the dataset a small number of times over, not \
         one more: measured {measured} against a {DATASET_BYTES}-byte dataset"
    );

    // Measured at 1.06 allocations per chunk, and allowed half again. Two per
    // chunk — what this allowed first — is exactly what *one* restored scratch
    // `Vec` in the splitter loop costs, so a bound there absorbs the defect it
    // names: the splitter kept three, and only restoring all three failed it.
    const BLOCKS_PER_TWO_CHUNKS: u64 = 3;
    assert!(
        measured.blocks <= BLOCKS_PER_TWO_CHUNKS * CHUNKS / 2 + 256,
        "a chunked write must cost a small constant number of allocations per \
         chunk: measured {measured} over {CHUNKS} chunks, above the \
         {BLOCKS_PER_TWO_CHUNKS} per two chunks this bound allows"
    );

    // The file must still be the right file.
    let file = File::open(&path).unwrap();
    let back = file.dataset("t").unwrap().read_f64().unwrap();
    assert_eq!(back.len(), N0);
    assert_eq!(back[N0 - 1], (N0 - 1) as f64);
}

/// A filtered write builds one compressor, not one per chunk.
///
/// A zlib compressor holds ~300 KiB of hash tables, so building one per chunk
/// made writing this 8 MiB dataset allocate 743 MB — ninety times the data, and
/// by a wide margin the most expensive thing this crate did (issue #228). The
/// bound is on bytes because that is the axis a per-chunk codec moves; a count
/// bound alone would miss it, since the codec is a handful of allocations that
/// happen to be enormous.
#[test]
fn filtered_write_does_not_build_a_compressor_per_chunk() {
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: u64 = 512;
    const CHUNKS: u64 = (N0 as u64) / CHUNK_ELEMS;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deflate.h5");

    let ((), measured) = measure("filtered_write", || {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[N0 as u64])
            .with_chunks(&[CHUNK_ELEMS])
            .with_deflate(3);
        builder.write(&path).unwrap();
    });

    assert!(
        measured.bytes >= DATASET_BYTES as u64,
        "the write's own copy of the data is not in this measurement: {measured}"
    );

    // One compressor per chunk would be ~615 MiB here, seventy times this bound.
    assert!(
        measured.bytes < (DATASET_BYTES * 4) as u64,
        "a filtered write must build its compressor once, not per chunk: measured \
         {measured} against a {DATASET_BYTES}-byte dataset over {CHUNKS} chunks"
    );

    let file = File::open(&path).unwrap();
    let back = file.dataset("t").unwrap().read_f64().unwrap();
    assert_eq!(back.len(), N0);
    assert_eq!(back[N0 - 1], (N0 - 1) as f64);
}

/// The same for the read side, where a decoder per chunk cost 174 MB to read the
/// 8 MiB back (issue #228).
#[test]
fn filtered_read_does_not_build_a_decompressor_per_chunk() {
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deflate_read.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[512])
        .with_deflate(3);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();
    let (all, measured) = measure("filtered_read", || ds.read_raw().unwrap());

    assert!(
        measured.bytes >= DATASET_BYTES as u64,
        "the read's own output is not in this measurement: {measured}"
    );

    // One decoder per chunk would be ~85 MiB of Huffman state on top of the
    // decoded data, ten times this bound.
    assert!(
        measured.bytes < (DATASET_BYTES * 4) as u64,
        "a filtered read must build its decoder once, not per chunk: measured \
         {measured} against a {DATASET_BYTES}-byte dataset"
    );

    assert_eq!(all.len(), DATASET_BYTES);
}

/// Opening one child of a group costs the group's header, not one allocation per
/// child in it.
///
/// A group stores its children as one Link message each, and a lookup used to
/// build an entry with an owned name for every one of them before returning the
/// one it was asked for: 2,084 allocations and 310 KiB to open a single dataset
/// out of 1,024, paid again on the next open. That is what makes walking a group
/// quadratic in its size, and this is the rule that keeps the walk linear
/// (issue #228).
///
/// Three measurements, because the lookup is reached by three routes that are
/// separate code: a path from the file and a name within an opened group, and
/// each of those over a streaming file and a buffered one — `File::open`, the
/// default entry point, resolves through a different function than
/// `File::open_streaming` does, and a bound on one says nothing about the other.
#[test]
fn opening_one_child_does_not_allocate_per_child_of_the_group() {
    const OBJECTS: usize = 1024;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many.h5");
    let mut builder = FileBuilder::new();
    for i in 0..OBJECTS {
        builder
            .create_dataset(&format!("d{i:05}"))
            .with_i32_data(&[i as i32]);
    }
    builder.write(&path).unwrap();

    // The last child written, so the scan passes every link before it: the worst
    // case for a lookup that stops at its match, and the one where a per-child
    // cost cannot hide behind an early hit.
    let last = format!("d{:05}", OBJECTS - 1);
    let file = File::open_streaming(&path).unwrap();
    // Opened before the measurements: a buffered open reads the whole file, and
    // what is being measured is the lookup inside it.
    let buffered = File::open(&path).unwrap();

    let (by_path, path_measured) = measure("child_by_path", || file.dataset(&last).unwrap());
    let (by_name, name_measured) = measure("child_by_name", || file.root().dataset(&last).unwrap());
    let (buffered_by_path, buffered_measured) = measure("child_by_path_buffered", || {
        buffered.dataset(&last).unwrap()
    });

    for (what, measured) in [
        ("by path", path_measured),
        ("in the group", name_measured),
        ("by path in a buffered file", buffered_measured),
    ] {
        // Measured at 18 allocations: the group's header, the opened dataset's
        // header, and its handful of messages. One per link is 1,026, and the
        // bound sits between the two rather than near either.
        assert!(
            measured.blocks < (OBJECTS / 8) as u64,
            "opening one child {what} must not allocate per child of the group: \
             measured {measured} in a {OBJECTS}-child group"
        );

        // The other axis, which the count cannot see: the group's header is read
        // whole (about 30 stored bytes per child, so ~30 KiB here) because the
        // links must be scanned, but nothing may allocate several times over it.
        // The per-child entries this replaced cost ten times the header.
        assert!(
            measured.bytes < (OBJECTS * 128) as u64,
            "opening one child {what} must allocate about its group's header, not \
             a multiple of it: measured {measured} in a {OBJECTS}-child group"
        );

        // The handle each call returned — its parsed object header among it — is
        // in this measurement, so its size is a floor a measurement of nothing
        // cannot reach.
        assert!(
            measured.live_bytes >= 128,
            "the handle this open returned is not in its own measurement, so the \
             bounds above are bounding something other than the open: {measured}"
        );
    }

    // Every handle must still be the right dataset.
    assert_eq!(by_path.read_i32().unwrap(), vec![(OBJECTS - 1) as i32]);
    assert_eq!(by_name.read_i32().unwrap(), vec![(OBJECTS - 1) as i32]);
    assert_eq!(
        buffered_by_path.read_i32().unwrap(),
        vec![(OBJECTS - 1) as i32]
    );
}

/// Writing variable-length strings copies the text into the heap collections it
/// has to end up in, and not into a per-element buffer on the way.
///
/// The writer is handed `&[&str]` and used to own each one before staging it: a
/// second copy of the whole payload, in one allocation per string — 4 MiB in
/// 32,768 blocks to write 4 MiB of text (issue #228). The count is the axis that
/// names that defect, since a copy made one string at a time is one allocation
/// per string whatever it costs in bytes; the byte bound beside it is the one
/// that would still catch the same copy made in a single buffer.
#[test]
fn vlen_string_write_does_not_own_each_string_before_staging_it() {
    const N0: usize = 32 * 1024;
    const STR_LEN: usize = 128;
    const PAYLOAD_BYTES: usize = N0 * STR_LEN;

    // Built before the measurement: these are the caller's strings, and the
    // question is what the writer adds to them.
    let strings: Vec<String> = (0..N0)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");

    let ((), measured) = measure("vlen_string_write", || {
        let mut builder = FileBuilder::new();
        builder.create_dataset("labels").with_vlen_strings(&refs);
        builder.write(&path).unwrap();
    });

    // The collections this write builds hold the payload, so its size is a floor
    // a measurement of nothing cannot reach.
    assert!(
        measured.bytes >= PAYLOAD_BYTES as u64,
        "the heap collections this write built are not in its own measurement, so \
         the bounds below are measuring something other than the write: {measured}"
    );

    // Measured at 104 allocations for 32,768 strings: the collections, the
    // reference vector, the file image and the object headers. One per string is
    // the defect.
    assert!(
        measured.blocks < (N0 / 8) as u64,
        "a variable-length string write must not allocate per string: measured \
         {measured} over {N0} strings"
    );

    // Measured at 1.57 copies of the payload: the heap collections it is written
    // into, and the file image assembled around them, with the 16-byte reference
    // per element beside those. One more whole copy — the caller's strings owned
    // before staging, however few allocations that took — is 2.8, and this bound
    // sits below it.
    assert!(
        measured.bytes < (PAYLOAD_BYTES * 2) as u64,
        "a variable-length string write may hold the payload about once, not one \
         time more: measured {measured} against a {PAYLOAD_BYTES}-byte payload"
    );

    // The file must still be the right file.
    let file = File::open(&path).unwrap();
    let back = file.dataset("labels").unwrap().read_string().unwrap();
    assert_eq!(back.len(), N0);
    assert_eq!(back[N0 - 1], strings[N0 - 1]);
}

/// One in-place append allocates on the order of its own batch, not of the
/// dataset it grows.
///
/// This is the write path that runs many times over one file, so a per-call cost
/// that scales with what is already there is what turns a long append loop
/// quadratic. The measurement is of the *last* append, onto a dataset already
/// several hundred times the batch, so a term in the dataset's size cannot hide
/// in a term in the batch's.
#[test]
fn one_append_costs_its_batch_not_the_dataset() {
    const CHUNK_ELEMS: u64 = 512;
    const BATCH_BYTES: usize = CHUNK_ELEMS as usize * 8;
    const WARMUP_APPENDS: usize = 512;
    const DATASET_BYTES: usize = (WARMUP_APPENDS + 1) * BATCH_BYTES;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("growing.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&[0.0f64; CHUNK_ELEMS as usize])
        .with_shape(&[CHUNK_ELEMS])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[CHUNK_ELEMS]);
    builder.write(&path).unwrap();

    let batch: Vec<f64> = (0..CHUNK_ELEMS).map(|i| i as f64).collect();
    {
        // One barrier at close rather than one per append: an `fsync` apiece
        // would make this test's runtime, not its allocations, the thing to look
        // at (see `.config/nextest.toml`, which fails the run on either).
        let file = hdf5_pure::File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        let mut ds = file.dataset("t").unwrap();
        for _ in 0..WARMUP_APPENDS {
            ds.append(&batch).unwrap();
        }

        let ((), measured) = measure("append_in_place", || ds.append(&batch).unwrap());

        // The batch's own bytes are staged inside this measurement, so their size
        // is a floor a measurement of nothing cannot reach.
        assert!(
            measured.bytes >= BATCH_BYTES as u64,
            "the appended batch is not in this append's own measurement, so the \
             bounds below are measuring something other than the append: {measured}"
        );

        // Measured at 14 KiB and 28 allocations for a 4 KiB batch onto a 2 MiB
        // dataset: the batch staged and written, and the chunk index and object
        // header around it. The same append measured against datasets of 17,
        // 513 and 4,097 chunks costs 13.4, 14.3 and 15.3 KiB and 28 allocations
        // throughout — the growth is in the index's *depth*, not its size, so
        // eight batches' worth is a ceiling with room in it, and anything
        // proportional to the dataset (512 batches here) is far above it.
        assert!(
            measured.bytes < (BATCH_BYTES * 8) as u64,
            "one append must allocate on the order of its {BATCH_BYTES}-byte batch, \
             not of the {DATASET_BYTES}-byte dataset it grows: measured {measured}"
        );
        assert!(
            measured.blocks < 128,
            "one append must cost a bounded number of allocations, whatever the \
             dataset's size: measured {measured}"
        );
    }

    // The session is closed before reading: an open read-write file holds a lock
    // that a second open refuses on Windows.
    let file = File::open(&path).unwrap();
    let back = file.dataset("t").unwrap().read_f64().unwrap();
    // The dataset it was written with, plus the warmup appends and the measured
    // one.
    assert_eq!(back.len(), (WARMUP_APPENDS + 2) * CHUNK_ELEMS as usize);
    assert_eq!(back[back.len() - 1], (CHUNK_ELEMS - 1) as f64);
}

/// Gathering writes does not copy what it has already gathered on every new one.
///
/// Chunks are placed one after another, so each new write starts exactly where
/// the last one ended. Merging that by building a fresh buffer and copying the
/// old one into it makes the gathering quadratic in the number of writes it
/// holds: filling the budget that way copies hundreds of megabytes to write one
/// (issue #288). Extending the pending run in place makes it linear.
///
/// Measured through one large append rather than many small ones, because that
/// is what grows a single run to the whole gather budget: an append of this size
/// is split into one-megabyte batches, and within a batch the chunk writes are
/// contiguous and merge into one run that reaches the budget before it drains.
/// Many separate appends would not show it — each drains at its own barriers
/// while its run is still a few kilobytes — and a commit shows the same defect
/// but buries it under the header, link and index work that dominates at any
/// size worth measuring.
#[test]
fn gathering_writes_does_not_recopy_what_it_holds() {
    const CHUNK: usize = 512;
    const CHUNKS: usize = 1024;
    const RAW_BYTES: u64 = (CHUNK * CHUNKS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("buffered.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_file_space_strategy(hdf5_pure::FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(4096);
    builder
        .create_dataset("growing")
        .with_f64_data(&[1.0f64; CHUNK])
        .with_shape(&[CHUNK as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[CHUNK as u64]);
    builder.write(&path).unwrap();

    // Built outside the region: this is the caller's buffer, not the library's.
    let batch: Vec<f64> = (0..CHUNK * CHUNKS).map(|i| (i % CHUNK) as f64).collect();
    // Scoped: a `Dataset` owns a handle on the session, so the session outlives
    // `file` itself and the append is only certainly on disk once both are gone.
    let measured = {
        let file = File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        let mut ds = file.dataset("growing").unwrap();
        measure("gathered_append", || {
            ds.append(&batch).unwrap();
        })
        .1
    };

    // The appended data is in this measurement, so a measurement of nothing
    // cannot pass the ceiling below by default.
    assert!(
        measured.bytes >= RAW_BYTES,
        "the appended data is not in this measurement, so the bound below is \
         measuring something other than the append: {measured}"
    );

    // Measured at 5.2x the raw bytes when a pending run is extended in place, and
    // at 132x when each write rebuilds the run it lands in. The ceiling sits
    // between them with room on both sides, since the point is the shape of the
    // curve rather than one host's constant.
    assert!(
        measured.bytes < 32 * RAW_BYTES,
        "gathering must not recopy what it holds: measured {measured} against \
         {RAW_BYTES} bytes of appended data"
    );

    let file = File::open(&path).unwrap();
    let back = file.dataset("growing").unwrap().read_f64().unwrap();
    assert_eq!(back.len(), CHUNK * (CHUNKS + 1));
    assert_eq!(&back[CHUNK..CHUNK * 2], &batch[..CHUNK]);
}

/// A page buffer does not recopy what it holds when a write lands *inside* a run
/// it is already holding.
///
/// The companion to the bound above, through the configuration that reaches the
/// other fast path. `absorb` has two: extending a pending run at its end, which
/// one large append reaches by itself, and patching a write that falls wholly
/// inside a run already held, which needs a run grown to the byte budget and
/// then written into again. Only a buffer held *across* operations gets there —
/// every append here re-patches the object header and chunk index that the runs
/// before it left in the buffer — so the default gathering cannot bound this
/// path and this is the vehicle that can (issues #288 and #308).
#[test]
fn a_page_buffer_does_not_recopy_what_it_holds_across_operations() {
    const CHUNK: usize = 512;
    const APPENDS: usize = 256;
    const RAW_BYTES: u64 = (CHUNK * APPENDS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("page_buffered.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_file_space_strategy(hdf5_pure::FileSpaceStrategy::Page, true, 1)
        .with_file_space_page_size(4096);
    builder
        .create_dataset("growing")
        .with_f64_data(&[1.0f64; CHUNK])
        .with_shape(&[CHUNK as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[CHUNK as u64]);
    builder.write(&path).unwrap();

    // Built outside the region: this is the caller's buffer, not the library's.
    let batch: Vec<f64> = (0..CHUNK).map(|i| i as f64).collect();
    // Scoped: a `Dataset` owns a handle on the session, so the session outlives
    // `file` itself, and with a page buffer that is the difference between a
    // file with every append in it and one still holding them.
    let measured = {
        let file = File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new()
                .with_sync_policy(hdf5_pure::SyncPolicy::OnClose)
                .with_page_buffer_size(1 << 20),
        )
        .unwrap();
        let mut ds = file.dataset("growing").unwrap();
        measure("page_buffered_appends", || {
            for _ in 0..APPENDS {
                ds.append(&batch).unwrap();
            }
        })
        .1
    };

    // The appended data is in this measurement, so a measurement of nothing
    // cannot pass the ceiling below by default.
    assert!(
        measured.bytes >= RAW_BYTES,
        "the appended data is not in this measurement, so the bound below is \
         measuring something other than the appends: {measured}"
    );

    // Measured at 7.3x the raw bytes when a write inside a held run patches it in
    // place, and at 388x when each one rebuilds the run it lands in (7.7 MB
    // against 407 MB, to write one megabyte) — nearly two orders of magnitude
    // apart, so this bound need not be tight to separate them.
    assert!(
        measured.bytes < 64 * RAW_BYTES,
        "a page buffer must not recopy what it holds: measured {measured} against \
         {RAW_BYTES} bytes of appended data"
    );

    let file = File::open(&path).unwrap();
    let back = file.dataset("growing").unwrap().read_f64().unwrap();
    assert_eq!(back.len(), CHUNK * (APPENDS + 1));
    assert_eq!(&back[CHUNK..CHUNK * 2], &batch[..]);
}

/// A write too large for the gather budget is issued rather than copied into it.
///
/// Gathering holds a write so it can be merged with its neighbours, and a write
/// already larger than the whole budget has no neighbours it could be merged
/// with — it is flushed on the very next line. Absorbing it first therefore buys
/// nothing and costs a full copy of it, which for a staged commit is a second
/// copy of the dataset (issue #288). The bound is on bytes because that is the
/// only axis this moves: the allocation *count* is identical either way.
#[test]
fn a_write_larger_than_the_gather_budget_is_not_copied_into_it() {
    const ELEMS: usize = 2 * 1024 * 1024;
    const DATASET_BYTES: u64 = (ELEMS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big_commit.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("seed")
        .with_f64_data(&[1.0f64; 8])
        .with_shape(&[8]);
    builder.write(&path).unwrap();

    let data: Vec<f64> = (0..ELEMS).map(|i| i as f64).collect();
    let measured = {
        let file = File::open_rw(&path).unwrap();
        let root = file.root();
        let ((), m) = measure("big_staged_commit", || {
            root.create_dataset("big", |b| {
                b.with_f64_data(&data).with_shape(&[ELEMS as u64]);
            })
            .unwrap();
            file.commit().unwrap();
        });
        m
    };

    // The staged dataset is in this measurement, so a commit that did nothing
    // cannot reach this floor.
    assert!(
        measured.bytes >= DATASET_BYTES,
        "the staged dataset is not in this measurement: {measured}"
    );

    // Measured at 1.00x the dataset with the bypass and 2.00x without it: the
    // difference is exactly one more copy. Half again is ample headroom for the
    // headers and the link the commit also writes, and still nowhere near two.
    assert!(
        measured.bytes < DATASET_BYTES * 3 / 2,
        "a staged commit must not copy its data into the gather buffer as well: \
         measured {measured} against a {DATASET_BYTES}-byte dataset"
    );
}

/// A bounded read-write open holds a metadata-sized amount of the file, not a
/// copy of it — and the mirror beside it is what makes that a measurement rather
/// than a hope (issue #389).
///
/// This is the rule behind `MemoryStrategy::Auto` preferring the bounded engine:
/// the choice is only worth making if the two backings really differ by the size
/// of the file. The contrast is measured on the same file through the same
/// appends, so nothing but the strategy separates the two figures.
#[test]
fn open_rw_of_a_persisting_fsm_file_does_not_allocate_the_file() {
    // 16 MiB in 32 KiB chunks, persisting its free-space managers under the
    // default FsmAggr strategy: the ordinary growing file of issue #389.
    const CHUNK_ELEMS: usize = 4096;
    const CHUNKS: usize = 512;
    const APPENDS: usize = 4;
    const BATCH_BYTES: u64 = (CHUNK_ELEMS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fsm.h5");
    let data: Vec<u64> = (0..(CHUNK_ELEMS * CHUNKS) as u64).collect();
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(FileSpaceStrategy::FsmAggr, true, 0);
    builder
        .create_dataset("samples")
        .with_u64_data(&data)
        .with_shape(&[data.len() as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[CHUNK_ELEMS as u64]);
    builder.write(&path).unwrap();
    drop(data);

    let file_len = std::fs::metadata(&path).unwrap().len();
    let batch: Vec<u64> = (0..CHUNK_ELEMS as u64).collect();

    // One barrier at close rather than one per append; this test measures
    // allocations, not `fsync` cadence.
    let access = |strategy| {
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .with_memory_strategy(strategy)
    };

    let open_and_append = |name: &'static str, strategy| {
        let (file, measured) = measure(name, || {
            let file = File::open_rw_with_options(&path, access(strategy)).unwrap();
            let mut ds = file.dataset("samples").unwrap();
            for _ in 0..APPENDS {
                ds.append(&batch).unwrap();
            }
            drop(ds);
            file
        });
        let backing = file.edit_backing();
        file.close().unwrap();
        (backing, measured)
    };

    let (backing, bounded) =
        open_and_append("bounded_open_rw_of_a_large_file", MemoryStrategy::Auto);
    assert_eq!(
        backing,
        Some(EditBacking::Bounded),
        "Auto must reach the bounded engine on this file, or the bounds below are \
         measuring the mirror"
    );

    // The appended batches are staged inside this measurement, so their bytes are
    // a floor a measurement of nothing cannot reach — and the session itself is
    // returned from the region, so its retained allocations are in `live_bytes`.
    assert!(
        bounded.bytes >= BATCH_BYTES * APPENDS as u64,
        "the appended batches are not in this measurement, so the ceilings below \
         are bounding something other than the session: {bounded}"
    );
    assert!(
        bounded.blocks > 0,
        "no allocation was recorded for the open at all, so this region saw none \
         of the work it names: {bounded}"
    );

    // What a bounded session keeps is the metadata it is holding plus whatever an
    // append is building — a figure with no term in the file's length. Measured
    // at 3.9 KiB live and 101 KiB at peak against a 16 MiB file, where the mirror
    // below holds 32 MiB live, so a sixteenth of the file is orders above the one
    // and orders below the other.
    assert!(
        bounded.live_bytes < file_len / 16,
        "a bounded session must not retain a copy of the {file_len}-byte file: \
         {bounded}"
    );
    assert!(
        bounded.peak_bytes < file_len / 8,
        "a bounded session must not build a copy of the {file_len}-byte file even \
         transiently: {bounded}"
    );

    // The contrast that makes the ceilings above mean something: the same file,
    // the same appends, one strategy apart.
    let (backing, mirrored) =
        open_and_append("mirrored_open_rw_of_a_large_file", MemoryStrategy::Mirrored);
    assert_eq!(backing, Some(EditBacking::Mirrored));
    assert!(
        mirrored.live_bytes > file_len,
        "the mirror is supposed to hold the whole {file_len}-byte file, so a \
         measurement below that is not measuring the mirror and the bounded \
         bounds above are not a contrast with anything: {mirrored}"
    );
}

/// Repeated staged appends do not retain memory per commit (issue #412).
///
/// A staged append relocates: the commit writes the new chunks, rebuilds the
/// dataset's chunk index and moves its object header. The report was that a
/// process doing this in a loop grew without bound, which would mean each commit
/// left something behind — a vacated index the engine still held, a batch it did
/// not release. On the bounded engine a commit holds only what it is building,
/// so what is live after a run of commits must not scale with their number, and
/// the peak of that run must be the scratch of *one* commit rather than the sum
/// of them.
///
/// The mirror is the other backing and has its own rule below: it holds the file,
/// so its retained memory scales with the file and must not scale with anything
/// else.
#[test]
fn repeated_staged_append_commits_retain_nothing_per_commit() {
    const DATASETS: usize = 4;
    const CHUNK_ELEMS: u64 = 512;
    const BATCH_BYTES: usize = CHUNK_ELEMS as usize * 8;
    const WARMUP_COMMITS: usize = 128;
    const MEASURED_COMMITS: usize = 64;
    const APPENDED_BYTES: usize = MEASURED_COMMITS * DATASETS * BATCH_BYTES;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("growing.h5");
    let batch = write_growing_fixture(&path, DATASETS, CHUNK_ELEMS);
    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
        for _ in 0..WARMUP_COMMITS {
            commit_one_batch_per_dataset(&file, DATASETS, &batch);
        }

        let ((), measured) = measure("staged_append_commits", || {
            for _ in 0..MEASURED_COMMITS {
                commit_one_batch_per_dataset(&file, DATASETS, &batch);
            }
        });

        // Every batch is staged inside this region, so their sum is a floor a
        // measurement of nothing cannot reach.
        assert!(
            measured.bytes >= APPENDED_BYTES as u64,
            "the appended batches are not in the commits' own measurement, so the \
             bounds below are measuring something other than the commits: {measured}"
        );

        // Measured at 0 to 8 KiB live after 64 commits. A commit that kept even
        // one batch per dataset would leave 64 times this ceiling behind; one
        // that kept its rebuilt index would leave far more.
        assert!(
            measured.live_bytes < (2 * DATASETS * BATCH_BYTES) as u64,
            "{MEASURED_COMMITS} staged-append commits must not retain memory in \
             proportion to their number: measured {measured}"
        );

        // Measured at 115 KiB against 1 MiB appended. The peak is one commit's
        // scratch — four batches, the four rebuilt indexes and headers — and it
        // grows with the datasets' chunk count (218 KiB and 353 KiB after 512 and
        // 1024 warm-up commits), never with the number of commits measured. A run
        // that carried anything from one commit into the next would peak at the
        // sum of what it appended, above this ceiling four times over.
        assert!(
            measured.peak_bytes < (APPENDED_BYTES / 4) as u64,
            "the peak over {MEASURED_COMMITS} staged-append commits must be one \
             commit's scratch, not the {APPENDED_BYTES} bytes they appended \
             together: measured {measured}"
        );
    }

    // The session is closed before reading: an open read-write file holds a lock
    // that a second open refuses on Windows.
    assert_growing_fixture_contents(
        &path,
        DATASETS,
        CHUNK_ELEMS,
        WARMUP_COMMITS + MEASURED_COMMITS,
    );
}

/// The whole-file mirror holds the file and nothing that grows with the number
/// of commits made through it (issue #412).
///
/// Its memory scales with the file by design — that is the cost
/// [`MemoryStrategy::Mirrored`] states — so the rule is stated against the file
/// the run ends with: the image, with the headroom its growth leaves, and the
/// scratch of the last commit. An image that kept its superseded bytes, or a
/// commit that held on to a copy of it, would put the run's retained and peak
/// memory at a multiple of the file these ceilings do not allow.
#[test]
fn the_mirror_retains_the_file_not_the_commits_made_through_it() {
    const DATASETS: usize = 4;
    const CHUNK_ELEMS: u64 = 512;
    const COMMITS: usize = 192;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("growing.h5");
    let batch = write_growing_fixture(&path, DATASETS, CHUNK_ELEMS);

    // The open is inside the region so the image itself is in the measurement,
    // and the file is handed back out of it so that image is still live when the
    // region closes.
    let (file, measured) = measure("mirrored_staged_append_commits", || {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_sync_policy(SyncPolicy::OnClose)
                .with_memory_strategy(MemoryStrategy::Mirrored),
        )
        .unwrap();
        assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
        for _ in 0..COMMITS {
            commit_one_batch_per_dataset(&file, DATASETS, &batch);
        }
        file
    });
    let file_size = file.file_size();

    // The image is the floor: a measurement that does not contain the file is
    // measuring something other than the mirror.
    assert!(
        measured.live_bytes >= file_size,
        "the mirror's image of the {file_size}-byte file is not in its own \
         measurement, so the ceilings below are bounding something else: {measured}"
    );

    // Measured at 1.5 file sizes live and at peak for a 3.2 MiB file. The image
    // grows in steps, so up to two file sizes is the image alone, and a growth
    // step holds the old and new image together; a mirror that kept one
    // superseded image per commit would hold hundreds. The peak ceiling is not
    // tight enough to see one transient copy of the image inside a commit — that
    // fits under the growth step's headroom — only copies that accumulate.
    assert!(
        measured.live_bytes < 3 * file_size,
        "the mirror must retain its image of the {file_size}-byte file and the \
         headroom of that image's growth, not what {COMMITS} commits made through \
         it: measured {measured}"
    );
    assert!(
        measured.peak_bytes < 4 * file_size,
        "the peak over {COMMITS} mirrored commits must be a small multiple of the \
         {file_size}-byte image, not the images those commits built together: \
         measured {measured}"
    );

    drop(file);
    assert_growing_fixture_contents(&path, DATASETS, CHUNK_ELEMS, COMMITS);
}

/// `datasets` rank-1 unlimited f64 datasets of one `chunk_elems` chunk each, and
/// the batch the tests above append to them.
fn write_growing_fixture(path: &std::path::Path, datasets: usize, chunk_elems: u64) -> Vec<f64> {
    let mut builder = FileBuilder::new();
    let initial = vec![0.0f64; chunk_elems as usize];
    for i in 0..datasets {
        builder
            .create_dataset(&format!("d{i}"))
            .with_f64_data(&initial)
            .with_shape(&[chunk_elems])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[chunk_elems]);
    }
    builder.write(path).unwrap();
    (0..chunk_elems).map(|i| i as f64).collect()
}

/// One staged append of `batch` onto each dataset, then one commit — the loop
/// issue #412 describes.
fn commit_one_batch_per_dataset(file: &File, datasets: usize, batch: &[f64]) {
    for i in 0..datasets {
        file.dataset(&format!("d{i}"))
            .unwrap()
            .append_staged(|b| {
                b.append_f64(batch);
            })
            .unwrap();
    }
    file.commit().unwrap();
}

/// Every dataset holds its initial chunk plus one batch per commit, ending in
/// the batch's last element.
fn assert_growing_fixture_contents(
    path: &std::path::Path,
    datasets: usize,
    chunk_elems: u64,
    commits: usize,
) {
    let file = File::open(path).unwrap();
    for i in 0..datasets {
        let back = file.dataset(&format!("d{i}")).unwrap().read_f64().unwrap();
        assert_eq!(back.len(), (commits + 1) * chunk_elems as usize);
        assert_eq!(back[back.len() - 1], (chunk_elems - 1) as f64);
    }
}
