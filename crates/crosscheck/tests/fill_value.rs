// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for configurable fill values (issue #151).
//!
//! Two directions, both make-or-break:
//!
//! * The pure writer sets a fill value and the reference C library must report
//!   the same value through its own `H5Pget_fill_value` path.
//! * The reference C library writes a fill value — under both the default format
//!   (a version-2 Fill Value message) and the latest format (a version-3
//!   message) — and the pure reader must recover it.

use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder, ScaleOffset};
use tempfile::tempdir;

/// Read a dataset's typed fill value back through the reference C library.
fn c_fill_value<T: hdf5::H5Type>(path: &std::path::Path, name: &str) -> Option<T> {
    let file = hdf5::File::open(path).unwrap();
    let ds = file.dataset(name).unwrap();
    let fv = ds.dcpl().unwrap().get_fill_value_as::<T>().unwrap();
    file.close().unwrap();
    fv
}

// ---- pure writes, C library reads ------------------------------------------

#[test]
fn c_reads_pure_written_contiguous_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_contig.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_i32_data(&[10, 20, 30])
        .with_fill_value(-7_i32);
    fb.write(&path).unwrap();

    assert_eq!(c_fill_value::<i32>(&path, "d"), Some(-7));
}

#[test]
fn c_reads_pure_written_chunked_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_chunked.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[4])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[2])
        .with_fill_value(3.5_f64);
    fb.write(&path).unwrap();

    assert_eq!(c_fill_value::<f64>(&path, "d"), Some(3.5));
}

#[test]
fn c_sees_no_fill_value_when_pure_sets_none() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_none.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d").with_i32_data(&[1, 2, 3]);
    fb.write(&path).unwrap();

    // The crate writes HDF5's *default* fill message (neither the "defined" nor
    // the "undefined" bit set). The C library resolves that default to the type's
    // implicit zero, so it reports `Some(0)` — whereas the pure reader reports
    // `None` for the same message, since there is no *user-defined* value. Both
    // describe the same on-disk state: unwritten elements read back as zero.
    assert_eq!(c_fill_value::<i32>(&path, "d"), Some(0));
}

// ---- C library writes, pure reads ------------------------------------------

/// Create a dataset with a fill value using the reference C library under the
/// given format bounds, so the on-disk Fill Value message version is controlled:
/// the default (`Earliest`) bound writes a version-2 message, `V110`/latest a
/// version-3 message.
fn c_write_i32_fill(path: &std::path::Path, low: LibraryVersion, high: LibraryVersion) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, high))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .fill_value(-7_i32)
        .shape((3,))
        .create("d")
        .unwrap();
    ds.write(&[10_i32, 20, 30]).unwrap();
    file.close().unwrap();
}

#[test]
fn pure_reads_c_written_v2_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v2.h5");
    // Earliest..=V18 keeps the classic format, whose Fill Value message is v2.
    c_write_i32_fill(&path, LibraryVersion::Earliest, LibraryVersion::V18);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-7));
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

#[test]
fn pure_reads_c_written_v3_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v3.h5");
    // The latest format writes a version-3 Fill Value message.
    c_write_i32_fill(&path, LibraryVersion::V110, LibraryVersion::latest());

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-7));
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

/// Create a chunked f64 dataset with a fill value using the reference C library.
/// The handles are dropped when this returns, so the file is fully flushed before
/// the caller reopens it.
fn c_write_chunked_f64_fill(path: &std::path::Path) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<f64>()
        .fill_value(2.5_f64)
        .chunk((2,))
        .shape((4,))
        .create("d")
        .unwrap();
    ds.write(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    file.close().unwrap();
}

#[test]
fn pure_reads_c_written_chunked_f64_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked.h5");
    c_write_chunked_f64_fill(&path);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<f64>().unwrap(), Some(2.5));
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

/// Create an i32 dataset whose fill value is explicitly *undefined* (a version-3
/// message with the "undefined" bit set), with the reference C library.
fn c_write_i32_no_fill(path: &std::path::Path) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .no_fill_value()
        .shape((3,))
        .create("d")
        .unwrap();
    ds.write(&[1_i32, 2, 3]).unwrap();
    file.close().unwrap();
}

#[test]
fn pure_reads_none_when_c_sets_no_fill() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_no_fill.h5");
    c_write_i32_no_fill(&path);

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().fill_value::<i32>().unwrap(),
        None
    );
}

// ---- fill value alongside a filter -----------------------------------------

#[test]
fn c_reads_pure_scaleoffset_dataset_data_and_fill() {
    // A dataset fill value coexists with the scale-offset filter: the pure writer
    // emits the fill-undefined form of the filter (its own on-disk contract) plus
    // a defined dataset Fill Value message. The C library decodes the data through
    // the filter's own fill config (so the data is intact) and reports the dataset
    // fill value separately — the two do not collide.
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_so_fill.h5");
    let data: Vec<i32> = (0..16).collect();
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[16])
        .with_chunks(&[8])
        .with_scale_offset(ScaleOffset::Integer(0))
        .with_fill_value(-1_i32);
    fb.write(&path).unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.read_raw::<i32>().unwrap(), data);
    assert_eq!(
        ds.dcpl().unwrap().get_fill_value_as::<i32>().unwrap(),
        Some(-1)
    );
    file.close().unwrap();
}

// ---- the unwritten slots of an allocated chunk (issue #296) -----------------
//
// A chunk that covers the dataset's edge is stored whole, so some of its slots
// hold no data. `H5D_FILL_TIME_IFSET` — what both libraries record whenever a
// fill value is set — promises those slots hold the fill value. They are not
// reachable through the dataset's own extent, which is why writing zeros there
// stayed invisible: it takes a later `resize` (or another library's) to bring
// them inside it, and then every reader returns what is physically stored.

/// Grow `col` from `from` to `to` elements with the reference C library,
/// writing no new data, and read the whole dataset back through it.
fn c_extend_and_read(path: &std::path::Path, to: usize) -> Vec<u32> {
    {
        let f = hdf5::File::open_rw(path).unwrap();
        f.dataset("col").unwrap().resize((to,)).unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(path).unwrap();
    let v = f.dataset("col").unwrap().read_raw::<u32>().unwrap();
    f.close().unwrap();
    v
}

fn c_create_chunked(path: &std::path::Path, data: &[u32], chunk: usize, fill: Option<u32>) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let mut b = file
        .new_dataset::<u32>()
        .chunk((chunk,))
        .shape((hdf5::Extent::resizable(data.len()),));
    if let Some(f) = fill {
        b = b.fill_value(f);
    }
    let ds = b.create("col").unwrap();
    ds.write_raw(data).unwrap();
    file.close().unwrap();
}

fn pure_create_chunked(path: &std::path::Path, data: &[u32], chunk: u64, fill: Option<u32>) {
    let mut b = FileBuilder::new();
    let ds = b
        .create_dataset("col")
        .with_u32_data(data)
        .with_shape(&[data.len() as u64])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[chunk]);
    if let Some(f) = fill {
        ds.with_fill_value(f);
    }
    b.write(path).unwrap();
}

/// The whole point, stated as agreement with the reference: extend a dataset
/// into the tail of its last chunk and both writers' files must read the same.
#[test]
fn extending_into_a_partial_chunk_reads_the_fill_value() {
    let dir = tempdir().unwrap();
    let data = [1u32, 2, 3, 4, 5];
    let fill = 77u32;

    let c_path = dir.path().join("c.h5");
    c_create_chunked(&c_path, &data, 4, Some(fill));
    let want = c_extend_and_read(&c_path, 8);
    assert_eq!(
        want,
        vec![1, 2, 3, 4, 5, fill, fill, fill],
        "premise: the reference fills the unwritten slots of an allocated chunk",
    );

    let p_path = dir.path().join("pure.h5");
    pure_create_chunked(&p_path, &data, 4, Some(fill));
    assert_eq!(c_extend_and_read(&p_path, 8), want);
}

/// The same for a dataset grown through this crate's append, which rewrites the
/// trailing chunk and so decides the padding a second time.
#[test]
fn appending_leaves_the_fill_value_in_the_rest_of_the_chunk() {
    let dir = tempdir().unwrap();
    let fill = 77u32;

    let p_path = dir.path().join("appended.h5");
    pure_create_chunked(&p_path, &[1u32, 2], 4, Some(fill));
    {
        let f = File::open_rw(&p_path).unwrap();
        f.dataset("col")
            .unwrap()
            .append_staged(|b| {
                b.append_u32(&[3]);
            })
            .unwrap();
        f.commit().unwrap();
    }
    // Three live elements in a chunk of four; the fourth slot is the one at
    // issue.
    assert_eq!(c_extend_and_read(&p_path, 4), vec![1, 2, 3, fill]);
}

/// Without a fill value the slots stay zero, which is both what the reference
/// does and what this crate has always written — so every existing file and
/// every fixture that pins written bytes is unaffected.
#[test]
fn no_fill_value_still_pads_with_zeros() {
    let dir = tempdir().unwrap();
    let data = [1u32, 2, 3, 4, 5];

    let c_path = dir.path().join("c.h5");
    c_create_chunked(&c_path, &data, 4, None);
    let p_path = dir.path().join("pure.h5");
    pure_create_chunked(&p_path, &data, 4, None);

    let want = vec![1, 2, 3, 4, 5, 0, 0, 0];
    assert_eq!(c_extend_and_read(&c_path, 8), want);
    assert_eq!(c_extend_and_read(&p_path, 8), want);
}

/// A zero fill value is the zero pattern, and must not push the write onto the
/// tiling path for nothing.
#[test]
fn a_zero_fill_value_pads_with_zeros() {
    let dir = tempdir().unwrap();
    let data = [1u32, 2, 3, 4, 5];
    let p_path = dir.path().join("pure.h5");
    pure_create_chunked(&p_path, &data, 4, Some(0));
    assert_eq!(c_extend_and_read(&p_path, 8), vec![1, 2, 3, 4, 5, 0, 0, 0]);
}

/// The padding survives the filter pipeline: a filtered chunk is compressed
/// after the fill is laid down, so the fill has to be there before compression
/// rather than patched into the stored bytes.
#[test]
fn a_filtered_partial_chunk_is_padded_before_compression() {
    let dir = tempdir().unwrap();
    let fill = 77u32;
    let data: Vec<u32> = (0..5).collect();
    let p_path = dir.path().join("deflate.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("col")
        .with_u32_data(&data)
        .with_shape(&[5])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_fill_value(fill)
        .with_deflate(6);
    b.write(&p_path).unwrap();

    assert_eq!(
        c_extend_and_read(&p_path, 8),
        vec![0, 1, 2, 3, 4, fill, fill, fill]
    );
}

/// Scale-offset re-derives each chunk's value range from its contents, so the
/// padding is not merely stored through it — it participates in the encoding.
/// Reaching the fill value through the filter is the check that it was laid
/// down as element bytes and not appended as an afterthought.
#[test]
fn a_scale_offset_partial_chunk_is_padded_before_encoding() {
    let dir = tempdir().unwrap();
    let fill = 77u32;
    let data: Vec<u32> = (0..5).collect();
    let p_path = dir.path().join("so.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("col")
        .with_u32_data(&data)
        .with_shape(&[5])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_fill_value(fill)
        .with_scale_offset(ScaleOffset::Integer(0));
    b.write(&p_path).unwrap();

    assert_eq!(
        c_extend_and_read(&p_path, 8),
        vec![0, 1, 2, 3, 4, fill, fill, fill]
    );
}

/// `Dataset::append` grows the file immediately through the chunk-index engine
/// rather than staging into a commit, so it reaches the trailing chunk by a
/// different route than [`appending_leaves_the_fill_value_in_the_rest_of_the_chunk`]
/// and decides the padding from its own copy of the dataset's fill value.
#[test]
fn an_immediate_append_leaves_the_fill_value_in_the_rest_of_the_chunk() {
    let dir = tempdir().unwrap();
    let fill = 77u32;
    let path = dir.path().join("immediate.h5");
    pure_create_chunked(&path, &[1u32, 2], 4, Some(fill));
    {
        let f = File::open_rw(&path).unwrap();
        let mut ds = f.dataset("col").unwrap();
        ds.append(&[3u32]).unwrap();
    }
    assert_eq!(c_extend_and_read(&path, 4), vec![1, 2, 3, fill]);
}

/// `H5D_FILL_TIME_NEVER` says the library writes the fill value *nowhere*, so
/// the unwritten slots of an allocated chunk are not promised to hold it and
/// must not be filled with it. The read path already honours this (a
/// never-written dataset reads as zeros regardless of its fill value); the
/// write path has to make the same distinction, or a dataset that declares
/// nothing about its unwritten storage silently gains a claim.
#[test]
fn a_fill_time_of_never_still_pads_with_zeros() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("never.h5");
    // Only the C library can express this today, so it builds the fixture and
    // this crate's append rewrites the trailing chunk through it.
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<u32>()
            .chunk((4,))
            .shape((hdf5::Extent::resizable(2),))
            .fill_value(77u32)
            .fill_time(hdf5::dataset::FillTime::Never)
            .create("col")
            .unwrap();
        ds.write_raw(&[1u32, 2]).unwrap();
        file.close().unwrap();
    }
    // Premise: the fixture really does declare Never with a fill value set.
    {
        let f = hdf5::File::open(&path).unwrap();
        let dcpl = f.dataset("col").unwrap().dcpl().unwrap();
        assert_eq!(dcpl.fill_time(), hdf5::dataset::FillTime::Never);
        assert_eq!(dcpl.get_fill_value_as::<u32>().unwrap(), Some(77));
        f.close().unwrap();
    }
    {
        let f = File::open_rw(&path).unwrap();
        let mut ds = f.dataset("col").unwrap();
        ds.append(&[3u32]).unwrap();
    }
    assert_eq!(
        c_extend_and_read(&path, 4),
        vec![1, 2, 3, 0],
        "a fill value the file says is never written must not be written",
    );
}

/// `Dataset::write_staged` replaces a chunked dataset's values, rebuilding every
/// chunk including the partial one at the edge. It is a third route to the
/// splitter, and it reads the fill value from the dataset already on disk
/// rather than from anything staged — `write_staged` refuses to change the fill
/// value, so the existing message is the only source.
#[test]
fn overwriting_values_keeps_the_fill_value_in_the_rest_of_the_chunk() {
    let dir = tempdir().unwrap();
    let fill = 77u32;
    let path = dir.path().join("overwrite.h5");
    c_create_chunked(&path, &[1u32, 2, 3, 4, 5], 4, Some(fill));

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("col")
            .unwrap()
            .write_staged(|b| {
                b.with_u32_data(&[9u32, 8, 7, 6, 5]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert_eq!(
        c_extend_and_read(&path, 8),
        vec![9, 8, 7, 6, 5, fill, fill, fill]
    );
}

/// The fifth and last route to the splitter: a chunked dataset created *inside*
/// an `open_rw` session, which builds its chunk blob through the edit engine
/// rather than the whole-file writer and so carries its own copy of the staged
/// fill value.
#[test]
fn a_dataset_created_in_a_session_pads_with_its_fill_value() {
    let dir = tempdir().unwrap();
    let fill = 77u32;
    let path = dir.path().join("session.h5");
    // Any file to open; the dataset under test is added to it.
    FileBuilder::new().write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("col", |b| {
                b.with_u32_data(&[1u32, 2, 3, 4, 5])
                    .with_shape(&[5])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[4])
                    .with_fill_value(fill);
            })
            .unwrap();
        session.commit().unwrap();
    }

    assert_eq!(
        c_extend_and_read(&path, 8),
        vec![1, 2, 3, 4, 5, fill, fill, fill]
    );
}

/// Rank > 1, where the uncovered region of a partial chunk is not a trailing
/// run but a set of interior gaps — every row of the chunk that reaches past the
/// dataset's edge leaves one, and rows past the edge entirely are uncovered
/// whole.
///
/// Checked by comparing the *stored chunk bytes* against the reference's rather
/// than by extending and reading. That pins every gap individually instead of
/// the concatenation of them, and it needs no resize — which matters, because a
/// rank > 1 resizable chunked dataset is written unreadable today (#299), so a
/// fixture built the way the rank-1 tests are built could not run here at all.
fn assert_chunks_match_the_c_library(shape: &[usize], chunks: &[usize], fill: u32) {
    let dir = tempdir().unwrap();
    let n: usize = shape.iter().product();
    let data: Vec<u32> = (1..=n as u32).collect();

    let c_path = dir.path().join("c.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&c_path)
            .unwrap();
        let ds = file
            .new_dataset::<u32>()
            .chunk(chunks)
            .shape(shape)
            .fill_value(fill)
            .create("col")
            .unwrap();
        ds.write_raw(&data).unwrap();
        file.close().unwrap();
    }

    let p_path = dir.path().join("pure.h5");
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    let chunks_u64: Vec<u64> = chunks.iter().map(|&d| d as u64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("col")
        .with_u32_data(&data)
        .with_shape(&shape_u64)
        .with_chunks(&chunks_u64)
        .with_fill_value(fill);
    b.write(&p_path).unwrap();

    assert_eq!(
        stored_chunks(&p_path),
        stored_chunks(&c_path),
        "shape {shape:?} chunks {chunks:?}: stored chunk bytes differ from the reference",
    );
}

/// Every chunk's offset and its raw stored bytes, in a deterministic order.
fn stored_chunks(path: &std::path::Path) -> Vec<(Vec<u64>, Vec<u8>)> {
    let bytes = std::fs::read(path).unwrap();
    let f = File::open(path).unwrap();
    let ds = f.dataset("col").unwrap();
    let mut out: Vec<(Vec<u64>, Vec<u8>)> = ds
        .chunks()
        .unwrap()
        .into_iter()
        .map(|c| {
            let s = usize::try_from(c.address).unwrap();
            let n = usize::try_from(c.storage_size).unwrap();
            (c.offset, bytes[s..s + n].to_vec())
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn a_partial_chunk_is_filled_in_every_dimension() {
    // Overhang in the inner dimension only, the outer only, and both.
    assert_chunks_match_the_c_library(&[3, 3], &[2, 2], 77);
    assert_chunks_match_the_c_library(&[4, 3], &[2, 2], 77);
    assert_chunks_match_the_c_library(&[3, 4], &[2, 2], 77);
    // Rank 3, overhang in all three.
    assert_chunks_match_the_c_library(&[3, 3, 3], &[2, 2, 2], 77);
}

/// A one-byte element type, where the fill "pattern" is a single byte and any
/// element-alignment mistake in the tiling would be invisible.
#[test]
fn a_partial_chunk_of_single_byte_elements_is_filled() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("u8.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("col")
        .with_u8_data(&[1u8, 2, 3, 4, 5])
        .with_shape(&[5])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[4])
        .with_fill_value(9u8);
    b.write(&path).unwrap();

    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.dataset("col").unwrap().resize((8,)).unwrap();
        f.close().unwrap();
    }
    let got = hdf5::File::open(&path)
        .unwrap()
        .dataset("col")
        .unwrap()
        .read_raw::<u8>()
        .unwrap();
    assert_eq!(got, vec![1, 2, 3, 4, 5, 9, 9, 9]);
}
