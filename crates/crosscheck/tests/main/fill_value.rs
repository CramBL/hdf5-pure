#![cfg(feature = "__hdf5-1.10")]
//! Dataset fill values and creation metadata read through libhdf5 and the pure reader.

use std::collections::HashMap;
use std::str::FromStr;

use hdf5::Extent;
use hdf5::dataset::FillTime as ReferenceFillTime;
use hdf5::file::LibraryVersion;
#[cfg(target_pointer_width = "64")]
use hdf5::plist::dataset_create::Layout;
use hdf5::types::VarLenUnicode;
use hdf5_pure::AttrValue;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FillTime;
use hdf5_pure::MaxExtent;
use hdf5_pure::ScaleOffset;
use rstest::rstest;

use test_util_hdf5::dataset::{self, Filter, Unlimited};
use test_util_hdf5::file;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::default_policy(FillTime::IfSet, ReferenceFillTime::IfSet)]
#[case::allocate(FillTime::Alloc, ReferenceFillTime::Alloc)]
#[case::never(FillTime::Never, ReferenceFillTime::Never)]
fn creation_requests_preserve_reference_values_and_metadata(
    #[case] fill_time: FillTime,
    #[case] reference_fill_time: ReferenceFillTime,
    #[values(CreationTarget::ExistingFile, CreationTarget::NewFile)] target: CreationTarget,
) {
    let directory = tempfile::tempdir().unwrap();
    let reference_path = directory.path().join("reference.h5");
    let pure_path = directory.path().join("created.h5");
    let values = [11_i32, 13, 17, 19, 23, 29];
    {
        let reference = file::libhdf5_create_v110(&reference_path);
        {
            let dataset = reference
                .new_dataset::<i32>()
                .shape((Extent::resizable(2), Extent::fixed(3)))
                .chunk((2, 2))
                .fill_value(-7_i32)
                .fill_time(reference_fill_time)
                .create("data")
                .unwrap();
            dataset.write_raw(&values).unwrap();
            dataset
                .new_attr::<VarLenUnicode>()
                .create("note")
                .unwrap()
                .write_scalar(&VarLenUnicode::from_str("prepared attribute").unwrap())
                .unwrap();
        }
        reference.close().unwrap();
    }
    let configure = |dataset: &mut DatasetBuilder| {
        dataset
            .with_i32_data(&values)
            .with_shape(&[2, 3])
            .with_chunks(&[2, 2])
            .with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Fixed(3)])
            .with_fill_value(-7_i32)
            .with_fill_time(fill_time)
            .set_attr("note", AttrValue::VarLenString("prepared attribute".into()));
    };
    match target {
        CreationTarget::ExistingFile => {
            std::fs::copy(&reference_path, &pure_path).unwrap();
            let session = File::open_rw(&pure_path).unwrap();
            session.root().create_dataset("created", configure).unwrap();
            session.commit().unwrap();
        }
        CreationTarget::NewFile => {
            let mut builder = FileBuilder::new();
            configure(builder.create_dataset("created"));
            builder.write(&pure_path).unwrap();
        }
    }
    {
        let reference = hdf5::File::open(&reference_path).unwrap();
        let pure = hdf5::File::open(&pure_path).unwrap();
        let expected = reference.dataset("data").unwrap();
        let actual = pure.dataset("created").unwrap();
        assert_eq!(expected.shape(), [2, 3]);
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(expected.space().unwrap().maxdims(), [None, Some(3)]);
        assert_eq!(
            actual.space().unwrap().maxdims(),
            expected.space().unwrap().maxdims()
        );
        assert_eq!(expected.chunk(), Some(vec![2, 2]));
        assert_eq!(actual.chunk(), expected.chunk());
        assert_eq!(expected.read_raw::<i32>().unwrap(), values);
        assert_eq!(actual.read_raw::<i32>().unwrap(), values);
        let expected_properties = expected.dcpl().unwrap();
        let actual_properties = actual.dcpl().unwrap();
        assert_eq!(expected_properties.fill_time(), reference_fill_time);
        assert_eq!(
            actual_properties.fill_time(),
            expected_properties.fill_time()
        );
        assert_eq!(
            expected_properties.get_fill_value_as::<i32>().unwrap(),
            Some(-7)
        );
        assert_eq!(
            actual_properties.get_fill_value_as::<i32>().unwrap(),
            expected_properties.get_fill_value_as::<i32>().unwrap()
        );
        assert_eq!(expected.attr_names().unwrap(), ["note"]);
        assert_eq!(actual.attr_names().unwrap(), ["note"]);
        assert_eq!(
            expected
                .attr("note")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "prepared attribute"
        );
        assert_eq!(
            actual
                .attr("note")
                .unwrap()
                .read_scalar::<VarLenUnicode>()
                .unwrap()
                .as_str(),
            "prepared attribute"
        );
    }
    let reference = File::open(&reference_path).unwrap();
    let pure = File::open(&pure_path).unwrap();
    let expected = reference.dataset("data").unwrap();
    let actual = pure.dataset("created").unwrap();
    assert_eq!(expected.read_i32().unwrap(), values);
    assert_eq!(actual.read_i32().unwrap(), values);
    assert_eq!(expected.fill_time().unwrap(), fill_time);
    assert_eq!(actual.fill_time().unwrap(), fill_time);
    assert_eq!(expected.fill_value::<i32>().unwrap(), Some(-7));
    assert_eq!(actual.fill_value::<i32>().unwrap(), Some(-7));
    let attributes = HashMap::from([(
        "note".into(),
        AttrValue::VarLenString("prepared attribute".into()),
    )]);
    assert_eq!(expected.attrs().unwrap(), attributes);
    assert_eq!(actual.attrs().unwrap(), attributes);
}

#[derive(Clone, Copy, Debug)]
enum CreationTarget {
    ExistingFile,
    NewFile,
}

/// Read a dataset's typed fill value back through libhdf5.
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
    let dir = tempfile::tempdir().unwrap();
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure_chunked.h5");
    Unlimited::new("d", &[1.0, 2.0, 3.0, 4.0], 2)
        .fill_value(3.5_f64)
        .pure_create(&path);

    assert_eq!(c_fill_value::<f64>(&path, "d"), Some(3.5));
}

#[test]
fn c_sees_no_fill_value_when_pure_sets_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure_none.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d").with_i32_data(&[1, 2, 3]);
    fb.write(&path).unwrap();

    // The crate writes HDF5's *default* fill message (neither the "defined" nor
    // the "undefined" bit set). libhdf5 resolves that default to the type's
    // implicit zero, so it reports `Some(0)` — whereas the pure reader reports
    // `None` for the same message, since there is no *user-defined* value. Both
    // describe the same on-disk state: unwritten elements read back as zero.
    assert_eq!(c_fill_value::<i32>(&path, "d"), Some(0));
}

// ---- C library writes, pure reads ------------------------------------------

/// Create a dataset with a fill value using libhdf5 under configured
/// library-version bounds, controlling the on-disk Fill Value message version:
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
    let dir = tempfile::tempdir().unwrap();
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_v3.h5");
    // The latest format writes a version-3 Fill Value message.
    c_write_i32_fill(&path, LibraryVersion::V110, LibraryVersion::latest());

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-7));
    assert_eq!(ds.read_i32().unwrap(), vec![10, 20, 30]);
}

#[rstest]
#[case::v2(LibraryVersion::Earliest, LibraryVersion::V18)]
#[case::v3(LibraryVersion::V110, LibraryVersion::latest())]
fn pure_reads_c_written_fill_time_never(#[case] low: LibraryVersion, #[case] high: LibraryVersion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_never.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(low, high))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .fill_value(-7_i32)
            .fill_time(hdf5::dataset::FillTime::Never)
            .shape((3,))
            .create("d")
            .unwrap();
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_time().unwrap(), FillTime::Never);
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(-7));
}

#[rstest]
#[case::v2(LibraryVersion::Earliest, LibraryVersion::V18)]
#[case::v3(LibraryVersion::V110, LibraryVersion::latest())]
fn pure_reads_c_written_fill_time_alloc(#[case] low: LibraryVersion, #[case] high: LibraryVersion) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_alloc.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(low, high))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .fill_value(42_i32)
            .fill_time(hdf5::dataset::FillTime::Alloc)
            .shape((3,))
            .create("d")
            .unwrap();
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_time().unwrap(), FillTime::Alloc);
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(42));
}

#[rstest]
#[case::v2(LibraryVersion::Earliest, LibraryVersion::V18)]
#[case::v3(LibraryVersion::V110, LibraryVersion::latest())]
fn pure_reads_c_written_default_fill_time_as_if_set(
    #[case] low: LibraryVersion,
    #[case] high: LibraryVersion,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_default.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(low, high))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .fill_value(10_i32)
            .shape((3,))
            .create("d")
            .unwrap();
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_time().unwrap(), FillTime::IfSet);
    assert_eq!(ds.fill_value::<i32>().unwrap(), Some(10));
}

/// Create a chunked f64 dataset with a fill value using libhdf5.
/// The handles are dropped when this returns, so the file is fully flushed before
/// the caller reopens it.
fn c_write_chunked_f64_fill(path: &std::path::Path) {
    let file = file::libhdf5_create_v110(path);
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_chunked.h5");
    c_write_chunked_f64_fill(&path);

    let file = File::open(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    assert_eq!(ds.fill_value::<f64>().unwrap(), Some(2.5));
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[rstest]
#[case::rank_two_band(&[3, 5], &[2, 5])]
#[case::rank_three_band(&[3, 5, 3], &[2, 5, 3])]
#[case::rank_three_scatter(&[3, 5, 3], &[2, 3, 2])]
fn pure_reads_c_written_dense_edge_chunks(
    #[case] shape: &[usize],
    #[case] chunks: &[usize],
    #[values(false, true)] filtered: bool,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let path = test_util::temp::temp_path("dense_edges.h5");
    let expected: Vec<i32> = (0..i32::try_from(shape.iter().product::<usize>()).unwrap()).collect();
    {
        let file = file::libhdf5_create_v110(&path);
        let builder = file
            .new_dataset::<i32>()
            .shape(shape)
            .chunk(chunks)
            .fill_value(-7);
        let builder = if filtered { builder.shuffle() } else { builder };
        let dataset = builder.create("d").unwrap();
        dataset.write_raw(&expected).unwrap();
    }
    let file = backend.open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(
        dataset.chunk_shape().unwrap(),
        Some(chunks.iter().map(|&edge| edge as u64).collect())
    );
    assert_eq!(dataset.read_i32().unwrap(), expected);
    assert_eq!(dataset.read_i32().unwrap(), expected);
    let row_elements = shape[1..].iter().product::<usize>();
    assert_eq!(
        dataset.read_i32_rows(1, 1).unwrap(),
        expected[row_elements..2 * row_elements]
    );
    drop(dataset);
    file.close().unwrap();
}

/// Create an i32 dataset whose fill value is explicitly *undefined* (a version-3
/// message with the "undefined" bit set), with libhdf5.
fn c_write_i32_no_fill(path: &std::path::Path) {
    let file = file::libhdf5_create_v110(path);
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
    let dir = tempfile::tempdir().unwrap();
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
    // a defined dataset Fill Value message. libhdf5 decodes the data through
    // the filter's own fill config (so the data is intact) and reports the dataset
    // fill value separately — the two do not collide.
    let dir = tempfile::tempdir().unwrap();
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

/// Grow `col` from `from` to `to` elements with libhdf5,
/// writing no new data, and read the whole dataset back through it.
fn c_extend_and_read(path: &std::path::Path, to: usize) -> Vec<u32> {
    {
        let f = hdf5::File::open_rw(path).unwrap();
        f.dataset("col").unwrap().resize((to,)).unwrap();
        f.close().unwrap();
    }
    dataset::read_libhdf5(path, "col")
}

/// The whole point, stated as agreement with the reference: extend a dataset
/// into the tail of its last chunk and both writers' files must read the same.
#[test]
fn extending_into_a_partial_chunk_reads_the_fill_value() {
    let dir = tempfile::tempdir().unwrap();
    let data = [1u32, 2, 3, 4, 5];
    let fill = 77u32;

    let c_path = dir.path().join("c.h5");
    Unlimited::new("col", &data, 4)
        .fill_value(fill)
        .libhdf5_create(&c_path);
    let want = c_extend_and_read(&c_path, 8);
    assert_eq!(
        want,
        vec![1, 2, 3, 4, 5, fill, fill, fill],
        "premise: the reference fills the unwritten slots of an allocated chunk",
    );

    let p_path = dir.path().join("pure.h5");
    Unlimited::new("col", &data, 4)
        .fill_value(fill)
        .pure_create(&p_path);
    assert_eq!(c_extend_and_read(&p_path, 8), want);
}

/// The same for a dataset grown through this crate's append, which rewrites the
/// trailing chunk and so decides the padding a second time.
#[test]
fn appending_leaves_the_fill_value_in_the_rest_of_the_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let fill = 77u32;

    let p_path = dir.path().join("appended.h5");
    Unlimited::new("col", &[1u32, 2], 4)
        .fill_value(fill)
        .pure_create(&p_path);
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
    let dir = tempfile::tempdir().unwrap();
    let data = [1u32, 2, 3, 4, 5];

    let c_path = dir.path().join("c.h5");
    Unlimited::new("col", &data, 4).libhdf5_create(&c_path);
    let p_path = dir.path().join("pure.h5");
    Unlimited::new("col", &data, 4).pure_create(&p_path);

    let want = vec![1, 2, 3, 4, 5, 0, 0, 0];
    assert_eq!(c_extend_and_read(&c_path, 8), want);
    assert_eq!(c_extend_and_read(&p_path, 8), want);
}

/// A zero fill value is the zero pattern, and must not push the write onto the
/// tiling path for nothing.
#[test]
fn a_zero_fill_value_pads_with_zeros() {
    let dir = tempfile::tempdir().unwrap();
    let data = [1u32, 2, 3, 4, 5];
    let p_path = dir.path().join("pure.h5");
    Unlimited::new("col", &data, 4)
        .fill_value(0)
        .pure_create(&p_path);
    assert_eq!(c_extend_and_read(&p_path, 8), vec![1, 2, 3, 4, 5, 0, 0, 0]);
}

/// The padding survives the filter pipeline: a filtered chunk is compressed
/// after the fill is laid down, so the fill has to be there before compression
/// rather than patched into the stored bytes.
#[test]
fn a_filtered_partial_chunk_is_padded_before_compression() {
    let dir = tempfile::tempdir().unwrap();
    let fill = 77u32;
    let data: Vec<u32> = (0..5).collect();
    let p_path = dir.path().join("deflate.h5");
    Unlimited::new("col", &data, 4)
        .fill_value(fill)
        .filters(&[Filter::Deflate(DeflateLevel::Six)])
        .pure_create(&p_path);

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
    let dir = tempfile::tempdir().unwrap();
    let fill = 77u32;
    let data: Vec<u32> = (0..5).collect();
    let p_path = dir.path().join("so.h5");
    Unlimited::new("col", &data, 4)
        .fill_value(fill)
        .filters(&[Filter::ScaleOffset(ScaleOffset::Integer(0))])
        .pure_create(&p_path);

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
    let dir = tempfile::tempdir().unwrap();
    let fill = 77u32;
    let path = dir.path().join("immediate.h5");
    Unlimited::new("col", &[1u32, 2], 4)
        .fill_value(fill)
        .pure_create(&path);
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
#[cfg(target_endian = "little")]
fn a_fill_time_of_never_still_pads_with_zeros() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("never.h5");
    // Only the C library can express this today, so it builds the fixture and
    // this crate's append rewrites the trailing chunk through it.
    {
        let file = file::libhdf5_create_v110(&path);
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
#[cfg(target_endian = "little")]
fn overwriting_values_keeps_the_fill_value_in_the_rest_of_the_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let fill = 77u32;
    let path = dir.path().join("overwrite.h5");
    Unlimited::new("col", &[1u32, 2, 3, 4, 5], 4)
        .fill_value(fill)
        .libhdf5_create(&path);

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
    let dir = tempfile::tempdir().unwrap();
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
                    .with_maxshape(&[MaxExtent::Unlimited])
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
    let dir = tempfile::tempdir().unwrap();
    let n: usize = shape.iter().product();
    let data: Vec<u32> = (1..=n as u32).collect();

    let c_path = dir.path().join("c.h5");
    {
        let file = file::libhdf5_create_v110(&c_path);
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
#[cfg(target_endian = "little")]
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
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u8.h5");
    Unlimited::new("col", &[1u8, 2, 3, 4, 5], 4)
        .fill_value(9u8)
        .pure_create(&path);

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

#[test]
fn c_reads_pure_written_fill_time_never() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure_never.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("contig")
        .with_i32_data(&[10, 20, 30])
        .with_fill_time(FillTime::Never);
    fb.create_dataset("chunked")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shape(&[4])
        .with_chunks(&[2])
        .with_fill_time(FillTime::Never);
    fb.write(&path).unwrap();

    let f = hdf5::File::open(&path).unwrap();
    let contig_dcpl = f.dataset("contig").unwrap().dcpl().unwrap();
    assert_eq!(contig_dcpl.fill_time(), hdf5::dataset::FillTime::Never);
    let chunked_dcpl = f.dataset("chunked").unwrap().dcpl().unwrap();
    assert_eq!(chunked_dcpl.fill_time(), hdf5::dataset::FillTime::Never);
    f.close().unwrap();
}

#[test]
fn c_reads_pure_written_fill_time_alloc() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure_alloc.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_f64_data(&[1.5, 2.5])
        .with_fill_time(FillTime::Alloc);
    fb.write(&path).unwrap();

    let f = hdf5::File::open(&path).unwrap();
    let dcpl = f.dataset("d").unwrap().dcpl().unwrap();
    assert_eq!(dcpl.fill_time(), hdf5::dataset::FillTime::Alloc);
    f.close().unwrap();
}

#[test]
fn c_reads_pure_written_fill_time_never_with_defined_fill() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure_never_def.h5");
    let mut fb = FileBuilder::new();
    fb.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_fill_value(-7_i32)
        .with_fill_time(FillTime::Never);
    fb.write(&path).unwrap();

    let f = hdf5::File::open(&path).unwrap();
    let dcpl = f.dataset("d").unwrap().dcpl().unwrap();
    assert_eq!(dcpl.fill_time(), hdf5::dataset::FillTime::Never);
    assert_eq!(dcpl.get_fill_value_as::<i32>().unwrap(), Some(-7));
    assert_eq!(
        f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    f.close().unwrap();
}

#[test]
#[cfg(target_endian = "little")]
fn c_reads_pure_written_chunk_padding_under_fill_time_never() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure_never_pad.h5");
    let fill = 77u32;
    let data = [1u32, 2, 3, 4, 5];

    let mut b = FileBuilder::new();
    b.create_dataset("col")
        .with_u32_data(&data)
        .with_shape(&[5])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4])
        .with_fill_value(fill)
        .with_fill_time(FillTime::Never);
    b.write(&path).unwrap();

    assert_eq!(c_extend_and_read(&path, 8), vec![1, 2, 3, 4, 5, 0, 0, 0]);
}

#[cfg(target_pointer_width = "64")]
#[rstest]
#[case::unallocated_buffered(false, false, 4_294_967_360, 7.0)]
#[case::unallocated_streaming(false, true, 4_294_967_360, 7.0)]
#[case::allocated_buffered(true, false, 2, 13.0)]
#[case::allocated_streaming(true, true, 2, 13.0)]
fn row_windows_read_large_chunked_extents_written_by_the_c_library(
    #[case] allocated: bool,
    #[case] streaming: bool,
    #[case] start: u64,
    #[case] expected: f64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large.h5");
    {
        let file = file::libhdf5_create_v18(&path);
        let dataset = file
            .new_dataset::<f64>()
            .shape((4_294_967_424usize, 1))
            .chunk((4, 1))
            .fill_value(7.0)
            .create("values")
            .unwrap();
        if allocated {
            let values = ndarray::arr2(&[[11.0], [12.0], [13.0], [14.0]]);
            dataset.write_slice(&values, (0..4, ..)).unwrap();
        }
        drop(dataset);
        file.close().unwrap();
    }
    let file_bytes = std::fs::metadata(&path).unwrap().len();
    assert!(file_bytes < 1_048_576, "fixture has {file_bytes} bytes");
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    assert_eq!(
        file.dataset("values")
            .unwrap()
            .read_f64_rows(start, 1)
            .unwrap(),
        vec![expected],
    );
}

#[cfg(target_pointer_width = "64")]
#[rstest]
#[case::full_rows(1, 1)]
#[case::inner_split(6, 2)]
fn high_row_windows_read_chunks_written_by_the_c_library(
    #[case] columns: usize,
    #[case] chunk_columns: usize,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("high_rows.h5");
    let start = 4_294_967_360usize;
    let values = ndarray::Array2::from_shape_vec(
        (8, columns),
        (101..101 + 8 * columns)
            .map(|value| f64::from(u32::try_from(value).unwrap()))
            .collect(),
    )
    .unwrap();
    {
        let file = file::libhdf5_create_v18(&path);
        let dataset = file
            .new_dataset::<f64>()
            .shape((4_294_967_424usize, columns))
            .chunk((4, chunk_columns))
            .create("d")
            .unwrap();
        dataset
            .write_slice(&values, (start..start + 8, ..))
            .unwrap();
        drop(dataset);
        file.close().unwrap();
    }
    let file_bytes = std::fs::metadata(&path).unwrap().len();
    assert!(file_bytes < 1_048_576, "fixture has {file_bytes} bytes");
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.chunk_index().unwrap(), Some(ChunkIndex::BTreeV1));
    let expected = values
        .slice(ndarray::s![3..6, ..])
        .iter()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(
        dataset.read_f64_rows(start as u64 + 3, 3).unwrap(),
        expected
    );
    assert_eq!(
        dataset.read_f64_rows(start as u64 + 3, 3).unwrap(),
        expected
    );
    assert_eq!(
        dataset.read_f64_rows(start as u64 + 1, 1).unwrap(),
        values.row(1).to_vec()
    );
}

#[cfg(target_pointer_width = "64")]
#[rstest]
#[case::huge_then_zero(&[2, 4_294_967_296, 0], 1, 1)]
#[case::zero_then_huge(&[2, 0, 4_294_967_296], 1, 1)]
#[case::zero_rows_huge_then_zero(&[2, 4_294_967_296, 0], 1, 0)]
#[case::zero_rows_zero_then_huge(&[2, 0, 4_294_967_296], 1, 0)]
#[case::high_row(&[4_294_967_424, 0], 4_294_967_360, 1)]
#[case::many_rows(&[4_294_967_424, 0], 1, 4_294_967_296)]
fn empty_row_windows_read_large_extents_written_by_the_c_library(
    #[case] shape: &[usize],
    #[case] start: u64,
    #[case] count: u64,
    #[values(Layout::Compact, Layout::Contiguous)] layout: Layout,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_rows.h5");
    {
        let file = file::libhdf5_create_v18(&path);
        let dataset = file
            .new_dataset::<f64>()
            .shape(shape)
            .layout(layout)
            .create("values")
            .unwrap();
        assert_eq!(dataset.read_raw::<f64>().unwrap(), Vec::<f64>::new());
        drop(dataset);
        file.close().unwrap();
    }
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.read_f64().unwrap(), Vec::<f64>::new());
    assert_eq!(
        dataset.read_f64_rows(start, count).unwrap(),
        Vec::<f64>::new()
    );
}
