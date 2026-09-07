// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! Shared object header messages (SOHM), written by the reference C library and
//! read back by hdf5-pure (issue #417).
//!
//! A file created with `H5Pset_shared_mesg_index` stores one copy of each message
//! it shares in a fractal heap the superblock extension names, and every object
//! past the first carries an eight-byte heap ID in place of the message. Nothing
//! in the referring bytes says what the message *was*: a heap ID decodes as a
//! perfectly well-formed datatype, dataspace or attribute if a reader mistakes it
//! for content. So the whole file — its attributes, its element types, its shapes
//! and its data — is what these fixtures assert, against the values the C library
//! put there.
//!
//! Two things the reader must not depend on are varied deliberately: whether the
//! index is a **list** or a **version 2 B-tree** (forced by driving the message
//! count past the list maximum), and how many objects share one message.

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::{SharedMessageIndex, SharedMessageType};
use hdf5_pure::{AttrValue, Datatype, DatatypeByteOrder, File};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// The attribute two or more objects carry identically, which is what puts it in
/// the shared-message heap. Its value is the assertion: a reader that followed a
/// heap ID to the wrong place returns *something*, and only the value says which.
const SHARED_ATTR: &str = "units";
const SHARED_ATTR_VALUE: i64 = -12345;

/// A second attribute, on one object only, so a fixture also proves that a
/// private attribute beside a shared one still reads as itself.
const PRIVATE_ATTR: &str = "private";
const PRIVATE_ATTR_VALUE: i64 = 77;

/// The element values every shared-shape dataset holds, distinct per dataset by
/// its first element so a mixed-up dataspace or datatype shows up as data.
const ROWS: usize = 4;

fn dataset_values(seed: i32) -> Vec<i32> {
    (0..ROWS as i32).map(|i| seed * 100 + i).collect()
}

/// The i32 type every fixture uses, as this crate decodes it. Sharing a datatype
/// does not change its encoding, only where it is stored, so a resolved heap
/// reference must produce exactly what an inline type produces.
fn i32_type() -> Datatype {
    Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        signed: true,
        bit_offset: 0,
        bit_precision: 32,
    }
}

/// A fixture file plus the directory that keeps it alive.
struct Fixture {
    _dir: TempDir,
    path: PathBuf,
}

/// Create a file whose single shared-message index covers every shareable type,
/// with the given list/B-tree phase-change thresholds.
///
/// `min_message_size` is set to 1 so that even a short attribute or dataspace is
/// shared; the C library's own default (250) shares almost nothing in a small
/// fixture.
fn create_shared_file(path: &Path, max_list: u32, min_btree: u32) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::latest()))
        .with_fcpl(|p| {
            p.shared_mesg_phase_change(max_list, min_btree)
                .shared_mesg_indexes(&[SharedMessageIndex {
                    message_types: SharedMessageType::ALL,
                    min_message_size: 1,
                }])
        })
        .create(path)
        .expect("create a file with a shared-message index")
}

/// Write `count` groups, each carrying the same attribute, plus `count` datasets
/// of the same shape and type. Everything past the first user of each message is
/// stored as a reference into the shared-message heap.
fn write_shared_objects(file: &hdf5::File, count: usize) {
    for i in 0..count {
        let group = file.create_group(&format!("g{i}")).expect("create group");
        group
            .new_attr::<i64>()
            .shape(())
            .create(SHARED_ATTR)
            .expect("create shared attribute")
            .write_scalar(&SHARED_ATTR_VALUE)
            .expect("write shared attribute");
        if i == 0 {
            group
                .new_attr::<i64>()
                .shape(())
                .create(PRIVATE_ATTR)
                .expect("create private attribute")
                .write_scalar(&PRIVATE_ATTR_VALUE)
                .expect("write private attribute");
        }

        let dataset = file
            .new_dataset::<i32>()
            .shape([ROWS])
            .create(format!("d{i}").as_str())
            .expect("create dataset");
        dataset
            .write(dataset_values(i as i32).as_slice())
            .expect("write dataset");
    }
}

/// A file whose shared-message index is a **list**: the phase-change maximum is
/// far above the handful of distinct messages the fixture creates.
fn list_fixture(count: usize) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sohm_list.h5");
    {
        let file = create_shared_file(&path, 50, 40);
        write_shared_objects(&file, count);
        file.close().unwrap();
    }
    Fixture { _dir: dir, path }
}

/// How many objects [`btree_fixture`] writes at the root. Kept under this
/// crate's dense-link threshold so the root group stays compact: a dense root is
/// refused by the in-place editor before any shared-message screen runs, which
/// would leave the B-tree index unexercised by the edit test below.
const BTREE_FIXTURE_OBJECTS: usize = 3;

/// A file whose shared-message index has been converted to a **version 2
/// B-tree**: the list maximum is two, and the fixture creates far more distinct
/// shared messages than that (one dataspace per distinct shape, plus the
/// datatype and the attribute), which is what makes the C library convert.
fn btree_fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sohm_btree.h5");
    {
        let file = create_shared_file(&path, 2, 1);
        write_shared_objects(&file, BTREE_FIXTURE_OBJECTS);
        // Distinct shapes, so each contributes its own dataspace message to the
        // index rather than sharing one already there. They live in a subgroup so
        // the root keeps few links.
        let wide = file.create_group("wide").unwrap();
        for rank in 1..=6usize {
            wide.new_dataset::<i32>()
                .shape([rank, rank + 1])
                .create(format!("w{rank}").as_str())
                .expect("create dataset with a distinct shape");
        }
        file.close().unwrap();
    }
    Fixture { _dir: dir, path }
}

/// Every group's shared attribute, and every dataset's type, shape and data,
/// read through hdf5-pure.
fn assert_reads_back(path: &Path, count: usize) {
    let file = File::open(path).expect("hdf5-pure opens a file with shared messages");

    for i in 0..count {
        let group = file
            .group(&format!("g{i}"))
            .unwrap_or_else(|e| panic!("group g{i}: {e}"));
        let attrs = group.attrs().expect("read group attributes");
        assert_eq!(
            attrs.get(SHARED_ATTR),
            Some(&AttrValue::I64(SHARED_ATTR_VALUE)),
            "g{i}/{SHARED_ATTR}"
        );
        if i == 0 {
            assert_eq!(
                attrs.get(PRIVATE_ATTR),
                Some(&AttrValue::I64(PRIVATE_ATTR_VALUE)),
                "the private attribute beside the shared one"
            );
        }

        let dataset = file
            .dataset(&format!("d{i}"))
            .unwrap_or_else(|e| panic!("dataset d{i}: {e}"));
        assert_eq!(dataset.datatype().unwrap(), i32_type(), "d{i} datatype");
        assert_eq!(dataset.shape().unwrap(), vec![ROWS as u64], "d{i} shape");
        assert_eq!(
            dataset.read_i32().unwrap(),
            dataset_values(i as i32),
            "d{i} data"
        );
    }
}

/// The baseline: a file whose shared-message index is a list, read whole.
///
/// Before the heap lookup existed, every object past the first failed with
/// `FormatError::UnsupportedSohmReference`, so this is the case the issue is
/// about.
#[test]
fn a_list_indexed_shared_message_file_reads_back_whole() {
    let fixture = list_fixture(4);
    assert_reads_back(&fixture.path, 4);
}

/// The same file read through the streaming backend, which resolves references
/// by reading windows of the file rather than indexing a whole-file image. The
/// two backends have separate resolvers and separate heap readers.
#[test]
fn the_streaming_backend_resolves_shared_messages_too() {
    let fixture = list_fixture(3);
    let file = File::open_streaming(&fixture.path).expect("streaming open");
    for i in 0..3i32 {
        let group = file.group(&format!("g{i}")).unwrap();
        assert_eq!(
            group.attrs().unwrap().get(SHARED_ATTR),
            Some(&AttrValue::I64(SHARED_ATTR_VALUE))
        );
        let dataset = file.dataset(&format!("d{i}")).unwrap();
        assert_eq!(dataset.datatype().unwrap(), i32_type());
        assert_eq!(dataset.read_i32().unwrap(), dataset_values(i));
    }
}

/// The reader must not depend on which kind of index the file keeps: the heap ID
/// in a reference locates the message directly, and the list-to-B-tree conversion
/// changes only how a *writer* finds an equal message to share.
#[test]
fn a_btree_indexed_shared_message_file_reads_back_whole() {
    let fixture = btree_fixture();
    // The fixture is only a B-tree fixture if the file really carries the low
    // list maximum it was created with; the C library reports it back off the
    // reopened file's creation property list.
    let phase = hdf5::File::open(&fixture.path)
        .unwrap()
        .fcpl()
        .unwrap()
        .get_shared_mesg_phase_change()
        .unwrap();
    assert_eq!(phase.max_list, 2, "the list-to-B-tree threshold");

    assert_reads_back(&fixture.path, BTREE_FIXTURE_OBJECTS);
    let file = File::open(&fixture.path).unwrap();
    for rank in 1..=6usize {
        let dataset = file.dataset(&format!("wide/w{rank}")).unwrap();
        assert_eq!(
            dataset.shape().unwrap(),
            vec![rank as u64, rank as u64 + 1],
            "w{rank} shape"
        );
        assert_eq!(dataset.datatype().unwrap(), i32_type());
    }
}

/// A message shared by many objects is one message: every user must read the
/// same value, and the count of users is not something the heap entry records
/// per user.
#[test]
fn one_message_shared_by_many_objects_reads_the_same_everywhere() {
    let fixture = list_fixture(12);
    assert_reads_back(&fixture.path, 12);
}

/// A chunked, filtered dataset shares its filter pipeline and its fill value with
/// the next one like it — the other two message types `H5Pset_shared_mesg_index`
/// covers — so the second dataset's pipeline is a heap reference and its chunks
/// cannot be inflated without following it.
#[test]
fn shared_filter_pipelines_and_fill_values_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sohm_filters.h5");
    let data: Vec<i32> = (0..64).collect();
    {
        let file = create_shared_file(&path, 50, 40);
        for i in 0..3 {
            let dataset = file
                .new_dataset::<i32>()
                .deflate(4)
                .chunk([16])
                .shape([data.len()])
                .fill_value(-1i32)
                .create(format!("c{i}").as_str())
                .expect("create filtered dataset");
            dataset.write(data.as_slice()).unwrap();
        }
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    for i in 0..3 {
        let dataset = file.dataset(&format!("c{i}")).unwrap();
        assert_eq!(
            dataset.filters().len(),
            1,
            "c{i} must report its shared deflate pipeline"
        );
        assert_eq!(dataset.read_i32().unwrap(), data, "c{i} data");
    }
}

/// Every commit on a shared-message file walks that file's index — list or
/// B-tree — to check that nothing it moves or removes is named there, so a
/// misread index shows up as a failed commit rather than as a stranded record.
///
/// Neither fixture's index names an object header (the reference C library puts
/// even a single-user message straight in the heap here), so both commits go
/// through; what they prove is that the walk reads both index kinds.
#[test]
fn a_commit_on_a_shared_message_file_walks_its_index_and_proceeds() {
    for fixture in [list_fixture(4), btree_fixture()] {
        {
            // Scoped: the write session holds the file's exclusive lock, which
            // Windows enforces, so it has to be dropped before the file is
            // reopened for reading.
            let session =
                File::open_rw(&fixture.path).expect("open a shared-message file for edit");
            session.root().create_group("added").unwrap();
            session.commit().expect("commit on a shared-message file");
        }

        let file = File::open(&fixture.path).unwrap();
        assert!(file.group("added").is_ok());
        // The edit left every shared message where it was.
        assert_eq!(
            file.group("g0").unwrap().attrs().unwrap().get(SHARED_ATTR),
            Some(&AttrValue::I64(SHARED_ATTR_VALUE))
        );
    }
}

/// `repack` rewrites a shared-message file into one that shares nothing: every
/// message it resolves is written back inline, which is a faithful file (sharing
/// is a storage choice, not something a reader can name) and one this crate can
/// then edit freely.
#[test]
fn repack_rewrites_a_shared_message_file_without_sharing() {
    let fixture = list_fixture(4);
    let dst = fixture.path.with_file_name("repacked.h5");
    hdf5_pure::repack(&fixture.path, &dst, &hdf5_pure::RepackOptions::default())
        .expect("repack a shared-message file");

    assert_reads_back(&dst, 4);
    hdf5::File::open(&dst).expect("the C library opens the repacked file");
}
