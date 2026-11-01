#![cfg(feature = "__hdf5-1.10")]
//! Native interoperability for dense attribute storage and name indexes.
//!
//! Native operations run in the explicit `dense-attributes` fixture binary.
//! The parent checks the exit status and typed response before asserting the
//! observed attributes. Native rejection and an explicitly permitted assertion
//! abort are distinct from infrastructure or protocol failures.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use hdf5_pure::{AttrValue, FileBuilder};
use rstest::rstest;

use test_util::btree_v2;
use test_util::bytes;
use test_util::checksum;
use test_util::fractal_heap;
use test_util_hdf5::dense_attr;

use test_util::process::{self, Fixture};

use crate::test_util::dense_attributes::{
    self, AbortPolicy, Attribute, Lookup, Operation, Outcome, Rejection, Response, Stage,
};

#[derive(Debug, PartialEq, Eq)]
enum CReads {
    Aborted,
    Attrs(usize),
    Error(Rejection),
}

fn c_reads(path: &Path) -> CReads {
    c_child(path, Operation::Count, AbortPolicy::Forbid).verdict
}

struct CDetail {
    verdict: CReads,
    /// Total data sizes in bytes, keyed by attribute name.
    sizes: Vec<(String, usize)>,
    /// Wrapping sums of `i64` values, keyed by attribute name.
    sums: Vec<(String, i64)>,
}

impl CDetail {
    fn new(verdict: CReads) -> Self {
        Self {
            verdict,
            sizes: Vec::new(),
            sums: Vec::new(),
        }
    }

    #[track_caller]
    fn size_of(&self, name: &str) -> Option<usize> {
        self.sizes.iter().find(|(n, _)| n == name).map(|(_, s)| *s)
    }

    #[track_caller]
    fn sum_of(&self, name: &str) -> Option<i64> {
        self.sums.iter().find(|(n, _)| n == name).map(|(_, s)| *s)
    }
}

fn c_reads_in_detail(path: &Path) -> CDetail {
    c_child(path, Operation::Details, AbortPolicy::Forbid)
}

fn c_child(path: &Path, operation: Operation, abort: AbortPolicy) -> CDetail {
    assert!(path.is_file(), "missing dense-attribute fixture: {path:?}");
    let mut command = process::command(env!("CARGO_BIN_EXE_dense-attributes")).unwrap();
    command
        .arg(path)
        .arg(serde_json::to_string(&operation).unwrap());
    let output = Fixture::spawn(command, Duration::from_secs(30))
        .unwrap()
        .finish()
        .unwrap_or_else(|error| panic!("dense-attribute fixture failed: {error}"));
    let outcome = dense_attributes::observe(&operation, abort, output)
        .unwrap_or_else(|error| panic!("dense-attribute protocol failed: {error}"));
    match outcome {
        Outcome::Aborted => CDetail::new(CReads::Aborted),
        Outcome::Completed(Response::Rejected(rejection)) => CDetail::new(CReads::Error(rejection)),
        Outcome::Completed(Response::Count(count)) => CDetail::new(CReads::Attrs(count)),
        Outcome::Completed(Response::Details(attributes)) => {
            let mut detail = CDetail::new(CReads::Attrs(attributes.len()));
            for attribute in attributes {
                match attribute {
                    Attribute::Integer { name, size, sum } => {
                        detail.sizes.push((name.clone(), size));
                        detail.sums.push((name, sum));
                    }
                    Attribute::Other { name, size } => detail.sizes.push((name, size)),
                }
            }
            detail
        }
        Outcome::Completed(Response::Lookup(attributes)) => {
            let sums: Vec<_> = attributes
                .into_iter()
                .filter_map(|attribute| match attribute {
                    Lookup::Found { name, sum } => Some((name, sum)),
                    Lookup::Missing(_) => None,
                })
                .collect();
            CDetail {
                verdict: CReads::Attrs(sums.len()),
                sizes: Vec::new(),
                sums,
            }
        }
    }
}

fn c_looks_up_by_name(path: &Path, names: &[String]) -> CDetail {
    c_child(path, Operation::Lookup(names.to_vec()), AbortPolicy::Forbid)
}

fn c_inserts_then_reads(path: &Path) -> CDetail {
    c_child(path, Operation::InsertHuge, AbortPolicy::Forbid)
}

fn c_inserts_managed_then_reads(path: &Path) -> CDetail {
    c_child(path, Operation::InsertManaged, AbortPolicy::Forbid)
}

/// Both sides of the managed/huge threshold, against the library that has to read
/// them. The interesting one is the byte *past* it: the attribute's whole message
/// moves out of the direct block the name index points into, and is reachable
/// only if the heap ID, the huge-objects B-tree and the header fields all agree.
#[test]
fn c_reads_both_sides_of_the_managed_object_threshold() {
    let payload = dense_attr::largest_managed_payload();
    let dir = tempfile::tempdir().unwrap();

    let managed = dir.path().join("managed.h5");
    dense_attr::nine_attrs(payload).write(&managed).unwrap();
    let detail = c_reads_in_detail(&managed);
    assert_eq!(detail.verdict, CReads::Attrs(9));
    assert_eq!(
        detail.size_of("big"),
        Some(payload),
        "libhdf5 read back a different managed attribute than was written"
    );

    let huge = dir.path().join("huge.h5");
    dense_attr::nine_attrs(payload + 1).write(&huge).unwrap();
    assert_eq!(
        fractal_heap::huge_object_count(&std::fs::read(&huge).unwrap()),
        1,
        "one byte past the threshold must actually select huge storage, or this \
         test is comparing two managed heaps"
    );
    let detail = c_reads_in_detail(&huge);
    assert_eq!(detail.verdict, CReads::Attrs(9));
    assert_eq!(
        detail.size_of("big"),
        Some(payload + 1),
        "libhdf5 could not resolve the huge object"
    );
}

/// An `i64` attribute of `len` elements counting up from `first`, so both its
/// serialized size and its contents are known exactly.
fn counting_attr(first: i64, len: usize) -> AttrValue {
    AttrValue::I64Array((first..first + len as i64).collect())
}

/// The wrapping sum [`counting_attr`] implies, which is what the child reports.
fn counting_sum(first: i64, len: usize) -> i64 {
    (first..first + len as i64)
        .reduce(i64::wrapping_add)
        .unwrap_or(0)
}

#[derive(Clone, Copy)]
enum Producer {
    Native,
    Pure,
}

// The native producer forces dense storage even below the attribute-count threshold.
#[rstest]
#[case::pure_lone(Producer::Pure, 0)]
#[case::pure_mixed(Producer::Pure, 2)]
#[case::native_lone(Producer::Native, 0)]
#[case::native_mixed(Producer::Native, 2)]
fn c_reads_a_lone_huge_attribute(#[case] producer: Producer, #[case] others: usize) {
    let dir = tempfile::tempdir().unwrap();
    let len = 12_000usize;
    let path = dir.path().join("lone.h5");
    match producer {
        Producer::Pure => {
            let mut builder = FileBuilder::new();
            builder.set_attr("big", counting_attr(0, len));
            for i in 0..others {
                builder.set_attr(&format!("a{i}"), AttrValue::I64(i as i64));
            }
            builder.create_dataset("x").with_f64_data(&[1.0]);
            builder.write(&path).unwrap();
        }
        Producer::Native => {
            let file = hdf5::File::with_options()
                .with_fapl(|properties| properties.libver_latest())
                .with_fcpl(|properties| properties.attr_phase_change(0, 0))
                .create(&path)
                .unwrap();
            file.new_attr::<i64>()
                .shape([len])
                .create("big")
                .unwrap()
                .write(&(0..len as i64).collect::<Vec<_>>())
                .unwrap();
            for i in 0..others {
                file.new_attr::<i64>()
                    .create(format!("a{i}").as_str())
                    .unwrap()
                    .write_scalar(&(i as i64))
                    .unwrap();
            }
            file.close().unwrap();
        }
    }
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(fractal_heap::huge_object_count(&bytes), 1);
    assert_eq!(fractal_heap::managed_object_count(&bytes), others as u64);

    let detail = c_reads_in_detail(&path);
    assert_eq!(detail.verdict, CReads::Attrs(others + 1));
    let mut sizes = BTreeMap::from([("big".to_string(), len * size_of::<i64>())]);
    let mut sums = BTreeMap::from([("big".to_string(), counting_sum(0, len))]);
    for i in 0..others {
        sizes.insert(format!("a{i}"), size_of::<i64>());
        sums.insert(format!("a{i}"), i as i64);
    }
    assert_eq!(detail.sizes.into_iter().collect::<BTreeMap<_, _>>(), sizes);
    assert_eq!(detail.sums.into_iter().collect::<BTreeMap<_, _>>(), sums);
}

/// The heap header's `next_huge_object_id` holds the *last* ID assigned, not the
/// next one free, because the C library pre-increments it. Nothing that only
/// reads a file can tell the two apart — libhdf5 resolves a huge object through
/// the B-tree and never consults the counter. It consults it when *inserting*, so
/// an off-by-one there produces a file that reads back perfectly and then
/// collides ("record is already in B-tree") the first time anything adds to it.
///
/// This is the only test that exercises that field, and it also covers the header
/// accounting an insert has to build on: `free_space`, `managed_space_in_heap`
/// and the two object counts.
///
/// It pins the direction that matters rather than the exact value. A counter
/// below the last ID assigned collides, and this fails; one above merely skips an
/// ID, which the C library tolerates, so this would not catch it.
#[test]
fn c_adds_a_huge_attribute_to_a_heap_this_crate_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("insert.h5");
    let len = 12_000usize;

    let mut builder = FileBuilder::new();
    builder.set_attr("big", counting_attr(0, len));
    for i in 0..8 {
        builder.set_attr(&format!("a{i}"), AttrValue::I64(i as i64));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();
    assert_eq!(
        fractal_heap::huge_object_count(&std::fs::read(&path).unwrap()),
        1
    );

    let detail = c_inserts_then_reads(&path);
    assert_eq!(
        detail.verdict,
        CReads::Attrs(10),
        "libhdf5 could not add a huge attribute to this heap"
    );
    assert_eq!(detail.size_of("big"), Some(len * 8));
    assert_eq!(detail.size_of("inserted"), Some(len * 8));
}

/// A heap that mixes both storage classes: the direct block is sized by the
/// managed objects alone while the name index spans all of them, so libhdf5 has
/// to follow each heap ID to a different kind of destination.
#[test]
fn c_reads_a_heap_mixing_managed_and_huge_objects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.h5");

    let mut builder = FileBuilder::new();
    for i in 0..6 {
        builder.set_attr(&format!("small{i}"), counting_attr(i as i64 * 100, 10));
    }
    for i in 0..3 {
        builder.set_attr(
            &format!("big{i}"),
            counting_attr(i as i64 * 100_000, 10_000),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(fractal_heap::huge_object_count(&bytes), 3);
    assert_eq!(fractal_heap::managed_object_count(&bytes), 6);

    let detail = c_reads_in_detail(&path);
    assert_eq!(detail.verdict, CReads::Attrs(9));
    for i in 0..6 {
        let name = format!("small{i}");
        assert_eq!(detail.size_of(&name), Some(10 * 8));
        assert_eq!(detail.sum_of(&name), Some(counting_sum(i as i64 * 100, 10)));
    }
    for i in 0..3 {
        let name = format!("big{i}");
        assert_eq!(detail.size_of(&name), Some(10_000 * 8));
        assert_eq!(
            detail.sum_of(&name),
            Some(counting_sum(i as i64 * 100_000, 10_000)),
            "libhdf5 read the wrong bytes for huge object {i}"
        );
    }
}

/// Many huge objects in one heap, so the huge-objects B-tree holds more than the
/// single record the simplest case would. IDs are assigned in attribute order and
/// the B-tree is searched by ID, so a heap that only works for one object fails
/// here — and distinct lengths and values mean a record matched to the wrong ID
/// reads back wrong rather than coincidentally right.
#[test]
fn c_reads_a_heap_of_many_huge_objects() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many_huge.h5");
    let len = |i: usize| 9_000 + i * 100;

    let mut builder = FileBuilder::new();
    for i in 0..12 {
        builder.set_attr(
            &format!("h{i:02}"),
            counting_attr(i as i64 * 1_000_000, len(i)),
        );
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    assert_eq!(
        fractal_heap::huge_object_count(&std::fs::read(&path).unwrap()),
        12
    );

    let detail = c_reads_in_detail(&path);
    assert_eq!(detail.verdict, CReads::Attrs(12));
    for i in 0..12 {
        let name = format!("h{i:02}");
        assert_eq!(
            detail.size_of(&name),
            Some(len(i) * 8),
            "huge object {i} came back with the wrong length"
        );
        assert_eq!(
            detail.sum_of(&name),
            Some(counting_sum(i as i64 * 1_000_000, len(i))),
            "huge object {i} came back with the wrong contents"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum HeapCorruption {
    /// Changes only a header checksum byte.
    Checksum,
    /// Lowers the managed-object limit and recomputes the header checksum.
    ManagedLimit,
}

/// A valid dense-attribute image with a managed object larger than the lowered limit.
struct DenseHeapFixture {
    bytes: Vec<u8>,
    header: std::ops::Range<usize>,
    max_managed: u32,
    payload: usize,
}

impl DenseHeapFixture {
    fn at_managed_boundary() -> Self {
        let payload = dense_attr::largest_managed_payload();
        let bytes = dense_attr::nine_attrs(payload).finish().unwrap();
        let offsets = fractal_heap::header_offsets(&bytes);
        let &[at] = offsets.as_slice() else {
            panic!("expected one dense attribute heap, got {offsets:?}");
        };
        let len = usize::try_from(fractal_heap::encoded_header_len(&bytes, at)).unwrap();
        let header = at..at + len;
        let max_managed = bytes::u32_at(&bytes, at + MAX_MANAGED_AT);
        let checksum_at = header.end - checksum::CHECKSUM;
        assert_eq!(
            bytes::u32_at(&bytes, checksum_at),
            checksum::lookup3(&bytes[header.start..checksum_at])
        );
        assert_eq!(fractal_heap::huge_object_count(&bytes), 0);
        assert_eq!(fractal_heap::managed_object_count(&bytes), 9);
        assert!(payload > usize::try_from(LOWERED_LIMIT).unwrap());
        assert!(max_managed > LOWERED_LIMIT);
        assert_eq!(
            (u32::BITS - max_managed.leading_zeros()).div_ceil(8),
            (u32::BITS - LOWERED_LIMIT.leading_zeros()).div_ceil(8)
        );
        let object_length = big_managed_object_length(&bytes, at, max_managed);
        assert!(object_length > u64::from(LOWERED_LIMIT));
        assert!(object_length <= u64::from(max_managed));
        Self {
            bytes,
            header,
            max_managed,
            payload,
        }
    }

    fn corrupt(&self, corruption: HeapCorruption) -> Vec<u8> {
        let mut bytes = self.bytes.clone();
        match corruption {
            HeapCorruption::Checksum => bytes[self.header.end - checksum::CHECKSUM] ^= 1,
            HeapCorruption::ManagedLimit => {
                let at = self.header.start + MAX_MANAGED_AT;
                bytes[at..at + size_of::<u32>()].copy_from_slice(&LOWERED_LIMIT.to_le_bytes());
                checksum::restamp(&mut bytes, self.header.start, self.header.len());
            }
        }
        bytes
    }
}

// Attribute-name records and managed heap IDs follow sections `subsubsec_fmt4_infra_btrees_v2`
// and `subsec_fmt4_infra_fractalheap`, version 4.0.
fn big_managed_object_length(file: &[u8], heap: usize, max_managed: u32) -> u64 {
    let tree = bytes::sole_signature(file, btree_v2::SIGNATURE);
    assert_eq!(btree_v2::sole_btree_depth(file), 0);
    assert_eq!(
        bytes::u8_at(file, tree + btree_v2::SIGNATURE.len() + 1),
        ATTRIBUTE_NAME_INDEX_TYPE
    );
    let id_len = usize::from(bytes::u16_at(file, heap + HEAP_ID_LENGTH_AT));
    assert_eq!(id_len, ATTRIBUTE_HEAP_ID_LENGTH);
    let record_len = id_len + 1 + 2 * size_of::<u32>();
    assert_eq!(
        usize::from(bytes::u16_at(
            file,
            tree + btree_v2::NODE_PREFIX + size_of::<u32>()
        )),
        record_len
    );
    let records = btree_v2::name_index_leaf_records(file, 9);
    let matches: Vec<_> = records
        .iter()
        .enumerate()
        .filter(|(_, (_, hash))| *hash == checksum::lookup3(b"big"))
        .map(|(index, _)| index)
        .collect();
    let &[index] = matches.as_slice() else {
        panic!("expected one name-index record for big, got {matches:?}");
    };
    assert_eq!(records[index].0, 0);
    let leaf = bytes::sole_signature(file, btree_v2::LEAF_SIGNATURE);
    assert_eq!(
        bytes::u8_at(file, leaf + btree_v2::LEAF_SIGNATURE.len() + 1),
        ATTRIBUTE_NAME_INDEX_TYPE
    );
    let id = bytes::slice_at(
        file,
        leaf + btree_v2::NODE_PREFIX + index * record_len,
        id_len,
    );
    assert_eq!(id[0], MANAGED_ID_FLAGS);
    let table_at = heap
        + MAX_MANAGED_AT
        + size_of::<u32>()
        + HEAP_LENGTH_AND_ADDRESS_FIELDS * size_of::<u64>();
    let max_direct_at = table_at + size_of::<u16>() + size_of::<u64>();
    assert!(bytes::u64_at(file, max_direct_at) > u64::from(max_managed));
    let offset_bytes =
        usize::from(bytes::u16_at(file, max_direct_at + size_of::<u64>())).div_ceil(8);
    let length_bytes =
        usize::try_from((u32::BITS - max_managed.leading_zeros()).div_ceil(8)).unwrap();
    assert_eq!(1 + offset_bytes + length_bytes, id_len);
    bytes::uint_at(id, 1 + offset_bytes, length_bytes)
}

#[rstest]
#[case::checksum(HeapCorruption::Checksum)]
#[case::managed_limit(HeapCorruption::ManagedLimit)]
fn c_observes_checksum_and_managed_limit_corruption(#[case] corruption: HeapCorruption) {
    let fixture = DenseHeapFixture::at_managed_boundary();
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good.h5");
    std::fs::write(&good, &fixture.bytes).unwrap();
    assert_eq!(c_reads(&good), CReads::Attrs(9));
    assert_nine_native_attributes(&good, fixture.payload);
    let bad = dir.path().join("bad.h5");
    let corrupted = fixture.corrupt(corruption);
    let checksum_at = fixture.header.end - checksum::CHECKSUM;
    let stored = bytes::u32_at(&corrupted, checksum_at);
    let computed = checksum::lookup3(&corrupted[fixture.header.start..checksum_at]);
    let limit_at = fixture.header.start + MAX_MANAGED_AT;
    match corruption {
        HeapCorruption::Checksum => {
            assert_eq!(bytes::u32_at(&corrupted, limit_at), fixture.max_managed);
            assert_ne!(stored, computed);
            assert_eq!(corrupted[..checksum_at], fixture.bytes[..checksum_at]);
        }
        HeapCorruption::ManagedLimit => {
            assert_eq!(bytes::u32_at(&corrupted, limit_at), LOWERED_LIMIT);
            assert_eq!(stored, computed);
            assert_eq!(corrupted[..limit_at], fixture.bytes[..limit_at]);
            let limit_end = limit_at + size_of::<u32>();
            assert_eq!(
                corrupted[limit_end..checksum_at],
                fixture.bytes[limit_end..checksum_at]
            );
        }
    }
    assert_eq!(
        corrupted[fixture.header.end..],
        fixture.bytes[fixture.header.end..]
    );
    assert_eq!(fractal_heap::huge_object_count(&corrupted), 0);
    assert_eq!(fractal_heap::managed_object_count(&corrupted), 9);
    std::fs::write(&bad, &corrupted).unwrap();
    assert_native_corruption(&bad, corruption, fixture.payload);
    let past = dense_attr::nine_attrs(fixture.payload + 1)
        .finish()
        .unwrap();
    assert_eq!(fractal_heap::huge_object_count(&past), 1);
    assert_eq!(fractal_heap::managed_object_count(&past), 8);
}

// H5C_load_entry in H5C.c and H5HF__man_op_real in H5HFman.c, HDF5 1.10.11, report
// checksum and size errors. H5A__attr_release_table in H5Aint.c, HDF5 2.2.0, can assert during cleanup.
/// Checks the native outcome against the corruption's permitted observations.
///
/// The size case permits native acceptance, a size-specific rejection, or the
/// assertion abort permitted by the child protocol. Acceptance requires every
/// attribute name, data size, and integer sum to match the valid control.
fn assert_native_corruption(path: &Path, corruption: HeapCorruption, payload: usize) {
    let detail = c_child(path, Operation::Count, AbortPolicy::AttributeRelease);
    match (corruption, detail.verdict) {
        (_, CReads::Aborted) => {}
        (HeapCorruption::Checksum, CReads::Error(rejection)) => {
            assert_eq!(rejection.stage, Stage::Names);
            assert!(
                rejection.frames.iter().any(|frame| frame.major == "Cache"
                    && frame.minor == "ReadError"
                    && frame.description.contains("checksum")),
                "{rejection:?}"
            );
        }
        (HeapCorruption::ManagedLimit, CReads::Error(rejection)) => {
            assert_eq!(rejection.stage, Stage::Names);
            assert!(
                rejection.frames.iter().any(|frame| frame.major == "Heap"
                    && frame.minor == "BadRange"
                    && frame.function == "H5HF__man_op_real"
                    && frame.description == "fractal heap object should be standalone"),
                "{rejection:?}"
            );
            assert!(
                rejection
                    .frames
                    .iter()
                    .all(|frame| !frame.description.contains("checksum")),
                "{rejection:?}"
            );
        }
        (HeapCorruption::ManagedLimit, CReads::Attrs(9)) => {
            assert_nine_native_attributes(path, payload)
        }
        (corruption, observed) => panic!(
            "unexpected native result for {corruption:?} under HDF5 {:?}: {observed:?}",
            hdf5::library_version()
        ),
    }
}

fn assert_nine_native_attributes(path: &Path, payload: usize) {
    let CDetail {
        verdict,
        sizes,
        sums,
    } = c_reads_in_detail(path);
    assert_eq!(verdict, CReads::Attrs(9));
    let mut expected_sizes = BTreeMap::from([("big".to_string(), payload)]);
    for i in 0..8 {
        expected_sizes.insert(format!("a{i}"), size_of::<i64>());
    }
    assert_eq!(
        sizes.into_iter().collect::<BTreeMap<_, _>>(),
        expected_sizes
    );
    assert_eq!(
        sums.into_iter().collect::<BTreeMap<_, _>>(),
        (0..8i64).map(|i| (format!("a{i}"), i)).collect()
    );
}

// The managed-size field follows the flags in section `subsec_fmt4_infra_fractalheap`, version 4.0.
const MAX_MANAGED_AT: usize = 10;
const LOWERED_LIMIT: u32 = 1_000;
const ATTRIBUTE_NAME_INDEX_TYPE: u8 = 8;
const ATTRIBUTE_HEAP_ID_LENGTH: usize = 8;
const HEAP_ID_LENGTH_AT: usize = 5;
const HEAP_LENGTH_AND_ADDRESS_FIELDS: usize = 12;
const MANAGED_ID_FLAGS: u8 = 0;

/// Attribute counts that need a multi-level name index, against the library
/// that defines the shape. 61,680 was the ceiling while the index was a single
/// leaf grown to fit — one more produced a file that *aborted* libhdf5 rather
/// than being rejected by it, which is exactly why the accepted side has to be
/// confirmed here and not just against our own reader.
///
/// Each count crosses a different level boundary of a 512-byte-node tree: 570
/// is the first that needs depth 2, 10,260 the first that needs depth 3, and
/// 70,000 is well past the old single-leaf limit.
#[test]
fn c_reads_attribute_counts_that_need_a_multi_level_index() {
    let build = |n: usize| {
        let mut builder = FileBuilder::new();
        for i in 0..n {
            builder.set_attr(&format!("a{i:06}"), AttrValue::I64(i as i64));
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder
    };

    let dir = tempfile::tempdir().unwrap();
    for count in [570, 10_260, 61_681, 70_000] {
        let path = dir.path().join(format!("count_{count}.h5"));
        build(count).write(&path).unwrap();
        assert_eq!(
            c_reads(&path),
            CReads::Attrs(count),
            "libhdf5 could not read {count} attributes"
        );

        // Iteration walks the tree in order and would be satisfied by a tree
        // whose records sit in the wrong order; opening by name makes libhdf5
        // descend it, comparing at each internal node. Sampled across the range
        // so the descents take different paths.
        let sampled: Vec<String> = [0, 1, count / 3, count / 2, count - 2, count - 1]
            .iter()
            .map(|i| format!("a{i:06}"))
            .collect();
        let detail = c_looks_up_by_name(&path, &sampled);
        assert_eq!(
            detail.verdict,
            CReads::Attrs(sampled.len()),
            "libhdf5 could not look up attributes by name among {count}"
        );
        for (i, name) in [0, 1, count / 3, count / 2, count - 2, count - 1]
            .iter()
            .zip(&sampled)
        {
            assert_eq!(
                detail.sum_of(name),
                Some(*i as i64),
                "{name} read back the wrong value among {count} attributes"
            );
        }
    }
}

/// Managed attributes whose bytes outgrow the root direct block, against the
/// library that defines how a heap grows instead: a root indirect block over a
/// doubling table of direct blocks, and — once the root's own row of indirect
/// blocks is used up — nested indirect blocks below that.
///
/// Both shapes have to be crosschecked separately. A root indirect block is one
/// level of navigation; a nested one is the recursion, and it is reached only by
/// objects large enough to skip every row of smaller blocks, which is a different
/// path through the emitter as well as through libhdf5.
///
/// Values are read back, not just counted: an attribute placed at the wrong heap
/// offset still yields a well-formed message from a neighbouring block.
#[test]
fn c_reads_a_heap_whose_managed_blocks_need_indirect_blocks() {
    let dir = tempfile::tempdir().unwrap();

    // (name, elements per attribute, attribute count, indirect blocks expected)
    let shapes: [(&str, usize, usize, usize); 2] = [
        // Small attributes, enough of them to spill out of one starting-size
        // block into the table's lower rows.
        ("shallow", 10, 60, 1),
        // Each attribute nearly fills the table's largest direct block, so the
        // root's four of those run out and the walk descends into a nested
        // indirect block for the rest.
        ("nested", 8_000, 12, 2),
    ];

    for (name, len, count, indirect) in shapes {
        let path = dir.path().join(format!("{name}.h5"));
        let mut builder = FileBuilder::new();
        for i in 0..count {
            builder.set_attr(
                &format!("a{i:04}"),
                counting_attr(i as i64 * 1_000_000, len),
            );
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        builder.write(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            fractal_heap::huge_object_count(&bytes),
            0,
            "{name}: these attributes must stay managed, or this is not a test \
             about the managed blocks"
        );
        assert_eq!(fractal_heap::managed_object_count(&bytes), count as u64);
        assert!(
            fractal_heap::root_indirect_rows(&bytes) > 0,
            "{name}: the root should have grown into an indirect block"
        );
        assert!(
            fractal_heap::indirect_block_count(&bytes) >= indirect,
            "{name}: expected at least {indirect} indirect blocks, found {}",
            fractal_heap::indirect_block_count(&bytes)
        );

        let detail = c_reads_in_detail(&path);
        assert_eq!(detail.verdict, CReads::Attrs(count), "{name}");
        for i in 0..count {
            let attr = format!("a{i:04}");
            assert_eq!(detail.size_of(&attr), Some(len * 8), "{name}: {attr}");
            assert_eq!(
                detail.sum_of(&attr),
                Some(counting_sum(i as i64 * 1_000_000, len)),
                "{name}: {attr} came back with the wrong contents"
            );
        }

        // And libhdf5 can go on to place a managed attribute of its own in the
        // table, which reading alone never checks: that placement starts from the
        // heap header's allocation iterator, and a wrong one lands the new block
        // on top of the attributes already stored.
        let detail = c_inserts_managed_then_reads(&path);
        assert_eq!(
            detail.verdict,
            CReads::Attrs(count + 1),
            "{name}: libhdf5 could not add a managed attribute to this heap"
        );
        assert_eq!(detail.sum_of("inserted"), Some((0..16).sum::<i64>()));
        for i in 0..count {
            let attr = format!("a{i:04}");
            assert_eq!(
                detail.sum_of(&attr),
                Some(counting_sum(i as i64 * 1_000_000, len)),
                "{name}: {attr} was overwritten by the attribute libhdf5 added"
            );
        }
    }
}

/// A multi-megabyte dense heap of individually small attributes is inside what
/// the emitter can encode, and libhdf5 must accept it — including the doubling
/// table of direct blocks the emitter grows to hold it.
#[test]
fn c_reads_a_multi_megabyte_dense_heap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big_total.h5");

    let mut builder = FileBuilder::new();
    for i in 0..40 {
        builder.set_attr(&format!("a{i}"), AttrValue::AsciiString("y".repeat(60_000)));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    assert!(std::fs::metadata(&path).unwrap().len() > 2_000_000);
    assert_eq!(c_reads(&path), CReads::Attrs(40));
}

/// Attribute names that share a `jenkins_lookup3` hash with the name beside
/// them, each pair listed in the order that is the *reverse* of how the two
/// compare — the order a writer that breaks a hash tie by insertion index gets
/// wrong.
///
/// Ten pairs rather than one because a colliding pair only exercises an internal
/// node when the split happens to fall between its two records, and which pair
/// that is depends on where every other name's hash lands.
const COLLIDING_PAIRS: [(&str, &str); 10] = [
    ("k69209", "k155448"),
    ("k92415", "k257792"),
    ("k76770", "k310249"),
    ("k42661", "k327803"),
    ("k5073", "k491506"),
    ("k7843", "k559364"),
    ("k62716", "k403994"),
    ("k70075", "k405037"),
    ("k59238", "k391908"),
    ("k53867", "k394939"),
];

/// Set every name in [`COLLIDING_PAIRS`] on the root group, first of each pair
/// first, and pad to `total` attributes. Each value is the attribute's index, so
/// a lookup that lands on the wrong record reports the wrong sum rather than
/// merely succeeding.
fn colliding_attrs(total: usize) -> (FileBuilder, Vec<String>, Vec<i64>) {
    let mut builder = FileBuilder::new();
    let mut names = Vec::new();
    let mut values = Vec::new();
    for (first, second) in COLLIDING_PAIRS {
        for name in [first, second] {
            let value = names.len() as i64;
            builder.set_attr(name, AttrValue::I64(value));
            names.push(name.to_string());
            values.push(value);
        }
    }
    assert!(total >= names.len(), "no room for the colliding pairs");
    for i in names.len()..total {
        builder.set_attr(&format!("pad{i:06}"), AttrValue::I64(i as i64));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);
    (builder, names, values)
}

/// Have libhdf5 open each colliding name and check it got that name's own value.
#[track_caller]
fn c_finds_each(path: &std::path::Path, names: &[String], values: &[i64], context: &str) {
    let detail = c_looks_up_by_name(path, names);
    assert_eq!(
        detail.verdict,
        CReads::Attrs(names.len()),
        "{context}: libhdf5 could not open every colliding name"
    );
    for (name, value) in names.iter().zip(values) {
        assert_eq!(
            detail.sum_of(name),
            Some(*value),
            "{context}: {name} came back with another record's value"
        );
    }
}

/// How many attributes put a colliding name in the root of a depth-2 index.
///
/// Whether a colliding pair straddles a node boundary is a property of where
/// every name's hash falls, so it is a search, not a calculation. This count came
/// out of running one over 20..=760: it is the smallest set of [`colliding_attrs`]
/// that is *both* depth 2 and has a colliding name as its root record. Smaller
/// sets straddle at depth 1 (30 is the first), and neighbouring counts do not
/// straddle at all — 600 and 710 both promote a padding name. The tests assert
/// the property still holds, so a change in how the tree is planned reports
/// itself rather than quietly reverting them to searching one flat leaf.
const STRADDLING_TOTAL: usize = 709;

/// Assert the index in `bytes` is the deep tree with a promoted colliding record
/// that the multi-level tests mean to search, rather than something shallower
/// that would satisfy their lookups without descending anything.
#[track_caller]
fn assert_a_colliding_name_is_promoted(bytes: &[u8], colliding: usize, context: &str) {
    assert_eq!(
        btree_v2::sole_btree_depth(bytes),
        2,
        "{context}: the index is not the multi-level tree this test exists to search"
    );
    let root = btree_v2::root_records(bytes);
    // A record's creation order is the attribute's index in the order it was set,
    // and `colliding_attrs` sets every colliding name before any padding.
    assert!(
        root.iter().any(|(order, _)| (*order as usize) < colliding),
        "{context}: no colliding name reached the root, so nothing here descends \
         past one; re-run the search behind STRADDLING_TOTAL"
    );
}

/// The name index a hash collision has to survive is not always one leaf.
///
/// [`c_finds_attributes_whose_names_hash_alike`] covers a ten-attribute set,
/// whose whole index is a single leaf node — every record sits in one flat run of
/// bytes and libhdf5 never descends anything. Since #195 the index is a tree of
/// 512-byte nodes, so a large set puts records in internal nodes too, and a
/// colliding pair can straddle a split: one record promoted into a parent, its
/// twin in a child below.
///
/// What that buys is a call site, not a second way for the writer to be wrong.
/// The order this crate emits comes from one flat sort before the tree is
/// planned, so any error in it shows up in the single-leaf test too. The half
/// that only a deep tree reaches is libhdf5's: resolving a record inside a `BTIN`
/// node runs the same hash-then-`strcmp` compare from `H5B2__find`'s descent,
/// against a record this crate promoted, and a pair ordered the wrong way sends
/// it down the branch that cannot hold what it is looking for.
#[test]
fn c_finds_colliding_names_in_a_multi_level_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("colliding_multi_level.h5");

    let (builder, names, values) = colliding_attrs(STRADDLING_TOTAL);
    let bytes = builder.finish().unwrap();
    assert_a_colliding_name_is_promoted(&bytes, names.len(), "multi-level");
    std::fs::write(&path, &bytes).unwrap();

    c_finds_each(&path, &names, &values, "multi-level");
}

/// Repacking a colliding attribute set yields a file libhdf5 can still search by
/// name.
///
/// [`hdf5_pure::repack`] is the remediation the issue points affected files at,
/// and it reaches the index builder by a different route than `FileBuilder::write`
/// does: it reads a parsed attribute set back out of a source heap and emits a
/// fresh heap and index for it. What makes that a *repair* is that the order it
/// emits is a function of the names alone, never of the order they arrived in —
/// pinned by
/// `file_writer::tests::the_name_index_order_does_not_depend_on_insertion_order`,
/// which is where a change to the sort shows up.
///
/// Be clear about what this test can and cannot catch. Its source file is one
/// this crate just wrote, so the attributes come back already in index order and
/// a sort that dropped its tie-break would still leave them right; the sort is
/// not what this guards, and neither is the repair — that is
/// [`c_repairs_a_file_written_before_the_fix`], on a file an affected version
/// actually wrote. What this guards is `repack` itself over an attribute set
/// large enough to need a multi-level index: every name present, every value its
/// own, and the tree deep enough that libhdf5 has to descend it.
#[test]
fn c_finds_colliding_names_after_a_repack() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("colliding_src.h5");
    let dst = dir.path().join("colliding_repacked.h5");

    let (builder, names, values) = colliding_attrs(STRADDLING_TOTAL);
    builder.write(&src).unwrap();
    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::default()).unwrap();

    let bytes = std::fs::read(&dst).unwrap();
    assert_a_colliding_name_is_promoted(&bytes, names.len(), "repacked");
    c_finds_each(&dst, &names, &values, "repacked");
}

/// A file written before the fix is repaired by rewriting it, which is what the
/// changelog tells anyone holding one to do.
///
/// The fixture is not synthesized: it was written by 0.28.0 itself, checked out
/// at its release tag and handed [`COLLIDING_PAIRS`] in the order above. That
/// version broke a name-hash tie by insertion index, so ten of its twenty
/// attributes are indexed on the wrong side of their twin. Regenerate it the same
/// way if it ever needs to change — a hand-patched file would only prove that a
/// hand-patched file repairs.
///
/// All three halves of the claim are asserted, because each is a separate thing
/// that could stop being true: libhdf5 really cannot find ten of the names, this
/// crate really can read every one of them with its value intact (which is *why*
/// the file is repairable rather than lost), and `repack` really does hand
/// libhdf5 a file it can search. Without the first assertion the other two would
/// pass just as well on a file that was never broken.
///
/// What this does *not* guard is the tie-break itself, and the reason is worth
/// knowing: `repack` re-sets the attributes it read in name order (`copy_attrs` in
/// `src/repack/mod.rs`), so the rebuild is handed a list on which a hash-only stable sort
/// already produces name order within a hash group. Deleting the tie-break leaves
/// this test green. The sort is guarded by
/// [`c_finds_colliding_names_in_a_multi_level_index`] and its single-leaf sibling;
/// what this guards is that the remedy the changelog names still works end to
/// end, which no amount of reasoning about the sort establishes.
#[test]
fn c_repairs_a_file_written_before_the_fix() {
    let affected = test_util::data("pure/dense_attrs_colliding_v0_28_0.h5");
    let names: Vec<String> = COLLIDING_PAIRS
        .iter()
        .flat_map(|(first, second)| [first.to_string(), second.to_string()])
        .collect();
    let values: Vec<i64> = (0..names.len() as i64).collect();

    // Half of each pair is on the wrong side of its twin, so libhdf5's binary
    // search walks away from it: it opens the ten that happen to sit where it
    // looks and reports the rest missing.
    let detail = c_looks_up_by_name(&affected, &names);
    assert_eq!(
        detail.verdict,
        CReads::Attrs(names.len() / 2),
        "the fixture is no longer the broken file this test needs"
    );

    // The bytes are all there and in the right place — only the index order is
    // wrong — so a reader that walks the index start to finish sees everything.
    let file = hdf5_pure::File::open(&affected).unwrap();
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), names.len());
    for (name, value) in names.iter().zip(&values) {
        assert_eq!(
            attrs.get(name),
            Some(&AttrValue::I64(*value)),
            "{name}: this crate misread the affected file"
        );
    }
    drop(file);

    let dir = tempfile::tempdir().unwrap();
    let repaired = dir.path().join("repaired.h5");
    hdf5_pure::repack(&affected, &repaired, &hdf5_pure::RepackOptions::default()).unwrap();
    c_finds_each(&repaired, &names, &values, "repaired");
}

/// libhdf5 must find both halves of a name-hash collision.
///
/// Its search compares the 32-bit name hash and, on a tie, `strcmp`s the name it
/// pulls back out of the heap. Two names that hash alike therefore have to be
/// stored in `strcmp` order or the descent takes a wrong branch and one of them
/// is unfindable — while iteration, which never compares anything, still reports
/// both. The zero-padded names the other tests here use cannot expose this: their
/// insertion order and their `strcmp` order are the same by construction.
#[test]
fn c_finds_attributes_whose_names_hash_alike() {
    // Distinct names with the same jenkins_lookup3 hash, set in the reverse of
    // the order they compare in.
    const FIRST: &str = "k69209";
    const SECOND: &str = "k155448";

    let mut builder = FileBuilder::new();
    builder.set_attr(FIRST, AttrValue::I64(11));
    builder.set_attr(SECOND, AttrValue::I64(22));
    for i in 0..8 {
        builder.set_attr(&format!("f{i}"), AttrValue::I64(i));
    }
    builder.create_dataset("x").with_f64_data(&[1.0]);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("colliding_names.h5");
    builder.write(&path).unwrap();

    let names = [FIRST.to_string(), SECOND.to_string()];
    let detail = c_looks_up_by_name(&path, &names);
    assert_eq!(
        detail.verdict,
        CReads::Attrs(names.len()),
        "libhdf5 could not open both colliding names"
    );
    assert_eq!(detail.sum_of(FIRST), Some(11));
    assert_eq!(detail.sum_of(SECOND), Some(22));
}
