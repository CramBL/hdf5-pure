#![cfg(feature = "__hdf5-1.10")]
//! Reading dense (fractal-heap) group link storage that libhdf5
//! writes, across group sizes that grow the heap's root indirect block past a
//! single direct block.
//!
//! Regression for a fractal-heap reader bug: the doubling-table row classifier
//! used the `start_root_rows` header field as the direct/indirect boundary
//! instead of the computed `max_direct_rows`. Once a dense group's heap grew a
//! root indirect block with more direct-block rows than that hint (empirically
//! ~150 short links), rows that are actually direct were mis-read as indirect
//! and the reader recursed into a direct block expecting an indirect-block
//! signature, failing with `InvalidFractalHeapSignature`. These groups are read
//! through both the buffered and the streaming reader.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::Source;
use hdf5_pure_core::__private::{
    BaseAddressExt, FileRange, FileRegion, PhysicalAddress, StoredAddressExt,
};
use hdf5_pure_core::{BaseAddress, StoredAddress};
use hdf5_pure_format::__private::{
    BTREE_V2_ATTRIBUTE_NAME, BTREE_V2_LINK_NAME, BTreeV2Header, BTreeV2Layout, FractalHeapBlock,
    FractalHeapBlockLayout, FractalHeapHeaderLocator, LengthWidth, OffsetWidth,
};
use rstest::rstest;

/// Write a group of `n` single-i32 datasets `g/d{i}` with the latest format, so
/// the C library stores the links densely in a fractal heap.
fn write_dense_group(path: &std::path::Path, n: usize) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_latest())
        .create(path)
        .unwrap();
    let g = file.create_group("g").unwrap();
    for i in 0..n {
        g.new_dataset::<i32>()
            .shape((1,))
            .create(format!("d{i}").as_str())
            .unwrap()
            .write(&[i as i32])
            .unwrap();
    }
    file.close().unwrap();
}

#[test]
fn reads_c_written_dense_link_groups_across_sizes() {
    let dir = tempfile::tempdir().unwrap();
    // 16: spills compact storage into a single fractal-heap direct block.
    // 150: the root grows to a 2-row indirect block — both rows direct — which
    // is the size that used to fail outright. 600: a larger root indirect block
    // with several direct-block rows, exercising the corrected boundary.
    for &n in &[16usize, 150, 600] {
        let path = dir.path().join(format!("dense_{n}.h5"));
        write_dense_group(&path, n);

        // Buffered reader: every link resolves to the right dataset value.
        let buffered = File::open(&path).unwrap();
        for i in 0..n {
            let v = buffered
                .dataset(format!("g/d{i}").as_str())
                .unwrap()
                .read_i32()
                .unwrap();
            assert_eq!(v, vec![i as i32], "buffered n={n} d{i}");
        }

        // Streaming reader: exercises the on-demand indirect-block window sizing
        // (`indirect_block_entries_len`), which shares the same row classifier.
        let streaming = File::open_streaming(&path).unwrap();
        for i in 0..n {
            let v = streaming
                .dataset(format!("g/d{i}").as_str())
                .unwrap()
                .read_i32()
                .unwrap();
            assert_eq!(v, vec![i as i32], "streaming n={n} d{i}");
        }
    }
}

#[rstest]
#[case::leaf(16)]
#[case::internal(150)]
#[case::larger(600)]
fn borrowed_nodes_search_c_written_name_indexes(
    #[case] count: usize,
    #[values(false, true)] relocated: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("indexed.h5");
    write_dense_group(&path, count);
    let original = std::fs::read(&path).unwrap();
    let base = if relocated { 512 } else { 0 };
    let mut bytes = vec![0; base];
    bytes.extend_from_slice(&original);
    let at = test_util::bytes::sole_signature(&bytes, test_util::btree_v2::SIGNATURE);
    let header = BTreeV2Header::parse(&bytes, at, 8, 8).unwrap();
    assert_eq!(header.tree_type, BTREE_V2_LINK_NAME);
    assert_eq!(header.total_records, count as u64);
    assert_eq!(header.depth == 0, count == 16);
    let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
    let wanted = hdf5_pure_format::__private::jenkins_lookup3(format!("d{}", count - 1).as_bytes());
    let mut node = layout.root().unwrap();
    let mut visited = 0;
    let found = loop {
        visited += 1;
        let start = base + usize::try_from(node.address().get()).unwrap();
        let view = layout.parse(&bytes[start..], node).unwrap();
        let mut left = 0;
        let mut right = view.record_count();
        let mut matched = false;
        while left < right {
            let mid = left + (right - left) / 2;
            let hash = u32::from_le_bytes(view.record(mid).unwrap()[..4].try_into().unwrap());
            match hash.cmp(&wanted) {
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Equal => {
                    matched = true;
                    break;
                }
                std::cmp::Ordering::Greater => right = mid,
            }
        }
        if matched {
            break true;
        }
        let Some(child) = view.child(left).unwrap() else {
            break false;
        };
        node = child;
    };
    assert!(found);
    assert!(visited <= usize::from(header.depth) + 1);
}

#[rstest]
#[case::leaf(16)]
#[case::internal(150)]
#[case::larger(600)]
fn indexed_named_lookups_preserve_relocated_dense_groups(
    #[case] count: usize,
    #[values((0, 0), (0, 512), (512, 512), (512, 1024), (512, 0))] placement: (usize, usize),
    #[values(false, true)] streaming: bool,
) {
    let (userblock, signature_offset) = placement;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("named.h5");
    let reference = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_latest())
        .with_fcpl(|properties| properties.userblock(userblock as u64))
        .create(&path)
        .unwrap();
    let group = reference.create_group("g").unwrap();
    for index in 0..count {
        group
            .new_dataset::<i32>()
            .shape(())
            .create(format!("d{index}").as_str())
            .unwrap()
            .write_scalar(&(index as i32))
            .unwrap();
    }
    drop(group);
    reference.close().unwrap();
    let original = std::fs::read(&path).unwrap();
    let mut bytes = vec![0; signature_offset];
    bytes.extend_from_slice(&original[userblock..]);
    std::fs::write(&path, bytes).unwrap();
    let reference = hdf5::File::open(&path).unwrap();
    let reader = open_dense_reader(&path, streaming);
    let name = format!("g/d{}", count - 1);
    assert_eq!(
        reader.dataset(&name).unwrap().read_i32().unwrap(),
        [i32::try_from(count - 1).unwrap()]
    );
    assert_eq!(
        reference.dataset(&name).unwrap().read_raw::<i32>().unwrap(),
        [i32::try_from(count - 1).unwrap()]
    );
    let Error::Format(error) = reader.dataset("g/absent").unwrap_err() else {
        panic!("expected missing path");
    };
    assert_eq!(error, FormatError::PathNotFound("g/absent".into()));
}

#[rstest]
#[case::leaf(16)]
#[case::internal(150)]
#[case::larger(600)]
fn dense_named_lookup_reads_at_most_one_btree_node_per_depth(
    #[case] count: usize,
    #[values(false, true)] absent: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reads.h5");
    write_dense_group(&path, count);
    let bytes = std::fs::read(&path).unwrap();
    let (_, header) = dense_index(&bytes);
    let node_reads = Arc::new(AtomicUsize::new(0));
    let reader = File::from_source(DenseNodeCountingSource {
        bytes,
        node_reads: Arc::clone(&node_reads),
    })
    .unwrap();
    node_reads.store(0, Ordering::Relaxed);
    if absent {
        let Error::Format(error) = reader.dataset("g/absent").unwrap_err() else {
            panic!("expected missing path");
        };
        assert_eq!(error, FormatError::PathNotFound("g/absent".into()));
    } else {
        assert_eq!(reader.dataset("g/d0").unwrap().read_i32().unwrap(), [0]);
    }
    let visited = node_reads.load(Ordering::Relaxed);
    assert!(visited > 0);
    assert!(
        visited <= usize::from(header.depth) + 1,
        "{visited} node reads for depth {}",
        header.depth
    );
}

#[rstest]
#[case::hard_soft(DenseLink::Hard, DenseLink::Soft)]
#[case::soft_hard(DenseLink::Soft, DenseLink::Hard)]
#[case::external_hard(DenseLink::External, DenseLink::Hard)]
#[case::hard_external(DenseLink::Hard, DenseLink::External)]
#[case::soft_external(DenseLink::Soft, DenseLink::External)]
#[case::external_soft(DenseLink::External, DenseLink::Soft)]
fn indexed_colliding_names_preserve_link_targets(
    #[case] first_type: DenseLink,
    #[case] second_type: DenseLink,
    #[values(false, true)] streaming: bool,
) {
    let names = ["k69209", "k155448"];
    assert_eq!(
        hdf5_pure_format::__private::jenkins_lookup3(names[0].as_bytes()),
        hdf5_pure_format::__private::jenkins_lookup3(names[1].as_bytes())
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("collisions.h5");
    write_dense_group(&path, 150);
    let reference = hdf5::File::open_rw(&path).unwrap();
    let group = reference.group("g").unwrap();
    for ((name, link_type), target) in names
        .into_iter()
        .zip([first_type, second_type])
        .zip(["d0", "d1"])
    {
        match link_type {
            DenseLink::Hard => group.link_hard(target, name).unwrap(),
            DenseLink::Soft => group.link_soft(target, name).unwrap(),
            DenseLink::External => group.link_external("absent.h5", "/value", name).unwrap(),
        }
    }
    drop(group);
    reference.close().unwrap();
    let reference = hdf5::File::open(&path).unwrap();
    let group = reference.group("g").unwrap();
    let reader = open_dense_reader(&path, streaming);
    for ((name, link_type), expected) in
        names.into_iter().zip([first_type, second_type]).zip([0, 1])
    {
        assert!(group.link_exists(name));
        let path = format!("g/{name}");
        if matches!(link_type, DenseLink::External) {
            let Error::Format(error) = reader.dataset(&path).unwrap_err() else {
                panic!("expected external link error");
            };
            assert_eq!(
                error,
                FormatError::ExternalLinkUnsupported {
                    path,
                    filename: "absent.h5".into(),
                    object_path: "/value".into()
                }
            );
        } else {
            assert_eq!(
                reader.dataset(&path).unwrap().read_i32().unwrap(),
                [expected]
            );
            assert_eq!(
                group.dataset(name).unwrap().read_raw::<i32>().unwrap(),
                [expected]
            );
        }
    }
}

#[rstest]
#[case::internal_match(false, false)]
#[case::corrupt_leaf(true, false)]
#[case::invalid_pointer(true, true)]
fn indexed_matches_validate_only_the_selected_path(
    #[case] selected_leaf: bool,
    #[case] invalid_pointer: bool,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("internal.h5");
    write_dense_group(&path, 150);
    let mut bytes = std::fs::read(&path).unwrap();
    let (_, header) = dense_index(&bytes);
    assert_eq!(header.depth, 1);
    let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
    let root = layout.root().unwrap();
    let view = layout
        .parse(&bytes[root.address().get() as usize..], root)
        .unwrap();
    let child = view.child(0).unwrap().unwrap();
    let hash = if selected_leaf {
        let leaf = layout
            .parse(&bytes[child.address().get() as usize..], child)
            .unwrap();
        u32::from_le_bytes(
            leaf.record(0).unwrap()[..LINK_NAME_HASH_LEN]
                .try_into()
                .unwrap(),
        )
    } else {
        u32::from_le_bytes(
            view.record(0).unwrap()[..LINK_NAME_HASH_LEN]
                .try_into()
                .unwrap(),
        )
    };
    let index = (0..150)
        .find(|index| {
            hdf5_pure_format::__private::jenkins_lookup3(format!("d{index}").as_bytes()) == hash
        })
        .unwrap();
    if invalid_pointer {
        let start = root.address().get() as usize;
        let pointer = start
            + test_util::btree_v2::NODE_PREFIX
            + usize::from(header.num_records_in_root) * usize::from(header.record_size);
        bytes[pointer..pointer + usize::from(OffsetWidth::Eight.get())].fill(0xff);
        test_util::checksum::restamp(&mut bytes, start, layout.encoded_len(root).unwrap());
    } else {
        bytes[child.address().get() as usize] = b'X';
    }
    std::fs::write(&path, &bytes).unwrap();
    let reader = open_dense_reader(&path, streaming);
    if selected_leaf {
        let Error::Format(error) = reader.dataset(&format!("g/d{index}")).unwrap_err() else {
            panic!("expected selected path error");
        };
        assert_eq!(error, FormatError::InvalidBTreeV2Signature);
    } else {
        assert_eq!(
            reader
                .dataset(&format!("g/d{index}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            [index]
        );
    }
    if invalid_pointer {
        return;
    }
    let Error::Format(error) = reader.group("g").unwrap().datasets().unwrap_err() else {
        panic!("expected corrupt node error");
    };
    assert_eq!(error, FormatError::InvalidBTreeV2Signature);
}

#[rstest]
#[case::wrong_type(
    BTREE_HEADER_TYPE_OFFSET,
    BTREE_V2_ATTRIBUTE_NAME,
    FormatError::InvalidBTreeNodeType(BTREE_V2_ATTRIBUTE_NAME)
)]
#[case::short_record(
    BTREE_HEADER_RECORD_SIZE_OFFSET,
    4,
    FormatError::InvalidBTreeV2Signature
)]
fn indexed_lookup_rejects_malformed_name_indexes(
    #[case] offset: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad-index.h5");
    write_dense_group(&path, 16);
    let mut bytes = std::fs::read(&path).unwrap();
    let (at, _) = dense_index(&bytes);
    bytes[at + offset] = value;
    let header_len =
        hdf5_pure_format::__private::btree_v2_header_size(OffsetWidth::Eight, LengthWidth::Eight);
    test_util::checksum::restamp(&mut bytes, at, header_len);
    std::fs::write(&path, bytes).unwrap();
    let error = open_dense_reader(&path, streaming)
        .dataset("g/d0")
        .unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected malformed index error");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::header(true)]
#[case::node(false)]
fn indexed_lookup_reports_metadata_checksums(
    #[case] header_checksum: bool,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checksum.h5");
    write_dense_group(&path, 16);
    let mut bytes = std::fs::read(&path).unwrap();
    let (at, header) = dense_index(&bytes);
    let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
    let (start, len) = if header_checksum {
        (
            at,
            hdf5_pure_format::__private::btree_v2_header_size(
                OffsetWidth::Eight,
                LengthWidth::Eight,
            ),
        )
    } else {
        let root = layout.root().unwrap();
        (
            root.address().get() as usize,
            layout.encoded_len(root).unwrap(),
        )
    };
    let end = start + len - INDEX_CHECKSUM_LEN;
    let computed = hdf5_pure_format::__private::jenkins_lookup3(&bytes[start..end]);
    bytes[end] ^= 1;
    let expected = u32::from_le_bytes(bytes[end..end + INDEX_CHECKSUM_LEN].try_into().unwrap());
    std::fs::write(&path, bytes).unwrap();
    let Error::Format(error) = open_dense_reader(&path, streaming)
        .dataset("g/d0")
        .unwrap_err()
    else {
        panic!("expected checksum error");
    };
    assert_eq!(error, FormatError::ChecksumMismatch { expected, computed });
}

#[rstest]
#[case::matched("g/d0", true)]
#[case::unrelated("g/d1", false)]
fn indexed_lookup_validates_prefixes_when_the_name_hash_matches(
    #[case] name: &str,
    #[case] malformed_match: bool,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prefix.h5");
    write_dense_group(&path, 16);
    let mut bytes = std::fs::read(&path).unwrap();
    let at = test_util::bytes::sole_signature(&bytes, &[1, 0, 2, b'd', b'0']);
    bytes[at] = 2;
    repair_direct_heap_checksum(&mut bytes);
    std::fs::write(&path, bytes).unwrap();
    let reader = open_dense_reader(&path, streaming);
    if malformed_match {
        let Error::Format(error) = reader.dataset(name).unwrap_err() else {
            panic!("expected bad prefix");
        };
        assert_eq!(error, FormatError::InvalidLinkVersion(2));
    } else {
        assert_eq!(reader.dataset(name).unwrap().read_i32().unwrap(), [1]);
    }
    let Error::Format(error) = reader.group("g").unwrap().datasets().unwrap_err() else {
        panic!("expected bad prefix in enumeration");
    };
    assert_eq!(error, FormatError::InvalidLinkVersion(2));
}

#[rstest]
#[case::invalid_utf8(true)]
#[case::literal_replacement(false)]
fn dense_lookup_preserves_replacement_character_names(
    #[case] invalid_utf8: bool,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lossy.h5");
    write_dense_group(&path, 16);
    let wanted = "\u{fffd}0";
    if invalid_utf8 {
        let mut bytes = std::fs::read(&path).unwrap();
        let name_at = test_util::bytes::sole_signature(&bytes, b"d0");
        bytes[name_at] = 0xff;
        let (_, header) = dense_index(&bytes);
        assert_eq!(header.depth, 0);
        let start = header.root_node_address.get() as usize + test_util::btree_v2::NODE_PREFIX;
        let end = start + usize::from(header.num_records_in_root) * usize::from(header.record_size);
        let old_hash = hdf5_pure_format::__private::jenkins_lookup3(b"d0");
        let hash = hdf5_pure_format::__private::jenkins_lookup3(&[0xff, b'0']);
        let mut records: Vec<_> = bytes[start..end]
            .chunks_exact(usize::from(header.record_size))
            .map(<[u8]>::to_vec)
            .collect();
        let record = records
            .iter_mut()
            .find(|record| record[..LINK_NAME_HASH_LEN] == old_hash.to_le_bytes())
            .unwrap();
        record[..LINK_NAME_HASH_LEN].copy_from_slice(&hash.to_le_bytes());
        records.sort_by_key(|record| {
            u32::from_le_bytes(record[..LINK_NAME_HASH_LEN].try_into().unwrap())
        });
        bytes[start..end].copy_from_slice(&records.concat());
        let node_at = start - test_util::btree_v2::NODE_PREFIX;
        test_util::checksum::restamp(&mut bytes, node_at, end + INDEX_CHECKSUM_LEN - node_at);
        repair_direct_heap_checksum(&mut bytes);
        std::fs::write(&path, bytes).unwrap();
    } else {
        let reference = hdf5::File::open_rw(&path).unwrap();
        reference
            .group("g")
            .unwrap()
            .link_hard("d0", wanted)
            .unwrap();
        reference.close().unwrap();
    }
    let reference = hdf5::File::open(&path).unwrap();
    assert!(
        reference
            .group("g")
            .unwrap()
            .member_names()
            .unwrap()
            .iter()
            .any(|name| name == wanted)
    );
    let reader = open_dense_reader(&path, streaming);
    let names = reader.group("g").unwrap().datasets().unwrap();
    assert!(names.iter().any(|name| name == wanted));
    assert_eq!(
        reader
            .dataset(&format!("g/{wanted}"))
            .unwrap()
            .read_i32()
            .unwrap(),
        [0]
    );
}

fn repair_direct_heap_checksum(bytes: &mut [u8]) {
    let at = test_util::fractal_heap::header_offsets(bytes)[0];
    let image = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap();
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(at as u64),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    let header = locator.parse(image).unwrap().parse(image).unwrap();
    let heap = FractalHeapBlockLayout::new(
        &header,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        StoredAddress::new(at as u64),
    )
    .unwrap();
    let Some(FractalHeapBlock::Direct(block)) = heap.root().unwrap() else {
        panic!("expected a direct heap root");
    };
    let reference = block
        .at(PhysicalAddress::new(block.address().get()))
        .unwrap();
    reference.parse_prefix(image).unwrap();
    let start = reference.range().start().get() as usize;
    let end = reference.range().end().get() as usize;
    let checksum_at = reference.prefix_range().unwrap().end().get() as usize;
    bytes[checksum_at..checksum_at + DIRECT_BLOCK_CHECKSUM_LEN].fill(0);
    let checksum = hdf5_pure_format::__private::jenkins_lookup3(&bytes[start..end]);
    bytes[checksum_at..checksum_at + DIRECT_BLOCK_CHECKSUM_LEN]
        .copy_from_slice(&checksum.to_le_bytes());
}

fn dense_index(bytes: &[u8]) -> (usize, BTreeV2Header) {
    let at = test_util::bytes::sole_signature(bytes, test_util::btree_v2::SIGNATURE);
    (at, BTreeV2Header::parse(bytes, at, 8, 8).unwrap())
}

#[derive(Clone, Copy)]
enum DenseLink {
    External,
    Hard,
    Soft,
}

struct DenseNodeCountingSource {
    bytes: Vec<u8>,
    node_reads: Arc<AtomicUsize>,
}

impl Source for DenseNodeCountingSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        let start = usize::try_from(offset).unwrap();
        let end = start + buffer.len();
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(FormatError::UnexpectedSourceEof {
                expected: end as u64,
                available: self.len(),
            })?;
        if bytes.starts_with(test_util::btree_v2::LEAF_SIGNATURE)
            || bytes.starts_with(test_util::btree_v2::INTERNAL_SIGNATURE)
        {
            self.node_reads.fetch_add(1, Ordering::Relaxed);
        }
        buffer.copy_from_slice(bytes);
        Ok(())
    }
}

fn open_dense_reader(path: &Path, streaming: bool) -> File {
    if streaming {
        File::open_streaming(path)
    } else {
        File::open(path)
    }
    .unwrap()
}

#[rstest]
#[case::direct(16, 8, 0, false, 0, false)]
#[case::indirect(150, 8, 512, false, 1, false)]
#[case::nested(800, 1024, 0, false, 2, false)]
#[case::attributes(32, 8, 512, true, 0, false)]
#[case::huge_only(16, 70_000, 0, false, 0, true)]
fn caller_supplied_regions_parse_c_written_heap_blocks(
    #[case] count: usize,
    #[case] name_len: usize,
    #[case] userblock: u64,
    #[case] attributes: bool,
    #[case] minimum_indirect: usize,
    #[case] huge_only: bool,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("blocks.h5");
    let file = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_latest())
        .with_fcpl(|properties| properties.userblock(userblock))
        .create(&path)
        .unwrap();
    let group = file.create_group("g").unwrap();
    for index in 0..count {
        let prefix = format!("d{index}_");
        let name = format!(
            "{prefix}{}",
            "x".repeat(name_len.saturating_sub(prefix.len()))
        );
        if attributes {
            group
                .new_attr::<i32>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&i32::try_from(index).unwrap())
                .unwrap();
        } else {
            group
                .new_dataset::<i32>()
                .shape(())
                .create(name.as_str())
                .unwrap()
                .write_scalar(&i32::try_from(index).unwrap())
                .unwrap();
        }
    }
    drop(group);
    file.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let image = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        &bytes,
    )
    .unwrap();
    let positions = test_util::fractal_heap::header_offsets(&bytes);
    assert_eq!(positions.len(), 1);
    let header_at = positions[0] as u64;
    let base = BaseAddress::new(userblock);
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(header_at),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    let header_ref = locator
        .parse(image.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap();
    let header = header_ref
        .parse(image.subregion(header_ref.parse_range()).unwrap())
        .unwrap();
    let heap = FractalHeapBlockLayout::new(
        &header,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        base.relative(header_at).unwrap(),
    )
    .unwrap();
    let mut pending = heap.root().unwrap().into_iter().collect::<Vec<_>>();
    let mut direct = Vec::new();
    let mut indirect = Vec::new();
    let mut visited = BTreeSet::new();
    while let Some(block) = pending.pop() {
        match block {
            FractalHeapBlock::Direct(block) => {
                assert!(visited.insert(block.address()), "duplicate managed block");
                let reference = block
                    .at(PhysicalAddress::new(
                        base.absolute(block.address()).unwrap(),
                    ))
                    .unwrap();
                let region = image.subregion(reference.parse_range()).unwrap();
                let view = reference.parse(region).unwrap();
                assert_eq!(view.bytes().len() as u64, block.decoded_size());
                assert!(!view.data().is_empty());
                reference
                    .parse_prefix(image.subregion(reference.prefix_range().unwrap()).unwrap())
                    .unwrap();
                direct.push((block.address(), reference.range().length()));
            }
            FractalHeapBlock::Indirect(block) => {
                assert!(visited.insert(block.address()), "duplicate managed block");
                let reference = block
                    .at(PhysicalAddress::new(
                        base.absolute(block.address()).unwrap(),
                    ))
                    .unwrap();
                let view = reference
                    .parse(image.subregion(reference.parse_range()).unwrap())
                    .unwrap();
                pending.extend(view.children().collect::<Result<Vec<_>, _>>().unwrap());
                indirect.push(block.address());
            }
        }
    }
    assert_eq!(
        direct.iter().map(|(_, length)| length).sum::<u64>(),
        header.allocated_managed_space
    );
    assert!(
        indirect.len() >= minimum_indirect,
        "indirect count: {}",
        indirect.len()
    );
    if huge_only {
        assert_eq!(heap.root().unwrap(), None);
        assert_eq!(header.huge_objects_count, count as u64);
        assert_eq!(direct, Vec::<(StoredAddress, u64)>::new());
    } else {
        assert!(!direct.is_empty());
        assert_eq!(header.managed_objects_count, count as u64);
    }
    let reader = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let group = reader.group("g").unwrap();
    if attributes {
        assert_eq!(group.attrs().unwrap().len(), count);
    } else {
        assert_eq!(group.datasets().unwrap().len(), count);
        let index = count - 1;
        let prefix = format!("d{index}_");
        let name = format!(
            "{prefix}{}",
            "x".repeat(name_len.saturating_sub(prefix.len()))
        );
        assert_eq!(
            group.dataset(&name).unwrap().read_i32().unwrap(),
            vec![i32::try_from(index).unwrap()]
        );
    }
    let reopened = hdf5::File::open(&path).unwrap();
    let group = reopened.group("g").unwrap();
    assert_eq!(
        if attributes {
            group.attr_names().unwrap().len()
        } else {
            group.member_names().unwrap().len()
        },
        count
    );
}

#[test]
fn edits_reject_corrupt_checksummed_direct_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt_direct.h5");
    write_dense_group(&path, 16);
    let mut bytes = std::fs::read(&path).unwrap();
    let positions = test_util::fractal_heap::header_offsets(&bytes);
    assert_eq!(positions.len(), 1);
    let at = positions[0];
    let locator = FractalHeapHeaderLocator::new(
        PhysicalAddress::new(at as u64),
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    let image = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        &bytes,
    )
    .unwrap();
    let reference = locator.parse(image).unwrap();
    let header = reference.parse(image).unwrap();
    let heap = FractalHeapBlockLayout::new(
        &header,
        OffsetWidth::Eight,
        LengthWidth::Eight,
        BaseAddress::ZERO.relative(at as u64).unwrap(),
    )
    .unwrap();
    let Some(FractalHeapBlock::Direct(block)) = heap.root().unwrap() else {
        panic!("expected direct root")
    };
    let reference = block
        .at(PhysicalAddress::new(block.address().get()))
        .unwrap();
    let view = reference.parse_checked(image).unwrap();
    let block_at = reference.range().start().get() as usize;
    let block_end = reference.range().end().get() as usize;
    let checksum_at = view.bytes().len() - view.data().len() - DIRECT_BLOCK_CHECKSUM_LEN;
    let stored = u32::from_le_bytes(
        view.bytes()[checksum_at..checksum_at + DIRECT_BLOCK_CHECKSUM_LEN]
            .try_into()
            .unwrap(),
    );
    bytes[block_end - 1] ^= 1;
    let mut zeroed = bytes[block_at..block_end].to_vec();
    zeroed[checksum_at..checksum_at + DIRECT_BLOCK_CHECKSUM_LEN].fill(0);
    let computed = hdf5_pure_format::__private::jenkins_lookup3(&zeroed);
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("g/d0")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![0]
    );
    assert_eq!(
        File::open_streaming(&path)
            .unwrap()
            .dataset("g/d0")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![0]
    );
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("g/d0").unwrap();
        let error = session.commit().unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected format error, got {error:?}")
        };
        assert_eq!(
            error,
            FormatError::ChecksumMismatch {
                expected: stored,
                computed
            }
        );
    }
    assert_eq!(
        &std::fs::read(&path).unwrap()[block_at..block_end],
        &bytes[block_at..block_end]
    );
}

// "Fractal Heap" in the format specification, version 4.0, defines the four-byte
// direct-block checksum.
const DIRECT_BLOCK_CHECKSUM_LEN: usize = 4;

// "Version 2 B-trees" in the format specification, version 4.0, defines these header offsets,
// the four-byte name hash of type 5 records, and the trailing checksum width.
const BTREE_HEADER_TYPE_OFFSET: usize = 5;
const BTREE_HEADER_RECORD_SIZE_OFFSET: usize = 10;
const LINK_NAME_HASH_LEN: usize = 4;
const INDEX_CHECKSUM_LEN: usize = 4;
