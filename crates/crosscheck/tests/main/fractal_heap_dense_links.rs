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

use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, PhysicalAddress};
use hdf5_pure_core::{BaseAddress, StoredAddress};
use hdf5_pure_format::__private::{
    BTREE_V2_LINK_NAME, BTreeV2Header, BTreeV2Layout, FractalHeapBlock, FractalHeapBlockLayout,
    FractalHeapHeaderLocator, LengthWidth, OffsetWidth,
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
