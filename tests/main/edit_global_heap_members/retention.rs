use core::num::NonZeroUsize;
use std::fs;
use std::ops::Range;

use hdf5_pure::{
    AddressRange, AttrValue, EditStorageConflict, Error, File, FileAccessProperties, FileLocking,
    FormatError, PhysicalAddress,
};
use rstest::rstest;
use test_util::dense_attributes;
use test_util::global_heap::MIN_COLLECTION_SIZE;
use test_util_hdf5::__private::{
    EnvelopeDamage, EnvelopeSource, ResidualGroupSource, RetainedContinuationSource,
};
use test_util_hdf5::global_heap_fixture::members::ParentImage;

#[rstest]
#[case::numeric_child(ResidualGroupSource::Keep, true)]
#[case::removed_child(ResidualGroupSource::Gone, false)]
#[case::unknown_bytes(ResidualGroupSource::Unknown, false)]
#[case::unsupported_header(ResidualGroupSource::Unsupported, false)]
#[case::malformed_header(ResidualGroupSource::Malformed, false)]
#[case::legacy_storage(ResidualGroupSource::Legacy, false)]
#[case::unresolved_link(ResidualGroupSource::Unresolved, false)]
#[case::residual_byte_boundary(
    ResidualGroupSource::Pad(NonZeroUsize::new(RESIDUAL_BYTE_LIMIT).unwrap()),
    true
)]
#[case::residual_byte_limit(
    ResidualGroupSource::Pad(NonZeroUsize::new(RESIDUAL_BYTE_LIMIT + 1).unwrap()),
    false
)]
#[case::residual_group_boundary(
    ResidualGroupSource::Repeat(NonZeroUsize::new(RESIDUAL_GROUP_LIMIT).unwrap()),
    true
)]
#[case::residual_group_limit(
    ResidualGroupSource::Repeat(NonZeroUsize::new(RESIDUAL_GROUP_LIMIT + 1).unwrap()),
    false
)]
fn unlinked_group_references_control_parent_retirement(
    #[case] source: ResidualGroupSource,
    #[case] retired: bool,
    #[values(false, true)] reconstruct: bool,
) {
    let path = test_util::temp::temp_path("public-residual-groups.h5");
    let (parent, residual, keeper) = source.write(&path);
    let original = fs::read(&path).unwrap();
    let remaining = &original[residual as usize..];
    match source {
        ResidualGroupSource::Pad(length) => {
            assert_eq!(remaining.len(), length.get());
            assert_eq!(&remaining[..4], b"OHDR");
            assert!(
                remaining[RESIDUAL_GROUP_LENGTH..]
                    .iter()
                    .all(|&byte| byte == 0)
            );
        }
        ResidualGroupSource::Repeat(count) => {
            let (chunks, tail) = remaining.as_chunks::<RESIDUAL_GROUP_LENGTH>();
            assert_eq!(chunks.len(), count.get());
            assert!(tail.is_empty());
            assert!(
                chunks
                    .iter()
                    .all(|chunk| chunk.as_slice() == &remaining[..RESIDUAL_GROUP_LENGTH])
            );
            assert_eq!(&remaining[..4], b"OHDR");
        }
        _ => {}
    }
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["gone", "keep"]);
    assert_eq!(
        file.dataset("gone").unwrap().read_string().unwrap(),
        ["removed"]
    );
    file.root().delete("gone").unwrap();
    if reconstruct {
        file.dataset("keep")
            .unwrap()
            .set_attr("a", AttrValue::I64(9))
            .unwrap();
    }
    file.commit().unwrap();
    let free = file.reusable_free_sections().unwrap();
    assert_eq!(
        free.sections()
            .iter()
            .any(|section| section.address().get() <= parent
                && section.end_address().get() >= parent + MIN_COLLECTION_SIZE as u64),
        retired
    );
    let after = fs::read(&path).unwrap();
    assert_eq!(
        &after[residual as usize..original.len()],
        &original[residual as usize..]
    );
    let extent = keeper.start().get() as usize..keeper.end().get() as usize;
    assert_eq!(&after[extent.clone()], &original[extent]);
    if !retired {
        ParentImage::read(&original, parent).assert_unchanged(&after);
        assert_allocated(&file, parent..parent + MIN_COLLECTION_SIZE as u64);
    }
    assert_allocated(&file, residual..original.len() as u64);
    file.root().create_group("added").unwrap();
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    let after = fs::read(&path).unwrap();
    assert_eq!(&after[residual as usize..original.len()], remaining);
    if !retired {
        ParentImage::read(&original, parent).assert_unchanged(&after);
        assert_allocated(&file, parent..parent + MIN_COLLECTION_SIZE as u64);
    }
    assert_allocated(&file, residual..original.len() as u64);
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["keep"]);
    assert_eq!(file.root().groups().unwrap(), ["added"]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
    if reconstruct {
        assert_eq!(
            file.dataset("keep").unwrap().attrs().unwrap()["a"],
            AttrValue::I64(9)
        );
    }
    file.close().unwrap();
}

#[rstest]
#[case::malformed_records(EnvelopeDamage::Records, true)]
#[case::unsupported_version(EnvelopeDamage::Version, false)]
#[case::unknown_storage(EnvelopeDamage::Unknown, false)]
fn healthy_parent_retirement_preserves_damaged_neighbors(
    #[case] damage: EnvelopeDamage,
    #[case] retired: bool,
) {
    let path = test_util::temp::temp_path("public-damaged-neighbor.h5");
    let source = EnvelopeSource::write(&path, damage);
    let original = fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    let error = file.dataset("bad").unwrap().read_string().unwrap_err();
    let hdf5_pure::Error::Format(original_error) = error else {
        panic!("{error:?}");
    };
    assert_eq!(
        original_error,
        match damage {
            EnvelopeDamage::Version => hdf5_pure::FormatError::InvalidGlobalHeapVersion(2),
            _ => hdf5_pure::FormatError::DuplicateGlobalHeapObjectIndex {
                collection_address: source.damaged,
                index: 7
            },
        }
    );
    file.root().delete("gone").unwrap();
    file.commit().unwrap();
    let after = fs::read(&path).unwrap();
    let start = source.damaged as usize;
    assert_eq!(
        &after[start..start + MIN_COLLECTION_SIZE],
        &original[start..start + MIN_COLLECTION_SIZE]
    );
    assert_eq!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .any(|section| section.address().get() <= source.healthy
                && section.end_address().get() >= source.healthy + MIN_COLLECTION_SIZE as u64),
        retired
    );
    assert_allocated(
        &file,
        source.damaged..source.damaged + MIN_COLLECTION_SIZE as u64,
    );
    if !retired {
        ParentImage::read(&original, source.healthy).assert_unchanged(&after);
        assert_allocated(
            &file,
            source.healthy..source.healthy + MIN_COLLECTION_SIZE as u64,
        );
    }
    if let Some(range) = source.opaque {
        let range = range.start().get() as usize..range.end().get() as usize;
        assert_eq!(&after[range.clone()], &original[range]);
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["bad", "keep"]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
    let error = file.dataset("bad").unwrap().read_string().unwrap_err();
    let hdf5_pure::Error::Format(error) = error else {
        panic!("{error:?}");
    };
    assert_eq!(error, original_error);
    file.close().unwrap();
}

#[rstest]
#[case::overlapping_parents(ParentConflict::Overlap)]
#[case::parent_beyond_source(ParentConflict::BeyondSource)]
fn conflicting_parent_extents_reject_public_deletion_without_changing_storage(
    #[case] conflict: ParentConflict,
) {
    let path = test_util::temp::temp_path("public-conflicting-parent.h5");
    let source = EnvelopeSource::write(
        &path,
        match conflict {
            ParentConflict::BeyondSource => EnvelopeDamage::BeyondSource,
            ParentConflict::Overlap => EnvelopeDamage::Overlap,
        },
    );
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    let original = fs::read(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    let range = |start, length| AddressRange::new(PhysicalAddress::new(start), length).unwrap();
    for _ in 0..2 {
        file.root().delete("gone").unwrap();
        file.root().create_group("added").unwrap();
        assert!(file.has_staged_edits());
        let error = file.commit().unwrap_err();
        match (conflict, error) {
            (ParentConflict::Overlap, Error::EditStorageConflict(actual)) => assert_eq!(
                actual,
                EditStorageConflict::Overlap {
                    first: range(source.damaged, 2 * MIN_COLLECTION_SIZE as u64),
                    second: range(source.healthy, MIN_COLLECTION_SIZE as u64),
                }
            ),
            (
                ParentConflict::BeyondSource,
                Error::Format(FormatError::UnexpectedSourceEof {
                    expected,
                    available,
                }),
            ) => {
                assert_eq!(expected, source.damaged + 16 * MIN_COLLECTION_SIZE as u64);
                assert_eq!(available, original.len() as u64);
            }
            (_, error) => panic!("{error:?}"),
        }
        assert!(!file.has_staged_edits());
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(file.reusable_free_sections().unwrap(), free);
        assert!(file.root().groups().unwrap().is_empty());
        assert_eq!(file.root().datasets().unwrap(), ["bad", "gone", "keep"]);
        assert_eq!(
            file.dataset("gone").unwrap().read_string().unwrap(),
            ["removed"]
        );
        assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
        file.commit().unwrap();
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(file.reusable_free_sections().unwrap(), free);
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert!(file.root().groups().unwrap().is_empty());
    assert_eq!(file.root().datasets().unwrap(), ["bad", "gone", "keep"]);
    assert_eq!(
        file.dataset("gone").unwrap().read_string().unwrap(),
        ["removed"]
    );
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum ParentConflict {
    BeyondSource,
    Overlap,
}

#[test]
fn unrelated_edits_preserve_complete_continuation_allocations() {
    let path = test_util::temp::temp_path("public-retained-continuation.h5");
    let source = RetainedContinuationSource::write(&path);
    let original = fs::read(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    file.root()
        .set_attr("obsolete", AttrValue::VarLenAsciiString("temporary".into()))
        .unwrap();
    file.commit().unwrap();
    file.root().remove_attr("obsolete").unwrap();
    file.root().create_group("added").unwrap();
    file.commit().unwrap();
    let after = fs::read(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    for extent in [source.data, source.child, source.continuation, source.group] {
        let range = extent.start().get() as usize..extent.end().get() as usize;
        assert_eq!(&after[range.clone()], &original[range]);
        assert!(
            free.sections()
                .iter()
                .all(|section| section.end_address() <= extent.start()
                    || section.address() >= extent.end())
        );
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("retained/child").unwrap().read_i32().unwrap(),
        [31]
    );
    file.close().unwrap();
}

#[test]
fn unrelated_edits_preserve_dense_allocations_beyond_used_bytes() {
    let path = test_util::temp::temp_path("public-retained-dense.h5");
    let source = dense_attributes::Fixture::Dataset.build_with_layout();
    assert!(source.storage.direct_used < source.storage.direct.len());
    assert!(source.storage.leaf_used < source.storage.leaf.len());
    fs::write(&path, &source.bytes).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    file.root()
        .set_attr("obsolete", AttrValue::VarLenAsciiString("temporary".into()))
        .unwrap();
    file.commit().unwrap();
    file.root().remove_attr("obsolete").unwrap();
    file.root().create_group("added").unwrap();
    file.commit().unwrap();
    let after = fs::read(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    for extent in [
        source.storage.heap,
        source.storage.direct,
        source.storage.leaf,
        source.storage.index,
        source.data,
        source.dataset,
    ] {
        assert_eq!(&after[extent.clone()], &source.bytes[extent.clone()]);
        assert!(
            free.sections()
                .iter()
                .all(|section| section.end_address().get() <= extent.start as u64
                    || section.address().get() >= extent.end as u64)
        );
    }
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_f64().unwrap(), [1.0, 2.0, 3.0]);
    let expected = dense_attributes::Fixture::Dataset
        .attributes()
        .map(|(name, value)| (name, AttrValue::F64(value)))
        .collect();
    assert_eq!(dataset.attrs().unwrap(), expected);
    drop(dataset);
    file.close().unwrap();
}

#[test]
fn overwriting_session_strings_retires_the_previous_owned_parent() {
    let path = test_util::temp::temp_path("public-session-parent.h5");
    test_util_hdf5::__private::FailureSource::session(&path, false);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    file.dataset("labels")
        .unwrap()
        .write_staged(|builder| {
            builder.with_vlen_strings(&["first"]);
        })
        .unwrap();
    file.commit().unwrap();
    let original = fs::read(&path).unwrap();
    let parent =
        test_util::global_heap::collection_ranges(&original, test_util::widths::Widths::EIGHT)
            .into_iter()
            .find(|range| {
                ParentImage::read(&original, range.start).indices().len() == 1
                    && ParentImage::read_members(&original, range.start)
                        .iter()
                        .any(|(_, _, data)| data == b"first")
            })
            .unwrap();
    file.dataset("labels")
        .unwrap()
        .write_staged(|builder| {
            builder.with_vlen_strings(&["second"]);
        })
        .unwrap();
    file.commit().unwrap();
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .any(|section| section.address().get() <= parent.start
                && section.end_address().get() >= parent.end)
    );
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        ["second"]
    );
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        ["second"]
    );
    assert_eq!(
        file.root().attrs().unwrap()["changed"],
        AttrValue::VarLenAsciiString("before".into())
    );
    file.close().unwrap();
}

fn assert_allocated(file: &File, extent: Range<u64>) {
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| section.end_address().get() <= extent.start
                || section.address().get() >= extent.end)
    );
}

/// The residual-byte boundary exercised by public qualification tests.
const RESIDUAL_BYTE_LIMIT: usize = 1_048_576;
/// The residual-group boundary exercised by public qualification tests.
const RESIDUAL_GROUP_LIMIT: usize = 64;
/// The complete compact-group header emitted by the independent residual fixture.
const RESIDUAL_GROUP_LENGTH: usize = 53;
