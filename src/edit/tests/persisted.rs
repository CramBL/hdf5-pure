use std::path::Path;

use hdf5_pure_space::__private::Extent;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::file_space::persisted::{
    EOF, HEADER, HEADER_LEN, ManagerSlot, PersistedImage, PersistedStrategy, SECTION, STRIDE,
};

use crate::Error;
use crate::FormatError;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::WriteEngine;
use crate::file_lock::FileLocking;
use crate::source::MetadataCacheConfig;

#[test]
fn strict_persisted_later_failure_precedes_allocator_construction() {
    let path = temp::temp_path("later-manager.h5");
    let mut fixture = PersistedImage::new(&path, PersistedStrategy::Page, 16);
    fixture.manager(ManagerSlot::Metadata, &[(SECTION, 32)]);
    fixture.manager(ManagerSlot::Unknown, &[(EOF - 16, 32)]);
    let bytes = fixture.write(&path);
    let built = std::cell::Cell::new(false);
    let Err(error) = WriteEngine::open_imaged_with_policy(
        &path,
        Some(FileLocking::Enabled),
        PersistedFreeSpacePolicy::Strict,
        |handle, _| {
            built.set(true);
            Ok(Box::new(WriteEngine::read_mirror(handle)?))
        },
    ) else {
        panic!("expected invalid section geometry")
    };
    let Error::PersistedFreeSpace {
        manager_slot,
        header_address,
        section_list_address,
        section,
        source,
    } = error
    else {
        panic!("{error:?}")
    };
    assert_eq!(
        (manager_slot, header_address, section_list_address, section),
        (
            7,
            StoredAddress::new(HEADER + 7 * STRIDE),
            Some(StoredAddress::new(HEADER + 7 * STRIDE + HEADER_LEN)),
            Some((StoredAddress::new(EOF - 16), 32))
        )
    );
    let Error::Format(FormatError::InvalidFreeSpaceManager(reason)) = *source else {
        panic!("{source:?}")
    };
    assert_eq!(
        reason,
        format!(
            "the free section at {}, 32 bytes long, exceeds the file allocation",
            EOF - 16
        )
    );
    assert!(!built.get());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[rstest]
#[case::same_manager(false)]
#[case::different_managers(true)]
fn strict_persisted_adjacent_sections_are_admitted(#[case] separate: bool) {
    let path = temp::temp_path("adjacent.h5");
    let mut fixture = PersistedImage::new(&path, PersistedStrategy::FsmAggr, 128);
    if separate {
        fixture.manager(ManagerSlot::Metadata, &[(SECTION, 32)]);
        fixture.manager(ManagerSlot::Unknown, &[(SECTION + 32, 32)]);
    } else {
        fixture.manager(ManagerSlot::Metadata, &[(SECTION, 32), (SECTION + 32, 32)]);
    }
    fixture.manager(ManagerSlot::LocalHeap, &[]);
    let bytes = fixture.write(&path);
    let engine = strict_engine(&path);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [Extent::new(SECTION, 64).unwrap()]
    );
    assert_eq!(
        engine.persist.as_ref().unwrap().old_blocks.len(),
        if separate { 7 } else { 5 }
    );
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[rstest]
#[case::metadata(ManagerSlot::Metadata, SECTION + 16, 8, Admission::Metadata)]
#[case::raw(ManagerSlot::Raw, SECTION + 16, 8, Admission::Raw)]
#[case::generic_page(ManagerSlot::Generic, SECTION, 4096, Admission::Raw)]
#[case::generic_fragment(ManagerSlot::Generic, SECTION + 16, 8, Admission::Unclassified)]
#[case::unknown(ManagerSlot::Unknown, SECTION + 16, 8, Admission::Unclassified)]
fn strict_persisted_page_roles_preserve_admission(
    #[case] slot: ManagerSlot,
    #[case] address: u64,
    #[case] length: u64,
    #[case] admission: Admission,
) {
    let path = temp::temp_path("page-role.h5");
    let mut fixture = PersistedImage::new(&path, PersistedStrategy::Page, 128);
    fixture.manager(slot, &[(address, length)]);
    fixture.manager(ManagerSlot::LocalHeap, &[]);
    let bytes = fixture.write(&path);
    let engine = strict_engine(&path);
    let pages = engine.space.paged_sections().unwrap();
    let extent = [Extent::new(address, length).unwrap()];
    assert_eq!(
        pages.metadata(),
        if matches!(admission, Admission::Metadata) {
            &extent[..]
        } else {
            &[]
        }
    );
    assert_eq!(
        pages.raw(),
        if matches!(admission, Admission::Raw) {
            &extent[..]
        } else {
            &[]
        }
    );
    assert_eq!(
        pages.unclassified(),
        if matches!(admission, Admission::Unclassified) {
            &extent[..]
        } else {
            &[]
        }
    );
    assert_eq!(engine.persist.as_ref().unwrap().old_blocks.len(), 5);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

fn strict_engine(path: &Path) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Admission {
    Metadata,
    Raw,
    Unclassified,
}
