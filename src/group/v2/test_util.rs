#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::test_source::ObservedSource;
use crate::group::v2 as group_v2;
use crate::link_info::LinkInfoMessage;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPath;
use crate::source::PhysicalBytes;
use crate::source::PhysicalSource;
use crate::superblock::Superblock;
use crate::width::FormatWidths;

/// Checks complete link names, source requests, and resident allocations.
///
/// # Panics
///
/// Panics if metadata cannot be read, the names or addresses differ, the
/// source walk requests the tracker more than once, or the allocation bound
/// fails. Each caller supplies a distinct allocation region name.
pub(super) fn assert_dense_walk(bytes: Vec<u8>, region: &str) {
    const COUNT: usize = 12;
    let (link_info, fh_addr, widths) = dense_link_info(&bytes);

    let heap = FractalHeapHeader::parse_bytes(
        &bytes[fh_addr.get().to_usize().unwrap()..],
        widths.offsets,
        widths.lengths,
    )
    .unwrap();
    assert_eq!(heap.huge_objects_count, COUNT as u64);

    let walk_resident = || {
        group_v2::resolve_dense_entries(
            PhysicalBytes::for_test(
                &bytes,
                AddressContext::from_effective_base(BaseAddress::new(0)),
            )
            .stored()
            .unwrap(),
            AccessMode::ReadOnly,
            &link_info,
            fh_addr,
            widths,
        )
        .unwrap()
    };
    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    let buffered = {
        let (entries, work) = ::test_util::allocation::measure(region, walk_resident);
        // Cumulative allocations reveal repeated index construction even
        // when each temporary index is freed before the next record.
        assert!((COUNT as u64..=100).contains(&work.blocks), "{work}");
        entries
    };
    #[cfg(not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))))]
    let buffered = {
        let _ = region;
        walk_resident()
    };
    assert_eq!(buffered.len(), COUNT, "the walk must still read every link");
    let source = ObservedSource::new(&bytes);
    let streamed = group_v2::resolve_dense_entries_from_source(
        &PhysicalSource::for_test(
            &source,
            AddressContext::from_effective_base(BaseAddress::new(0)),
        )
        .stored(),
        AccessMode::ReadOnly,
        &link_info,
        fh_addr,
        widths,
    )
    .unwrap();
    assert_eq!(streamed.len(), COUNT);
    assert_eq!(source.reads_at(heap.btree_huge_objects_address.get()), 1);

    let mut buffered: Vec<_> = buffered
        .iter()
        .map(|entry| (entry.name.as_str(), entry.object_header_address))
        .collect();
    let mut streamed: Vec<_> = streamed
        .iter()
        .map(|entry| (entry.name.as_str(), entry.object_header_address))
        .collect();
    let mut expected = test_util_hdf5::huge_links::link_names();
    buffered.sort_unstable_by_key(|entry| entry.0);
    streamed.sort_unstable_by_key(|entry| entry.0);
    expected.sort();
    assert_eq!(buffered, streamed);
    let names: Vec<_> = buffered.into_iter().map(|(name, _)| name).collect();
    assert_eq!(names, expected);
}

/// Returns group `g`'s dense-link info, heap address, and file widths.
fn dense_link_info(bytes: &[u8]) -> (LinkInfoMessage, StoredAddress, FormatWidths) {
    let superblock = crate::superblock::read(bytes).unwrap().superblock;
    let widths = test_widths(&superblock);
    let group_addr = group_v2::resolve_path_any(
        PhysicalBytes::for_test(
            bytes,
            AddressContext::from_effective_base(superblock.base_address),
        ),
        AccessMode::ReadOnly,
        &superblock,
        widths,
        &ObjectPath::parse("g"),
    )
    .unwrap();
    let header = ObjectHeader::parse_with_widths(
        bytes,
        AccessMode::ReadOnly,
        group_addr.to_usize().unwrap(),
        widths,
    )
    .unwrap();
    let link_info = group_v2::find_link_info(&header, widths.offsets).unwrap();
    let fh_addr = link_info
        .fractal_heap_address()
        .expect("this many long-named links are stored densely");
    (link_info, fh_addr, widths)
}

fn test_widths(superblock: &Superblock) -> FormatWidths {
    FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    }
}
