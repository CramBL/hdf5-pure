#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::fractal_heap;
use crate::group_v2;
use crate::link_info::LinkInfoMessage;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPath;
use crate::source::BytesSource;
use crate::source::PhysicalBytes;
use crate::source::PhysicalSource;
use crate::superblock::Superblock;
use crate::width::FormatWidths;

/// Asserts complete link names and one huge-index parse for each dense-link walk.
///
/// # Panics
///
/// Panics if metadata cannot be read, the names differ, or either walk's parse
/// count differs from one.
pub(super) fn assert_dense_walk(bytes: Vec<u8>) {
    const COUNT: usize = 12;
    let (link_info, fh_addr, widths) = dense_link_info(&bytes);

    fractal_heap::reset_huge_index_decodes();
    let buffered = group_v2::resolve_dense_entries(
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
    .unwrap();
    assert_eq!(buffered.len(), COUNT, "the walk must still read every link");
    assert_eq!(
        fractal_heap::huge_index_decodes(),
        1,
        "the buffered walk parsed the huge-object index per link rather than per walk"
    );

    fractal_heap::reset_huge_index_decodes();
    let source = BytesSource::new(bytes);
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
    assert_eq!(
        fractal_heap::huge_index_decodes(),
        1,
        "the streaming walk parsed the huge-object index per link rather than per walk"
    );

    let mut buffered: Vec<_> = buffered.iter().map(|entry| entry.name.as_str()).collect();
    let mut streamed: Vec<_> = streamed.iter().map(|entry| entry.name.as_str()).collect();
    let mut expected = test_util_hdf5::huge_links::link_names();
    buffered.sort();
    streamed.sort();
    expected.sort();
    assert_eq!(buffered, expected);
    assert_eq!(streamed, expected);
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
        .fractal_heap_address
        .expect("this many long-named links are stored densely");
    (link_info, fh_addr, widths)
}

fn test_widths(superblock: &Superblock) -> FormatWidths {
    FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    }
}
