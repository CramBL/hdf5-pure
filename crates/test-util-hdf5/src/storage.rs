//! Serialized storage extents and free-space assertions for controlled fixtures.
//!
//! Offsets are absolute byte positions. Each byte slice contains the image of the file passed
//! alongside it.

use std::collections::BTreeSet;

use hdf5_pure::DecodedStoredAddress;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::MessageType;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_format::__private::LinkMessage;
use hdf5_pure_format::__private::LinkTarget;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use test_util::checksum;

/// Returns allocations for the superblock, root and extension headers, and free-space managers.
///
/// Object headers must use version 2. Child objects are inspected separately with
/// [`header_extents`].
///
/// # Panics
///
/// Panics if an address is outside `bytes`, a structure cannot be parsed, or an allocation size
/// cannot be represented.
pub fn file_metadata_extents(file: &File, bytes: &[u8]) -> Vec<(u64, u64)> {
    let sb = file.superblock();
    let mut extents = manager_extents(file, bytes);
    extents.push((
        sb.base_address.get(),
        u64::try_from(format::serialize_superblock(sb).unwrap().len()).unwrap(),
    ));
    extents.extend(header_extents(
        file,
        bytes,
        sb.base_address.absolute(sb.root_group_address).unwrap(),
    ));
    if let Some(extension) = sb.superblock_extension_address
        && !extension.is_undefined(sb.offset_size)
    {
        extents.extend(header_extents(
            file,
            bytes,
            sb.base_address.absolute(extension).unwrap(),
        ));
    }
    extents
}

/// Returns each persisted manager's header and complete section-list allocation.
///
/// # Panics
///
/// Panics if a manager header cannot be read from `bytes` or an address cannot be represented.
pub fn manager_extents(file: &File, bytes: &[u8]) -> Vec<(u64, u64)> {
    let Some(info) = file.file_space_info().unwrap() else {
        return Vec::new();
    };
    let superblock = file.superblock();
    let widths = format::FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    info.manager_addrs
        .iter()
        .filter_map(|address| match address {
            DecodedStoredAddress::Defined(address) => Some(*address),
            DecodedStoredAddress::Undefined => None,
        })
        .flat_map(|address| {
            let at = superblock.base_address.absolute(address).unwrap();
            let header =
                FreeSpaceManagerHeader::parse(widths, &bytes[usize::try_from(at).unwrap()..])
                    .unwrap();
            vec![
                (at, format::free_space_manager_header_len(widths)),
                (
                    superblock
                        .base_address
                        .absolute(header.section_list_addr())
                        .unwrap(),
                    header.section_list_allocated(),
                ),
            ]
        })
        .collect()
}

/// Returns the sorted allocations of a version 2 object header and its continuation blocks.
///
/// The first allocation includes the checksum following chunk 0, as defined in "Version 2 Data
/// Object Header Prefix" of the [format specification, version 4.0][spec].
///
/// # Panics
///
/// Panics if a header or continuation block cannot be read, a continuation address is repeated,
/// or an address or size cannot be represented.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub fn header_extents(file: &File, bytes: &[u8], address: u64) -> Vec<(u64, u64)> {
    let superblock = file.superblock();
    let widths = format::FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let at = usize::try_from(address).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[at..]).unwrap();
    let chunk0_end = at + prefix.len + usize::try_from(prefix.chunk0_size).unwrap();
    let mut extents = vec![(
        address,
        u64::try_from(chunk0_end + checksum::CHECKSUM - at).unwrap(),
    )];
    let mut regions = std::iter::once(at + prefix.len..chunk0_end).collect::<Vec<_>>();
    let mut seen = BTreeSet::from([address]);
    while let Some(region) = regions.pop() {
        let records = &bytes[region];
        let mut pos = 0;
        while let Some(record) = prefix.prefix.layout.next_message(records, pos).unwrap() {
            pos = record.body_range.end;
            if record.msg_type != MessageType::OBJECT_HEADER_CONTINUATION {
                continue;
            }
            let continuation =
                ObjectHeaderContinuation::parse_with_widths(record.body, widths).unwrap();
            let at = superblock
                .base_address
                .absolute(continuation.address())
                .unwrap();
            assert!(seen.insert(at), "header continuation cycle at {at}");
            let len = continuation.length();
            let start = usize::try_from(at).unwrap();
            let end = start + usize::try_from(len).unwrap();
            let messages = format::continuation_block_messages(&bytes[start..end]).unwrap();
            regions.push(start + messages.start..start + messages.end);
            extents.push((at, len));
        }
    }
    extents.sort_unstable();
    extents
}

/// Returns the object header address of a compact hard link in the root group.
///
/// # Panics
///
/// Panics if the root header or a link cannot be parsed, an address cannot be represented, or
/// `link_name` does not identify a compact hard link.
pub fn root_object_address(file: &File, bytes: &[u8], link_name: &str) -> u64 {
    let sb = file.superblock();
    let root = ObjectHeader::parse_with_base_and_widths(
        bytes,
        AccessMode::ReadOnly,
        usize::try_from(sb.base_address.absolute(sb.root_group_address).unwrap()).unwrap(),
        format::FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
        sb.base_address,
    )
    .unwrap();
    root.messages
        .iter()
        .filter(|message| message.msg_type == MessageType::LINK)
        .map(|message| LinkMessage::parse_with_offset_width(&message.data, sb.offset_size).unwrap())
        .find_map(|link| match link.link_target {
            LinkTarget::Hard {
                object_header_address,
            } if link.name == link_name => {
                Some(sb.base_address.absolute(object_header_address).unwrap())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing compact root hard link {link_name:?}"))
}

/// Asserts that allocated extents are disjoint and outside the userblock and reusable free space.
///
/// Uses session accounting for writable files and persisted free space for read-only files.
///
/// # Panics
///
/// Panics if free space cannot be read, an extent is empty or exceeds the physical file length,
/// an extent overlaps the userblock or another extent, or an allocated byte is reusable.
pub fn assert_unavailable(file: &File, extents: &[(u64, u64)]) {
    let free = match file.space_accounting() {
        Ok(accounting) => accounting.reusable_free_space,
        Err(Error::ReadOnly) => file
            .persisted_free_space()
            .unwrap()
            .into_iter()
            .map(|(at, len)| (at + file.superblock().base_address.get(), len))
            .collect(),
        Err(error) => panic!("free-space inspection failed: {error:?}"),
    };
    let base = file.superblock().base_address.get();
    let mut sorted = extents.to_vec();
    sorted.sort_unstable();
    let mut end = base;
    for (at, len) in sorted {
        assert!(len > 0);
        assert!(
            at >= end,
            "allocated ranges overlap or enter the userblock at {at}, previous end {end}"
        );
        end = at.checked_add(len).unwrap();
        assert!(end <= file.file_size());
        assert!(
            free.iter()
                .all(|&(free, size)| end <= free || free + size <= at),
            "live range {at}..{end} overlaps free space {:?}",
            free
        );
    }
}

/// Asserts that persisted free-space managers and free sections are absent.
///
/// # Panics
///
/// Panics if a manager address is defined, persisted free space cannot be read, or a persisted
/// free section is present.
pub fn assert_no_managers(file: &File) {
    if let Some(info) = file.file_space_info().unwrap() {
        assert!(
            info.manager_addrs
                .iter()
                .all(|address| *address == DecodedStoredAddress::Undefined)
        );
    }
    assert_eq!(file.persisted_free_space().unwrap(), []);
}
