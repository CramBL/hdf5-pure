use super::FreeSection;
use super::section_list::{FSSE_SIGNATURE, FSSE_VERSION};
use crate::address::{StoredAddress, StoredAddressExt};
use crate::bytes;
use crate::checksum;
use crate::width::FormatWidths;

/// A free section at `addr` spanning `size` bytes.
pub(super) fn section(addr: u64, size: u64) -> FreeSection {
    FreeSection {
        addr: StoredAddress::new(addr),
        size,
    }
}

pub(super) fn widths(offset_size: u8, length_size: u8) -> FormatWidths {
    FormatWidths::from_sizes(offset_size, length_size).unwrap()
}

/// Returns the section list of `payload` for the header at `fshd_addr`, with its prefix and its
/// checksum.
pub(super) fn section_list(widths: FormatWidths, fshd_addr: u64, payload: &[u8]) -> Vec<u8> {
    let mut list = [&FSSE_SIGNATURE[..], &[FSSE_VERSION]].concat();
    bytes::try_write_offset(&mut list, fshd_addr, widths.offsets).unwrap();
    list.extend_from_slice(payload);
    let checksum = checksum::jenkins_lookup3(&list);
    list.extend_from_slice(&checksum.to_le_bytes());
    list
}

pub(super) const FRACTAL_HEAP_FSM_CLIENT_ID: u8 = 0;
