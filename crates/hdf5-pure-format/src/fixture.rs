//! The test helpers the parsers that read from a [`FileRegion`] share.

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use test_util::widths::Widths;

use crate::width::FormatWidths;

/// Returns the region that holds `bytes` from the physical address `start`.
pub(crate) fn region(start: u64, bytes: &[u8]) -> FileRegion<'_> {
    FileRegion::new(
        FileRange::new(PhysicalAddress::new(start), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap()
}

/// Converts a test's [`Widths`] to the [`FormatWidths`] a parser takes.
pub(crate) fn format_widths(Widths { offset, length }: Widths) -> FormatWidths {
    FormatWidths::from_sizes(offset.try_into().unwrap(), length.try_into().unwrap()).unwrap()
}
