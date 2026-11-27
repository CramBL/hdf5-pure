use alloc::vec::Vec;

use hdf5_pure_core::AddressRange;

use super::plan::ExtensibleArrayPlan;
use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::chunk_record::IndexSlots;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

impl ExtensibleArrayPlan {
    /// Checks disjoint complete ranges at the supplied stored addresses.
    ///
    /// Addresses follow [`blocks`](Self::blocks) order and may be in any address order. Starts
    /// must be defined, and exclusive ends must fit the address width. An end may equal the
    /// undefined address value. The caller establishes ownership and placement policy.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ExtensibleArrayBlockCount`] if the number of addresses differs
    /// from the number of blocks, [`FormatError::DefinedAddressIsUndefined`] if a start is
    /// undefined, [`FormatError::AddressTooLarge`] if a start or end exceeds the address width,
    /// [`FormatError::OffsetOverflow`] if an end exceeds [`u64::MAX`], or
    /// [`FormatError::ExtensibleArrayBlockOverlap`] if complete ranges overlap.
    pub fn place(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<ExtensibleArrayPlacement<'_>, FormatError> {
        if addresses.len() != self.blocks().len() {
            return Err(FormatError::ExtensibleArrayBlockCount {
                expected: self.blocks().len(),
                actual: addresses.len(),
            });
        }
        let mut ranges = Vec::with_capacity(addresses.len());
        for (&address, length) in addresses.iter().zip(self.block_lengths()) {
            ranges.push(stored_block_range(self.widths().offsets, address, length)?);
        }
        let mut ordered: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(index, range)| (range.start(), range.end(), index))
            .collect();
        ordered.sort_unstable();
        for pair in ordered.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(FormatError::ExtensibleArrayBlockOverlap {
                    first: pair[0].2,
                    second: pair[1].2,
                });
            }
        }
        Ok(ExtensibleArrayPlacement { plan: self, ranges })
    }

    /// Serializes adjacent complete blocks at the stored `address`.
    ///
    /// Checks the complete stored range and writes each block in [`blocks`](Self::blocks) order
    /// into its slice of a single output buffer. The caller establishes allocation ownership.
    ///
    /// # Errors
    ///
    /// Returns the address errors of [`Self::place`] and record errors of
    /// [`ExtensibleArrayPlacement::encoding`]. Returns [`FormatError::ValueTooLargeForPlatform`]
    /// if the total length does not fit [`usize`], [`FormatError::AllocationFailed`] if a range,
    /// reference, or output reservation fails, or [`FormatError::Internal`] if the ranges or
    /// encoded length differ from the planned total.
    pub(super) fn encode_contiguous(
        &self,
        slots: &IndexSlots<'_>,
        address: StoredAddress,
    ) -> Result<Vec<u8>, FormatError> {
        let whole = stored_block_range(self.widths().offsets, address, self.total_len())?;
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(self.blocks().len())
            .map_err(|source| FormatError::AllocationFailed {
                values: self.blocks().len(),
                source,
            })?;
        let mut cursor = whole.start();
        for length in self.block_lengths() {
            let range = AddressRange::new(cursor, length)?;
            cursor = range.end();
            ranges.push(range);
        }
        let placement = ExtensibleArrayPlacement { plan: self, ranges };
        let encoding = placement.encoding(slots)?;
        let length = self.total_len().to_usize()?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|source| FormatError::AllocationFailed {
                values: length,
                source,
            })?;
        output.resize(length, 0);
        let mut remainder = output.as_mut_slice();
        for block in encoding.blocks() {
            let written = block.write_into(remainder)?;
            remainder = &mut remainder[written..];
        }
        if cursor != whole.end() || !remainder.is_empty() {
            return Err(FormatError::Internal(alloc::format!(
                "contiguous EA encoding does not fill its planned {length} bytes"
            )));
        }
        Ok(output)
    }
}

/// Disjoint complete block ranges with stored addresses that fit an Extensible Array plan.
///
/// The plan is borrowed, and ranges follow its descriptor order. Allocation ownership and
/// placement policy are established by the caller.
#[derive(Debug)]
pub struct ExtensibleArrayPlacement<'a> {
    plan: &'a ExtensibleArrayPlan,
    ranges: Vec<AddressRange<StoredAddress>>,
}

impl ExtensibleArrayPlacement<'_> {
    /// Returns the borrowed geometry and block descriptors.
    pub fn plan(&self) -> &ExtensibleArrayPlan {
        self.plan
    }
    /// Returns stored starts and complete lengths in the plan's descriptor order.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.ranges
            .iter()
            .map(|range| (range.start(), range.length()))
    }

    pub(super) fn ranges(&self) -> &[AddressRange<StoredAddress>] {
        &self.ranges
    }
}

/// Checks a block's start address and exclusive end against the file's offset width.
///
/// # Errors
///
/// Returns [`FormatError::DefinedAddressIsUndefined`] if the start is undefined,
/// [`FormatError::AddressTooLarge`] if either endpoint exceeds the offset width, or
/// [`FormatError::OffsetOverflow`] if the exclusive end exceeds [`u64::MAX`].
pub(super) fn stored_block_range(
    offsets: OffsetWidth,
    address: StoredAddress,
    length: u64,
) -> Result<AddressRange<StoredAddress>, FormatError> {
    let address = DecodedStoredAddress::Defined(address).to_stored(offsets)?;
    let range = AddressRange::new(address, length)?;
    if range.end() > StoredAddress::undefined(offsets) {
        return Err(FormatError::AddressTooLarge {
            address: range.end().get(),
            offset_size: offsets.get(),
        });
    }
    Ok(range)
}

#[cfg(test)]
#[path = "tests/placement.rs"]
mod tests;
