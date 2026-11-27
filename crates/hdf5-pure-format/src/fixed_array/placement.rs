use hdf5_pure_core::AddressRange;

use super::FixedArrayPlan;
use crate::address::StoredAddress;
use crate::bytes;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// Disjoint complete Fixed Array block ranges whose stored addresses fit the encoding width.
///
/// A paged data block occupies one range, including its prefix, bitmap, pages, and checksums.
/// Allocation ownership and placement policy are established by the caller.
#[derive(Clone, Copy, Debug)]
pub struct FixedArrayPlacement {
    plan: FixedArrayPlan,
    header: AddressRange<StoredAddress>,
    data_block: AddressRange<StoredAddress>,
}

impl FixedArrayPlacement {
    pub(super) fn new(
        plan: FixedArrayPlan,
        header: StoredAddress,
        data_block: StoredAddress,
    ) -> Result<Self, FormatError> {
        let widths = plan.widths();
        let [header_length, data_block_length] = plan.block_sizes();
        let header = Self::range(widths.offsets, header, header_length)?;
        let data_block = Self::range(widths.offsets, data_block, data_block_length)?;
        if header.overlaps(data_block) {
            return Err(FormatError::FixedArrayBlockOverlap {
                header: header.start().get(),
                data_block: data_block.start().get(),
            });
        }
        Ok(Self {
            plan,
            header,
            data_block,
        })
    }

    /// Returns each block's stored address and complete length, header first.
    ///
    /// The order is independent of the blocks' address order.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> {
        self.ranges()
            .into_iter()
            .map(|range| (range.start(), range.length()))
    }

    pub(super) fn plan(&self) -> FixedArrayPlan {
        self.plan
    }

    pub(super) fn ranges(&self) -> [AddressRange<StoredAddress>; 2] {
        [self.header, self.data_block]
    }

    fn range(
        offsets: OffsetWidth,
        address: StoredAddress,
        length: u64,
    ) -> Result<AddressRange<StoredAddress>, FormatError> {
        let maximum = StoredAddress::undefined(offsets).get();
        bytes::require_defined_offset(address.get(), offsets)?;
        let range = AddressRange::new(address, length)?;
        if range.end().get() > maximum {
            return Err(FormatError::AddressTooLarge {
                address: range.end().get(),
                offset_size: offsets.get(),
            });
        }
        Ok(range)
    }
}
