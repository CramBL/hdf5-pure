use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::error::FormatError;

use super::plan::BTreeV2Plan;
use super::prepared::CheckedNodeEncoding;

impl BTreeV2Plan {
    /// Assigns independently selected stored addresses to the planned nodes.
    ///
    /// Pass one address per length returned by [`Self::block_lengths`], in the same order.
    /// Complete node ranges must be disjoint and fit the plan's address width. A node's
    /// exclusive end may equal the undefined address value. The caller allocates the header
    /// separately.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::BTreeV2NodeCount`] if the address count differs from the node
    /// count, and [`FormatError::BTreeV2NodeOverlap`] if complete node ranges overlap.
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a node's exclusive end exceeds [`u64::MAX`],
    /// [`FormatError::AddressTooLarge`] if an address or exclusive end exceeds the plan's address
    /// width, and [`FormatError::DefinedAddressIsUndefined`] if a node starts at the undefined
    /// address. Returns [`FormatError::ValueTooLargeForPlatform`] if the node size exceeds
    /// [`usize::MAX`], and [`FormatError::Internal`] if the node geometry or counts are inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{FormatError, LengthWidth, OffsetWidth, StoredAddress};
    /// use hdf5_pure_format::__private::{BTreeV2Header, BTreeV2Plan, BTREE_V2_LINK_NAME};
    ///
    /// let plan = BTreeV2Plan::new(BTREE_V2_LINK_NAME, 50, 11, 512, OffsetWidth::Two).unwrap();
    /// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [512; 3]);
    /// let addresses = [4096, 1024, 8192].map(StoredAddress::new);
    /// let placement = plan.place(&addresses)?;
    /// let header = placement.serialize_header(LengthWidth::Eight)?;
    /// let header = BTreeV2Header::parse(&header, 0, 2, 8)?;
    /// assert_eq!(header.root_node_address, addresses[2]);
    /// let records = [0; 550];
    /// for ((address, length), encoded) in placement.blocks().zip(placement.serialize_nodes(&records)?) {
    ///     let (stored_address, bytes) = encoded?;
    ///     assert_eq!(stored_address, address);
    ///     assert_eq!(bytes.len() as u64, length);
    /// }
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn place(&self, addresses: &[StoredAddress]) -> Result<BTreeV2Placement<'_>, FormatError> {
        if addresses.len() != self.nodes().len() {
            return Err(FormatError::BTreeV2NodeCount {
                expected: self.nodes().len(),
                actual: addresses.len(),
            });
        }
        let mut ranges = Vec::with_capacity(addresses.len());
        for (index, &address) in addresses.iter().enumerate() {
            ranges.push((
                address.get(),
                self.block_end(address, u64::from(self.node_size()))?,
                index,
            ));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(FormatError::BTreeV2NodeOverlap {
                    first: pair[0].2,
                    second: pair[1].2,
                });
            }
        }
        Ok(BTreeV2Placement {
            addresses: addresses.to_vec(),
            encoding: CheckedNodeEncoding::new(self)?,
        })
    }

    /// Returns the exclusive end of a block at a defined address in the plan's address width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] for an undefined starting address,
    /// [`FormatError::AddressTooLarge`] if either endpoint exceeds the address width, or
    /// [`FormatError::OffsetOverflow`] if adding the length overflows.
    pub(super) fn block_end(
        &self,
        address: StoredAddress,
        length: u64,
    ) -> Result<u64, FormatError> {
        let maximum = StoredAddress::undefined(self.offset_width()).get();
        if address.get() == maximum {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: self.offset_width().get(),
            });
        }
        if address.get() > maximum {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: self.offset_width().get(),
            });
        }
        let end = address.checked_add(length)?.get();
        if end > maximum {
            return Err(FormatError::AddressTooLarge {
                address: end,
                offset_size: self.offset_width().get(),
            });
        }
        Ok(end)
    }
}

/// A version 2 B-tree plan with checked, disjoint stored node ranges.
///
/// Created by [`BTreeV2Plan::place`]. The caller allocates storage for each complete node and
/// writes the header separately. Node placement excludes the header's allocation. Node geometry
/// and count-field widths are checked before placement is returned.
pub struct BTreeV2Placement<'a> {
    addresses: Vec<StoredAddress>,
    encoding: CheckedNodeEncoding<'a>,
}

impl BTreeV2Placement<'_> {
    /// Returns each node's stored address and complete serialized length in placement order.
    ///
    /// The order is the same as [`BTreeV2Plan::block_lengths`].
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.addresses
            .iter()
            .copied()
            .zip(self.encoding.plan().block_lengths())
    }

    pub(super) fn addresses(&self) -> &[StoredAddress] {
        &self.addresses
    }

    pub(super) fn encoding(&self) -> &CheckedNodeEncoding<'_> {
        &self.encoding
    }
}
