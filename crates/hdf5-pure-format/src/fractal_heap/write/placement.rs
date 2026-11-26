use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::convert::{self, Narrow};
use crate::error::FormatError;

use super::plan::ManagedHeapPlan;

impl ManagedHeapPlan {
    /// Assigns independently selected stored addresses to the planned blocks.
    ///
    /// Pass one address per length returned by [`Self::block_lengths`], in the same order.
    /// Complete blocks must fit the encoded address width and occupy disjoint ranges. A block's
    /// exclusive end may equal the undefined address value.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FractalHeapBlockCount`] if the address count differs from the block
    /// count, and [`FormatError::FractalHeapBlockOverlap`] if complete block ranges overlap.
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a block's exclusive end exceeds `u64`,
    /// [`FormatError::AddressTooLarge`] if an address or exclusive end exceeds the encoded width,
    /// and [`FormatError::DefinedAddressIsUndefined`] if a block starts at the undefined address.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{ManagedHeapPlan, OffsetWidth};
    ///
    /// let plan = ManagedHeapPlan::group_links(&[491, 1], OffsetWidth::Eight).unwrap();
    /// let addresses = [128, 1024, 2048].map(StoredAddress::new);
    /// let placement = plan.place(&addresses).unwrap();
    /// assert_eq!(placement.root_address(), addresses[0]);
    /// assert_eq!(placement.blocks().map(|block| block.0).collect::<Vec<_>>(), addresses);
    /// ```
    pub fn place(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<ManagedHeapPlacement<'_>, FormatError> {
        let expected = self.block_lengths().len();
        if addresses.len() != expected {
            return Err(FormatError::FractalHeapBlockCount {
                expected,
                actual: addresses.len(),
            });
        }
        let mut ranges = Vec::with_capacity(expected);
        for (index, (&address, length)) in addresses.iter().zip(self.block_lengths()).enumerate() {
            self.address_at(address, 0)?;
            let end = address
                .get()
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow {
                    offset: address.get(),
                    length,
                })?;
            self.require_end(end)?;
            ranges.push((address.get(), end, index));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(FormatError::FractalHeapBlockOverlap {
                    first: pair[0].2,
                    second: pair[1].2,
                });
            }
        }
        Ok(ManagedHeapPlacement {
            plan: self,
            addresses: addresses.to_vec(),
        })
    }

    /// Serializes the blocks at `region_address` with their owner set to `heap_header_address`.
    ///
    /// Both addresses are stored file addresses. The returned buffer has
    /// [`region_size`](Self::region_size) bytes. Unused direct-block data bytes are zeroed, and
    /// each block includes its checksum. The caller sets the heap header's direct-block checksum
    /// flag.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the object count differs from the plan and
    /// [`FormatError::DataSizeMismatch`] if an object's length differs from its planned size.
    /// Returns [`FormatError::OffsetOverflow`] if a block address or the region end overflows
    /// `u64`, [`FormatError::AddressTooLarge`] if an address or the region extent exceeds the
    /// encoded file-address width, and [`FormatError::DefinedAddressIsUndefined`] if a required
    /// address equals that width's undefined value.
    pub fn serialize(
        &self,
        objects: &[&[u8]],
        region_address: StoredAddress,
        heap_header_address: StoredAddress,
    ) -> Result<Vec<u8>, FormatError> {
        self.check_objects(objects)?;
        self.address_at(region_address, 0)?;
        self.address_at(heap_header_address, 0)?;
        let end = region_address.get().checked_add(self.region_size()).ok_or(
            FormatError::OffsetOverflow {
                offset: region_address.get(),
                length: self.region_size(),
            },
        )?;
        self.require_end(end)?;
        let mut region = vec![0; self.region_size().to_usize()?];
        for index in 0..self.block_lengths().len() {
            let data = self.encode_block(index, objects, heap_header_address, |child| {
                self.address_at(region_address, self.block_offset(child))
            })?;
            place(&mut region, self.block_offset(index), &data)?;
        }
        Ok(region)
    }
}

/// A managed fractal-heap plan with checked, disjoint stored block ranges.
///
/// Created by [`ManagedHeapPlan::place`]. Object heap offsets and heap statistics are those of
/// the borrowed plan. The caller allocates storage for each complete block and writes the heap
/// header separately.
pub struct ManagedHeapPlacement<'a> {
    plan: &'a ManagedHeapPlan,
    addresses: Vec<StoredAddress>,
}

impl ManagedHeapPlacement<'_> {
    /// Returns the root block's stored address for the heap header.
    pub fn root_address(&self) -> StoredAddress {
        self.addresses[self.plan.root_index()]
    }

    /// Returns each block's stored address and complete serialized length in placement order.
    ///
    /// The order is the same as [`ManagedHeapPlan::block_lengths`].
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.addresses
            .iter()
            .copied()
            .enumerate()
            .map(|(index, address)| (address, self.plan.block_length(index)))
    }

    /// Serializes each block separately with its owner set to `heap_header_address`.
    ///
    /// Encodes one block per iterator step, in the order returned by [`Self::blocks`]. Each item
    /// contains the block's stored address and encoded bytes. Unused direct-block data bytes are
    /// zeroed, and each block includes its checksum. The caller sets the heap header's
    /// direct-block checksum flag.
    ///
    /// Pass objects in the plan's input order, with their planned lengths. Object lengths and
    /// the heap header address are checked before the iterator is returned.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the object count differs from the plan and
    /// [`FormatError::DataSizeMismatch`] if an object's length differs from its planned size.
    /// Returns [`FormatError::AddressTooLarge`] if `heap_header_address` exceeds the encoded
    /// file-address width, and [`FormatError::DefinedAddressIsUndefined`] if it equals that
    /// width's undefined value.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{ManagedHeapPlan, OffsetWidth};
    ///
    /// let plan = ManagedHeapPlan::group_links(&[491, 1], OffsetWidth::Eight).unwrap();
    /// let addresses = [128, 1024, 2048].map(StoredAddress::new);
    /// let placement = plan.place(&addresses).unwrap();
    /// let objects: &[&[u8]] = &[&[7; 491], &[9]];
    /// let blocks = placement.serialize_blocks(objects, StoredAddress::new(64)).unwrap();
    /// for ((address, length), encoded) in placement.blocks().zip(blocks) {
    ///     let (stored_address, bytes) = encoded.unwrap();
    ///     assert_eq!(stored_address, address);
    ///     assert_eq!(bytes.len() as u64, length);
    /// }
    /// ```
    pub fn serialize_blocks<'b>(
        &'b self,
        objects: &'b [&'b [u8]],
        heap_header_address: StoredAddress,
    ) -> Result<
        impl ExactSizeIterator<Item = Result<(StoredAddress, Vec<u8>), FormatError>> + 'b,
        FormatError,
    > {
        self.plan.check_objects(objects)?;
        self.plan.address_at(heap_header_address, 0)?;
        Ok(self
            .addresses
            .iter()
            .copied()
            .enumerate()
            .map(move |(index, address)| {
                self.plan
                    .encode_block(index, objects, heap_header_address, |child| {
                        Ok(self.addresses[child])
                    })
                    .map(|data| (address, data))
            }))
    }
}

fn place(region: &mut [u8], offset: u64, data: &[u8]) -> Result<(), FormatError> {
    let range = convert::slice_range(offset, data.len() as u64)?;
    let available = region.len();
    region
        .get_mut(range)
        .ok_or(FormatError::UnexpectedEof {
            expected: offset.to_usize()?,
            available,
        })?
        .copy_from_slice(data);
    Ok(())
}
