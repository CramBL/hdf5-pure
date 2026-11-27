use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::__private as format_boundary;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::attribute_info::AttributeCreationOrderIndex;
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_message::AttributeMessage;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

use super::attributes::{DenseAttrCreationOrder, EncodedDenseAttribute, EncodedDenseAttrs};
use super::fixup::FixupRange;
use super::geometry::DenseAttrGeometry;
use super::placement::DenseAttrPlacement;

/// The managed-object size limit of the default eight-byte-width attribute heap.
///
/// The writer uses an eight-byte version 0 managed heap ID with a flag byte, a five-byte
/// offset, and a two-byte length. Larger messages are stored and indexed as huge objects,
/// as defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
pub const DENSE_ATTR_MAX_MANAGED_OBJECT: usize =
    format_boundary::attribute_heap_max_managed_object(DENSE_ATTR_OFFSET_WIDTH);

/// Encoded dense attribute payloads and the geometry used to emit them.
///
/// The heap and index offsets, creation metadata, and total length are independent of the
/// heap's file address. [`build`](Self::build) writes the retained layout at a chosen address.
/// Cloning copies encoded messages while sharing immutable geometry.
#[derive(Clone)]
pub struct DenseAttrPlan {
    /// Each attribute's version 3 message bytes and creation index, in the caller's order.
    attributes: Vec<EncodedDenseAttribute>,
    geometry: Arc<DenseAttrGeometry>,
}

/// Encodes attributes and plans their heap and indexes before file placement.
///
/// Attributes that fit [`DENSE_ATTR_MAX_MANAGED_OBJECT`] are stored as managed
/// objects in the heap's direct blocks. Larger ones are stored as *huge*
/// objects, whose bytes sit outside the managed blocks and whose address and
/// length are indexed by a huge-objects v2 B-tree.
///
/// # Errors
///
/// Returns [`FormatError::AttributeFieldTooLong`] if the name with its NUL terminator, the encoded
/// datatype, or the encoded dataspace exceeds 65,535 bytes.
/// Returns [`FormatError::DuplicateAttributeName`] if names repeat,
/// [`FormatError::DuplicateAttributeCreationIndex`] if tracked indexes repeat, or
/// [`FormatError::InvalidAttributeCreationCounter`] if a tracked index is at or above the
/// next creation index. Returns [`FormatError::Internal`] if the counts of attributes and
/// tracked indexes differ or if the writer cannot plan one of the indexes. Returns
/// [`FormatError::DenseAttributeHeapTooLarge`] if managed storage exceeds the heap address
/// space, [`FormatError::ValueTooLargeForPlatform`] if
/// storage does not fit a [`usize`], [`FormatError::ValueTooLargeForRepresentation`] if an
/// attribute count or position does not fit a [`u32`], or [`FormatError::OffsetOverflow`] if
/// the total length exceeds [`u64`].
/// Propagates other managed-heap planning errors.
pub fn dense_attrs_plan(
    attrs: &[AttributeMessage],
    creation: DenseAttrCreationOrder,
) -> Result<DenseAttrPlan, FormatError> {
    DenseAttrPlan::new(EncodedDenseAttrs::new(
        FormatWidths {
            offsets: DENSE_ATTR_OFFSET_WIDTH,
            lengths: DENSE_ATTR_LENGTH_WIDTH,
        },
        attrs,
        creation,
    )?)
}

impl DenseAttrPlan {
    pub(super) fn new(encoded: EncodedDenseAttrs) -> Result<Self, FormatError> {
        let (geometry, attributes) = DenseAttrGeometry::prepare(encoded)?;
        Ok(Self {
            attributes,
            geometry: Arc::new(geometry),
        })
    }

    pub(super) fn geometry(&self) -> &DenseAttrGeometry {
        &self.geometry
    }

    pub(super) fn attributes(&self) -> &[EncodedDenseAttribute] {
        &self.attributes
    }

    pub(super) fn fixup_bytes_mut(&mut self, field: &FixupRange) -> Result<&mut [u8], FormatError> {
        EncodedDenseAttribute::fixup_bytes_mut(&mut self.attributes, field)
    }

    /// Returns the allocation kind of each block in placement order.
    ///
    /// The order matches [`Self::block_lengths`]. Huge-object payloads follow the metadata.
    pub fn block_kinds(&self) -> impl ExactSizeIterator<Item = DenseAttrBlockKind> + '_ {
        let metadata = self.geometry.block_lengths().len() - self.geometry.huge_count();
        (0..self.geometry.block_lengths().len()).map(move |index| {
            if index < metadata {
                DenseAttrBlockKind::Metadata
            } else {
                DenseAttrBlockKind::HugeObject
            }
        })
    }

    /// Returns the complete lengths of independently placed blocks.
    ///
    /// The order is the heap header, managed heap blocks, name index, optional creation-order
    /// index, optional huge-object index, and huge-object payloads. Each index includes its
    /// header followed by its nodes. Huge objects retain the input attribute order.
    /// [`Self::place`] takes addresses in this order.
    pub fn block_lengths(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.geometry.block_lengths().iter().copied()
    }

    /// Places each heap block, index block, and huge object at its supplied address.
    ///
    /// Pass one address per entry of [`Self::block_lengths`], in the same order. Blocks may be
    /// separated by gaps or placed in a different address order. Every complete block must
    /// have a defined start address, a representable exclusive end, and no overlap with
    /// another block. The exclusive end may equal the width's undefined-address value.
    /// The caller chooses alignment and page boundaries.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DenseAttributeBlockCount`] if the number of addresses differs
    /// from the planned count, or [`FormatError::DenseAttributeBlockOverlap`] if blocks
    /// overlap. Returns [`FormatError::DefinedAddressIsUndefined`] if a start address is
    /// undefined, [`FormatError::AddressTooLarge`] if a start or exclusive end exceeds the
    /// retained address width, or [`FormatError::OffsetOverflow`] if an exclusive end exceeds
    /// [`u64::MAX`]. Propagates heap-ID encoding, heap-ID parsing, and index position conversion
    /// errors.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::FormatError;
    /// use hdf5_pure_format::__private::{dense_attrs_plan, DenseAttrCreationOrder};
    ///
    /// let plan = dense_attrs_plan(&[], DenseAttrCreationOrder::Untracked)?;
    /// let mut next = 0x8000;
    /// let addresses: Vec<_> = plan.block_lengths().map(|length| {
    ///     let address = StoredAddress::new(next);
    ///     next += length + 256;
    ///     address
    /// }).collect();
    /// let placement = plan.place(&addresses)?;
    /// assert_eq!(placement.attribute_info().fractal_heap_address(), Some(addresses[0]));
    /// for ((address, length), block) in placement.blocks().zip(placement.serialize_blocks()?) {
    ///     let block = block?;
    ///     assert_eq!(block.address, address);
    ///     assert_eq!(block.bytes.len() as u64, length);
    /// }
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn place(
        &self,
        addresses: &[StoredAddress],
    ) -> Result<DenseAttrPlacement<'_>, FormatError> {
        DenseAttrPlacement::new(self, addresses)
    }

    /// Returns the reserved length, including heap blocks, indexes, and huge objects.
    pub fn blob_len(&self) -> u64 {
        self.geometry.offsets().total_len
    }

    /// Returns the object-header metadata for a heap placed at `heap_address`.
    ///
    /// The message length is independent of the chosen address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `heap_address` is undefined,
    /// [`FormatError::AddressTooLarge`] if the start or reserved exclusive end exceeds the
    /// retained address width, or [`FormatError::OffsetOverflow`] if the extent exceeds
    /// [`u64::MAX`].
    pub fn attribute_info_for_heap(
        &self,
        heap_address: StoredAddress,
    ) -> Result<AttributeInfoMessage, FormatError> {
        self.extent(heap_address)?;
        AttributeInfoMessage::new(
            self.geometry.widths().offsets,
            self.geometry.indexes().next_index(),
            DecodedStoredAddress::Defined(heap_address),
            DecodedStoredAddress::Defined(
                heap_address.checked_add(self.geometry.offsets().btree_off)?,
            ),
            if self.geometry.indexes().indexed() {
                AttributeCreationOrderIndex::Defined(
                    heap_address.checked_add(self.geometry.offsets().corder_bthd_off)?,
                )
            } else {
                AttributeCreationOrderIndex::Absent
            },
        )
    }

    /// Checks the heap's base address and reserved end.
    ///
    /// # Errors
    ///
    /// Returns the start and exclusive-end errors of [`Self::block_end`].
    pub(super) fn extent(&self, base: StoredAddress) -> Result<(), FormatError> {
        self.block_end(base, self.geometry().offsets().total_len)
            .map(|_| ())
    }

    /// Checks a defined block start and returns its representable exclusive end.
    ///
    /// The exclusive end may equal the address width's undefined marker.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if `address` is undefined,
    /// [`FormatError::AddressTooLarge`] if the start or end exceeds the retained width, or
    /// [`FormatError::OffsetOverflow`] if the end exceeds [`u64::MAX`].
    pub(super) fn block_end(
        &self,
        address: StoredAddress,
        length: u64,
    ) -> Result<u64, FormatError> {
        let offsets = self.geometry().widths().offsets;
        let maximum = StoredAddress::undefined(offsets).get();
        if address.get() == maximum {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: offsets.get(),
            });
        }
        if address.get() > maximum {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: offsets.get(),
            });
        }
        let end = address.checked_add(length)?.get();
        if end > maximum {
            return Err(FormatError::AddressTooLarge {
                address: end,
                offset_size: offsets.get(),
            });
        }
        Ok(end)
    }
}

/// The allocation kind of an independently placed dense-attribute block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseAttrBlockKind {
    /// An encoded attribute stored outside the managed heap blocks.
    HugeObject,
    /// A heap header, managed heap block, B-tree header, or B-tree node.
    Metadata,
}

/// Returns the minimum nonzero byte width needed to encode `value`.
pub(super) fn encoded_size_width(value: u64) -> usize {
    (64 - value.leading_zeros() as usize).div_ceil(8).max(1)
}

pub(super) const DENSE_ATTR_OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;
pub(super) const DENSE_ATTR_LENGTH_WIDTH: LengthWidth = LengthWidth::Eight;
