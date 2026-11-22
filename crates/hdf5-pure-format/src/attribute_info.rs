//! The version 0 Attribute Info message (type 0x0015).

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use hdf5_pure_core::AttributeInfoField;

use crate::address::{DecodedStoredAddress, StoredAddress, StoredAddressExt};
use crate::bytes;
use crate::bytes::ByteCursor;
use crate::codec::{Decode, Encode};
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// An object header's version 0 metadata for attribute storage and creation order.
///
/// "The Attribute Info Message" of the [format specification, version 4.0][spec] defines the
/// message. The parser reads the maximum creation index if the tracking bit is set and the
/// creation-order B-tree address if the indexing bit is set.
///
/// All stored addresses fit the width supplied at construction or parsing. The writer uses that
/// width to encode the body.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttributeInfoMessage {
    offset_width: OffsetWidth,
    flags: u8,
    max_creation_index: u16,
    fractal_heap_address: StoredAddress,
    btree_name_index_address: StoredAddress,
    btree_creation_order_address: StoredAddress,
}

impl AttributeInfoMessage {
    /// Constructs a message with addresses checked at `offset_width`.
    ///
    /// Pass a maximum creation index to enable tracking. Use [`AttributeCreationOrderIndex`]
    /// to specify whether the creation-order index field is present and whether its address is
    /// defined. Pass [`DecodedStoredAddress::Undefined`] for an unallocated heap or name B-tree.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if a defined address exceeds the selected width,
    /// or [`FormatError::DefinedAddressIsUndefined`] if it equals the width's undefined marker.
    ///
    /// # Examples
    ///
    /// Creates a message for dense attributes with a creation-order index.
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::{DecodedStoredAddress, OffsetWidth, StoredAddress};
    /// use hdf5_pure_format::__private::{AttributeCreationOrderIndex, AttributeInfoMessage};
    ///
    /// let index = StoredAddress::new(0x3000);
    /// let info = AttributeInfoMessage::new(
    ///     OffsetWidth::Two,
    ///     Some(4),
    ///     DecodedStoredAddress::Defined(StoredAddress::new(0x1000)),
    ///     DecodedStoredAddress::Defined(StoredAddress::new(0x2000)),
    ///     AttributeCreationOrderIndex::Defined(index),
    /// )?;
    /// assert_eq!(info.creation_order_index().defined_address(), Some(index));
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn new(
        offset_width: OffsetWidth,
        max_creation_index: Option<u16>,
        fractal_heap_address: DecodedStoredAddress,
        btree_name_index_address: DecodedStoredAddress,
        creation_order_index: AttributeCreationOrderIndex,
    ) -> Result<Self, FormatError> {
        Ok(Self::from_stored(
            offset_width,
            max_creation_index,
            fractal_heap_address.to_stored(offset_width)?,
            btree_name_index_address.to_stored(offset_width)?,
            creation_order_index.to_stored(offset_width)?,
        ))
    }

    /// Constructs a message for attributes stored in the object header.
    ///
    /// Uses undefined heap and name-index addresses. When `indexed` is `true`, includes
    /// a creation-order index with an undefined address. Pass a maximum creation index to
    /// include the tracking field.
    pub fn compact(
        offset_width: OffsetWidth,
        max_creation_index: Option<u16>,
        indexed: bool,
    ) -> Self {
        let undefined = StoredAddress::undefined(offset_width);
        Self::from_stored(
            offset_width,
            max_creation_index,
            undefined,
            undefined,
            indexed.then_some(undefined),
        )
    }

    /// Parses a version 0 Attribute Info body at the file's address width.
    ///
    /// Ignores bytes after the last field, including version 1 object header padding.
    /// "Version 1 Data Object Header Prefix" of the [format specification, version 4.0][spec]
    /// defines that padding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeInfoVersion`] for versions other than 0,
    /// [`FormatError::InvalidAttributeInfoFlags`] if a reserved flag bit is set, or
    /// [`FormatError::IncompleteAttributeInfo`] if `data` is shorter than the required fields.
    /// The truncation error reports the required length and the first incomplete field.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
    pub fn parse_with_offset_width(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<AttributeInfoMessage, FormatError> {
        Self::decode(data, offset_width)
    }

    /// Sets the next creation index, or disables tracking if `maximum` is `None`.
    pub fn set_max_creation_index(&mut self, maximum: Option<u16>) {
        self.flags =
            (self.flags & !TRACK_CREATION_ORDER) | maximum.map_or(0, |_| TRACK_CREATION_ORDER);
        self.max_creation_index = maximum.unwrap_or_default();
    }

    /// Checks the address at the retained width before updating the creation-order index.
    ///
    /// Pass [`AttributeCreationOrderIndex::Absent`] to omit the index address field.
    ///
    /// # Errors
    ///
    /// Returns the address errors documented by [`Self::new`].
    pub fn set_creation_order_index(
        &mut self,
        index: AttributeCreationOrderIndex,
    ) -> Result<(), FormatError> {
        let address = index.to_stored(self.offset_width)?;
        self.flags =
            (self.flags & !INDEX_CREATION_ORDER) | address.map_or(0, |_| INDEX_CREATION_ORDER);
        self.btree_creation_order_address =
            address.unwrap_or_else(|| StoredAddress::undefined(self.offset_width));
        Ok(())
    }

    /// Returns the next creation index to assign if the object tracks creation order.
    pub fn max_creation_index(&self) -> Option<u16> {
        self.tracks_creation_order()
            .then_some(self.max_creation_index)
    }

    /// Returns the address of the dense attribute heap, or `None` if undefined.
    pub fn fractal_heap_address(&self) -> Option<StoredAddress> {
        (!self.fractal_heap_address.is_undefined(self.offset_width))
            .then_some(self.fractal_heap_address)
    }

    /// Returns the address of the name B-tree, or `None` if undefined.
    pub fn btree_name_index_address(&self) -> Option<StoredAddress> {
        (!self
            .btree_name_index_address
            .is_undefined(self.offset_width))
        .then_some(self.btree_name_index_address)
    }

    /// Returns whether the index field is present, including when its address is undefined.
    pub fn indexes_creation_order(&self) -> bool {
        self.flags & INDEX_CREATION_ORDER != 0
    }

    /// Returns the index field's presence and address.
    pub fn creation_order_index(&self) -> AttributeCreationOrderIndex {
        if !self.indexes_creation_order() {
            AttributeCreationOrderIndex::Absent
        } else if self
            .btree_creation_order_address
            .is_undefined(self.offset_width)
        {
            AttributeCreationOrderIndex::Undefined
        } else {
            AttributeCreationOrderIndex::Defined(self.btree_creation_order_address)
        }
    }

    /// Returns the length of the body in bytes without object header padding.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the body at the start of `output` and returns its length in bytes.
    ///
    /// Writes exactly [`Self::encoded_size`] bytes and preserves the rest of the slice.
    ///
    /// # Errors
    ///
    /// Returns [`AttributeInfoOutputTooSmall`] before writing any bytes if the slice is shorter
    /// than the body.
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, AttributeInfoOutputTooSmall> {
        self.encode_into(output)
    }

    /// Returns the version 0 body without object header padding.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![0; self.encoded_size()];
        // The allocation has the exact length required by the encoder.
        self.write_into(&mut bytes)
            .expect("exact Attribute Info length");
        bytes
    }

    /// Builds the message from addresses acquired or checked at `offset_width`.
    fn from_stored(
        offset_width: OffsetWidth,
        max_creation_index: Option<u16>,
        fractal_heap_address: StoredAddress,
        btree_name_index_address: StoredAddress,
        btree_creation_order_address: Option<StoredAddress>,
    ) -> Self {
        Self {
            offset_width,
            flags: max_creation_index.map_or(0, |_| TRACK_CREATION_ORDER)
                | btree_creation_order_address.map_or(0, |_| INDEX_CREATION_ORDER),
            max_creation_index: max_creation_index.unwrap_or_default(),
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address: btree_creation_order_address
                .unwrap_or_else(|| StoredAddress::undefined(offset_width)),
        }
    }

    fn tracks_creation_order(&self) -> bool {
        self.flags & TRACK_CREATION_ORDER != 0
    }
}

impl<'input> Decode<'input> for AttributeInfoMessage {
    type Context = OffsetWidth;
    type Error = FormatError;

    fn decode(input: &'input [u8], offset_width: OffsetWidth) -> Result<Self, Self::Error> {
        let mut cursor = ByteCursor::new(input);
        let [version, flags] = cursor.array::<PREFIX_SIZE>().map_err(incomplete_prefix)?;
        if version != VERSION {
            return Err(FormatError::InvalidAttributeInfoVersion(version));
        }
        if flags & !(TRACK_CREATION_ORDER | INDEX_CREATION_ORDER) != 0 {
            return Err(FormatError::InvalidAttributeInfoFlags(flags));
        }

        let tracked = flags & TRACK_CREATION_ORDER != 0;
        let indexed = flags & INDEX_CREATION_ORDER != 0;
        let address_start = PREFIX_SIZE + usize::from(tracked) * MAXIMUM_INDEX_SIZE;
        let address_count = REQUIRED_ADDRESS_COUNT + usize::from(indexed);
        let address_size = usize::from(offset_width.get());
        let body_size = address_start + address_count * address_size;
        let incomplete = |available| {
            incomplete_body(
                bytes::UnexpectedEof::new(body_size, available),
                address_start,
                address_size,
            )
        };
        let maximum_bytes = if tracked {
            Some(
                cursor
                    .take_array::<MAXIMUM_INDEX_SIZE>()
                    .map_err(|error| incomplete(error.available))?,
            )
        } else {
            None
        };
        let address_bytes = cursor.remaining();
        let StoredAddresses {
            fractal_heap,
            name_index,
            creation_order_index,
        } = match offset_width {
            OffsetWidth::Two => StoredAddresses::read(address_bytes, indexed, |bytes| {
                u64::from(u16::from_le_bytes(bytes))
            }),
            OffsetWidth::Four => StoredAddresses::read(address_bytes, indexed, |bytes| {
                u64::from(u32::from_le_bytes(bytes))
            }),
            OffsetWidth::Eight => StoredAddresses::read(address_bytes, indexed, u64::from_le_bytes),
        }
        .ok_or_else(|| incomplete(input.len()))?;
        Ok(Self::from_stored(
            offset_width,
            maximum_bytes.copied().map(u16::from_le_bytes),
            fractal_heap,
            name_index,
            creation_order_index,
        ))
    }
}

impl Encode for AttributeInfoMessage {
    type Error = AttributeInfoOutputTooSmall;

    fn encoded_len(&self) -> usize {
        PREFIX_SIZE
            + MAXIMUM_INDEX_SIZE * usize::from(self.tracks_creation_order())
            + usize::from(self.offset_width.get())
                * (REQUIRED_ADDRESS_COUNT + usize::from(self.indexes_creation_order()))
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let len = self.encoded_len();
        let available = output.len();
        let record = output.get_mut(..len).ok_or(AttributeInfoOutputTooSmall {
            required: len,
            available,
        })?;
        let Self {
            offset_width,
            flags,
            max_creation_index,
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address,
        } = *self;

        // The encoded length includes both prefix bytes.
        let (prefix, fields) = record
            .split_first_chunk_mut::<PREFIX_SIZE>()
            .expect("two prefix bytes");
        *prefix = [VERSION, flags];

        let addresses = if flags & TRACK_CREATION_ORDER != 0 {
            // The encoded length includes the maximum when tracking is enabled.
            let (field, addresses) = fields
                .split_first_chunk_mut::<MAXIMUM_INDEX_SIZE>()
                .expect("maximum field");
            *field = max_creation_index.to_le_bytes();
            addresses
        } else {
            fields
        };
        let width = usize::from(offset_width.get());
        let stored = [
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address,
        ];
        for (field, address) in addresses.chunks_exact_mut(width).zip(stored) {
            field.copy_from_slice(&address.get().to_le_bytes()[..width]);
        }
        Ok(len)
    }
}

/// Addresses in their stored representation, including the undefined marker.
///
/// The creation-order index is `None` when its field is absent.
struct StoredAddresses {
    fractal_heap: StoredAddress,
    name_index: StoredAddress,
    creation_order_index: Option<StoredAddress>,
}

impl StoredAddresses {
    /// Acquires the complete array of addresses before decoding its elements.
    ///
    /// Includes the creation-order index address if `indexed` is true. Returns `None` if the
    /// required array extends past `input`. The caller supplies the decoder for the file's
    /// `N`-byte addresses.
    fn read<const N: usize>(
        input: &[u8],
        indexed: bool,
        decode: impl Fn([u8; N]) -> u64,
    ) -> Option<Self> {
        const { assert!(N > 0) };
        let (addresses, _) = input.as_chunks::<N>();
        let (fractal_heap, name_index, creation_order_index) = if indexed {
            let [fractal_heap, name_index, creation_order_index] =
                addresses.first_chunk::<{ REQUIRED_ADDRESS_COUNT + 1 }>()?;
            (fractal_heap, name_index, Some(creation_order_index))
        } else {
            let [fractal_heap, name_index] = addresses.first_chunk::<REQUIRED_ADDRESS_COUNT>()?;
            (fractal_heap, name_index, None)
        };
        let read = |bytes: &[u8; N]| StoredAddress::new(decode(*bytes));
        Some(Self {
            fractal_heap: read(fractal_heap),
            name_index: read(name_index),
            creation_order_index: creation_order_index.map(read),
        })
    }
}

/// Reports the missing version or flags byte in a truncated prefix.
#[cold]
fn incomplete_prefix(error: bytes::UnexpectedEof) -> FormatError {
    let field = if error.available == 0 {
        AttributeInfoField::Version
    } else {
        AttributeInfoField::Flags
    };
    incomplete_field(error, field, error.available, PREFIX_FIELD_SIZE)
}

/// Locates the first incomplete payload field after a complete prefix.
#[cold]
fn incomplete_body(
    error: bytes::UnexpectedEof,
    address_start: usize,
    address_size: usize,
) -> FormatError {
    if error.available < address_start {
        return incomplete_field(
            error,
            AttributeInfoField::MaximumCreationIndex,
            PREFIX_SIZE,
            MAXIMUM_INDEX_SIZE,
        );
    }

    let address_index = (error.available - address_start) / address_size;
    let field = match address_index {
        0 => AttributeInfoField::FractalHeapAddress,
        1 => AttributeInfoField::NameIndexAddress,
        _ => AttributeInfoField::CreationOrderIndexAddress,
    };
    incomplete_field(
        error,
        field,
        address_start + address_index * address_size,
        address_size,
    )
}

fn incomplete_field(
    error: bytes::UnexpectedEof,
    field: AttributeInfoField,
    field_offset: usize,
    field_size: usize,
) -> FormatError {
    FormatError::IncompleteAttributeInfo {
        expected: error.expected,
        available: error.available,
        field,
        field_offset,
        field_size,
        field_available: error.available - field_offset,
    }
}

/// The presence and address of an attribute creation-order index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttributeCreationOrderIndex {
    /// The object has no creation-order index field.
    Absent,
    /// The object stores a defined address for the index B-tree.
    Defined(StoredAddress),
    /// The object stores the undefined address for the index B-tree.
    Undefined,
}

impl AttributeCreationOrderIndex {
    /// Returns the B-tree address when the index field is present and the address is defined.
    pub fn defined_address(self) -> Option<StoredAddress> {
        match self {
            Self::Defined(address) => Some(address),
            Self::Absent | Self::Undefined => None,
        }
    }

    /// Converts a present index address to its stored value at `offset_width`.
    ///
    /// Returns `None` for an absent field or the width's undefined marker for a present field
    /// with an undefined address.
    ///
    /// # Errors
    ///
    /// Returns the address errors documented by [`AttributeInfoMessage::new`].
    fn to_stored(self, offset_width: OffsetWidth) -> Result<Option<StoredAddress>, FormatError> {
        match self {
            Self::Absent => Ok(None),
            Self::Defined(address) => DecodedStoredAddress::Defined(address)
                .to_stored(offset_width)
                .map(Some),
            Self::Undefined => Ok(Some(StoredAddress::undefined(offset_width))),
        }
    }
}

/// An error writing an Attribute Info body to a slice shorter than the body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttributeInfoOutputTooSmall {
    /// The length of the body in bytes.
    pub required: usize,
    /// The length of the supplied output slice in bytes.
    pub available: usize,
}

impl fmt::Display for AttributeInfoOutputTooSmall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Attribute Info encoding needs {} bytes, destination has {}",
            self.required, self.available
        )
    }
}

impl core::error::Error for AttributeInfoOutputTooSmall {}

/// The size of either prefix field in "The Attribute Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const PREFIX_FIELD_SIZE: usize = core::mem::size_of::<u8>();
/// The combined size of the Version and Flags fields in "The Attribute Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const PREFIX_SIZE: usize = 2 * PREFIX_FIELD_SIZE;
/// The number of mandatory addresses in "The Attribute Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const REQUIRED_ADDRESS_COUNT: usize = 2;
/// The size of the Maximum Creation Index field in "The Attribute Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const MAXIMUM_INDEX_SIZE: usize = core::mem::size_of::<u16>();

/// The version defined in "The Attribute Info Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const VERSION: u8 = 0;
/// The flag for the Maximum Creation Index field defined in "The Attribute Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const TRACK_CREATION_ORDER: u8 = 0x01;
/// The flag for the Attribute Creation Order v2 B-tree Address field defined in
/// "The Attribute Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const INDEX_CREATION_ORDER: u8 = 0x02;

#[cfg(test)]
mod tests {
    use super::{AttributeCreationOrderIndex, AttributeInfoMessage};
    use crate::address::{StoredAddress, StoredAddressExt};
    use crate::width::OffsetWidth;

    #[test]
    fn inactive_creation_order_fields_have_canonical_values() {
        let mut info = AttributeInfoMessage::compact(OffsetWidth::Two, Some(17), true);
        info.set_creation_order_index(AttributeCreationOrderIndex::Defined(StoredAddress::new(19)))
            .unwrap();

        info.set_max_creation_index(None);
        info.set_creation_order_index(AttributeCreationOrderIndex::Absent)
            .unwrap();

        assert_eq!(info.max_creation_index, 0);
        assert_eq!(
            info.btree_creation_order_address,
            StoredAddress::undefined(OffsetWidth::Two)
        );
        assert_eq!(
            info,
            AttributeInfoMessage::compact(OffsetWidth::Two, None, false)
        );
    }
}
