//! Stored counts and global heap IDs for variable-length strings and sequences.

use core::fmt;

use hdf5_pure_core::StoredAddress;

use crate::bytes::Fields;
use crate::decode::Decode;
use crate::error::FormatError;
use crate::fixed_record::{
    self, BoundStoredAddress, FixedRecord, FixedWire, Offset2, Offset4, Offset8, OffsetEncoding,
    StoredAddressCodec, U32Le,
};
use crate::width::OffsetWidth;

/// A variable-length value's stored element count and global heap ID.
///
/// The count and object index are four-byte unsigned integers. The collection address fits
/// the selected [`OffsetWidth`], including its all-ones undefined value. Encoding preserves
/// every field, including addresses and indices supplied with a zero count. The count is in
/// base elements, so a sequence of two two-byte integers has count two and four payload bytes.
///
/// The parser checks wire framing, and the constructor checks whether the address fits its
/// width. The caller resolves the collection address, checks the object index against the heap
/// collection, and interprets the payload. A zero count alone does not establish a null heap
/// reference.
///
/// The global heap ID is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_format::__private::VariableLengthDescriptor;
/// use hdf5_pure_format::{FormatError, OffsetWidth};
///
/// let descriptor = VariableLengthDescriptor::new(
///     OffsetWidth::Two, 2, StoredAddress::new(0x1234), 7,
/// )?;
/// let mut output = [0; 12];
/// assert_eq!(descriptor.write_into(&mut output)?, 10);
/// assert_eq!(output, [2, 0, 0, 0, 0x34, 0x12, 7, 0, 0, 0, 0, 0]);
/// let (parsed, consumed) = VariableLengthDescriptor::parse(OffsetWidth::Two, &output)?;
/// assert_eq!(parsed, descriptor);
/// assert_eq!(consumed, 10);
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct VariableLengthDescriptor {
    fields: DescriptorValue,
}

impl VariableLengthDescriptor {
    /// Parses one descriptor from the start of `raw` and returns the bytes consumed.
    ///
    /// Reads all fields as little-endian integers. The count and index each occupy four bytes,
    /// and the collection address occupies `offsets` bytes. Any suffix remains unexamined.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `raw` is shorter than
    /// [`Self::serialized_size`] for `offsets`.
    ///
    /// # Examples
    ///
    /// The parser retains the stored heap reference when the count is zero:
    ///
    /// ```
    /// use hdf5_pure_format::__private::VariableLengthDescriptor;
    /// use hdf5_pure_format::{FormatError, OffsetWidth};
    ///
    /// let raw = [0, 0, 0, 0, 0x34, 0x12, 7, 0, 0, 0];
    /// let (descriptor, consumed) = VariableLengthDescriptor::parse(OffsetWidth::Two, &raw)?;
    /// assert_eq!(descriptor.element_count(), 0);
    /// assert_eq!(descriptor.collection_address().get(), 0x1234);
    /// assert_eq!(descriptor.object_index(), 7);
    /// assert_eq!(consumed, raw.len());
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn parse(offsets: OffsetWidth, raw: &[u8]) -> Result<(Self, usize), FormatError> {
        let mut fields = Fields::new(raw, 0);
        let descriptor = Self::decode(&mut fields, offsets)?;
        Ok((descriptor, fields.pos()))
    }

    /// Returns eight bytes plus the file's address width.
    ///
    /// The file's length width does not affect descriptor size.
    pub fn serialized_size(offsets: OffsetWidth) -> usize {
        match offsets {
            OffsetWidth::Two => DescriptorWire::<Offset2>::SIZE,
            OffsetWidth::Four => DescriptorWire::<Offset4>::SIZE,
            OffsetWidth::Eight => DescriptorWire::<Offset8>::SIZE,
        }
    }

    /// Creates a descriptor whose stored address fits `offsets` exactly.
    ///
    /// Preserves zero and undefined addresses and the full count and index. A narrower field's
    /// undefined address must be supplied at that width, such as `0xFFFF` for two-byte addresses.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if `collection_address` exceeds the largest
    /// unsigned integer representable in `offsets` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::__private::VariableLengthDescriptor;
    /// use hdf5_pure_format::{FormatError, OffsetWidth};
    /// use hdf5_pure_core::StoredAddress;
    ///
    /// let descriptor = VariableLengthDescriptor::new(
    ///     OffsetWidth::Two, 0, StoredAddress::undefined(OffsetWidth::Two), u32::MAX,
    /// )?;
    /// assert_eq!(descriptor.collection_address().get(), 0xFFFF);
    /// assert_eq!(descriptor.object_index(), u32::MAX);
    /// assert_eq!(
    ///     VariableLengthDescriptor::new(
    ///         OffsetWidth::Two, 0, StoredAddress::undefined(OffsetWidth::Eight), 0,
    ///     ),
    ///     Err(FormatError::AddressTooLarge { address: u64::MAX, offset_size: 2 }),
    /// );
    /// # Ok::<(), FormatError>(())
    /// ```
    #[inline]
    pub fn new(
        offsets: OffsetWidth,
        element_count: u32,
        collection_address: StoredAddress,
        object_index: u32,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            fields: match offsets {
                OffsetWidth::Two => DescriptorValue::Two(DescriptorFields::new(
                    element_count,
                    collection_address,
                    object_index,
                )?),
                OffsetWidth::Four => DescriptorValue::Four(DescriptorFields::new(
                    element_count,
                    collection_address,
                    object_index,
                )?),
                OffsetWidth::Eight => DescriptorValue::Eight(DescriptorFields::new(
                    element_count,
                    collection_address,
                    object_index,
                )?),
            },
        })
    }

    /// Writes the descriptor into the start of `output` and returns the bytes written.
    ///
    /// Writes little-endian fields at the stored address width and preserves any output suffix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is shorter than [`Self::encoded_size`].
    /// The output is unchanged on error.
    #[inline]
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        match &self.fields {
            DescriptorValue::Two(fields) => DescriptorCodec::<Offset2>::write(output, fields),
            DescriptorValue::Four(fields) => DescriptorCodec::<Offset4>::write(output, fields),
            DescriptorValue::Eight(fields) => DescriptorCodec::<Offset8>::write(output, fields),
        }
    }

    /// Returns the stored number of base elements.
    pub const fn element_count(&self) -> u32 {
        match &self.fields {
            DescriptorValue::Two(fields) => fields.element_count,
            DescriptorValue::Four(fields) => fields.element_count,
            DescriptorValue::Eight(fields) => fields.element_count,
        }
    }

    /// Returns the stored base-relative collection address, including zero or undefined values.
    pub const fn collection_address(&self) -> StoredAddress {
        match &self.fields {
            DescriptorValue::Two(fields) => fields.collection_address.get(),
            DescriptorValue::Four(fields) => fields.collection_address.get(),
            DescriptorValue::Eight(fields) => fields.collection_address.get(),
        }
    }

    /// Returns the full four-byte index without checking membership in a collection.
    pub const fn object_index(&self) -> u32 {
        match &self.fields {
            DescriptorValue::Two(fields) => fields.object_index,
            DescriptorValue::Four(fields) => fields.object_index,
            DescriptorValue::Eight(fields) => fields.object_index,
        }
    }

    /// Returns the width used to read or write the collection address.
    pub const fn offset_width(&self) -> OffsetWidth {
        match self.fields {
            DescriptorValue::Two(_) => OffsetWidth::Two,
            DescriptorValue::Four(_) => OffsetWidth::Four,
            DescriptorValue::Eight(_) => OffsetWidth::Eight,
        }
    }

    /// Returns the descriptor's size at its stored address width.
    pub fn encoded_size(&self) -> usize {
        Self::serialized_size(self.offset_width())
    }
}

impl<'de> Decode<'de> for VariableLengthDescriptor {
    type Context = OffsetWidth;
    type Error = FormatError;

    fn decode(fields: &mut Fields<'de>, offsets: Self::Context) -> Result<Self, Self::Error> {
        Ok(Self {
            fields: match offsets {
                OffsetWidth::Two => DescriptorValue::Two(DescriptorCodec::<Offset2>::read(fields)?),
                OffsetWidth::Four => {
                    DescriptorValue::Four(DescriptorCodec::<Offset4>::read(fields)?)
                }
                OffsetWidth::Eight => {
                    DescriptorValue::Eight(DescriptorCodec::<Offset8>::read(fields)?)
                }
            },
        })
    }
}

impl fmt::Debug for VariableLengthDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { fields } = self;
        match fields {
            DescriptorValue::Eight(fields) => fields.fmt_descriptor(f),
            DescriptorValue::Four(fields) => fields.fmt_descriptor(f),
            DescriptorValue::Two(fields) => fields.fmt_descriptor(f),
        }
    }
}

/// A descriptor with its collection address bounded by the selected width.
#[derive(Clone, Copy, Eq, PartialEq)]
enum DescriptorValue {
    Eight(DescriptorFields<Offset8>),
    Four(DescriptorFields<Offset4>),
    Two(DescriptorFields<Offset2>),
}

// H5T__vlen_disk_write in H5Tvlen.c, HDF5 2.2.0, encodes the count with UINT32ENCODE.
// The global heap ID index occupies four bytes ("Global Heap", format specification, version 4.0).
fixed_record::fixed_record! {
    #[derive(Clone, Copy, Eq, PartialEq)]
    DescriptorCodec<O: OffsetEncoding> => DescriptorWire for DescriptorFields {
        element_count: U32Le,
        collection_address: StoredAddressCodec<O>,
        object_index: U32Le,
    }
}

impl<O: OffsetEncoding> DescriptorFields<O> {
    /// Creates fields with a collection address bounded by `O`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if `collection_address` exceeds
    /// [`OffsetEncoding::MAX`] for `O`.
    fn new(
        element_count: u32,
        collection_address: StoredAddress,
        object_index: u32,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            element_count,
            collection_address: BoundStoredAddress::new(collection_address)?,
            object_index,
        })
    }

    fn fmt_descriptor(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            element_count,
            collection_address,
            object_index,
        } = self;
        f.debug_struct("VariableLengthDescriptor")
            .field("element_count", element_count)
            .field("collection_address", &collection_address.get())
            .field("object_index", object_index)
            .field("offsets", &O::WIDTH)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use core::mem;

    use rstest::rstest;
    use test_util::variable_length_descriptor::Address;

    use super::*;
    use crate::address::StoredAddressExt;

    #[test]
    fn descriptor_record_sizes_fit_flat_field_tuples() {
        assert_eq!(
            mem::size_of::<DescriptorFields<Offset2>>(),
            mem::size_of::<(u32, StoredAddress, u32)>()
        );
        assert_eq!(
            mem::size_of::<DescriptorFields<Offset4>>(),
            mem::size_of::<(u32, StoredAddress, u32)>()
        );
        assert_eq!(
            mem::size_of::<DescriptorFields<Offset8>>(),
            mem::size_of::<(u32, StoredAddress, u32)>()
        );
        assert!(
            mem::size_of::<VariableLengthDescriptor>()
                <= mem::size_of::<(u32, StoredAddress, u32, OffsetWidth)>()
        );
    }

    #[rstest]
    #[case::two(Address::Two(0x1234))]
    #[case::four(Address::Four(0x1234_5678))]
    #[case::eight(Address::Eight(0x1234_5678_9abc_def0))]
    fn decoding_advances_from_the_current_position_and_preserves_following_fields(
        #[case] address: Address,
    ) {
        let offsets = OffsetWidth::try_from(address.width()).unwrap();
        let descriptor = address.descriptor(0, u32::MAX);
        let mut raw = vec![0xab; 3];
        raw.extend_from_slice(&descriptor);
        raw.extend_from_slice(&[0x34, 0x12]);
        let mut fields = Fields::new(&raw, 3);
        assert_eq!(
            VariableLengthDescriptor::decode(&mut fields, offsets).unwrap(),
            VariableLengthDescriptor::new(offsets, 0, StoredAddress::new(address.get()), u32::MAX)
                .unwrap()
        );
        assert_eq!(fields.pos(), 3 + descriptor.len());
        assert_eq!(fields.u16().unwrap(), 0x1234);
        assert_eq!(fields.pos(), raw.len());
    }
}
