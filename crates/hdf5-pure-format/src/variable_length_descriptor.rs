//! Stored counts and global heap IDs for variable-length strings and sequences.

use hdf5_pure_core::StoredAddress;

use crate::address::StoredAddressExt;
use crate::bytes;
use crate::bytes::ByteCursor;
use crate::bytes::UnexpectedEof;
use crate::codec::Encode;
use crate::error::FormatError;
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariableLengthDescriptor {
    element_count: u32,
    collection_address: StoredAddress,
    object_index: u32,
    offsets: OffsetWidth,
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
        let mut cursor = ByteCursor::new(raw);
        let descriptor = match offsets {
            OffsetWidth::Two => Self::read_fields(cursor.take_array::<10>()?, offsets)?,
            OffsetWidth::Four => Self::read_fields(cursor.take_array::<12>()?, offsets)?,
            OffsetWidth::Eight => Self::read_fields(cursor.take_array::<16>()?, offsets)?,
        };
        Ok((descriptor, cursor.position()))
    }

    /// Reads the count, stored address, and index in wire order.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if a field is incomplete, relative to `record`.
    #[inline]
    fn read_fields(record: &[u8], offsets: OffsetWidth) -> Result<Self, UnexpectedEof> {
        let mut cursor = ByteCursor::new(record);
        Ok(Self {
            element_count: u32::from_le_bytes(*cursor.take_array()?),
            collection_address: StoredAddress::new(match offsets {
                OffsetWidth::Two => u64::from(u16::from_le_bytes(*cursor.take_array()?)),
                OffsetWidth::Four => u64::from(u32::from_le_bytes(*cursor.take_array()?)),
                OffsetWidth::Eight => u64::from_le_bytes(*cursor.take_array()?),
            }),
            object_index: u32::from_le_bytes(*cursor.take_array()?),
            offsets,
        })
    }

    /// Returns eight bytes plus the file's address width.
    ///
    /// The file's length width does not affect descriptor size.
    pub fn serialized_size(offsets: OffsetWidth) -> usize {
        ELEMENT_COUNT_BYTES + usize::from(offsets.get()) + OBJECT_INDEX_BYTES
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
        if collection_address > StoredAddress::undefined(offsets) {
            return Err(FormatError::AddressTooLarge {
                address: collection_address.get(),
                offset_size: offsets.get(),
            });
        }
        Ok(Self {
            element_count,
            collection_address,
            object_index,
            offsets,
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
        Ok(self.encode_into(output)?)
    }

    /// Returns the stored number of base elements.
    pub const fn element_count(&self) -> u32 {
        self.element_count
    }

    /// Returns the stored base-relative collection address, including zero or undefined values.
    pub const fn collection_address(&self) -> StoredAddress {
        self.collection_address
    }

    /// Returns the full four-byte index without checking membership in a collection.
    pub const fn object_index(&self) -> u32 {
        self.object_index
    }

    /// Returns the width used to read or write the collection address.
    pub const fn offset_width(&self) -> OffsetWidth {
        self.offsets
    }

    /// Returns the descriptor's size at its stored address width.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }
}

impl Encode for VariableLengthDescriptor {
    type Error = UnexpectedEof;

    /// Returns eight bytes plus the retained address width.
    #[inline]
    fn encoded_len(&self) -> usize {
        Self::serialized_size(self.offsets)
    }

    /// Acquires the complete destination and each field before writing the descriptor.
    ///
    /// Writes little-endian fields, preserves the output suffix, and returns the bytes written.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if `output` is shorter than [`Self::encoded_size`],
    /// leaving the entire output unchanged.
    #[inline]
    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let available = output.len();
        let Self {
            element_count,
            collection_address,
            object_index,
            offsets,
        } = *self;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the address fits the width retained by the descriptor"
        )]
        match offsets {
            OffsetWidth::Two => {
                let record = bytes::ByteCursorMut::new(output).take_array::<10>()?;
                let Some((count_bytes, rest)) =
                    record.split_first_chunk_mut::<ELEMENT_COUNT_BYTES>()
                else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                let Some((address_bytes, rest)) = rest.split_first_chunk_mut::<2>() else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                let Some(index_bytes) = rest.first_chunk_mut::<OBJECT_INDEX_BYTES>() else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                *count_bytes = element_count.to_le_bytes();
                *address_bytes = (collection_address.get() as u16).to_le_bytes();
                *index_bytes = object_index.to_le_bytes();
            }
            OffsetWidth::Four => {
                let record = bytes::ByteCursorMut::new(output).take_array::<12>()?;
                let Some((count_bytes, rest)) =
                    record.split_first_chunk_mut::<ELEMENT_COUNT_BYTES>()
                else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                let Some((address_bytes, rest)) = rest.split_first_chunk_mut::<4>() else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                let Some(index_bytes) = rest.first_chunk_mut::<OBJECT_INDEX_BYTES>() else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                *count_bytes = element_count.to_le_bytes();
                *address_bytes = (collection_address.get() as u32).to_le_bytes();
                *index_bytes = object_index.to_le_bytes();
            }
            OffsetWidth::Eight => {
                let record = bytes::ByteCursorMut::new(output).take_array::<16>()?;
                let Some((count_bytes, rest)) =
                    record.split_first_chunk_mut::<ELEMENT_COUNT_BYTES>()
                else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                let Some((address_bytes, rest)) = rest.split_first_chunk_mut::<8>() else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                let Some(index_bytes) = rest.first_chunk_mut::<OBJECT_INDEX_BYTES>() else {
                    return Err(UnexpectedEof::new(
                        Self::serialized_size(offsets),
                        available,
                    ));
                };
                *count_bytes = element_count.to_le_bytes();
                *address_bytes = collection_address.get().to_le_bytes();
                *index_bytes = object_index.to_le_bytes();
            }
        }
        Ok(self.encoded_len())
    }
}

// `H5T__vlen_disk_write` in `H5Tvlen.c`, HDF5 2.2.0, encodes the count with `UINT32ENCODE`.
const ELEMENT_COUNT_BYTES: usize = 4;

// `H5VL__native_blob_put` in `H5VLnative_blob.c`, HDF5 2.2.0, encodes the object index with
// `UINT32ENCODE`.
const OBJECT_INDEX_BYTES: usize = 4;

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest::test_runner::FileFailurePersistence;
    use test_util::variable_length_descriptor::Address;

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128,
            failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
                env!("CARGO_MANIFEST_DIR"), "/tests/proptest-regressions/variable_length_descriptor.txt"
            )))),
            ..ProptestConfig::default()
        })]

        #[test]
        fn descriptor_values_match_independent_bytes_and_preserve_buffers(
            address in prop_oneof![
                any::<u16>().prop_map(Address::Two),
                any::<u32>().prop_map(Address::Four),
                any::<u64>().prop_map(Address::Eight),
            ],
            count in any::<u32>(),
            index in any::<u32>(),
            prefix in proptest::collection::vec(any::<u8>(), 0..16),
            suffix in proptest::collection::vec(any::<u8>(), 0..16),
            fill in any::<u8>(),
            available in 0usize..24,
        ) {
            let offsets = OffsetWidth::try_from(address.width()).unwrap();
            let expected = address.descriptor(count, index);
            let input = [prefix.as_slice(), expected.as_slice(), suffix.as_slice()].concat();
            let (value, consumed) = VariableLengthDescriptor::parse(offsets, &input[prefix.len()..]).unwrap();
            prop_assert_eq!(consumed, expected.len());
            prop_assert_eq!(value.element_count(), count);
            prop_assert_eq!(value.collection_address(), StoredAddress::new(address.get()));
            prop_assert_eq!(value.object_index(), index);
            prop_assert_eq!(value.offset_width(), offsets);
            prop_assert_eq!(value, VariableLengthDescriptor::new(offsets, count, StoredAddress::new(address.get()), index).unwrap());
            let mut output = [prefix.as_slice(), vec![fill; available].as_slice(), suffix.as_slice()].concat();
            let original = output.clone();
            prop_assert_eq!(value.encoded_len(), expected.len());
            let written = value.encode_into(&mut output[prefix.len()..prefix.len() + available]);
            if available < expected.len() {
                prop_assert_eq!(written, Err(UnexpectedEof { expected: expected.len(), available }));
                prop_assert_eq!(output, original);
            } else {
                prop_assert_eq!(written.unwrap(), expected.len());
                prop_assert_eq!(&output[..prefix.len()], prefix.as_slice());
                prop_assert_eq!(&output[prefix.len()..prefix.len() + expected.len()], &expected);
                prop_assert_eq!(&output[prefix.len() + expected.len()..], &original[prefix.len() + expected.len()..]);
            }
        }

    }
}
