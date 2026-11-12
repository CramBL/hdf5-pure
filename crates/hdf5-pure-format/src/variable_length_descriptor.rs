//! Stored counts and global heap IDs for variable-length strings and sequences.

use hdf5_pure_core::StoredAddress;

use crate::bytes;
use crate::bytes::Fields;
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
        let size = Self::serialized_size(offsets);
        bytes::ensure_len(raw, 0, size)?;
        let mut fields = Fields::new(raw, 0);
        let element_count = fields.u32()?;
        let collection_address = fields.address(offsets)?;
        let object_index = fields.u32()?;
        Ok((
            Self {
                element_count,
                collection_address,
                object_index,
                offsets,
            },
            size,
        ))
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
    pub fn new(
        offsets: OffsetWidth,
        element_count: u32,
        collection_address: StoredAddress,
        object_index: u32,
    ) -> Result<Self, FormatError> {
        if collection_address.get() > StoredAddress::undefined(offsets).get() {
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
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        let Self {
            element_count,
            collection_address,
            object_index,
            offsets,
        } = *self;
        let size = self.encoded_size();
        bytes::ensure_len(output, 0, size)?;
        let index_at = ELEMENT_COUNT_BYTES + usize::from(offsets.get());
        output[..ELEMENT_COUNT_BYTES].copy_from_slice(&element_count.to_le_bytes());
        output[ELEMENT_COUNT_BYTES..index_at]
            .copy_from_slice(&collection_address.get().to_le_bytes()[..usize::from(offsets.get())]);
        output[index_at..size].copy_from_slice(&object_index.to_le_bytes());
        Ok(size)
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
        Self::serialized_size(self.offsets)
    }
}

// H5T__vlen_disk_write in H5Tvlen.c, HDF5 2.2.0, encodes the count with UINT32ENCODE.
const ELEMENT_COUNT_BYTES: usize = 4;

// The global heap ID index occupies four bytes ("Global Heap", format specification, version 4.0).
const OBJECT_INDEX_BYTES: usize = 4;
