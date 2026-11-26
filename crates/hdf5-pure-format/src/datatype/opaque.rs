use crate::bytes::ByteCursor;
use crate::error::FormatError;

/// An opaque tag with a checked aligned property length and borrowed logical bytes.
///
/// The parser accepts a full property without a terminator. The encoder reserves a terminator
/// for every nonempty tag.
pub(super) struct OpaqueTagWire<'a> {
    encoded_len: u8,
    logical: &'a [u8],
}

impl<'a> OpaqueTagWire<'a> {
    /// Borrows the logical tag bytes from a complete aligned property.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOpaqueTagLength`] if the declared length is not a multiple
    /// of eight. Returns [`FormatError::UnexpectedEof`] if the property is incomplete, with
    /// byte counts relative to the cursor's original input. Returns
    /// [`FormatError::InvalidOpaqueTagPadding`] for a nonzero byte after the first NUL, with
    /// the byte's offset relative to the start of the property.
    pub(super) fn parse(encoded_len: u8, cursor: &mut ByteCursor<'a>) -> Result<Self, FormatError> {
        if !encoded_len.is_multiple_of(OPAQUE_TAG_ALIGNMENT) {
            return Err(FormatError::InvalidOpaqueTagLength {
                length: encoded_len,
            });
        }
        let property = cursor.take(usize::from(encoded_len))?;
        // `H5O__dtype_encode_helper` fills aligned tags without a terminator
        // (`H5Odtype.c`, HDF5 2.2.0).
        let terminator = property.iter().position(|&byte| byte == 0);
        let logical = if let Some(terminator) = terminator {
            if let Some(offset) = property[terminator + 1..]
                .iter()
                .position(|&byte| byte != 0)
            {
                let offset = terminator + 1 + offset;
                return Err(FormatError::InvalidOpaqueTagPadding {
                    offset,
                    byte: property[offset],
                });
            }
            &property[..terminator]
        } else {
            property
        };
        Ok(Self {
            encoded_len,
            logical,
        })
    }

    /// Determines the aligned property length for a tag, including its terminator when nonempty.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OpaqueTagTooLong`] if the tag exceeds 247 bytes, or
    /// [`FormatError::InvalidOpaqueTagByte`] for a NUL byte in the tag.
    pub(super) fn encode(logical: &'a [u8]) -> Result<Self, FormatError> {
        if logical.len() > OPAQUE_TAG_MAX_LEN {
            return Err(FormatError::OpaqueTagTooLong {
                length: logical.len(),
            });
        }
        if let Some(offset) = logical.iter().position(|&byte| byte == 0) {
            return Err(FormatError::InvalidOpaqueTagByte {
                offset,
                byte: logical[offset],
            });
        }
        let encoded_len = u8::try_from(Self::property_size(logical.len())).map_err(|_error| {
            FormatError::OpaqueTagTooLong {
                length: logical.len(),
            }
        })?;
        Ok(Self {
            encoded_len,
            logical,
        })
    }

    /// Returns the complete property length in bytes, including any terminator and padding.
    pub(super) fn encoded_len(&self) -> u8 {
        self.encoded_len
    }

    /// Borrows the tag bytes before any terminator and padding.
    pub(super) fn logical(&self) -> &'a [u8] {
        self.logical
    }

    /// Rounds a nonempty tag's length including its terminator up to a multiple of eight.
    ///
    /// Returns zero for an empty tag. Pass a logical tag length of at most 247 bytes.
    pub(super) fn property_size(tag_len: usize) -> usize {
        if tag_len == 0 {
            0
        } else {
            let alignment = usize::from(OPAQUE_TAG_ALIGNMENT);
            (tag_len + alignment) & !(alignment - 1)
        }
    }
}

/// The alignment of an opaque tag property, defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// `H5O__dtype_decode_helper` requires an aligned recorded length (`H5Odtype.c`, HDF5 2.2.0).
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const OPAQUE_TAG_ALIGNMENT: u8 = 8;
/// The maximum logical tag length with space for its terminator in an aligned one-byte length.
///
/// The length field and padding are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const OPAQUE_TAG_MAX_LEN: usize = 247;
