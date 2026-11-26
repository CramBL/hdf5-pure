use super::frame::FractalHeapIdType;
use crate::bytes;
use crate::codec::Encode;
use crate::error::FormatError;

/// A checked version 0 tiny fractal-heap ID encoding that borrows the object's bytes.
///
/// Both forms store the object length minus one. At ID widths up to 18 bytes, the normal
/// form stores it in the first byte's low four bits. At larger ID widths, the extended
/// form stores the high four bits there and the low eight bits in the second byte.
/// The object immediately follows the prefix.
///
/// The forms are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug)]
pub struct TinyHeapIdEncoding<'a> {
    payload: &'a [u8],
    id_width: u16,
}

impl<'a> TinyHeapIdEncoding<'a> {
    /// Prepares a tiny ID at the heap's declared ID width using a borrowed payload.
    ///
    /// The normal form can represent up to 16 payload bytes. The extended form can
    /// represent up to 4096. The payload must also fit the bytes after the prefix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TinyHeapIdWidthTooSmall`] if `id_width` is 0 or 1,
    /// [`FormatError::EmptyTinyHeapIdPayload`] if `payload` is empty,
    /// [`FormatError::TinyHeapIdPayloadLengthTooLarge`] if the length exceeds the form's
    /// length field limit, or [`FormatError::TinyHeapIdPayloadExceedsCapacity`] if the
    /// payload exceeds the space after the prefix. Checks these conditions in that order.
    pub fn new(id_width: u16, payload: &'a [u8]) -> Result<Self, FormatError> {
        let width = usize::from(id_width);
        let form = TinyIdForm::for_width(width);
        let capacity = width.saturating_sub(form.prefix_len());
        if capacity == 0 {
            return Err(FormatError::TinyHeapIdWidthTooSmall(id_width));
        }
        let length = payload.len();
        if length == 0 {
            return Err(FormatError::EmptyTinyHeapIdPayload);
        }
        let maximum = form.max_payload_len();
        if length > maximum {
            return Err(FormatError::TinyHeapIdPayloadLengthTooLarge { length, maximum });
        }
        if length > capacity {
            return Err(FormatError::TinyHeapIdPayloadExceedsCapacity { length, capacity });
        }
        Ok(Self { payload, id_width })
    }

    /// Returns the declared ID width in bytes, including padding after the object.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes one complete tiny ID to the start of `output` and returns its byte count.
    ///
    /// Writes zeros after the payload through the declared ID width and leaves any
    /// output suffix unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] with the required ID width and available
    /// bytes if `output` is too short. Leaves all of `output` unchanged on error.
    ///
    /// # Examples
    ///
    /// A 19-byte ID uses the extended form for a two-byte payload.
    ///
    /// ```
    /// use hdf5_pure_format::__private::TinyHeapIdEncoding;
    ///
    /// let id = TinyHeapIdEncoding::new(19, b"hi")?;
    /// let mut output = [0xa5; 20];
    /// assert_eq!(id.encoded_size(), 19);
    /// assert_eq!(id.write_into(&mut output)?, 19);
    /// assert_eq!(&output[..4], &[0x20, 1, b'h', b'i']);
    /// assert_eq!(&output[4..19], &[0; 15]);
    /// assert_eq!(output[19], 0xa5);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }
}

impl Encode for TinyHeapIdEncoding<'_> {
    type Error = bytes::UnexpectedEof;

    fn encoded_len(&self) -> usize {
        usize::from(self.id_width)
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let width = self.encoded_len();
        let available = output.len();
        let id = output
            .get_mut(..width)
            .ok_or_else(|| bytes::UnexpectedEof::new(width, available))?;
        let form = TinyIdForm::for_width(width);
        let (prefix, body) = id.split_at_mut(form.prefix_len());
        let (payload, padding) = body.split_at_mut(self.payload.len());
        form.write_prefix(prefix, self.payload.len());
        payload.copy_from_slice(self.payload);
        // `H5HF__tiny_insert` in `H5HFtiny.c`, HDF5 2.2.0, fills the remaining ID bytes with zeros.
        padding.fill(0);
        Ok(width)
    }
}

pub(super) enum TinyIdForm {
    Extended,
    Normal,
}

impl TinyIdForm {
    pub(super) fn for_width(width: usize) -> Self {
        if width <= TINY_NORMAL_ID_MAX_LEN {
            Self::Normal
        } else {
            Self::Extended
        }
    }

    fn prefix_len(&self) -> usize {
        match self {
            Self::Normal => 1,
            Self::Extended => 2,
        }
    }

    fn max_payload_len(&self) -> usize {
        match self {
            Self::Normal => TINY_NORMAL_PAYLOAD_MAX_LEN,
            Self::Extended => TINY_EXTENDED_PAYLOAD_MAX_LEN,
        }
    }

    /// Reads the object with `cursor` positioned immediately after `first` in the ID.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if the extended length byte or the object is incomplete.
    pub(super) fn parse<'a>(
        &self,
        first: u8,
        cursor: &mut bytes::ByteCursor<'a>,
    ) -> Result<&'a [u8], bytes::UnexpectedEof> {
        let length_bits = usize::from(first & TINY_LENGTH_MASK);
        let encoded_length = match self {
            Self::Normal => length_bits,
            Self::Extended => (length_bits << u8::BITS) | usize::from(cursor.u8()?),
        };
        cursor.take(encoded_length + 1)
    }

    /// Writes the version, type, and length into exactly [`Self::prefix_len`] bytes.
    ///
    /// The caller checks that `payload_len` is positive and fits the form's length field.
    fn write_prefix(&self, prefix: &mut [u8], payload_len: usize) {
        let [low, high, ..] = (payload_len - 1).to_le_bytes();
        let tag = FractalHeapIdType::Tiny.first_byte();
        match self {
            Self::Normal => prefix.copy_from_slice(&[tag | low]),
            Self::Extended => prefix.copy_from_slice(&[tag | high, low]),
        }
    }
}

/// The largest heap ID width in bytes that uses the normal tiny ID form.
///
/// The normal form represents objects of up to 16 bytes, as defined in "Fractal Heap" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const TINY_NORMAL_ID_MAX_LEN: usize = 18;

// The low four bits hold the length less one in the normal form and its high four bits in the
// extended form ("Fractal Heap", specification 4.0).
const TINY_LENGTH_MASK: u8 = 0x0F;
const TINY_NORMAL_PAYLOAD_MAX_LEN: usize = TINY_LENGTH_MASK as usize + 1;
const TINY_EXTENDED_PAYLOAD_MAX_LEN: usize = TINY_NORMAL_PAYLOAD_MAX_LEN << u8::BITS;
