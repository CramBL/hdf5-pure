//! Checked version 1 Link message fields and their bounded encoding.

use alloc::vec;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt;
use core::ops::Range;

use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_message::{EXTERNAL_LINK, HARD_LINK, LinkMessage, SOFT_LINK};
use crate::link_message_read::{
    ASCII, CHARSET_PRESENT, CREATION_ORDER_LEN, CREATION_ORDER_PRESENT, EXTERNAL_VERSION_FLAGS,
    LinkMessageView, LinkTargetView, PREFIX_LEN, TYPE_PRESENT, UTF8, VERSION,
};
use crate::link_message_v1::{self, LinkMessageV1, LinkName};
use crate::link_storage::{LinkStorageError, OwnedLink};
use crate::width::{OffsetWidth, UintWidth};

/// A version 1 Link body ready to write at a checked offset width.
///
/// The preparation borrows the name and any symbolic or user-defined target bytes. It holds
/// the exact body length and the encoding choices selected by the source message.
/// "The Link Message" of the [format specification, version 4.0][spec] defines the fields.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Debug)]
pub struct PreparedLink<'a> {
    fields: LinkFields<'a>,
    offset_width: OffsetWidth,
    len: usize,
}

impl<'a> PreparedLink<'a> {
    pub(crate) fn for_wire(
        link: &'a LinkMessageV1,
        offset_width: OffsetWidth,
    ) -> Result<Self, LinkStorageError> {
        Self::from_checked(
            offset_width,
            LinkFields {
                target: link.target().as_view(),
                value_len: link.target_value_len(),
                creation_order: link.creation_order(),
                charset: link.charset(),
                name: link.name().as_bytes(),
                name_width: link.name().width(),
            },
        )
        .map_err(encoding_error)
    }

    pub(crate) fn for_owned(
        link: &'a OwnedLink,
        offset_width: OffsetWidth,
    ) -> Result<Self, LinkStorageError> {
        Self::from_checked(
            offset_width,
            LinkFields {
                target: link.target_view(),
                value_len: link.target_value_len(),
                creation_order: link.creation_order(),
                charset: link.stored_charset(),
                name: link.name(),
                name_width: link.name_width(),
            },
        )
        .map_err(encoding_error)
    }

    pub(crate) fn for_view(
        link: &LinkMessageView<'a>,
        offset_width: OffsetWidth,
    ) -> Result<Self, LinkStorageError> {
        Self::from_checked(
            offset_width,
            LinkFields {
                target: link.target(),
                value_len: link.target_value_len(),
                creation_order: link.creation_order(),
                charset: link.charset(),
                name: link.name_bytes(),
                name_width: link.name_width(),
            },
        )
        .map_err(encoding_error)
    }

    pub(crate) fn for_text(
        link: &'a LinkMessage,
        offset_width: OffsetWidth,
    ) -> Result<Self, FormatError> {
        Self::from_text_fields(
            offset_width,
            &link.name,
            link.creation_order,
            canonical_charset(&link.charset),
            link.link_target.as_view(),
        )
    }

    /// Prepares a hard Link body from a borrowed name and a defined target address.
    ///
    /// Uses the smallest name-length width, omits the hard-link type field, and omits the
    /// character-set field for [`CharacterSet::Ascii`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLinkName`] for an invalid name component,
    /// [`FormatError::AddressTooLarge`] for an address wider than `offset_width`,
    /// [`FormatError::DefinedAddressIsUndefined`] for that width's undefined address, or
    /// [`FormatError::LinkMessageLengthOverflow`] if the complete length exceeds [`usize::MAX`].
    pub fn hard(
        offset_width: OffsetWidth,
        name: &'a str,
        object_header_address: StoredAddress,
        creation_order: Option<u64>,
        charset: CharacterSet,
    ) -> Result<Self, FormatError> {
        Self::from_text_fields(
            offset_width,
            name,
            creation_order,
            canonical_charset(&charset),
            LinkTargetView::Hard {
                object_header_address,
                explicit_type: false,
            },
        )
    }

    /// Returns the exact body length, excluding object header padding.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the body to the start of `output` and returns the number of bytes written.
    ///
    /// Leaves bytes beyond the body unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`LinkOutputTooSmall`] if `output` is shorter than [`Self::encoded_size`].
    /// Leaves all of `output` unchanged on error.
    ///
    /// # Examples
    ///
    /// Writes a hard link into a buffer with one extra byte.
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::PreparedLink;
    /// use hdf5_pure_format::{CharacterSet, OffsetWidth};
    ///
    /// let link = PreparedLink::hard(
    ///     OffsetWidth::Two,
    ///     "alias",
    ///     StoredAddress::new(0x1234),
    ///     None,
    ///     CharacterSet::Ascii,
    /// )?;
    /// let mut output = [0xa5; 11];
    /// assert_eq!(link.write_into(&mut output)?, link.encoded_size());
    /// assert_eq!(&output, b"\x01\x00\x05alias\x34\x12\xa5");
    /// assert_eq!(link.hard_link_address_range(), Some(8..10));
    /// # Ok::<(), Box<dyn core::error::Error>>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, LinkOutputTooSmall> {
        self.encode_into(output)
    }

    /// Returns the encoded body in an owned buffer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = vec![0; self.encoded_size()];
        self.write_body(&mut output);
        output
    }

    /// Returns the hard-link address range relative to the start of the encoded body.
    ///
    /// Returns [`None`] for a symbolic or user-defined target.
    pub fn hard_link_address_range(&self) -> Option<Range<usize>> {
        self.hard_offset_width()
            .map(|width| self.len - usize::from(width.get())..self.len)
    }

    pub(crate) fn hard_offset_width(&self) -> Option<OffsetWidth> {
        matches!(self.fields.target, LinkTargetView::Hard { .. }).then_some(self.offset_width)
    }

    pub(crate) fn name_bytes(&self) -> &'a [u8] {
        self.fields.name
    }

    pub(crate) fn creation_order(&self) -> Option<u64> {
        self.fields.creation_order
    }

    pub(crate) fn charset(&self) -> Option<CharacterSet> {
        self.fields.charset.clone()
    }

    pub(crate) fn target(&self) -> LinkTargetView<'a> {
        self.fields.target
    }

    pub(crate) fn name_width(&self) -> UintWidth {
        self.fields.name_width
    }

    pub(crate) fn target_value_len(&self) -> u16 {
        self.fields.value_len
    }

    /// Writes the body into a slice of exactly [`Self::encoded_size`] bytes.
    ///
    /// # Panics
    ///
    /// Panics if `output.len()` differs from [`Self::encoded_size`].
    pub(crate) fn write_body(&self, output: &mut [u8]) {
        let LinkFields {
            target,
            value_len,
            creation_order,
            ref charset,
            name,
            name_width,
        } = self.fields;
        let kind = target_type(target);
        let mut flags = name_width.flag_bits();
        if kind.is_some() {
            flags |= TYPE_PRESENT;
        }
        if creation_order.is_some() {
            flags |= CREATION_ORDER_PRESENT;
        }
        if charset.is_some() {
            flags |= CHARSET_PRESENT;
        }
        let mut output = write_field(output, &[VERSION, flags]);
        if let Some(kind) = kind {
            output = write_field(output, &[kind]);
        }
        if let Some(order) = creation_order {
            output = write_field(output, &order.to_le_bytes());
        }
        if let Some(charset) = charset {
            output = write_field(
                output,
                &[match charset {
                    CharacterSet::Ascii => ASCII,
                    CharacterSet::Utf8 => UTF8,
                }],
            );
        }
        output = write_field(
            output,
            &name.len().to_u64().to_le_bytes()[..usize::from(name_width.get())],
        );
        output = write_field(output, name);
        if !matches!(target, LinkTargetView::Hard { .. }) {
            output = write_field(output, &value_len.to_le_bytes());
        }
        match target {
            LinkTargetView::Hard {
                object_header_address,
                ..
            } => output.copy_from_slice(
                &object_header_address.get().to_le_bytes()[..usize::from(self.offset_width.get())],
            ),
            LinkTargetView::Soft { target_path } => output.copy_from_slice(target_path),
            LinkTargetView::External {
                filename,
                object_path,
            } => {
                let output = write_field(output, &[EXTERNAL_VERSION_FLAGS]);
                let output = write_field(output, filename);
                let output = write_field(output, &[0]);
                let output = write_field(output, object_path);
                output.copy_from_slice(&[0]);
            }
            LinkTargetView::UserDefined { value, .. } => output.copy_from_slice(value),
        }
    }

    fn from_text_fields(
        offset_width: OffsetWidth,
        name: &'a str,
        creation_order: Option<u64>,
        charset: Option<CharacterSet>,
        target: LinkTargetView<'a>,
    ) -> Result<Self, FormatError> {
        let name_bytes = name.as_bytes();
        if !LinkName::is_valid(name_bytes) {
            return Err(FormatError::InvalidLinkName(name.into()));
        }
        let value_len = checked_value_len(target).map_err(|error| match error {
            ValueLengthError::InvalidPath => FormatError::InvalidLinkTarget,
            ValueLengthError::Overflow => FormatError::LinkMessageLengthOverflow,
            ValueLengthError::TooLong(len) => FormatError::LengthTooLarge {
                length: len.to_u64(),
                length_size: VALUE_LENGTH_LEN,
            },
        })?;
        Self::from_checked(
            offset_width,
            LinkFields {
                target,
                value_len,
                creation_order,
                charset,
                name: name_bytes,
                name_width: UintWidth::smallest_for_len(name_bytes.len()),
            },
        )
    }

    /// Checks the hard address and computes the exact length from checked fields.
    ///
    /// The caller checks the name component and length width, and supplies the target's
    /// checked value length for a symbolic or user-defined target.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`DecodedStoredAddress::to_stored`] for a hard address or
    /// [`FormatError::LinkMessageLengthOverflow`] if the complete length exceeds [`usize::MAX`].
    fn from_checked(
        offset_width: OffsetWidth,
        fields: LinkFields<'a>,
    ) -> Result<Self, FormatError> {
        let target_len = if let LinkTargetView::Hard {
            object_header_address,
            ..
        } = fields.target
        {
            DecodedStoredAddress::Defined(object_header_address).to_stored(offset_width)?;
            usize::from(offset_width.get())
        } else {
            usize::from(VALUE_LENGTH_LEN) + usize::from(fields.value_len)
        };
        let framing = PREFIX_LEN
            + usize::from(target_type(fields.target).is_some())
            + usize::from(fields.creation_order.is_some()) * CREATION_ORDER_LEN
            + usize::from(fields.charset.is_some())
            + usize::from(fields.name_width.get());
        let len = fields
            .name
            .len()
            .checked_add(framing + target_len)
            .ok_or(FormatError::LinkMessageLengthOverflow)?;
        Ok(Self {
            fields,
            offset_width,
            len,
        })
    }
}

impl Encode for PreparedLink<'_> {
    type Error = LinkOutputTooSmall;

    fn encoded_len(&self) -> usize {
        self.len
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let available = output.len();
        let body = output.get_mut(..self.len).ok_or(LinkOutputTooSmall {
            required: self.len,
            available,
        })?;
        self.write_body(body);
        Ok(self.len)
    }
}

/// The destination is shorter than the prepared Link message body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkOutputTooSmall {
    /// The length of the body in bytes.
    pub required: usize,
    /// The length of the destination in bytes.
    pub available: usize,
}

impl fmt::Display for LinkOutputTooSmall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Link encoding needs {} bytes, destination has {}",
            self.required, self.available,
        )
    }
}

impl Error for LinkOutputTooSmall {}

#[derive(Debug)]
struct LinkFields<'a> {
    target: LinkTargetView<'a>,
    value_len: u16,
    creation_order: Option<u64>,
    charset: Option<CharacterSet>,
    name: &'a [u8],
    name_width: UintWidth,
}

fn target_type(target: LinkTargetView<'_>) -> Option<u8> {
    match target {
        LinkTargetView::Hard { explicit_type, .. } => explicit_type.then_some(HARD_LINK),
        LinkTargetView::Soft { .. } => Some(SOFT_LINK),
        LinkTargetView::External { .. } => Some(EXTERNAL_LINK),
        LinkTargetView::UserDefined { kind, .. } => Some(kind.get()),
    }
}

/// Checks a target and returns the length of its value, excluding the length field.
///
/// Checks a hard address at eight bytes and returns a value length of zero.
///
/// # Errors
///
/// Returns [`LinkStorageError::UndefinedAddress`] for an eight-byte undefined hard address,
/// [`LinkStorageError::InvalidTarget`] for an empty path or one containing a NUL,
/// [`LinkStorageError::TargetLength`] if the value exceeds [`u16::MAX`] bytes, or
/// [`LinkStorageError::LengthOverflow`] if computing the external value length exceeds
/// [`usize::MAX`].
pub(crate) fn initial_value_len(target: LinkTargetView<'_>) -> Result<u16, LinkStorageError> {
    if let LinkTargetView::Hard {
        object_header_address,
        ..
    } = target
    {
        DecodedStoredAddress::Defined(object_header_address)
            .to_stored(OffsetWidth::Eight)
            .map_err(encoding_error)?;
    }
    checked_value_len(target).map_err(|error| match error {
        ValueLengthError::InvalidPath => LinkStorageError::InvalidTarget,
        ValueLengthError::Overflow => LinkStorageError::LengthOverflow,
        ValueLengthError::TooLong(len) => LinkStorageError::TargetLength(len),
    })
}

fn encoding_error(error: FormatError) -> LinkStorageError {
    match error {
        FormatError::DefinedAddressIsUndefined { .. } => LinkStorageError::UndefinedAddress,
        FormatError::LinkMessageLengthOverflow => LinkStorageError::LengthOverflow,
        error => LinkStorageError::Format(error),
    }
}

enum ValueLengthError {
    InvalidPath,
    Overflow,
    TooLong(usize),
}

fn checked_value_len(target: LinkTargetView<'_>) -> Result<u16, ValueLengthError> {
    let len = match target {
        LinkTargetView::Hard { .. } => return Ok(0),
        LinkTargetView::Soft { target_path } => {
            if !link_message_v1::is_symbolic_path(target_path) {
                return Err(ValueLengthError::InvalidPath);
            }
            target_path.len()
        }
        LinkTargetView::External {
            filename,
            object_path,
        } => {
            if !link_message_v1::is_symbolic_path(filename)
                || !link_message_v1::is_symbolic_path(object_path)
            {
                return Err(ValueLengthError::InvalidPath);
            }
            filename
                .len()
                .checked_add(object_path.len())
                .and_then(|len| len.checked_add(EXTERNAL_FRAMING_LEN))
                .ok_or(ValueLengthError::Overflow)?
        }
        LinkTargetView::UserDefined { value, .. } => value.len(),
    };
    u16::try_from(len).map_err(|_error| ValueLengthError::TooLong(len))
}

fn write_field<'a>(output: &'a mut [u8], bytes: &[u8]) -> &'a mut [u8] {
    let (field, rest) = output.split_at_mut(bytes.len());
    field.copy_from_slice(bytes);
    rest
}

fn canonical_charset(charset: &CharacterSet) -> Option<CharacterSet> {
    (*charset != CharacterSet::Ascii).then(|| charset.clone())
}

// "The Link Message" of the format specification, version 4.0, defines these extents.
const VALUE_LENGTH_LEN: u8 = 2;
const EXTERNAL_FRAMING_LEN: usize = 3;
