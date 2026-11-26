//! Borrowed fields and target framing for version 1 Link messages.

use alloc::string::String;
use core::ops::Range;

use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::codec::Decode;
use crate::convert::Narrow;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_message::{EXTERNAL_LINK, HARD_LINK, LinkMessage, LinkTarget, SOFT_LINK};
use crate::link_message_v1::{self, LinkName, LinkProjectionError, LinkWireError, UserLinkKind};
use crate::link_message_write::PreparedLink;
use crate::width::{OffsetWidth, UintWidth};

/// The checked fields of a version 1 Link message, borrowed from the message body.
///
/// The view retains the name-length width, optional character set and creation order, and
/// whether a hard link stores its type explicitly. Names and target paths retain their raw bytes.
/// "The Link Message" of the [format specification, version 4.0][spec] defines the fields.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Clone, Debug, PartialEq)]
pub struct LinkMessageView<'a> {
    target: LinkTargetView<'a>,
    target_value_len: u16,
    creation_order: Option<u64>,
    charset: Option<CharacterSet>,
    name: &'a [u8],
    name_width: UintWidth,
    target_start: usize,
    consumed: usize,
}

impl<'a> LinkMessageView<'a> {
    /// Borrows the fields of an exact Link body, excluding object header padding.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::parse_prefix`] converted to [`LinkWireError`], or
    /// [`LinkWireError::TrailingBytes`] if bytes
    /// follow the target.
    pub fn parse(data: &'a [u8], offset_width: OffsetWidth) -> Result<Self, LinkWireError> {
        let link = Self::decode(data, offset_width)?;
        if link.consumed() != data.len() {
            return Err(LinkWireError::TrailingBytes);
        }
        Ok(link)
    }

    /// Reads the complete Link message at the start of `data`.
    ///
    /// Borrows the name and any target bytes. Ignores bytes after the target, including
    /// object header padding. Use [`Self::consumed`] to find the end of the message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] for missing bytes, with extents measured from
    /// the start of `data`. If a fixed field selected by the flags is incomplete, the required
    /// extent reaches the end of the name-length field.
    ///
    /// Reads both prefix bytes before checking the version and flags. Returns
    /// [`FormatError::InvalidLinkVersion`] for a version other than 1 or
    /// [`FormatError::InvalidLinkFlags`] for reserved flag bits. Checks the character set
    /// after reading the name length and before reading the name bytes. Returns
    /// [`FormatError::InvalidCharacterSet`] for an unknown character-set byte or
    /// [`FormatError::ValueTooLargeForPlatform`] if the name length exceeds [`usize::MAX`].
    ///
    /// Returns [`FormatError::InvalidLinkType`] for a reserved link kind. For a recognized
    /// kind, acquires the complete target before checking the name component and target
    /// contents. Returns [`FormatError::InvalidLinkNameBytes`] for an invalid name component,
    /// [`FormatError::DefinedAddressIsUndefined`] for an undefined hard address, or
    /// [`FormatError::InvalidLinkTarget`] for an invalid symbolic path or empty external filename.
    /// Returns [`FormatError::InvalidExternalLinkValue`] for missing external framing or
    /// [`FormatError::InvalidExternalLinkVersionFlags`] for an unsupported version or flag byte.
    ///
    /// # Examples
    ///
    /// Reads a soft link with a raw name byte and trailing padding.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{LinkMessageView, LinkTargetView};
    /// use hdf5_pure_format::OffsetWidth;
    ///
    /// let data = b"\x01\x08\x01\x01\xff\x02\x00/a\x00";
    /// let link = LinkMessageView::parse_prefix(data, OffsetWidth::Two)?;
    /// assert_eq!(link.name_bytes(), b"\xff");
    /// assert_eq!(link.target(), LinkTargetView::Soft { target_path: b"/a" });
    /// assert_eq!(link.consumed(), 9);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn parse_prefix(data: &'a [u8], offset_width: OffsetWidth) -> Result<Self, FormatError> {
        Self::decode(data, offset_width)
    }

    /// Prepares a body at `offset_width` using the borrowed fields and retained encoding choices.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] for the new width's undefined address,
    /// [`LinkWireError::Format`] if the hard address exceeds that width, or
    /// [`FormatError::LinkMessageLengthOverflow`] if the body length exceeds [`usize::MAX`].
    pub fn prepare(&self, offset_width: OffsetWidth) -> Result<PreparedLink<'a>, LinkWireError> {
        PreparedLink::for_view(self, offset_width)
    }

    /// Returns the stored target value length, or zero for a hard link.
    pub(crate) fn target_value_len(&self) -> u16 {
        self.target_value_len
    }

    /// Returns the decoded target with any path or user-defined value borrowed from the message.
    pub fn target(&self) -> LinkTargetView<'a> {
        self.target
    }

    /// Returns the stored creation order if the message includes that field.
    pub fn creation_order(&self) -> Option<u64> {
        self.creation_order
    }

    /// Returns the stored character set, including an explicit ASCII declaration.
    pub fn charset(&self) -> Option<CharacterSet> {
        self.charset.clone()
    }

    /// Returns the stored name component before text conversion.
    pub fn name_bytes(&self) -> &'a [u8] {
        self.name
    }

    /// Returns the width of the encoded name-length field.
    pub fn name_width(&self) -> UintWidth {
        self.name_width
    }

    /// Returns the number of bytes through the target, excluding trailing padding.
    pub fn consumed(&self) -> usize {
        self.consumed
    }

    /// Returns the hard-link address range relative to the start of the message body.
    ///
    /// Returns [`None`] for other target kinds.
    pub fn hard_link_address_range(&self) -> Option<Range<usize>> {
        matches!(self.target, LinkTargetView::Hard { .. })
            .then_some(self.target_start..self.consumed)
    }

    /// Converts a supported target and the name to an owned text-oriented link.
    ///
    /// Uses ASCII when the message omits its character-set field. Replaces invalid UTF-8 in
    /// the name and external paths, and requires valid UTF-8 in soft paths.
    ///
    /// # Errors
    ///
    /// Returns [`LinkProjectionError::UnsupportedKind`] for a user-defined target or
    /// [`LinkProjectionError::Format`] with [`FormatError::InvalidSoftLinkEncoding`] for a
    /// soft path with invalid UTF-8.
    pub fn to_text(&self) -> Result<LinkMessage, LinkProjectionError> {
        let link_target = self.target().to_text()?;
        Ok(LinkMessage {
            name: String::from_utf8_lossy(self.name_bytes()).into_owned(),
            link_target,
            creation_order: self.creation_order(),
            charset: self.charset().unwrap_or(CharacterSet::Ascii),
        })
    }
}

impl<'a> Decode<'a> for LinkMessageView<'a> {
    type Context = OffsetWidth;
    type Error = FormatError;

    fn decode(input: &'a [u8], width: OffsetWidth) -> Result<Self, Self::Error> {
        let prefix = LinkPrefix::parse(input)?;
        let (target, consumed, target_value_len) = prefix.read_target(width)?;
        let target_start = prefix.target_start();
        let LinkPrefix {
            name,
            name_width,
            link_type: _,
            creation_order,
            charset,
            target: _,
        } = prefix;
        Ok(Self {
            target,
            target_value_len,
            creation_order,
            charset,
            name,
            name_width,
            target_start,
            consumed,
        })
    }
}

/// The target of a decoded Link message, with variable-length data borrowed from its body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkTargetView<'a> {
    /// An object header at a defined stored address.
    Hard {
        /// The base-relative address of the object header.
        object_header_address: StoredAddress,
        /// Whether the message includes a hard-link type field.
        explicit_type: bool,
    },
    /// A path in this file.
    Soft {
        /// The path bytes before text conversion.
        target_path: &'a [u8],
    },
    /// An object path in another file.
    External {
        /// The file name bytes, excluding the terminating NUL.
        filename: &'a [u8],
        /// The object path bytes, excluding the terminating NUL.
        object_path: &'a [u8],
    },
    /// An application-defined link type with opaque data.
    UserDefined {
        /// The link class code.
        kind: UserLinkKind,
        /// The payload bytes carried by the link.
        value: &'a [u8],
    },
}

impl<'a> LinkTargetView<'a> {
    /// Reads the version and flags byte and two NUL-terminated paths from an external value.
    ///
    /// Pass just the value bytes after the external target's length field.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExternalLinkValue`] if a framing byte is missing,
    /// [`FormatError::InvalidExternalLinkVersionFlags`] if the first byte is nonzero, or
    /// [`FormatError::InvalidLinkTarget`] if either path is empty or the object path contains a NUL.
    fn external(value: &'a [u8]) -> Result<Self, FormatError> {
        let (&version_flags, paths) = value
            .split_first()
            .ok_or(FormatError::InvalidExternalLinkValue)?;
        if version_flags != EXTERNAL_VERSION_FLAGS {
            return Err(FormatError::InvalidExternalLinkVersionFlags(version_flags));
        }
        let separator = paths
            .iter()
            .position(|&byte| byte == 0)
            .ok_or(FormatError::InvalidExternalLinkValue)?;
        let (filename, rest) = paths.split_at(separator);
        let object_path = rest
            .get(1..)
            .and_then(|path| path.strip_suffix(&[0]))
            .ok_or(FormatError::InvalidExternalLinkValue)?;
        if filename.is_empty() || !link_message_v1::is_symbolic_path(object_path) {
            return Err(FormatError::InvalidLinkTarget);
        }
        Ok(Self::External {
            filename,
            object_path,
        })
    }

    pub(crate) fn to_text(self) -> Result<LinkTarget, LinkProjectionError> {
        Ok(match self {
            Self::Hard {
                object_header_address,
                explicit_type: _,
            } => LinkTarget::Hard {
                object_header_address,
            },
            Self::Soft { target_path } => {
                let path = core::str::from_utf8(target_path).map_err(|source| {
                    LinkProjectionError::Format(FormatError::InvalidSoftLinkEncoding { source })
                })?;
                LinkTarget::soft(String::from(path))
            }
            Self::External {
                filename,
                object_path,
            } => LinkTarget::external(
                String::from_utf8_lossy(filename).into_owned(),
                String::from_utf8_lossy(object_path).into_owned(),
            ),
            Self::UserDefined { kind, .. } => {
                return Err(LinkProjectionError::UnsupportedKind(kind));
            }
        })
    }
}

/// Fields through the borrowed name, with the cursor positioned at the start of the target.
pub(crate) struct LinkPrefix<'a> {
    pub(crate) name: &'a [u8],
    name_width: UintWidth,
    link_type: Option<u8>,
    creation_order: Option<u64>,
    charset: Option<CharacterSet>,
    target: ByteCursor<'a>,
}

impl<'a> LinkPrefix<'a> {
    /// Reads the header fields and name bytes before interpreting the target or the name component.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] for missing bytes before the target,
    /// [`FormatError::InvalidLinkVersion`] for a version other than 1,
    /// [`FormatError::InvalidLinkFlags`] for reserved flag bits,
    /// [`FormatError::InvalidCharacterSet`] for an unsupported character set, or
    /// [`FormatError::ValueTooLargeForPlatform`] for a name length above [`usize::MAX`].
    pub(crate) fn parse(data: &'a [u8]) -> Result<Self, FormatError> {
        let mut cursor = ByteCursor::new(data);
        let [version, flags] = cursor.array::<PREFIX_LEN>()?;
        if version != VERSION {
            return Err(FormatError::InvalidLinkVersion(version));
        }
        if flags & RESERVED_FLAGS != 0 {
            return Err(FormatError::InvalidLinkFlags(flags));
        }

        let name_width = UintWidth::from_flags(flags);
        let stores_type = flags & TYPE_PRESENT != 0;
        let tracks_order = flags & CREATION_ORDER_PRESENT != 0;
        let stores_charset = flags & CHARSET_PRESENT != 0;
        let incomplete = |error: UnexpectedEof| FormatError::UnexpectedEof {
            expected: PREFIX_LEN
                + usize::from(stores_type)
                + usize::from(tracks_order) * CREATION_ORDER_LEN
                + usize::from(stores_charset)
                + usize::from(name_width.get()),
            available: error.available,
        };
        let link_type = stores_type
            .then(|| cursor.u8())
            .transpose()
            .map_err(incomplete)?;
        let order = tracks_order
            .then(|| cursor.take_array::<CREATION_ORDER_LEN>())
            .transpose()
            .map_err(incomplete)?;
        let charset_code = stores_charset
            .then(|| cursor.u8())
            .transpose()
            .map_err(incomplete)?;
        let name_len = match name_width {
            UintWidth::One => cursor.u8().map(u64::from),
            UintWidth::Two => cursor.u16().map(u64::from),
            UintWidth::Four => cursor.u32().map(u64::from),
            UintWidth::Eight => cursor.u64(),
        }
        .map_err(incomplete)?;
        let charset = match charset_code {
            None => None,
            Some(ASCII) => Some(CharacterSet::Ascii),
            Some(UTF8) => Some(CharacterSet::Utf8),
            Some(code) => return Err(FormatError::InvalidCharacterSet(code)),
        };
        let name = cursor.take(name_len.to_usize()?)?;
        Ok(Self {
            link_type,
            creation_order: order.copied().map(u64::from_le_bytes),
            charset,
            name_width,
            name,
            target: cursor,
        })
    }

    pub(crate) fn target_start(&self) -> usize {
        self.target.position()
    }

    pub(crate) fn target_view(
        &self,
        width: OffsetWidth,
    ) -> Result<LinkTargetView<'a>, FormatError> {
        Ok(self.read_target(width)?.0)
    }

    pub(crate) fn discover_hard_link(
        &self,
        width: OffsetWidth,
    ) -> Result<Option<(&'a [u8], StoredAddress)>, FormatError> {
        Ok(self
            .read_hard_target_with_name_policy(width, LinkNamePolicy::Discovery)?
            .map(|(address, _)| (self.name, address)))
    }

    /// Checks the name component and reads the target, end position, and stored value length.
    ///
    /// The end position is relative to the start of the entire Link message. Returns a value
    /// length of zero for a hard address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the target is incomplete. Also returns the
    /// kind, name, and target errors of [`LinkMessageView::parse_prefix`].
    fn read_target(
        &self,
        width: OffsetWidth,
    ) -> Result<(LinkTargetView<'a>, usize, u16), FormatError> {
        let mut cursor = self.target;
        let (target, value_len) = match self.link_type.unwrap_or(HARD_LINK) {
            HARD_LINK => (
                LinkTargetView::Hard {
                    object_header_address: self.read_hard_address(
                        &mut cursor,
                        width,
                        LinkNamePolicy::Component,
                    )?,
                    explicit_type: self.link_type.is_some(),
                },
                0,
            ),
            kind @ (SOFT_LINK | EXTERNAL_LINK | USER_LINK_MIN..=u8::MAX) => {
                let len = cursor.u16()?;
                let value = cursor.take(usize::from(len))?;
                self.check_name(LinkNamePolicy::Component)?;
                let target = match kind {
                    SOFT_LINK if link_message_v1::is_symbolic_path(value) => {
                        LinkTargetView::Soft { target_path: value }
                    }
                    SOFT_LINK => return Err(FormatError::InvalidLinkTarget),
                    EXTERNAL_LINK => LinkTargetView::external(value)?,
                    _ => LinkTargetView::UserDefined {
                        kind: UserLinkKind::new(kind)?,
                        value,
                    },
                };
                (target, len)
            }
            kind => return Err(FormatError::InvalidLinkType(kind)),
        };
        Ok((target, cursor.position(), value_len))
    }

    pub(crate) fn read_hard_target(
        &self,
        width: OffsetWidth,
    ) -> Result<Option<(StoredAddress, Range<usize>)>, FormatError> {
        self.read_hard_target_with_name_policy(width, LinkNamePolicy::Component)
    }

    fn read_hard_target_with_name_policy(
        &self,
        width: OffsetWidth,
        name_policy: LinkNamePolicy,
    ) -> Result<Option<(StoredAddress, Range<usize>)>, FormatError> {
        match self.link_type.unwrap_or(HARD_LINK) {
            HARD_LINK => {
                let mut cursor = self.target;
                let address = self.read_hard_address(&mut cursor, width, name_policy)?;
                Ok(Some((address, self.target_start()..cursor.position())))
            }
            SOFT_LINK | EXTERNAL_LINK => {
                self.check_name(name_policy)?;
                Ok(None)
            }
            kind => Err(FormatError::InvalidLinkType(kind)),
        }
    }

    fn check_name(&self, policy: LinkNamePolicy) -> Result<(), FormatError> {
        if policy == LinkNamePolicy::Discovery || LinkName::is_valid(self.name) {
            Ok(())
        } else {
            Err(FormatError::InvalidLinkNameBytes(self.name.to_vec()))
        }
    }

    fn read_hard_address(
        &self,
        cursor: &mut ByteCursor<'_>,
        width: OffsetWidth,
        name_policy: LinkNamePolicy,
    ) -> Result<StoredAddress, FormatError> {
        let address = cursor.decoded_address(width)?;
        self.check_name(name_policy)?;
        match address {
            DecodedStoredAddress::Defined(address) => Ok(address),
            DecodedStoredAddress::Undefined => Err(FormatError::DefinedAddressIsUndefined {
                address: StoredAddress::undefined(width).get(),
                offset_size: width.get(),
            }),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum LinkNamePolicy {
    Component,
    Discovery,
}

// "The Link Message" of the format specification, version 4.0, defines these fields.
pub(crate) const VERSION: u8 = 1;
pub(crate) const PREFIX_LEN: usize = 2;
pub(crate) const CREATION_ORDER_PRESENT: u8 = 0x04;
pub(crate) const TYPE_PRESENT: u8 = 0x08;
pub(crate) const CHARSET_PRESENT: u8 = 0x10;
const RESERVED_FLAGS: u8 = 0xe0;
pub(crate) const CREATION_ORDER_LEN: usize = 8;
pub(crate) const EXTERNAL_VERSION_FLAGS: u8 = 0;
pub(crate) const USER_LINK_MIN: u8 = 65;
pub(crate) const ASCII: u8 = 0;
pub(crate) const UTF8: u8 = 1;
