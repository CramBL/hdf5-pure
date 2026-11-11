//! Version 1 Link messages with raw names, target bytes, and retained field widths.

use alloc::string::String;
use alloc::vec::Vec;

use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::bytes;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_message::{
    EXTERNAL_LINK, HARD_LINK, LinkMessage, LinkPrefix, LinkTarget, SOFT_LINK,
};
use crate::link_storage::LinkStorageError;
use crate::width::{self, OffsetWidth, UintWidth};

/// A version 1 Link message with checked fields and retained encoding choices.
///
/// Names and symbolic targets are stored as bytes. The declared character set is retained
/// independently of those bytes, including an explicit ASCII declaration. The name-length width,
/// explicit hard-link type field, and creation-order field are also retained.
///
/// Group-level creation-order policies and application-defined traversal are the caller's
/// responsibility.
///
/// The fields are defined in "The Link Message" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::{LinkMessageV1, LinkWireError, OffsetWidth, UintWidth};
///
/// let bytes = b"\x01\x09\xff\x01\x00u\x02\x00\x00\xff";
/// let link = LinkMessageV1::parse(bytes, OffsetWidth::Two)?;
/// assert_eq!(link.name().width(), UintWidth::Two);
/// assert_eq!(link.serialize(OffsetWidth::Two)?, bytes);
/// # Ok::<(), LinkWireError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
#[derive(Clone, Debug, PartialEq)]
pub struct LinkMessageV1 {
    target: LinkTargetV1,
    creation_order: Option<u64>,
    charset: Option<CharacterSet>,
    name: LinkName,
}

impl LinkMessageV1 {
    /// Parses one complete message, rejecting bytes after its target.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::parse_prefix`], or [`LinkStorageError::TrailingBytes`]
    /// wrapped in [`LinkWireError::Storage`] if bytes follow the target.
    pub fn parse(data: &[u8], offset_width: OffsetWidth) -> Result<Self, LinkWireError> {
        let (link, len) = Self::parse_prefix(data, offset_width)?;
        if len != data.len() {
            return Err(LinkStorageError::TrailingBytes.into());
        }
        Ok(link)
    }

    /// Parses the message at the start of `data` and returns its consumed length.
    ///
    /// Object-header padding and other bytes after the target are left unread.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Storage`] for invalid names, reserved flags, undefined hard-link
    /// addresses, invalid symbolic targets, or malformed external-link framing. Decoding errors,
    /// including reserved kinds, unknown versions or character sets, and truncated fields, are
    /// wrapped in [`LinkStorageError::Format`].
    pub fn parse_prefix(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<(Self, usize), LinkWireError> {
        let prefix = LinkPrefix::parse(data)?;
        let flags = data
            .get(1)
            .copied()
            .ok_or(LinkStorageError::TrailingBytes)?;
        if flags & RESERVED_FLAGS != 0 {
            return Err(LinkStorageError::Flags(flags).into());
        }
        let name = LinkName::with_width(prefix.name.to_vec(), prefix.name_width)?;
        let mut fields = bytes::Fields::new(data, prefix.target_pos);
        let (target, end) = if prefix.link_type_code == HARD_LINK {
            let object_header_address = match fields.decoded_address(offset_width)? {
                DecodedStoredAddress::Defined(address) => address,
                DecodedStoredAddress::Undefined => {
                    return Err(LinkStorageError::UndefinedAddress.into());
                }
            };
            (
                LinkTargetV1::Hard {
                    object_header_address,
                    explicit_type: prefix.has_link_type,
                },
                fields.pos(),
            )
        } else {
            if !matches!(
                prefix.link_type_code,
                SOFT_LINK | EXTERNAL_LINK | USER_LINK_MIN..=u8::MAX
            ) {
                return Err(FormatError::InvalidLinkType(prefix.link_type_code).into());
            }
            let len = usize::from(fields.u16()?);
            let pos = fields.pos();
            bytes::ensure_len(data, pos, len)?;
            let value = &data[pos..pos + len];
            let target = match prefix.link_type_code {
                SOFT_LINK => LinkTargetV1::Soft {
                    target_path: value.to_vec(),
                },
                EXTERNAL_LINK => LinkTargetV1::parse_external(value)?,
                kind => LinkTargetV1::UserDefined {
                    kind: UserLinkKind::new(kind)?,
                    value: UserLinkValue::new(value.to_vec())?,
                },
            };
            (target, pos + len)
        };
        let link = Self::new(
            name,
            target,
            prefix.creation_order,
            prefix.has_charset.then_some(prefix.charset),
        )?;
        Ok((link, end))
    }

    /// Creates a message from checked name bytes and a target.
    ///
    /// With `None`, the writer omits the character-set field and declares ASCII. With `Some`,
    /// the writer stores the field, including for [`CharacterSet::Ascii`]. Name and target bytes
    /// are not decoded as text.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Storage`] if a hard-link address is undefined at eight bytes,
    /// a symbolic path is empty or contains NUL, or a target value exceeds [`u16::MAX`] bytes.
    /// An external value includes its version byte and two terminating NUL bytes in this limit.
    /// A hard-link address is checked against the supplied width during encoding.
    pub fn new(
        name: LinkName,
        target: LinkTargetV1,
        creation_order: Option<u64>,
        charset: Option<CharacterSet>,
    ) -> Result<Self, LinkWireError> {
        target.encoded_len(OffsetWidth::Eight)?;
        Ok(Self {
            target,
            creation_order,
            charset,
            name,
        })
    }

    /// Returns the stored target, including explicit hard-link type-field presence.
    pub fn target(&self) -> &LinkTargetV1 {
        &self.target
    }

    /// Returns the stored creation index, or `None` if the field is absent.
    pub fn creation_order(&self) -> Option<u64> {
        self.creation_order
    }

    /// Returns the explicit character-set declaration, or `None` for omitted ASCII.
    pub fn charset(&self) -> Option<CharacterSet> {
        self.charset.clone()
    }

    /// Returns the raw name and its retained length-field width.
    pub fn name(&self) -> &LinkName {
        &self.name
    }

    /// Returns the body length with the retained optional fields and name-length width.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Storage`] if a hard-link address does not fit `offset_width`
    /// or is undefined at that width, or the body length exceeds [`usize::MAX`].
    pub fn encoded_len(&self, offset_width: OffsetWidth) -> Result<usize, LinkWireError> {
        let Self {
            target,
            creation_order,
            charset,
            name,
        } = self;
        let prefix = PREFIX_LEN
            + usize::from(target.stores_type())
            + usize::from(creation_order.is_some()) * CREATION_ORDER_LEN
            + usize::from(charset.is_some())
            + usize::from(name.width.get());
        let target_len = target.encoded_len(offset_width)?;
        prefix
            .checked_add(name.bytes.len())
            .and_then(|len| len.checked_add(target_len))
            .ok_or(LinkStorageError::LengthOverflow.into())
    }

    /// Serializes the body with the retained optional fields and name-length width.
    ///
    /// The returned bytes exclude object-header padding.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::encoded_len`].
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, LinkWireError> {
        let len = self.encoded_len(offset_width)?;
        let Self {
            target,
            creation_order,
            charset,
            name,
        } = self;
        let mut flags = name.width.flag_bits();
        if target.stores_type() {
            flags |= TYPE_PRESENT;
        }
        if creation_order.is_some() {
            flags |= CREATION_ORDER_PRESENT;
        }
        if charset.is_some() {
            flags |= CHARSET_PRESENT;
        }
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&[VERSION, flags]);
        if target.stores_type() {
            out.push(target.kind());
        }
        if let Some(order) = creation_order {
            out.extend_from_slice(&order.to_le_bytes());
        }
        if let Some(charset) = charset {
            out.push(match charset {
                CharacterSet::Ascii => ASCII,
                CharacterSet::Utf8 => UTF8,
            });
        }
        name.width.write(&mut out, name.bytes.len());
        out.extend_from_slice(&name.bytes);
        target.write(&mut out, offset_width)?;
        debug_assert_eq!(out.len(), len);
        Ok(out)
    }

    pub(crate) fn into_parts(self) -> (LinkTargetV1, Option<u64>, Option<CharacterSet>, LinkName) {
        let Self {
            target,
            creation_order,
            charset,
            name,
        } = self;
        (target, creation_order, charset, name)
    }
}

impl TryFrom<LinkMessageV1> for LinkMessage {
    type Error = LinkProjectionError;

    /// Converts a checked message to a supported link with text names and paths.
    ///
    /// Replaces invalid UTF-8 in names and external paths with the replacement character.
    /// Soft-link paths must be valid UTF-8. The character-set declaration is retained
    /// independently of text conversion.
    ///
    /// The converter discards the stored name-length width and explicit default-field presence.
    /// [`LinkMessage::serialize`] writes the smallest name-length width and omits default fields.
    ///
    /// # Errors
    ///
    /// Returns [`LinkProjectionError::UnsupportedKind`] for a user-defined link, or
    /// [`LinkProjectionError::Format`] containing [`FormatError::InvalidSoftLinkEncoding`]
    /// if the soft-link target is not UTF-8.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::__private::{LinkMessage, LinkMessageV1, OffsetWidth};
    ///
    /// let wire = LinkMessageV1::parse(b"\x01\x00\x01\xff\x34\x12", OffsetWidth::Two).unwrap();
    /// let text = LinkMessage::try_from(wire).unwrap();
    /// assert_eq!(text.name, "\u{fffd}");
    /// ```
    fn try_from(link: LinkMessageV1) -> Result<Self, Self::Error> {
        let (target, creation_order, charset, name) = link.into_parts();
        let link_target = match target {
            LinkTargetV1::Hard {
                object_header_address,
                explicit_type: _,
            } => LinkTarget::Hard {
                object_header_address,
            },
            LinkTargetV1::Soft { target_path } => {
                LinkTarget::soft(String::from_utf8(target_path).map_err(|error| {
                    LinkProjectionError::Format(FormatError::InvalidSoftLinkEncoding {
                        source: error.utf8_error(),
                    })
                })?)
            }
            LinkTargetV1::External {
                filename,
                object_path,
            } => LinkTarget::external(into_lossy_string(filename), into_lossy_string(object_path)),
            LinkTargetV1::UserDefined { kind, value: _ } => {
                return Err(LinkProjectionError::UnsupportedKind(kind));
            }
        };
        Ok(Self {
            name: into_lossy_string(name.into_bytes()),
            link_target,
            creation_order,
            charset: charset.unwrap_or(CharacterSet::Ascii),
        })
    }
}

fn into_lossy_string(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes)
        .unwrap_or_else(|error| String::from_utf8_lossy(error.as_bytes()).into_owned())
}

/// A raw link-name component whose length fits its stored field width.
///
/// A component is nonempty, differs from `.`, and contains neither NUL nor `/`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinkName {
    width: UintWidth,
    bytes: Vec<u8>,
}

impl LinkName {
    /// Creates a name with the smallest length-field width that fits its bytes.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::InvalidName`] wrapped in [`LinkWireError::Storage`] if the
    /// bytes are empty, equal `.`, or contain NUL or `/`.
    pub fn new(bytes: Vec<u8>) -> Result<Self, LinkWireError> {
        let width = UintWidth::smallest_for_len(bytes.len());
        Self::with_width(bytes, width)
    }

    /// Creates a name with an explicit length-field width.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new`], or [`LinkWireError::NameWidth`] if the length
    /// does not fit `width`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::__private::{LinkName, LinkWireError, UintWidth};
    ///
    /// let name = LinkName::with_width(b"x".to_vec(), UintWidth::Eight)?;
    /// assert_eq!(name.as_bytes(), b"x");
    /// assert_eq!(name.width(), UintWidth::Eight);
    /// # Ok::<(), LinkWireError>(())
    /// ```
    pub fn with_width(bytes: Vec<u8>, width: UintWidth) -> Result<Self, LinkWireError> {
        if bytes.is_empty() || bytes == b"." || bytes.contains(&0) || bytes.contains(&b'/') {
            return Err(LinkStorageError::InvalidName(bytes).into());
        }
        if UintWidth::smallest_for_len(bytes.len()).get() > width.get() {
            return Err(LinkWireError::NameWidth {
                length: bytes.len(),
                width,
            });
        }
        Ok(Self { width, bytes })
    }

    /// Returns the stored name-length field width.
    pub fn width(&self) -> UintWidth {
        self.width
    }

    /// Returns the name bytes without text conversion.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// The target fields for each version 1 Link-message kind.
///
/// [`LinkMessageV1::new`] checks symbolic paths and their encoded lengths before storing a target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkTargetV1 {
    /// A hard link to an object header.
    Hard {
        /// The base-relative object-header address.
        object_header_address: StoredAddress,
        /// Whether to store the optional type field with the hard-link kind.
        explicit_type: bool,
    },
    /// A soft link with an undecoded target path.
    Soft {
        /// The raw path bytes, excluding a terminating NUL.
        target_path: Vec<u8>,
    },
    /// A version 0 external link with zero flags and undecoded path bytes.
    External {
        filename: Vec<u8>,
        object_path: Vec<u8>,
    },
    /// An application-defined link with an opaque value.
    UserDefined {
        kind: UserLinkKind,
        value: UserLinkValue,
    },
}

impl LinkTargetV1 {
    fn parse_external(value: &[u8]) -> Result<Self, LinkStorageError> {
        let (&version_flags, strings) =
            value.split_first().ok_or(LinkStorageError::ExternalValue)?;
        if version_flags != EXTERNAL_VERSION_FLAGS {
            return Err(LinkStorageError::ExternalVersionFlags(version_flags));
        }
        let separator = strings
            .iter()
            .position(|&byte| byte == 0)
            .ok_or(LinkStorageError::ExternalValue)?;
        let (filename, rest) = strings.split_at(separator);
        let object_path = rest
            .get(1..)
            .and_then(|rest| rest.strip_suffix(&[0]))
            .ok_or(LinkStorageError::ExternalValue)?;
        Ok(Self::External {
            filename: filename.to_vec(),
            object_path: object_path.to_vec(),
        })
    }

    fn stores_type(&self) -> bool {
        !matches!(
            self,
            Self::Hard {
                explicit_type: false,
                ..
            }
        )
    }

    fn kind(&self) -> u8 {
        match self {
            Self::Hard { .. } => HARD_LINK,
            Self::Soft { .. } => SOFT_LINK,
            Self::External { .. } => EXTERNAL_LINK,
            Self::UserDefined { kind, .. } => kind.get(),
        }
    }

    fn encoded_len(&self, offset_width: OffsetWidth) -> Result<usize, LinkStorageError> {
        let value_len = match self {
            Self::Hard {
                object_header_address,
                ..
            } => {
                if object_header_address.get() > width::offset_width_max(offset_width) {
                    return Err(FormatError::AddressTooLarge {
                        address: object_header_address.get(),
                        offset_size: offset_width.get(),
                    }
                    .into());
                }
                if object_header_address.is_undefined(offset_width) {
                    return Err(LinkStorageError::UndefinedAddress);
                }
                return Ok(usize::from(offset_width.get()));
            }
            Self::Soft { target_path } => {
                Self::path(target_path)?;
                target_path.len()
            }
            Self::External {
                filename,
                object_path,
            } => {
                Self::path(filename)?;
                Self::path(object_path)?;
                filename
                    .len()
                    .checked_add(object_path.len())
                    .and_then(|len| len.checked_add(EXTERNAL_FRAMING_LEN))
                    .ok_or(LinkStorageError::LengthOverflow)?
            }
            Self::UserDefined { value, .. } => value.as_bytes().len(),
        };
        u16::try_from(value_len).map_err(|_error| LinkStorageError::TargetLength(value_len))?;
        value_len
            .checked_add(VALUE_LENGTH_LEN)
            .ok_or(LinkStorageError::LengthOverflow)
    }

    fn write(&self, out: &mut Vec<u8>, offset_width: OffsetWidth) -> Result<(), LinkStorageError> {
        let len = self.encoded_len(offset_width)?;
        if let Self::Hard {
            object_header_address,
            ..
        } = self
        {
            bytes::try_write_decoded_address(
                out,
                DecodedStoredAddress::Defined(*object_header_address),
                offset_width,
            )?;
            return Ok(());
        }
        let value_len = len - VALUE_LENGTH_LEN;
        let value_len =
            u16::try_from(value_len).map_err(|_error| LinkStorageError::TargetLength(value_len))?;
        out.extend_from_slice(&value_len.to_le_bytes());
        match self {
            Self::Hard { .. } => {}
            Self::Soft { target_path } => out.extend_from_slice(target_path),
            Self::External {
                filename,
                object_path,
            } => {
                out.push(EXTERNAL_VERSION_FLAGS);
                out.extend_from_slice(filename);
                out.push(0);
                out.extend_from_slice(object_path);
                out.push(0);
            }
            Self::UserDefined { value, .. } => out.extend_from_slice(value.as_bytes()),
        }
        Ok(())
    }

    fn path(path: &[u8]) -> Result<(), LinkStorageError> {
        if path.is_empty() || path.contains(&0) {
            return Err(LinkStorageError::InvalidTarget);
        }
        Ok(())
    }
}

/// A user-defined Link-message kind in `65..=255`.
///
/// Kind 64 denotes an external link and is outside this range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UserLinkKind(u8);

impl UserLinkKind {
    /// Checks that `kind` is in the user-defined range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLinkType`] if `kind` is less than 65.
    pub fn new(kind: u8) -> Result<Self, FormatError> {
        if kind < USER_LINK_MIN {
            return Err(FormatError::InvalidLinkType(kind));
        }
        Ok(Self(kind))
    }

    /// Returns the on-disk kind code.
    pub fn get(self) -> u8 {
        self.0
    }
}

/// An opaque user-defined link value of at most [`u16::MAX`] bytes.
///
/// Empty values, NUL bytes, and bytes outside UTF-8 are permitted. Enclosing storage may impose
/// a smaller size limit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UserLinkValue(Vec<u8>);

impl UserLinkValue {
    /// Creates an opaque value whose length fits the two-byte field.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::TargetLength`] if the value exceeds [`u16::MAX`] bytes.
    pub fn new(value: Vec<u8>) -> Result<Self, LinkStorageError> {
        u16::try_from(value.len()).map_err(|_error| LinkStorageError::TargetLength(value.len()))?;
        Ok(Self(value))
    }

    /// Returns the value bytes without interpreting their contents.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// An error converting a checked wire message to a supported link model.
///
/// The converter reports user-defined kinds separately from text-conversion errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkProjectionError {
    /// Text conversion failed for a supported target.
    Format(FormatError),
    /// The checked message contains a user-defined link kind.
    UnsupportedKind(UserLinkKind),
}

/// An error constructing, decoding, or encoding a version 1 Link message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkWireError {
    /// The name length does not fit the selected field width.
    NameWidth { length: usize, width: UintWidth },
    /// A structural or decoding error in the message fields.
    Storage(LinkStorageError),
}

impl From<LinkStorageError> for LinkWireError {
    fn from(error: LinkStorageError) -> Self {
        Self::Storage(error)
    }
}

impl From<FormatError> for LinkWireError {
    fn from(error: FormatError) -> Self {
        Self::Storage(error.into())
    }
}

const VERSION: u8 = 1;
const PREFIX_LEN: usize = 2;
const CREATION_ORDER_PRESENT: u8 = 0x04;
const TYPE_PRESENT: u8 = 0x08;
const CHARSET_PRESENT: u8 = 0x10;
const RESERVED_FLAGS: u8 = 0xe0;
const CREATION_ORDER_LEN: usize = 8;
const VALUE_LENGTH_LEN: usize = 2;
const EXTERNAL_FRAMING_LEN: usize = 3;
const EXTERNAL_VERSION_FLAGS: u8 = 0;
const USER_LINK_MIN: u8 = 65;
const ASCII: u8 = 0;
const UTF8: u8 = 1;
