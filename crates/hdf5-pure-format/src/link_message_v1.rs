//! Version 1 Link messages with raw names, target bytes, and retained field widths.

use alloc::string::String;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt;

use crate::address::StoredAddress;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::link_message_read::{LinkMessageView, LinkTargetView, USER_LINK_MIN};
use crate::link_message_write::{self, PreparedLink, StoredLink};
use crate::width::{OffsetWidth, UintWidth};

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
    target_value_len: u16,
    creation_order: Option<u64>,
    charset: Option<CharacterSet>,
    name: LinkName,
}

impl LinkMessageV1 {
    /// Parses one complete message, rejecting bytes after its target.
    ///
    /// Copies the name and target bytes after checking the complete body.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::parse_prefix`], or [`LinkWireError::TrailingBytes`] if bytes follow the target.
    pub fn parse(data: &[u8], offset_width: OffsetWidth) -> Result<Self, LinkWireError> {
        Ok(Self::from_view(&LinkMessageView::parse(
            data,
            offset_width,
        )?))
    }

    /// Parses the message at the start of `data` and returns its consumed length.
    ///
    /// Uses [`LinkMessageView::parse_prefix`] to check the message before copying its name and
    /// target. The returned length ends at the target, excluding object header padding.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Format`] for invalid names, reserved flags, undefined hard-link
    /// addresses, invalid symbolic targets, or malformed external-link framing. Decoding errors,
    /// including reserved kinds, unknown versions or character sets, and truncated fields, are
    /// wrapped in [`LinkWireError::Format`].
    pub fn parse_prefix(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<(Self, usize), LinkWireError> {
        let link = LinkMessageView::parse_prefix(data, offset_width)?;
        Ok((Self::from_view(&link), link.consumed()))
    }

    /// Creates a message from checked name bytes and a target.
    ///
    /// With `None`, the writer omits the character-set field and declares ASCII. With `Some`,
    /// the writer stores the field, including for [`CharacterSet::Ascii`]. Name and target bytes
    /// are not decoded as text.
    ///
    /// # Errors
    ///
    /// Returns [`LinkWireError::Format`] if a hard-link address is undefined at eight bytes,
    /// a symbolic path is empty or contains NUL, or a target value exceeds [`u16::MAX`] bytes.
    /// An external value includes its version byte and two terminating NUL bytes in this limit.
    /// A hard-link address is checked against the supplied width during encoding.
    pub fn new(
        name: LinkName,
        target: LinkTargetV1,
        creation_order: Option<u64>,
        charset: Option<CharacterSet>,
    ) -> Result<Self, LinkWireError> {
        let target_value_len = link_message_write::initial_value_len(target.as_view())?;
        Ok(Self {
            target,
            target_value_len,
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
    /// Returns [`LinkWireError::Format`] if a hard-link address does not fit `offset_width`
    /// or is undefined at that width, or the body length exceeds [`usize::MAX`].
    pub fn encoded_len(&self, offset_width: OffsetWidth) -> Result<usize, LinkWireError> {
        Ok(self.prepare(offset_width)?.encoded_size())
    }

    /// Serializes the body with the retained optional fields and name-length width.
    ///
    /// The returned bytes exclude object-header padding.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::encoded_len`].
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, LinkWireError> {
        Ok(self.prepare(offset_width)?.to_bytes())
    }

    /// Prepares a body with borrowed bytes and its retained field choices.
    ///
    /// # Errors
    ///
    /// Returns the address and body length errors of [`Self::encoded_len`].
    pub fn prepare(&self, offset_width: OffsetWidth) -> Result<PreparedLink<'_>, LinkWireError> {
        PreparedLink::for_stored(self, offset_width)
    }

    fn from_view(link: &LinkMessageView<'_>) -> Self {
        Self {
            target: LinkTargetV1::from_view(link),
            target_value_len: link.target_value_len(),
            creation_order: link.creation_order(),
            charset: link.charset(),
            name: LinkName::from_view(link),
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        LinkTargetV1,
        Option<u64>,
        Option<CharacterSet>,
        LinkName,
        u16,
    ) {
        let Self {
            target,
            target_value_len,
            creation_order,
            charset,
            name,
        } = self;
        (target, creation_order, charset, name, target_value_len)
    }
}

impl StoredLink for LinkMessageV1 {
    fn name_field(&self) -> (&[u8], UintWidth) {
        (self.name.as_bytes(), self.name.width())
    }
    fn target_field(&self) -> (LinkTargetView<'_>, u16) {
        (self.target.as_view(), self.target_value_len)
    }
    fn creation_index(&self) -> Option<u64> {
        self.creation_order
    }
    fn charset_field(&self) -> Option<CharacterSet> {
        self.charset.clone()
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
        let (target, creation_order, charset, name, _) = link.into_parts();
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
    /// Returns [`FormatError::InvalidLinkNameBytes`] wrapped in [`LinkWireError::Format`] if the
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
        if !Self::is_valid(&bytes) {
            return Err(FormatError::InvalidLinkNameBytes(bytes).into());
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

    pub(crate) fn from_view(link: &LinkMessageView<'_>) -> Self {
        Self {
            width: link.name_width(),
            bytes: link.name_bytes().to_vec(),
        }
    }

    pub(crate) fn from_prepared(link: &PreparedLink<'_>) -> Self {
        Self {
            width: link.name_width(),
            bytes: link.name_bytes().to_vec(),
        }
    }

    pub(crate) fn is_valid(bytes: &[u8]) -> bool {
        !bytes.is_empty() && bytes != b"." && !bytes.contains(&0) && !bytes.contains(&b'/')
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
    pub(crate) fn from_view(link: &LinkMessageView<'_>) -> Self {
        match link.target() {
            LinkTargetView::Hard {
                object_header_address,
                explicit_type,
            } => LinkTargetV1::Hard {
                object_header_address,
                explicit_type,
            },
            LinkTargetView::Soft { target_path } => LinkTargetV1::Soft {
                target_path: target_path.to_vec(),
            },
            LinkTargetView::External {
                filename,
                object_path,
            } => LinkTargetV1::External {
                filename: filename.to_vec(),
                object_path: object_path.to_vec(),
            },
            LinkTargetView::UserDefined { kind, value } => LinkTargetV1::UserDefined {
                kind,
                value: UserLinkValue(value.to_vec()),
            },
        }
    }

    pub(crate) fn as_view(&self) -> LinkTargetView<'_> {
        match self {
            Self::Hard {
                object_header_address,
                explicit_type,
            } => LinkTargetView::Hard {
                object_header_address: *object_header_address,
                explicit_type: *explicit_type,
            },
            Self::Soft { target_path } => LinkTargetView::Soft { target_path },
            Self::External {
                filename,
                object_path,
            } => LinkTargetView::External {
                filename,
                object_path,
            },
            Self::UserDefined { kind, value } => LinkTargetView::UserDefined {
                kind: *kind,
                value: value.as_bytes(),
            },
        }
    }
}

pub(crate) fn is_symbolic_path(path: &[u8]) -> bool {
    !path.is_empty() && !path.contains(&0)
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
    /// Returns [`LinkWireError::TargetLength`] if the value exceeds [`u16::MAX`] bytes.
    pub fn new(value: Vec<u8>) -> Result<Self, LinkWireError> {
        u16::try_from(value.len()).map_err(|_error| LinkWireError::TargetLength(value.len()))?;
        Ok(Self(value))
    }

    /// Returns the value bytes without interpreting their contents.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// An error converting a checked wire message to a supported link model.
///
/// [`Error::source`] returns the wrapped [`FormatError`] for [`Self::Format`].
/// [`Self::UnsupportedKind`] reports the stored kind code and has no source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkProjectionError {
    /// Text conversion failed for a supported target.
    Format(FormatError),
    /// The checked message contains a user-defined link kind.
    UnsupportedKind(UserLinkKind),
}

impl LinkProjectionError {
    pub(crate) fn into_format(self) -> FormatError {
        match self {
            Self::Format(error) => error,
            Self::UnsupportedKind(kind) => FormatError::InvalidLinkType(kind.get()),
        }
    }
}

impl fmt::Display for LinkProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => fmt::Display::fmt(error, f),
            Self::UnsupportedKind(kind) => write!(f, "unsupported link kind: {}", kind.get()),
        }
    }
}

impl Error for LinkProjectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::UnsupportedKind(_) => None,
        }
    }
}

/// An error constructing, decoding, or encoding a version 1 Link message.
///
/// [`Error::source`] returns the wrapped [`FormatError`] for [`Self::Format`].
/// [`Self::NameWidth`] reports the name length and field width in bytes and has no source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkWireError {
    /// The name length does not fit the selected field width.
    NameWidth {
        length: usize,
        width: UintWidth,
    },
    /// An error in a field, the message framing, or the encoded length.
    Format(FormatError),
    TargetLength(usize),
    TrailingBytes,
}

impl fmt::Display for LinkWireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NameWidth { length, width } => write!(
                f,
                "link name length {length} exceeds its {}-byte field",
                width.get()
            ),
            Self::Format(error) => fmt::Display::fmt(error, f),
            Self::TargetLength(length) => {
                write!(f, "link target length {length} exceeds its two-byte field")
            }
            Self::TrailingBytes => f.write_str("trailing bytes after link message fields"),
        }
    }
}

impl Error for LinkWireError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NameWidth { .. } | Self::TargetLength(_) | Self::TrailingBytes => None,
            Self::Format(error) => Some(error),
        }
    }
}

impl From<FormatError> for LinkWireError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}
