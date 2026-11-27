//! HDF5 Link message parsing (message type 0x0006).

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt;
use core::ops::Range;

use super::encode::PreparedLink;
use super::name::{LinkTargetV1, UserLinkKind};
use super::view::{LinkMessageView, LinkPrefix, LinkTargetView};
use super::wire::LinkMessageV1;
use crate::address::StoredAddress;
use crate::codec::DecodePrefix;
use crate::datatype::CharacterSet;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The type of a link in an HDF5 v2 group.
#[derive(Clone, Debug, PartialEq)]
pub enum LinkTarget {
    /// Hard link pointing to an object header address.
    Hard {
        object_header_address: StoredAddress,
    },
    /// Soft or external link pointing to a path.
    Indirect(Box<SymbolicLinkTarget>),
}

impl LinkTarget {
    /// Creates a soft link pointing to `target_path`.
    pub fn soft(target_path: String) -> Self {
        Self::Indirect(Box::new(SymbolicLinkTarget::Soft { target_path }))
    }

    /// Creates an external link pointing to `object_path` in `filename`.
    pub fn external(filename: String, object_path: String) -> Self {
        Self::Indirect(Box::new(SymbolicLinkTarget::External {
            filename,
            object_path,
        }))
    }
    fn as_view(&self) -> LinkTargetView<'_> {
        match self {
            Self::Hard {
                object_header_address,
            } => LinkTargetView::Hard {
                object_header_address: *object_header_address,
                explicit_type: false,
            },
            Self::Indirect(target) => match target.as_ref() {
                SymbolicLinkTarget::Soft { target_path } => LinkTargetView::Soft {
                    target_path: target_path.as_bytes(),
                },
                SymbolicLinkTarget::External {
                    filename,
                    object_path,
                } => LinkTargetView::External {
                    filename: filename.as_bytes(),
                    object_path: object_path.as_bytes(),
                },
            },
        }
    }
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<LinkTarget>() == 16);

/// The target of a soft or an external link.
#[derive(Clone, Debug, PartialEq)]
pub enum SymbolicLinkTarget {
    /// Soft (symbolic) link with a target path string.
    Soft { target_path: String },
    /// External link pointing to a file and object path within it.
    External {
        filename: String,
        object_path: String,
    },
}

/// A parsed HDF5 Link message (type 0x0006).
#[derive(Debug, Clone, PartialEq)]
pub struct LinkMessage {
    /// Name of this link.
    pub name: String,
    /// What this link points to.
    pub link_target: LinkTarget,
    /// Creation order, if tracked.
    pub creation_order: Option<u64>,
    /// Character set of the link name.
    pub charset: CharacterSet,
}

/// Whether the stored name bytes `raw` name the link `wanted`.
///
/// [`LinkMessage::parse`] decodes a name with `from_utf8_lossy`, so a name that
/// is not valid UTF-8 is matched against the replacement characters a caller
/// would have received from it, the only spelling such a link can be requested by.
/// A valid name compares as bytes and allocates nothing.
fn name_matches(raw: &[u8], wanted: &str) -> bool {
    if raw == wanted.as_bytes() {
        return true;
    }
    core::str::from_utf8(raw).is_err() && String::from_utf8_lossy(raw) == wanted
}

/// Returns `true` if the stored name matches `name` or the name prefix is malformed.
///
/// Reads through the name. An object-header filter can retain malformed prefixes for
/// the later lookup to report.
pub fn link_is_named(data: &[u8], name: &str) -> bool {
    match LinkPrefix::parse(data) {
        Ok(prefix) => name_matches(prefix.name, name),
        Err(_) => true,
    }
}

impl LinkMessage {
    /// Returns the stored name bytes from the prefix of a version 1 Link message.
    ///
    /// Parses through the name, stopping before the target data. The returned slice borrows
    /// `data` and preserves invalid UTF-8 bytes.
    ///
    /// The prefix is defined in "The Link Message" of the [format specification,
    /// version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLinkVersion`] for a version other than 1,
    /// [`FormatError::InvalidLinkFlags`] for reserved flag bits,
    /// [`FormatError::InvalidCharacterSet`] for an unknown character set, or
    /// [`FormatError::UnexpectedEof`] for a truncated prefix. Returns
    /// [`FormatError::ValueTooLargeForPlatform`] if the name length exceeds [`usize::MAX`].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_link
    pub fn name_bytes(data: &[u8]) -> Result<&[u8], FormatError> {
        LinkPrefix::parse(data).map(|prefix| prefix.name)
    }

    /// Returns a borrowed hard-link name and its defined address for discovery.
    ///
    /// The caller can check the returned name bytes for component syntax. Returns [`None`]
    /// for a soft or external link after reading through its name.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. Returns
    /// [`FormatError::InvalidLinkType`] for any kind other than hard, soft, or external,
    /// [`FormatError::UnexpectedEof`] for a truncated hard address, or
    /// [`FormatError::DefinedAddressIsUndefined`] for the undefined address at `offset_width`.
    pub fn discover_hard_link(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<Option<(&[u8], StoredAddress)>, FormatError> {
        LinkPrefix::parse(data)?.discover_hard_link(offset_width)
    }

    /// Returns the object header address of the link in the Link message `data`, if it is a
    /// hard link named `name`.
    ///
    /// Compares the name before reading the target. Returns [`None`] for a different name or
    /// a soft or external link.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] for an unsupported `offset_size` or the
    /// errors of [`Self::hard_link_address_if_named_with_offset_width`].
    pub fn hard_link_address_if_named(
        data: &[u8],
        offset_size: u8,
        name: &str,
    ) -> Result<Option<StoredAddress>, FormatError> {
        Self::hard_link_address_if_named_with_offset_width(
            data,
            OffsetWidth::try_from(offset_size)?,
            name,
        )
    }

    /// Returns the object header address of the hard link named `name` using a validated address
    /// width.
    ///
    /// Returns [`None`] when the name differs. For a matching soft or external link, checks
    /// the name component and returns [`None`] before reading its target.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. For a matching name, returns
    /// the errors of [`Self::hard_link_address_range`] for an invalid kind, name component,
    /// or hard address.
    pub fn hard_link_address_if_named_with_offset_width(
        data: &[u8],
        offset_width: OffsetWidth,
        name: &str,
    ) -> Result<Option<StoredAddress>, FormatError> {
        let prefix = LinkPrefix::parse(data)?;
        if !name_matches(prefix.name, name) {
            return Ok(None);
        }
        Ok(prefix
            .read_hard_target(offset_width)?
            .map(|(address, _)| address))
    }

    /// Returns the target of the link named `name`, or `None` for another name.
    ///
    /// The name is compared in the message bytes before the target is parsed.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. For a matching name, returns
    /// the errors of [`Self::parse_with_offset_width`] for invalid name components,
    /// invalid targets, or failed text conversion.
    pub fn target_if_named(
        data: &[u8],
        offset_width: OffsetWidth,
        name: &str,
    ) -> Result<Option<LinkTarget>, FormatError> {
        let prefix = LinkPrefix::parse(data)?;
        if !name_matches(prefix.name, name) {
            return Ok(None);
        }
        Ok(Some(
            prefix
                .target_view(offset_width)?
                .to_text()
                .map_err(LinkProjectionError::into_format)?,
        ))
    }

    /// Returns the byte range of the object header address in the Link message `data`, or `None`
    /// for a soft or an external link.
    ///
    /// A caller that points a hard link at another object in place writes the new address over
    /// this range. For soft and external links, checks the name and returns [`None`] before
    /// reading the target.
    ///
    /// # Errors
    ///
    /// Propagates prefix errors from [`Self::name_bytes`]. Returns [`FormatError::InvalidLinkType`]
    /// for any kind other than hard, soft, or external, [`FormatError::InvalidLinkNameBytes`] for
    /// an invalid name component, [`FormatError::UnexpectedEof`] for a truncated hard address,
    /// or [`FormatError::DefinedAddressIsUndefined`] for the undefined address at `offset_width`.
    pub fn hard_link_address_range(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<Option<Range<usize>>, FormatError> {
        Ok(LinkPrefix::parse(data)?
            .read_hard_target(offset_width)?
            .map(|(_, range)| range))
    }

    /// Serializes this message using the canonical field choices of [`Self::prepare`].
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::prepare`].
    pub fn serialize(&self, offset_width: OffsetWidth) -> Result<Vec<u8>, FormatError> {
        Ok(self.prepare(offset_width)?.to_bytes())
    }

    /// Prepares a Link body from the borrowed name and supported target.
    ///
    /// Uses the smallest name-length width and omits the hard-link type field and the ASCII
    /// character-set field. Preserves any creation-order field.
    ///
    /// # Errors
    ///
    /// Returns the name, address, and length-overflow errors of [`PreparedLink::hard`].
    /// Returns [`FormatError::InvalidLinkTarget`] for an empty path or one containing a NUL,
    /// or [`FormatError::LengthTooLarge`] if a symbolic target exceeds its two-byte length field.
    /// The external target size includes its version and flags byte and both NUL terminators.
    pub fn prepare(&self, offset_width: OffsetWidth) -> Result<PreparedLink<'_>, FormatError> {
        PreparedLink::from_text_fields(
            offset_width,
            &self.name,
            self.creation_order,
            &self.charset,
            self.link_target.as_view(),
        )
    }

    /// Parses the body of a Link message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and the
    /// errors of [`parse_with_offset_width`](Self::parse_with_offset_width) in the cases it lists.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<LinkMessage, FormatError> {
        Self::parse_with_offset_width(data, OffsetWidth::try_from(offset_size)?)
    }

    /// Parses and converts the complete Link message at the start of `data` to owned text.
    ///
    /// Uses the framing of [`LinkMessageView::parse_prefix`] and the text conversion of
    /// [`LinkMessageView::to_text`]. Ignores bytes after the target.
    ///
    /// # Errors
    ///
    /// Propagates the structural errors of [`LinkMessageView::parse_prefix`]. After decoding,
    /// returns [`FormatError::InvalidLinkType`] for a user-defined kind or
    /// [`FormatError::InvalidSoftLinkEncoding`] for a soft path with invalid UTF-8.
    pub fn parse_with_offset_width(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<LinkMessage, FormatError> {
        Ok(LinkMessageView::decode_prefix(data, offset_width)?
            .try_map(|view| view.to_text())
            .map_err(LinkProjectionError::into_format)?
            .into_value())
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

impl LinkMessageView<'_> {
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

impl LinkTargetView<'_> {
    fn to_text(self) -> Result<LinkTarget, LinkProjectionError> {
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
    fn into_format(self) -> FormatError {
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

#[cfg(test)]
#[path = "tests/projection.rs"]
mod tests;
