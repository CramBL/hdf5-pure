use alloc::vec::Vec;

use super::encode::PreparedLink;
use super::error::LinkWireError;
use super::view::{LinkMessageView, LinkTargetView};
use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::width::UintWidth;

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

    pub(super) fn is_valid(bytes: &[u8]) -> bool {
        !bytes.is_empty() && bytes != b"." && !bytes.contains(&0) && !bytes.contains(&b'/')
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// The target fields for each version 1 Link-message kind.
///
/// [`LinkMessageV1::new`](super::wire::LinkMessageV1::new) checks symbolic paths and their encoded
/// lengths before storing a target.
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
    pub(super) fn from_view(link: &LinkMessageView<'_>) -> Self {
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

    pub(super) fn as_view(&self) -> LinkTargetView<'_> {
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

pub(super) fn is_symbolic_path(path: &[u8]) -> bool {
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

pub(super) const USER_LINK_MIN: u8 = 65;
