use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::CStr;
use core::fmt;
use core::str::{self, Utf8Error};

use crate::FormatError;
use crate::display::{self, EscapedName, QuotedBytes};

/// A datatype member name containing no NUL bytes.
///
/// The stored bytes exclude the terminator and any padding. Names are compared by their bytes.
/// Use [`as_str`](Self::as_str) to interpret the name as UTF-8.
///
/// [`Display`](fmt::Display) escapes UTF-8 text or prints byte escapes for names outside UTF-8.
/// It elides long names. [`Debug`](fmt::Debug) quotes the complete name.
#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DatatypeMemberName(Box<[u8]>);

impl DatatypeMemberName {
    /// Copies a name from a byte slice or string.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DatatypeMemberNameContainsNul`] if the name contains a NUL byte.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::DatatypeMemberName;
    ///
    /// let text = DatatypeMemberName::new("ready")?;
    /// assert_eq!(text.as_str(), Ok("ready"));
    ///
    /// let name = DatatypeMemberName::new(b"ready\xff")?;
    /// assert_eq!(name.as_bytes(), b"ready\xff");
    /// assert!(name.as_str().is_err());
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn new(bytes: impl AsRef<[u8]>) -> Result<Self, FormatError> {
        let bytes = bytes.as_ref();
        Self::check(bytes)?;
        Ok(Self(bytes.into()))
    }

    /// Borrows the name bytes without adding a terminator.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Borrows the name as UTF-8 text.
    ///
    /// # Errors
    ///
    /// Returns the original [`Utf8Error`] from [`str::from_utf8`] if the name is invalid UTF-8.
    pub fn as_str(&self) -> Result<&str, Utf8Error> {
        str::from_utf8(self.as_bytes())
    }

    fn check(bytes: &[u8]) -> Result<(), FormatError> {
        if let Some(index) = bytes.iter().position(|&byte| byte == 0) {
            return Err(FormatError::DatatypeMemberNameContainsNul { index });
        }
        Ok(())
    }
}

impl AsRef<[u8]> for DatatypeMemberName {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl From<&CStr> for DatatypeMemberName {
    /// Copies the name up to the NUL terminator.
    fn from(name: &CStr) -> Self {
        Self(name.to_bytes().into())
    }
}

impl TryFrom<Vec<u8>> for DatatypeMemberName {
    type Error = FormatError;

    /// Takes ownership of the name bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DatatypeMemberNameContainsNul`] if the name contains a NUL byte.
    fn try_from(bytes: Vec<u8>) -> Result<Self, FormatError> {
        Self::check(&bytes)?;
        Ok(Self(bytes.into_boxed_slice()))
    }
}

impl fmt::Display for DatatypeMemberName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(name) = self.as_str() {
            return fmt::Display::fmt(&EscapedName(name), f);
        }
        display::write_escaped_name(
            f,
            self.as_bytes()
                .iter()
                .flat_map(|&byte| core::ascii::escape_default(byte).map(char::from)),
        )
    }
}

impl fmt::Debug for DatatypeMemberName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Ok(name) => fmt::Debug::fmt(name, f),
            Err(_) => write!(f, "b{}", QuotedBytes(self.as_bytes())),
        }
    }
}
