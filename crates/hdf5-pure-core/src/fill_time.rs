//! Dataset fill value write time policies.

use crate::FormatError;

/// The policy for when to write the fill value to dataset storage.
///
/// The fill value write time is stored in a dataset's fill value message. The C library calls
/// this setting `H5D_fill_time_t` and sets it with `H5Pset_fill_time`.
///
/// The policy is defined in "The Data Storage - Fill Value Message" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::FillTime;
///
/// assert_eq!(FillTime::default(), FillTime::IfSet);
/// assert!(FillTime::Alloc.is_alloc());
/// assert!(FillTime::Never.is_never());
/// assert!(FillTime::IfSet.is_if_set());
/// assert_eq!(FillTime::try_from(0), Ok(FillTime::Alloc));
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fvmessage
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FillTime {
    /// Always writes the fill value when storage space is allocated.
    ///
    /// The C library calls this value `H5D_FILL_TIME_ALLOC`.
    Alloc = 0,
    /// Never writes the fill value to storage.
    ///
    /// The C library calls this value `H5D_FILL_TIME_NEVER`.
    Never = 1,
    /// Writes the fill value when storage space is allocated only if the caller defines a fill value.
    ///
    /// The C library calls this value `H5D_FILL_TIME_IFSET`.
    #[default]
    IfSet = 2,
}

impl FillTime {
    /// Returns the raw format byte for this policy.
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Returns `true` if the policy is [`Alloc`].
    ///
    /// [`Alloc`]: Self::Alloc
    pub const fn is_alloc(self) -> bool {
        matches!(self, Self::Alloc)
    }

    /// Returns `true` if the policy is [`Never`].
    ///
    /// [`Never`]: Self::Never
    pub const fn is_never(self) -> bool {
        matches!(self, Self::Never)
    }

    /// Returns `true` if the policy is [`IfSet`].
    ///
    /// [`IfSet`]: Self::IfSet
    pub const fn is_if_set(self) -> bool {
        matches!(self, Self::IfSet)
    }
}

impl TryFrom<u8> for FillTime {
    type Error = FormatError;

    /// Converts a raw format byte into a fill time policy.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFillTime`] if `value` is not 0, 1, or 2.
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Alloc),
            1 => Ok(Self::Never),
            2 => Ok(Self::IfSet),
            other => Err(FormatError::InvalidFillTime(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(0, FillTime::Alloc)]
    #[case(1, FillTime::Never)]
    #[case(2, FillTime::IfSet)]
    fn parses_valid_fill_time(#[case] raw: u8, #[case] expected: FillTime) {
        assert_eq!(FillTime::try_from(raw).unwrap(), expected);
        assert_eq!(expected.to_u8(), raw);
    }

    #[rstest]
    #[case(3)]
    #[case(4)]
    #[case(255)]
    fn rejects_invalid_fill_time(#[case] raw: u8) {
        let err = FillTime::try_from(raw).unwrap_err();
        assert_eq!(err, FormatError::InvalidFillTime(raw));
    }

    #[test]
    fn default_is_if_set() {
        assert_eq!(FillTime::default(), FillTime::IfSet);
        assert!(FillTime::IfSet.is_if_set());
        assert!(FillTime::Alloc.is_alloc());
        assert!(FillTime::Never.is_never());
    }
}
