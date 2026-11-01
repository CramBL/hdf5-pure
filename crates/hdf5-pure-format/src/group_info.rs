//! Link storage thresholds and entry estimates from version 0 Group Info messages.

use alloc::vec::Vec;

/// The link thresholds and optional entry estimates of a version 0 Group Info message.
///
/// Entry estimates are independent of the link thresholds. Their presence is retained when
/// parsing and serializing, including estimates equal to the defaults.
///
/// The message is defined in "The Group Info Message" of the
/// [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private::{GroupEstimates, GroupInfo, GroupPhaseChange};
///
/// let info = GroupInfo::new(GroupPhaseChange::new(12, 8).unwrap())
///     .with_estimates(Some(GroupEstimates { entries: 7, name_length: 20 }));
/// let bytes = info.serialize();
/// assert_eq!(bytes, [0, 3, 12, 0, 8, 0, 7, 0, 20, 0]);
/// assert_eq!(GroupInfo::parse(&bytes).unwrap(), info);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupInfo {
    phase_change: GroupPhaseChange,
    estimates: Option<GroupEstimates>,
}

impl GroupInfo {
    /// Parses a complete message, retaining any stored entry estimates.
    ///
    /// # Errors
    ///
    /// Returns [`GroupInfoError::Version`] if the version is not zero,
    /// [`GroupInfoError::Flags`] if reserved flag bits are set, or
    /// [`GroupInfoError::Length`] if the byte count differs from the message length.
    /// Returns [`GroupInfoError::TrailingBytes`] if the byte count exceeds [`u16::MAX`],
    /// or [`GroupInfoError::Thresholds`] if the minimum dense count exceeds the maximum
    /// compact count.
    pub fn parse(bytes: &[u8]) -> Result<Self, GroupInfoError> {
        let phase_change = GroupPhaseChange::parse(bytes)?;
        let estimates = if bytes
            .get(1)
            .is_some_and(|flags| flags & STORES_ESTIMATES != 0)
        {
            let &[.., a, b, c, d] = bytes else {
                return Err(GroupInfoError::Length {
                    expected: GROUP_INFO_PREFIX_LENGTH + GROUP_INFO_PAIR_LENGTH,
                    actual: u16::try_from(bytes.len())
                        .map_err(|_error| GroupInfoError::TrailingBytes)?,
                });
            };
            Some(GroupEstimates {
                entries: u16::from_le_bytes([a, b]),
                name_length: u16::from_le_bytes([c, d]),
            })
        } else {
            None
        };
        Ok(Self {
            phase_change,
            estimates,
        })
    }

    /// Creates a message using default entry estimates.
    pub const fn new(phase_change: GroupPhaseChange) -> Self {
        Self {
            phase_change,
            estimates: None,
        }
    }

    /// Sets the optional stored entry estimates.
    ///
    /// Pass [`None`] to omit the estimate fields. The format defines defaults of four entries and
    /// eight bytes per link name.
    pub const fn with_estimates(mut self, estimates: Option<GroupEstimates>) -> Self {
        self.estimates = estimates;
        self
    }

    /// Returns the effective thresholds, including defaults for omitted fields.
    pub const fn phase_change(&self) -> GroupPhaseChange {
        self.phase_change
    }

    /// Returns the stored estimates, or [`None`] when the estimate fields are omitted.
    pub const fn estimates(&self) -> Option<GroupEstimates> {
        self.estimates
    }

    /// Serializes the message with default link thresholds omitted.
    ///
    /// Writes the estimate fields whenever [`Self::estimates`] returns [`Some`], including
    /// estimates equal to the defaults.
    pub fn serialize(&self) -> Vec<u8> {
        let Self {
            phase_change,
            estimates,
        } = self;
        let stores_phase_change = phase_change.max_compact != DEFAULT_MAX_COMPACT
            || phase_change.min_dense != DEFAULT_MIN_DENSE;
        let flags = if stores_phase_change {
            STORES_PHASE_CHANGE
        } else {
            0
        } | if estimates.is_some() {
            STORES_ESTIMATES
        } else {
            0
        };
        let mut bytes = Vec::with_capacity(
            usize::from(GROUP_INFO_PREFIX_LENGTH)
                + usize::from(GROUP_INFO_PAIR_LENGTH)
                    * (usize::from(stores_phase_change) + usize::from(estimates.is_some())),
        );
        bytes.extend_from_slice(&[GROUP_INFO_VERSION, flags]);
        if stores_phase_change {
            bytes.extend_from_slice(&phase_change.max_compact.to_le_bytes());
            bytes.extend_from_slice(&phase_change.min_dense.to_le_bytes());
        }
        if let Some(GroupEstimates {
            entries,
            name_length,
        }) = estimates
        {
            bytes.extend_from_slice(&entries.to_le_bytes());
            bytes.extend_from_slice(&name_length.to_le_bytes());
        }
        bytes
    }
}

/// The entry count and link name length estimates for sizing a group's object header.
///
/// The fields are defined in "The Group Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupEstimates {
    /// The estimated number of links in the group.
    pub entries: u16,
    /// The estimated length of a link name, in bytes.
    pub name_length: u16,
}

/// The link counts for transitions between compact and dense storage.
///
/// The version 0 Group Info message is defined in "The Group Info Message" of the
/// [format specification, version 4.0][spec].
///
/// libhdf5 defines defaults of 8 for the maximum compact count and 6 for the minimum dense
/// count (`H5G_CRT_GINFO_MAX_COMPACT` and `H5G_CRT_GINFO_MIN_DENSE` in `H5Gprivate.h`,
/// HDF5 2.2.0). [`Self::parse`] uses these defaults when the message omits the thresholds.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupPhaseChange {
    max_compact: u16,
    min_dense: u16,
}

impl GroupPhaseChange {
    /// Creates thresholds with a minimum dense count at most the maximum compact count.
    ///
    /// A zero maximum requires a zero minimum. A positive maximum with a zero minimum disables
    /// compact reversion by link count.
    ///
    /// # Errors
    ///
    /// Returns [`GroupInfoError::Thresholds`] if `min_dense` exceeds `max_compact`.
    pub const fn new(max_compact: u16, min_dense: u16) -> Result<Self, GroupInfoError> {
        if min_dense > max_compact {
            return Err(GroupInfoError::Thresholds {
                max_compact,
                min_dense,
            });
        }
        Ok(Self {
            max_compact,
            min_dense,
        })
    }

    /// Parses the thresholds, using defaults when the message omits them.
    ///
    /// Checks the message length, including any optional entry estimates.
    ///
    /// # Errors
    ///
    /// Returns [`GroupInfoError::Version`] if the version is not zero,
    /// [`GroupInfoError::Flags`] if reserved flag bits are set, or
    /// [`GroupInfoError::Length`] if the byte count differs from the message length.
    /// Returns [`GroupInfoError::TrailingBytes`] if the byte count exceeds [`u16::MAX`],
    /// or [`GroupInfoError::Thresholds`] if the minimum dense count exceeds the maximum
    /// compact count.
    pub fn parse(bytes: &[u8]) -> Result<Self, GroupInfoError> {
        let DeclaredLength {
            flags,
            expected,
            actual,
        } = DeclaredLength::read(bytes)?;
        if actual != u16::from(expected) {
            return Err(GroupInfoError::Length { expected, actual });
        }
        let (max_compact, min_dense) = if flags & STORES_PHASE_CHANGE != 0 {
            let &[_version, _flags, a, b, c, d, ..] = bytes else {
                return Err(GroupInfoError::Length { expected, actual });
            };
            (u16::from_le_bytes([a, b]), u16::from_le_bytes([c, d]))
        } else {
            (DEFAULT_MAX_COMPACT, DEFAULT_MIN_DENSE)
        };
        Self::new(max_compact, min_dense)
    }

    /// Returns the length of the message at the start of `bytes`, computed from its flags byte.
    ///
    /// The bytes past that length are left unread, such as the padding after a message in a
    /// version 1 object header.
    ///
    /// # Errors
    ///
    /// Returns [`GroupInfoError::Version`] if the version is not zero,
    /// [`GroupInfoError::Flags`] if reserved flag bits are set, or
    /// [`GroupInfoError::Length`] if `bytes` is shorter than the message.
    /// Returns [`GroupInfoError::TrailingBytes`] if the byte count exceeds [`u16::MAX`].
    pub fn encoded_len(bytes: &[u8]) -> Result<usize, GroupInfoError> {
        let DeclaredLength {
            flags: _,
            expected,
            actual,
        } = DeclaredLength::read(bytes)?;
        if actual < u16::from(expected) {
            return Err(GroupInfoError::Length { expected, actual });
        }
        Ok(usize::from(expected))
    }

    /// Returns the maximum number of links to store in the object header.
    pub fn max_compact(&self) -> u16 {
        self.max_compact
    }

    /// Returns the minimum number of links to store in the fractal heap.
    pub fn min_dense(&self) -> u16 {
        self.min_dense
    }
}

/// The flags byte of a Group Info message, the message length for those flags, and the number of
/// bytes passed.
struct DeclaredLength {
    flags: u8,
    expected: u8,
    actual: u16,
}

impl DeclaredLength {
    /// Parses the version and the flags byte at the start of `bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`GroupInfoError::Version`] if the version is not zero,
    /// [`GroupInfoError::Flags`] if reserved flag bits are set, or
    /// [`GroupInfoError::Length`] if `bytes` is shorter than two bytes.
    /// Returns [`GroupInfoError::TrailingBytes`] if the byte count exceeds [`u16::MAX`].
    fn read(bytes: &[u8]) -> Result<Self, GroupInfoError> {
        let actual = u16::try_from(bytes.len()).map_err(|_error| GroupInfoError::TrailingBytes)?;
        let &[version, flags, ..] = bytes else {
            return Err(GroupInfoError::Length {
                expected: GROUP_INFO_PREFIX_LENGTH,
                actual,
            });
        };
        if version != GROUP_INFO_VERSION {
            return Err(GroupInfoError::Version(version));
        }
        if flags & !(STORES_PHASE_CHANGE | STORES_ESTIMATES) != 0 {
            return Err(GroupInfoError::Flags(flags));
        }
        let expected = GROUP_INFO_PREFIX_LENGTH
            + GROUP_INFO_PAIR_LENGTH
                * (u8::from(flags & STORES_PHASE_CHANGE != 0)
                    + u8::from(flags & STORES_ESTIMATES != 0));
        Ok(Self {
            flags,
            expected,
            actual,
        })
    }
}

/// An error in the encoding or thresholds of a Group Info message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupInfoError {
    /// Reserved flag bits are set in the stored flags byte.
    Flags(u8),
    /// The byte count differs from the required message length.
    Length { expected: u8, actual: u16 },
    /// The minimum dense count exceeds the maximum compact count.
    Thresholds { max_compact: u16, min_dense: u16 },
    /// The supplied byte count exceeds the parser's [`u16`] length range.
    TrailingBytes,
    /// The stored version byte is unsupported.
    Version(u8),
}

// "The Group Info Message" in the format specification, version 4.0, defines version and
// flags bytes before the optional fields.
const GROUP_INFO_PREFIX_LENGTH: u8 = 2;
// "The Group Info Message" in the format specification, version 4.0, defines version zero.
const GROUP_INFO_VERSION: u8 = 0;
// "The Group Info Message" in the format specification, version 4.0, defines bit zero as
// the flag for stored phase-change thresholds.
const STORES_PHASE_CHANGE: u8 = 0x01;
// "The Group Info Message" in the format specification, version 4.0, defines bit one as
// the flag for stored entry estimates.
const STORES_ESTIMATES: u8 = 0x02;
// "The Group Info Message" in the format specification, version 4.0, defines each optional
// pair as two two-byte integers.
const GROUP_INFO_PAIR_LENGTH: u8 = 4;
// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, defines the default maximum
// compact threshold.
const DEFAULT_MAX_COMPACT: u16 = 8;
// `H5G_CRT_GINFO_MIN_DENSE` in `H5Gprivate.h`, HDF5 2.2.0, defines the default minimum
// dense threshold.
const DEFAULT_MIN_DENSE: u16 = 6;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::defaults(8, 6, None, &[0, 0])]
    #[case::thresholds(12, 8, None, &[0, 1, 12, 0, 8, 0])]
    #[case::one_way(12, 0, None, &[0, 1, 12, 0, 0, 0])]
    #[case::always_dense(0, 0, None, &[0, 1, 0, 0, 0, 0])]
    #[case::equal(1, 1, None, &[0, 1, 1, 0, 1, 0])]
    #[case::high_bytes(513, 257, None, &[0, 1, 1, 2, 1, 1])]
    #[case::maximum(u16::MAX, u16::MAX, None, &[0, 1, 255, 255, 255, 255])]
    #[case::estimates(8, 6, Some(GroupEstimates { entries: 7, name_length: 20 }), &[0, 2, 7, 0, 20, 0])]
    #[case::zero_estimates(8, 6, Some(GroupEstimates { entries: 0, name_length: 0 }), &[0, 2, 0, 0, 0, 0])]
    #[case::explicit_default_estimates(8, 6, Some(GroupEstimates { entries: 4, name_length: 8 }), &[0, 2, 4, 0, 8, 0])]
    #[case::both(12, 8, Some(GroupEstimates { entries: 513, name_length: u16::MAX }), &[0, 3, 12, 0, 8, 0, 1, 2, 255, 255])]
    fn group_info_encoding_preserves_independent_estimates(
        #[case] max_compact: u16,
        #[case] min_dense: u16,
        #[case] estimates: Option<GroupEstimates>,
        #[case] bytes: &[u8],
    ) {
        let phase_change = GroupPhaseChange::new(max_compact, min_dense).unwrap();
        let info = GroupInfo::new(phase_change).with_estimates(estimates);
        assert_eq!(info.phase_change(), phase_change);
        assert_eq!(info.estimates(), estimates);
        assert_eq!(info.serialize(), bytes);
        assert_eq!(GroupInfo::parse(bytes).unwrap(), info);
    }

    #[rstest]
    #[case::thresholds(&[0, 1, 8, 0, 6, 0], &[0, 0])]
    #[case::both(&[0, 3, 8, 0, 6, 0, 7, 0, 20, 0], &[0, 2, 7, 0, 20, 0])]
    fn encoding_omits_default_thresholds_and_preserves_estimates(
        #[case] bytes: &[u8],
        #[case] expected: &[u8],
    ) {
        assert_eq!(GroupInfo::parse(bytes).unwrap().serialize(), expected);
    }

    #[rstest]
    #[case::inverted(3, 4)]
    #[case::zero_maximum(0, 1)]
    fn constructing_inverted_thresholds_fails(#[case] max_compact: u16, #[case] min_dense: u16) {
        assert_eq!(
            GroupPhaseChange::new(max_compact, min_dense).unwrap_err(),
            GroupInfoError::Thresholds {
                max_compact,
                min_dense
            }
        );
    }

    #[rstest]
    #[case::defaults(&[0, 0, 0, 0, 0, 0, 0, 0], Ok(2))]
    #[case::phase_change(&[0, 1, 12, 0, 4, 0, 0, 0], Ok(6))]
    #[case::both(&[0, 3, 12, 0, 4, 0, 7, 0, 20, 0, 0, 0, 0, 0, 0, 0], Ok(10))]
    #[case::truncated(&[0, 1, 12, 0], Err(GroupInfoError::Length { expected: 6, actual: 4 }))]
    #[case::version(&[1, 0, 0, 0, 0, 0, 0, 0], Err(GroupInfoError::Version(1)))]
    fn the_encoded_length_excludes_bytes_past_the_declared_fields(
        #[case] bytes: &[u8],
        #[case] expected: Result<usize, GroupInfoError>,
    ) {
        assert_eq!(GroupPhaseChange::encoded_len(bytes), expected);
    }

    #[rstest]
    #[case::defaults(&[0, 0], 8, 6)]
    #[case::phase_change(&[0, 1, 12, 0, 4, 0], 12, 4)]
    #[case::estimates(&[0, 2, 7, 0, 20, 0], 8, 6)]
    #[case::both(&[0, 3, 12, 0, 4, 0, 7, 0, 20, 0], 12, 4)]
    fn reads_phase_change_thresholds(
        #[case] bytes: &[u8],
        #[case] max_compact: u16,
        #[case] min_dense: u16,
    ) {
        let thresholds = GroupPhaseChange::parse(bytes).unwrap();
        assert_eq!(
            thresholds,
            GroupPhaseChange {
                max_compact,
                min_dense
            }
        );
        assert_eq!(thresholds.max_compact(), max_compact);
        assert_eq!(thresholds.min_dense(), min_dense);
    }

    #[rstest]
    #[case::prefix(&[0], GroupInfoError::Length { expected: 2, actual: 1 })]
    #[case::phase_change(&[0, 1], GroupInfoError::Length { expected: 6, actual: 2 })]
    #[case::estimates(&[0, 2], GroupInfoError::Length { expected: 6, actual: 2 })]
    #[case::both(&[0, 3, 12, 0, 8, 0, 7, 0], GroupInfoError::Length { expected: 10, actual: 8 })]
    #[case::trailing(&[0, 0, 0], GroupInfoError::Length { expected: 2, actual: 3 })]
    #[case::oversized(&vec![0; 65_536], GroupInfoError::TrailingBytes)]
    #[case::flags(&[0, 0x80], GroupInfoError::Flags(0x80))]
    #[case::version(&[1, 0], GroupInfoError::Version(1))]
    #[case::thresholds(&[0, 1, 3, 0, 4, 0], GroupInfoError::Thresholds { max_compact: 3, min_dense: 4 })]
    fn rejects_malformed_group_info(#[case] bytes: &[u8], #[case] expected: GroupInfoError) {
        assert_eq!(GroupPhaseChange::parse(bytes).unwrap_err(), expected);
        assert_eq!(GroupInfo::parse(bytes).unwrap_err(), expected);
    }
}
