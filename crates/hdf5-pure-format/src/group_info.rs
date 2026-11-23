//! Link storage thresholds and entry estimates from version 0 Group Info messages.

use alloc::vec;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt;
use core::num::NonZeroU16;

use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::codec::{Decode, Encode};
use crate::error::FormatError;

/// The optional link thresholds and entry estimates of a version 0 Group Info message.
///
/// The parser and writer preserve each pair of fields independently, including pairs whose
/// values equal the defaults. Use [`Self::stored_phase_change`] for the stored thresholds or
/// [`Self::phase_change`] for the effective thresholds.
///
/// The message is defined in "The Group Info Message" of the
/// [format specification, version 4.0][spec].
///
/// # Examples
///
/// Stores the default link thresholds explicitly alongside the entry estimates.
///
/// ```
/// use hdf5_pure_format::__private::{GroupEstimates, GroupInfo, GroupPhaseChange};
///
/// let thresholds = GroupPhaseChange::DEFAULT;
/// let estimates = GroupEstimates { entries: 7, name_length: 20 };
/// let info = GroupInfo::new(Some(thresholds), Some(estimates));
/// let bytes = info.serialize();
/// assert_eq!(bytes, [0, 3, 8, 0, 6, 0, 7, 0, 20, 0]);
/// let parsed = GroupInfo::parse(&bytes)?;
/// assert_eq!(parsed.stored_phase_change(), Some(thresholds));
/// assert_eq!(parsed.estimates(), Some(estimates));
/// # Ok::<(), hdf5_pure_format::FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupInfo {
    flags: u8,
    phase_change: GroupPhaseChange,
    estimates: GroupEstimates,
}

impl GroupInfo {
    /// Constructs a message that stores each supplied pair, including values equal to the defaults.
    ///
    /// Pass [`None`] to omit a pair.
    pub const fn new(
        phase_change: Option<GroupPhaseChange>,
        estimates: Option<GroupEstimates>,
    ) -> Self {
        let (phase_change_flag, phase_change) = match phase_change {
            Some(value) => (STORES_PHASE_CHANGE, value),
            None => (0, GroupPhaseChange::DEFAULT),
        };
        let (estimates_flag, estimates) = match estimates {
            Some(value) => (STORES_ESTIMATES, value),
            None => (
                0,
                GroupEstimates {
                    entries: 0,
                    name_length: 0,
                },
            ),
        };
        Self {
            flags: phase_change_flag | estimates_flag,
            phase_change,
            estimates,
        }
    }

    /// Parses one complete, unpadded Group Info body.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidGroupInfoLength`] if the prefix is incomplete.
    /// For a complete prefix, checks the version and flags before the body length,
    /// returning [`FormatError::InvalidGroupInfoVersion`] for a version other than 0,
    /// [`FormatError::InvalidGroupInfoFlags`] for reserved flag bits, or
    /// [`FormatError::InvalidGroupInfoLength`] for a truncated body or extra bytes.
    /// After checking the length, returns [`FormatError::InvalidGroupInfoThresholds`]
    /// if the stored minimum dense count exceeds the maximum compact count.
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        Self::decode(bytes, ())
    }

    /// Parses a Group Info body at the start of a slice that may include object header padding.
    ///
    /// Ignores all bytes after the fields selected by the flags.
    ///
    /// # Errors
    ///
    /// Returns the errors documented by [`Self::parse`] for malformed or incomplete fields.
    pub fn parse_padded(bytes: &[u8]) -> Result<Self, FormatError> {
        GroupInfoBody::read(bytes)?.decode()
    }

    /// Reads the length of the Group Info body at the start of `bytes`.
    ///
    /// Checks the prefix and presence of all selected fields. Leaves the field values
    /// uninterpreted and ignores any bytes beyond the body.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidGroupInfoVersion`] for a version other than 0,
    /// [`FormatError::InvalidGroupInfoFlags`] for reserved flag bits, or
    /// [`FormatError::InvalidGroupInfoLength`] if the prefix or body is incomplete.
    pub fn unpadded_body_len(bytes: &[u8]) -> Result<usize, FormatError> {
        Ok(GroupInfoBody::read(bytes)?.encoded_len())
    }

    /// Returns the stored thresholds, or [`None`] when the message omits them.
    pub const fn stored_phase_change(&self) -> Option<GroupPhaseChange> {
        if self.flags & STORES_PHASE_CHANGE != 0 {
            Some(self.phase_change)
        } else {
            None
        }
    }

    /// Returns the effective thresholds, including defaults for omitted fields.
    pub const fn phase_change(&self) -> GroupPhaseChange {
        self.phase_change
    }

    /// Returns the stored estimates, or [`None`] when the estimate fields are omitted.
    pub const fn estimates(&self) -> Option<GroupEstimates> {
        if self.flags & STORES_ESTIMATES != 0 {
            Some(self.estimates)
        } else {
            None
        }
    }

    /// Returns the exact unpadded body length, including four bytes for each stored pair.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the unpadded body to the start of `output` and returns the number of bytes written.
    ///
    /// Leaves bytes beyond the body unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GroupInfoOutputTooSmall`] and leaves `output` unchanged if the body is larger
    /// than the destination.
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, GroupInfoOutputTooSmall> {
        self.encode_into(output)
    }

    /// Returns the unpadded body, including each stored pair of fields.
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![0; self.encoded_len()];
        self.write_body(&mut bytes);
        bytes
    }

    const fn prefix(&self) -> [u8; GROUP_INFO_PREFIX_LENGTH] {
        [GROUP_INFO_VERSION, self.flags]
    }

    /// Writes the body into a slice of exactly [`Self::encoded_size`] bytes.
    fn write_body(&self, body: &mut [u8]) {
        let (prefix, fields) = body.split_at_mut(GROUP_INFO_PREFIX_LENGTH);
        prefix.copy_from_slice(&self.prefix());
        let (pairs, _) = fields.as_chunks_mut::<GROUP_INFO_PAIR_LENGTH>();
        let values = [
            self.stored_phase_change()
                .map(|phase_change| [phase_change.max_compact(), phase_change.min_dense()]),
            self.estimates()
                .map(|estimates| [estimates.entries, estimates.name_length]),
        ];
        for (pair, [first, second]) in pairs.iter_mut().zip(values.into_iter().flatten()) {
            let [a, b] = first.to_le_bytes();
            let [c, d] = second.to_le_bytes();
            *pair = [a, b, c, d];
        }
    }

    /// A message with both optional pairs omitted.
    pub const DEFAULT: Self = Self::new(None, None);
    /// The two-byte body of [`Self::DEFAULT`].
    pub const DEFAULT_BODY: [u8; GROUP_INFO_PREFIX_LENGTH] = Self::DEFAULT.prefix();
}

impl<'input> Decode<'input> for GroupInfo {
    type Context = ();
    type Error = FormatError;

    fn decode(input: &'input [u8], (): ()) -> Result<Self, Self::Error> {
        let body = GroupInfoBody::read(input)?;
        let expected = body.encoded_len();
        if input.len() != expected {
            return Err(FormatError::InvalidGroupInfoLength {
                expected,
                actual: input.len(),
            });
        }
        body.decode()
    }
}

impl Encode for GroupInfo {
    type Error = GroupInfoOutputTooSmall;

    fn encoded_len(&self) -> usize {
        GROUP_INFO_PREFIX_LENGTH + self.flags.count_ones() as usize * GROUP_INFO_PAIR_LENGTH
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let required = self.encoded_len();
        let available = output.len();
        let body = output.get_mut(..required).ok_or(GroupInfoOutputTooSmall {
            required,
            available,
        })?;
        self.write_body(body);
        Ok(required)
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

impl GroupEstimates {
    fn from_fields([a, b, c, d]: [u8; GROUP_INFO_PAIR_LENGTH]) -> Self {
        Self {
            entries: u16::from_le_bytes([a, b]),
            name_length: u16::from_le_bytes([c, d]),
        }
    }
}

/// The link counts for transitions between compact and dense storage.
///
/// The version 0 Group Info message is defined in "The Group Info Message" of the
/// [format specification, version 4.0][spec].
///
/// libhdf5 defines defaults of 8 for the maximum compact count and 6 for the minimum dense
/// count (`H5G_CRT_GINFO_MAX_COMPACT` and `H5G_CRT_GINFO_MIN_DENSE` in `H5Gprivate.h`,
/// HDF5 2.2.0). [`GroupInfo::phase_change`] uses these defaults when the message omits the thresholds.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupPhaseChange(Option<GroupThresholds>);

impl GroupPhaseChange {
    /// Creates thresholds with a minimum dense count at most the maximum compact count.
    ///
    /// A zero maximum requires a zero minimum. A positive maximum with a zero minimum disables
    /// compact reversion by link count.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidGroupInfoThresholds`] if `min_dense` exceeds `max_compact`.
    pub const fn new(max_compact: u16, min_dense: u16) -> Result<Self, FormatError> {
        if min_dense > max_compact {
            return Err(FormatError::InvalidGroupInfoThresholds {
                max_compact,
                min_dense,
            });
        }
        Ok(match NonZeroU16::new(max_compact) {
            Some(max_compact) => Self::from_thresholds(GroupThresholds {
                max_compact,
                min_dense,
            }),
            None => Self::ALWAYS_DENSE,
        })
    }

    /// Constructs a policy that permits compact storage with the supplied thresholds.
    pub const fn from_thresholds(thresholds: GroupThresholds) -> Self {
        Self(Some(thresholds))
    }

    /// Returns the checked thresholds, or [`None`] for [`Self::ALWAYS_DENSE`].
    pub const fn thresholds(&self) -> Option<GroupThresholds> {
        self.0
    }

    /// Returns the largest compact link count, or zero for [`Self::ALWAYS_DENSE`].
    pub const fn max_compact(&self) -> u16 {
        match self.0 {
            Some(thresholds) => thresholds.max_compact().get(),
            None => 0,
        }
    }

    /// Returns the reversion threshold, or zero when compact reversion is disabled.
    pub const fn min_dense(&self) -> u16 {
        match self.0 {
            Some(thresholds) => thresholds.min_dense(),
            None => 0,
        }
    }

    fn from_fields([a, b, c, d]: [u8; GROUP_INFO_PAIR_LENGTH]) -> Result<Self, FormatError> {
        Self::new(u16::from_le_bytes([a, b]), u16::from_le_bytes([c, d]))
    }

    /// Uses dense storage for nonempty groups.
    pub const ALWAYS_DENSE: Self = Self(None);
    /// Allows compact storage through eight links and reversion from dense storage below six links.
    pub const DEFAULT: Self = Self::from_thresholds(GroupThresholds::DEFAULT);
}

/// An ordered threshold pair with a positive compact limit.
///
/// The minimum dense count is at most the maximum compact count. A zero minimum disables
/// reversion from dense to compact storage by link count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupThresholds {
    max_compact: NonZeroU16,
    min_dense: u16,
}

impl GroupThresholds {
    /// Constructs thresholds from constants checked at compile time.
    ///
    /// `MAX` must be positive and `MIN` must be at most `MAX`.
    pub const fn from_constants<const MAX: u16, const MIN: u16>() -> Self {
        const { assert!(MIN <= MAX, "min_dense must not exceed max_compact") };
        Self {
            max_compact: const { NonZeroU16::new(MAX).expect("max_compact must be positive") },
            min_dense: MIN,
        }
    }

    /// Returns the largest link count that permits compact storage.
    pub const fn max_compact(&self) -> NonZeroU16 {
        self.max_compact
    }

    /// Returns the minimum dense link count. Zero disables compact reversion by link count.
    pub const fn min_dense(&self) -> u16 {
        self.min_dense
    }

    /// The default compact limit of eight and reversion threshold of six links.
    pub const DEFAULT: Self = Self::from_constants::<DEFAULT_MAX_COMPACT, DEFAULT_MIN_DENSE>();
}

/// The destination is too short to write the Group Info body.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupInfoOutputTooSmall {
    /// The length of the body in bytes.
    pub required: usize,
    /// The length of the destination in bytes.
    pub available: usize,
}

impl fmt::Display for GroupInfoOutputTooSmall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Group Info encoding needs {} bytes, destination has {}",
            self.required, self.available,
        )
    }
}

impl Error for GroupInfoOutputTooSmall {}

impl From<GroupInfoOutputTooSmall> for FormatError {
    /// Reports a short destination as [`FormatError::UnexpectedEof`], preserving both lengths.
    fn from(error: GroupInfoOutputTooSmall) -> Self {
        Self::UnexpectedEof {
            expected: error.required,
            available: error.available,
        }
    }
}

enum GroupInfoBody<'a> {
    Both(&'a [u8; 2 * GROUP_INFO_PAIR_LENGTH]),
    Estimates(&'a [u8; GROUP_INFO_PAIR_LENGTH]),
    Neither,
    PhaseChange(&'a [u8; GROUP_INFO_PAIR_LENGTH]),
}

impl<'a> GroupInfoBody<'a> {
    fn read(bytes: &'a [u8]) -> Result<Self, FormatError> {
        let mut cursor = ByteCursor::new(bytes);
        let [version, flags] = cursor
            .array::<GROUP_INFO_PREFIX_LENGTH>()
            .map_err(invalid_length)?;
        if version != GROUP_INFO_VERSION {
            return Err(FormatError::InvalidGroupInfoVersion(version));
        }
        let body = match flags {
            0 => Self::Neither,
            STORES_PHASE_CHANGE => Self::PhaseChange(cursor.take_array().map_err(invalid_length)?),
            STORES_ESTIMATES => Self::Estimates(cursor.take_array().map_err(invalid_length)?),
            STORES_BOTH => Self::Both(cursor.take_array().map_err(invalid_length)?),
            _ => return Err(FormatError::InvalidGroupInfoFlags(flags)),
        };
        Ok(body)
    }

    fn encoded_len(&self) -> usize {
        GROUP_INFO_PREFIX_LENGTH
            + match self {
                Self::Neither => 0,
                Self::Estimates(pair) | Self::PhaseChange(pair) => pair.len(),
                Self::Both(pairs) => pairs.len(),
            }
    }

    fn decode(self) -> Result<GroupInfo, FormatError> {
        let (phase_change, estimates) = match self {
            Self::Neither => (None, None),
            Self::PhaseChange(pair) => (Some(GroupPhaseChange::from_fields(*pair)?), None),
            Self::Estimates(pair) => (None, Some(GroupEstimates::from_fields(*pair))),
            Self::Both(&[a, b, c, d, e, f, g, h]) => (
                Some(GroupPhaseChange::from_fields([a, b, c, d])?),
                Some(GroupEstimates::from_fields([e, f, g, h])),
            ),
        };
        Ok(GroupInfo::new(phase_change, estimates))
    }
}

#[cold]
fn invalid_length(error: UnexpectedEof) -> FormatError {
    FormatError::InvalidGroupInfoLength {
        expected: error.expected,
        actual: error.available,
    }
}

// "The Group Info Message" in the format specification, version 4.0, defines version and
// flags bytes before the optional fields.
const GROUP_INFO_PREFIX_LENGTH: usize = 2;
// "The Group Info Message" in the format specification, version 4.0, defines version zero.
const GROUP_INFO_VERSION: u8 = 0;
// "The Group Info Message" in the format specification, version 4.0, defines bit zero as
// the flag for stored phase-change thresholds.
const STORES_PHASE_CHANGE: u8 = 0x01;
// "The Group Info Message" in the format specification, version 4.0, defines bit one as
// the flag for stored entry estimates.
const STORES_ESTIMATES: u8 = 0x02;
const STORES_BOTH: u8 = STORES_PHASE_CHANGE | STORES_ESTIMATES;
// "The Group Info Message" in the format specification, version 4.0, defines each optional
// pair as two two-byte integers.
const GROUP_INFO_PAIR_LENGTH: usize = 4;
// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, defines the default maximum
// compact threshold.
const DEFAULT_MAX_COMPACT: u16 = 8;
// `H5G_CRT_GINFO_MIN_DENSE` in `H5Gprivate.h`, HDF5 2.2.0, defines the default minimum
// dense threshold.
const DEFAULT_MIN_DENSE: u16 = 6;
