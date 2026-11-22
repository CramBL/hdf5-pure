//! Checked thresholds for compact and dense group link storage.

use core::num::NonZeroU16;

use hdf5_pure_format::__private::GroupInfo;
use hdf5_pure_format::__private::GroupPhaseChange;
use hdf5_pure_format::__private::GroupThresholds;

#[cfg(feature = "std")]
use crate::{FormatError, MessageType, OBJECT_HEADER_MESSAGE_MAX};

/// The thresholds for transitions between compact and dense group link storage.
///
/// Compact storage holds links in the object header. Dense storage holds links in a fractal heap.
/// The default thresholds are eight links for compact storage and six for dense storage.
/// libhdf5 configures these thresholds with `H5Pset_link_phase_change`.
///
/// # Examples
///
/// ```
/// use hdf5_pure::LinkPhaseChange;
///
/// const POLICY: LinkPhaseChange = LinkPhaseChange::builder()
///     .max_compact::<12>()
///     .min_dense::<8>()
///     .build();
/// assert_eq!((POLICY.max_compact(), POLICY.min_dense()), (12, 8));
///
/// const ONE_WAY: LinkPhaseChange = LinkPhaseChange::builder()
///     .max_compact::<12>()
///     .disable_compact_reversion()
///     .build();
/// assert_eq!((ONE_WAY.max_compact(), ONE_WAY.min_dense()), (12, 0));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkPhaseChange {
    /// Uses dense storage for nonempty groups.
    AlwaysDense,
    /// Uses positive compact thresholds with optional reversion from dense storage.
    Thresholds(LinkThresholds),
}

impl LinkPhaseChange {
    /// Starts construction of a policy with positive compact thresholds.
    ///
    /// Choose a maximum with [`LinkPhaseChangeBuilder::max_compact`], then a reversion policy
    /// with [`LinkThresholdsBuilder::min_dense`] or
    /// [`LinkThresholdsBuilder::disable_compact_reversion`].
    pub const fn builder() -> LinkPhaseChangeBuilder {
        LinkPhaseChangeBuilder { _private: () }
    }

    /// Converts effective Group Info thresholds to a storage policy.
    pub(crate) fn from_group_phase_change(value: GroupPhaseChange) -> Self {
        match value.thresholds() {
            Some(thresholds) => Self::Thresholds(LinkThresholds { thresholds }),
            None => Self::AlwaysDense,
        }
    }

    /// Parses Group Info, excluding version 1 object header padding.
    ///
    /// Retains stored entry estimates independently of the link thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidGroupInfoVersion`], [`FormatError::InvalidGroupInfoFlags`],
    /// [`FormatError::InvalidGroupInfoLength`], or [`FormatError::InvalidGroupInfoThresholds`]
    /// for malformed metadata. Returns [`FormatError::ObjectHeaderMessageTooLarge`] if `bytes`
    /// exceeds the message length range.
    #[cfg(feature = "std")]
    pub(crate) fn stored_group_info(
        bytes: &[u8],
        header_version: u8,
    ) -> Result<GroupInfo, FormatError> {
        if bytes.len() > OBJECT_HEADER_MESSAGE_MAX {
            return Err(FormatError::ObjectHeaderMessageTooLarge {
                message_type: MessageType::GROUP_INFO.to_u16(),
                size: bytes.len(),
            });
        }
        if header_version == PADDED_HEADER_VERSION {
            GroupInfo::parse_padded(bytes)
        } else {
            GroupInfo::parse(bytes)
        }
    }

    /// Builds Group Info for a new group with only non-default thresholds stored.
    pub(crate) fn group_info(self) -> GroupInfo {
        let phase_change = self.group_phase_change();
        GroupInfo::new(
            (phase_change != GroupPhaseChange::DEFAULT).then_some(phase_change),
            None,
        )
    }

    pub(crate) const fn group_phase_change(self) -> GroupPhaseChange {
        match self {
            Self::AlwaysDense => GroupPhaseChange::ALWAYS_DENSE,
            Self::Thresholds(thresholds) => {
                GroupPhaseChange::from_thresholds(thresholds.thresholds)
            }
        }
    }

    /// Returns the maximum compact link count, or zero for [`Self::AlwaysDense`].
    pub const fn max_compact(self) -> u16 {
        self.group_phase_change().max_compact()
    }

    /// Returns the minimum dense link count, or zero when compact reversion is disabled.
    ///
    /// Returns zero for [`Self::AlwaysDense`] as well.
    pub const fn min_dense(self) -> u16 {
        self.group_phase_change().min_dense()
    }

    /// The default maximum compact count of eight and minimum dense count of six.
    pub(crate) const DEFAULT: Self = Self::Thresholds(LinkThresholds {
        thresholds: GroupThresholds::DEFAULT,
    });
}

impl Default for LinkPhaseChange {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A positive compact threshold and a checked compact reversion policy.
///
/// Construct this value through [`LinkPhaseChange::builder`]. A positive minimum dense count is
/// at most the maximum compact count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkThresholds {
    thresholds: GroupThresholds,
}

impl LinkThresholds {
    /// Returns the largest link count that permits compact storage.
    pub const fn max_compact(self) -> NonZeroU16 {
        self.thresholds.max_compact()
    }

    /// Returns the condition for reverting from dense to compact storage.
    pub const fn compact_reversion(self) -> CompactReversion {
        match NonZeroU16::new(self.thresholds.min_dense()) {
            Some(min_dense) => CompactReversion::Below(min_dense),
            None => CompactReversion::Disabled,
        }
    }
}

/// The condition for reverting from dense to compact group link storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactReversion {
    /// Reverts when the link count falls below this positive threshold.
    Below(NonZeroU16),
    /// Keeps dense storage regardless of the link count.
    Disabled,
}

/// A link policy builder awaiting its maximum compact count.
///
/// Construct this builder with [`LinkPhaseChange::builder`].
///
/// # Examples
///
/// A policy requires a maximum and a reversion choice before it can be built:
///
/// ```compile_fail,E0599
/// use hdf5_pure::LinkPhaseChange;
///
/// const POLICY: LinkPhaseChange = LinkPhaseChange::builder().build();
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkPhaseChangeBuilder {
    _private: (),
}

impl LinkPhaseChangeBuilder {
    /// Sets the largest link count that permits compact storage.
    ///
    /// `MAX` must be positive. An invalid argument is rejected at compile time.
    ///
    /// # Examples
    ///
    /// Zero is excluded:
    ///
    /// ```compile_fail,E0080
    /// use hdf5_pure::LinkPhaseChange;
    ///
    /// const POLICY: LinkPhaseChange = LinkPhaseChange::builder()
    ///     .max_compact::<0>().min_dense::<1>().build();
    /// ```
    ///
    /// The maximum must fit in a [`u16`]:
    ///
    /// ```compile_fail
    /// use hdf5_pure::LinkPhaseChange;
    ///
    /// const POLICY: LinkPhaseChange = LinkPhaseChange::builder()
    ///     .max_compact::<65_537>().min_dense::<1>().build();
    /// ```
    pub const fn max_compact<const MAX: u16>(self) -> LinkThresholdsBuilder<MAX> {
        LinkThresholdsBuilder {
            thresholds: GroupThresholds::from_constants::<MAX, 0>(),
        }
    }
}

/// A link policy builder with a positive maximum compact count.
///
/// `COMPLETE` records whether a compact reversion policy has been chosen. Only a complete builder
/// has a [`build`](Self::build) method.
///
/// # Examples
///
/// ```compile_fail,E0599
/// use hdf5_pure::LinkPhaseChange;
///
/// const POLICY: LinkPhaseChange = LinkPhaseChange::builder().max_compact::<12>().build();
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkThresholdsBuilder<const MAX: u16, const COMPLETE: bool = false> {
    thresholds: GroupThresholds,
}

impl<const MAX: u16> LinkThresholdsBuilder<MAX> {
    /// Enables compact reversion below `MIN` links.
    ///
    /// `MIN` must be positive and at most `MAX`. An invalid argument is rejected at compile time.
    ///
    /// # Examples
    ///
    /// Zero is excluded. Use [`Self::disable_compact_reversion`] to disable reversion:
    ///
    /// ```compile_fail,E0080
    /// use hdf5_pure::LinkPhaseChange;
    ///
    /// const POLICY: LinkPhaseChange = LinkPhaseChange::builder()
    ///     .max_compact::<12>().min_dense::<0>().build();
    /// ```
    ///
    /// The minimum cannot exceed the maximum:
    ///
    /// ```compile_fail,E0080
    /// use hdf5_pure::LinkPhaseChange;
    ///
    /// const POLICY: LinkPhaseChange = LinkPhaseChange::builder()
    ///     .max_compact::<12>().min_dense::<13>().build();
    /// ```
    ///
    /// The minimum must fit in a [`u16`]:
    ///
    /// ```compile_fail
    /// use hdf5_pure::LinkPhaseChange;
    ///
    /// const POLICY: LinkPhaseChange = LinkPhaseChange::builder()
    ///     .max_compact::<65_535>().min_dense::<65_537>().build();
    /// ```
    pub const fn min_dense<const MIN: u16>(self) -> LinkThresholdsBuilder<MAX, true> {
        const { assert!(MIN > 0, "min_dense must be positive") };
        LinkThresholdsBuilder {
            thresholds: GroupThresholds::from_constants::<MAX, MIN>(),
        }
    }

    /// Keeps dense storage regardless of the link count after a transition to dense storage.
    pub const fn disable_compact_reversion(self) -> LinkThresholdsBuilder<MAX, true> {
        LinkThresholdsBuilder {
            thresholds: GroupThresholds::from_constants::<MAX, 0>(),
        }
    }
}

impl<const MAX: u16> LinkThresholdsBuilder<MAX, true> {
    /// Completes the policy with the chosen maximum and reversion condition.
    pub const fn build(self) -> LinkPhaseChange {
        LinkPhaseChange::Thresholds(LinkThresholds {
            thresholds: self.thresholds,
        })
    }
}

// "Version 1 Data Object Header Prefix" in the format specification, version 4.0, defines
// version one with padded message bodies.
#[cfg(feature = "std")]
const PADDED_HEADER_VERSION: u8 = 1;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn oversized_group_info_reports_the_object_message_limit() {
        assert_eq!(
            LinkPhaseChange::stored_group_info(&vec![0; 65_536], 2).unwrap_err(),
            FormatError::ObjectHeaderMessageTooLarge {
                message_type: MessageType::GROUP_INFO.to_u16(),
                size: 65_536,
            }
        );
    }

    #[rstest]
    #[case::omitted(&[0, 0], LinkPhaseChange::default())]
    #[case::thresholds(&[0, 1, 12, 0, 8, 0], LinkPhaseChange::builder().max_compact::<12>().min_dense::<8>().build())]
    #[case::always_dense(&[0, 1, 0, 0, 0, 0], LinkPhaseChange::AlwaysDense)]
    #[case::one_way(&[0, 1, 12, 0, 0, 0], LinkPhaseChange::builder().max_compact::<12>().disable_compact_reversion().build())]
    #[case::equal(&[0, 1, 12, 0, 12, 0], LinkPhaseChange::builder().max_compact::<12>().min_dense::<12>().build())]
    #[case::maximum(&[0, 1, 255, 255, 255, 255], LinkPhaseChange::builder().max_compact::<65_535>().min_dense::<65_535>().build())]
    #[case::maximum_one_way(&[0, 1, 255, 255, 0, 0], LinkPhaseChange::builder().max_compact::<65_535>().disable_compact_reversion().build())]
    #[case::minimum_positive(&[0, 1, 1, 0, 1, 0], LinkPhaseChange::builder().max_compact::<1>().min_dense::<1>().build())]
    #[case::minimum_one_way(&[0, 1, 1, 0, 0, 0], LinkPhaseChange::builder().max_compact::<1>().disable_compact_reversion().build())]
    fn parsed_group_info_converts_to_effective_policy(
        #[case] bytes: &[u8],
        #[case] expected: LinkPhaseChange,
    ) {
        let phase_change = GroupInfo::parse(bytes).unwrap().phase_change();
        let policy = LinkPhaseChange::from_group_phase_change(phase_change);
        assert_eq!(policy, expected);
        assert_eq!(policy.group_phase_change(), phase_change);
    }
}
