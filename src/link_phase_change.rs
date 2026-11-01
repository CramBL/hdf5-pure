//! Checked thresholds for compact and dense group link storage.

use core::num::NonZeroU16;

use hdf5_pure_format::__private::{GroupInfoError, GroupPhaseChange};

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

    /// Returns the maximum compact link count, or zero for [`Self::AlwaysDense`].
    pub const fn max_compact(self) -> u16 {
        match self {
            Self::AlwaysDense => 0,
            Self::Thresholds(thresholds) => thresholds.max_compact.get(),
        }
    }

    /// Returns the minimum dense link count, or zero when compact reversion is disabled.
    ///
    /// Returns zero for [`Self::AlwaysDense`] as well.
    pub const fn min_dense(self) -> u16 {
        match self {
            Self::AlwaysDense => 0,
            Self::Thresholds(thresholds) => match thresholds.compact_reversion {
                CompactReversion::Below(min_dense) => min_dense.get(),
                CompactReversion::Disabled => 0,
            },
        }
    }
}

impl TryFrom<GroupPhaseChange> for LinkPhaseChange {
    type Error = GroupInfoError;

    /// Converts parsed Group Info thresholds to a storage policy.
    ///
    /// # Errors
    ///
    /// Returns [`GroupInfoError::Thresholds`] if the minimum dense count exceeds the maximum
    /// compact count.
    fn try_from(value: GroupPhaseChange) -> Result<Self, Self::Error> {
        let max_compact = value.max_compact();
        let min_dense = value.min_dense();
        if min_dense > max_compact {
            return Err(GroupInfoError::Thresholds {
                max_compact,
                min_dense,
            });
        }
        Ok(match NonZeroU16::new(max_compact) {
            Some(max_compact) => Self::Thresholds(LinkThresholds {
                max_compact,
                compact_reversion: match NonZeroU16::new(min_dense) {
                    Some(min_dense) => CompactReversion::Below(min_dense),
                    None => CompactReversion::Disabled,
                },
            }),
            None => Self::AlwaysDense,
        })
    }
}

impl Default for LinkPhaseChange {
    fn default() -> Self {
        Self::builder()
            .max_compact::<DEFAULT_MAX_COMPACT>()
            .min_dense::<DEFAULT_MIN_DENSE>()
            .build()
    }
}

/// A positive compact threshold and a checked compact reversion policy.
///
/// Construct this value through [`LinkPhaseChange::builder`]. A positive minimum dense count is
/// at most the maximum compact count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkThresholds {
    max_compact: NonZeroU16,
    compact_reversion: CompactReversion,
}

impl LinkThresholds {
    /// Returns the largest link count that permits compact storage.
    pub const fn max_compact(self) -> NonZeroU16 {
        self.max_compact
    }

    /// Returns the condition for reverting from dense to compact storage.
    pub const fn compact_reversion(self) -> CompactReversion {
        self.compact_reversion
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
        const { assert!(MAX > 0, "max_compact must be positive") };
        LinkThresholdsBuilder {
            compact_reversion: CompactReversion::Disabled,
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
    compact_reversion: CompactReversion,
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
        const {
            assert!(MIN > 0, "min_dense must be positive");
            assert!(MIN <= MAX, "min_dense must not exceed max_compact");
        }
        LinkThresholdsBuilder {
            // The inline `const` assertion rejects every zero `MIN` at compilation.
            compact_reversion: CompactReversion::Below(NonZeroU16::new(MIN).unwrap()),
        }
    }

    /// Keeps dense storage regardless of the link count after a transition to dense storage.
    pub const fn disable_compact_reversion(self) -> LinkThresholdsBuilder<MAX, true> {
        LinkThresholdsBuilder {
            compact_reversion: CompactReversion::Disabled,
        }
    }
}

impl<const MAX: u16> LinkThresholdsBuilder<MAX, true> {
    /// Completes the policy with the chosen maximum and reversion condition.
    pub const fn build(self) -> LinkPhaseChange {
        LinkPhaseChange::Thresholds(LinkThresholds {
            // `max_compact` checks that `MAX` is positive before either reversion choice is available.
            max_compact: NonZeroU16::new(MAX).unwrap(),
            compact_reversion: self.compact_reversion,
        })
    }
}

// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, defines the default maximum.
const DEFAULT_MAX_COMPACT: u16 = 8;
// `H5G_CRT_GINFO_MIN_DENSE` in `H5Gprivate.h`, HDF5 2.2.0, defines the default minimum.
const DEFAULT_MIN_DENSE: u16 = 6;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::omitted(&[0, 0], LinkPhaseChange::default())]
    #[case::thresholds(&[0, 1, 12, 0, 8, 0], LinkPhaseChange::builder().max_compact::<12>().min_dense::<8>().build())]
    #[case::always_dense(&[0, 1, 0, 0, 0, 0], LinkPhaseChange::AlwaysDense)]
    #[case::one_way(&[0, 1, 12, 0, 0, 0], LinkPhaseChange::builder().max_compact::<12>().disable_compact_reversion().build())]
    #[case::equal(&[0, 1, 12, 0, 12, 0], LinkPhaseChange::builder().max_compact::<12>().min_dense::<12>().build())]
    #[case::maximum(&[0, 1, 255, 255, 255, 255], LinkPhaseChange::builder().max_compact::<65_535>().min_dense::<65_535>().build())]
    fn parsed_group_info_converts_to_effective_policy(
        #[case] bytes: &[u8],
        #[case] expected: LinkPhaseChange,
    ) {
        assert_eq!(
            LinkPhaseChange::try_from(GroupPhaseChange::parse(bytes).unwrap()).unwrap(),
            expected
        );
    }
}
