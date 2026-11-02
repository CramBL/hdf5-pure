//! Attribute storage selection from checked object-header thresholds.

use core::num::NonZeroU16;

use hdf5_pure_format::__private::AttributePhaseChange;

use crate::error::FormatError;

/// The effective compact and dense attribute thresholds for an object.
///
/// The minimum dense count is at most the maximum compact count. A zero maximum requires
/// dense storage for nonempty sets. A zero minimum disables compact reversion by count.
/// libhdf5 configures these thresholds with `H5Pset_attr_phase_change`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AttributePolicy {
    thresholds: Option<Thresholds>,
}

impl AttributePolicy {
    /// Parses stored thresholds, using the defaults when the prefix omits them.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributePhaseChange`] if the minimum dense count exceeds
    /// the maximum compact count.
    pub(crate) fn from_prefix(value: Option<AttributePhaseChange>) -> Result<Self, FormatError> {
        let Some(AttributePhaseChange {
            max_compact,
            min_dense,
        }) = value
        else {
            return Ok(Self::DEFAULT);
        };
        if min_dense > max_compact {
            return Err(FormatError::InvalidAttributePhaseChange {
                max_compact,
                min_dense,
            });
        }
        Ok(Self {
            thresholds: NonZeroU16::new(max_compact).map(|maximum| Thresholds {
                maximum,
                reversion: NonZeroU16::new(min_dense),
            }),
        })
    }

    /// Selects storage from the current form, the count change, and compact feasibility.
    ///
    /// Returns compact storage for an empty set, and dense storage for a nonempty set if
    /// compact encoding is infeasible or the maximum is zero. Otherwise, changes compact
    /// storage to dense only when the count increases above the maximum, and dense storage to
    /// compact only when the count decreases below a positive minimum.
    pub(crate) fn select(
        self,
        current: AttributeStorage,
        previous_count: usize,
        count: usize,
        compact_feasible: bool,
    ) -> AttributeStorage {
        if count == 0 {
            return AttributeStorage::Compact;
        }
        if !compact_feasible {
            return AttributeStorage::Dense;
        }
        let Some(Thresholds { maximum, reversion }) = self.thresholds else {
            return AttributeStorage::Dense;
        };
        match current {
            AttributeStorage::Compact
                if count > previous_count && count > usize::from(maximum.get()) =>
            {
                AttributeStorage::Dense
            }
            AttributeStorage::Dense
                if count >= previous_count
                    || reversion.is_none_or(|minimum| count >= usize::from(minimum.get())) =>
            {
                AttributeStorage::Dense
            }
            _ => AttributeStorage::Compact,
        }
    }

    /// The default maximum of eight compact attributes and minimum of six dense attributes.
    pub(crate) const DEFAULT: Self = Self {
        thresholds: Some(Thresholds {
            // The default maximum in `H5Opkg.h` is positive.
            maximum: NonZeroU16::new(DEFAULT_MAX_COMPACT).unwrap(),
            reversion: NonZeroU16::new(DEFAULT_MIN_DENSE),
        }),
    };
}

/// The active attribute storage form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttributeStorage {
    Compact,
    Dense,
}

/// Positive compact thresholds with optional reversion to compact storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Thresholds {
    maximum: NonZeroU16,
    /// The minimum dense count, or `None` if dense storage is retained by count.
    reversion: Option<NonZeroU16>,
}

/// The default maximum compact attribute count.
// `H5O_CRT_ATTR_MAX_COMPACT_DEF` in `H5Opkg.h`, HDF5 2.2.0, defines the default maximum.
pub(crate) const DEFAULT_MAX_COMPACT: u16 = 8;
// `H5O_CRT_ATTR_MIN_DENSE_DEF` in `H5Opkg.h`, HDF5 2.2.0, defines the default minimum.
const DEFAULT_MIN_DENSE: u16 = 6;

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::AttributePhaseChange;
    use rstest::rstest;

    use super::AttributePolicy;
    use super::AttributeStorage;
    use crate::error::FormatError;

    #[rstest]
    #[case::compact_at_max(32, 24, AttributeStorage::Compact, 32, true, AttributeStorage::Compact)]
    #[case::dense_above_max(32, 24, AttributeStorage::Compact, 33, true, AttributeStorage::Dense)]
    #[case::dense_at_min(32, 24, AttributeStorage::Dense, 24, true, AttributeStorage::Dense)]
    #[case::compact_below_min(32, 24, AttributeStorage::Dense, 23, true, AttributeStorage::Compact)]
    #[case::oversized_compact(32, 24, AttributeStorage::Compact, 1, false, AttributeStorage::Dense)]
    #[case::oversized_dense(32, 24, AttributeStorage::Dense, 23, false, AttributeStorage::Dense)]
    #[case::always_dense(0, 0, AttributeStorage::Compact, 1, true, AttributeStorage::Dense)]
    #[case::one_way(32, 0, AttributeStorage::Dense, 1, true, AttributeStorage::Dense)]
    #[case::one_way_compact(32, 0, AttributeStorage::Compact, 32, true, AttributeStorage::Compact)]
    #[case::empty(32, 24, AttributeStorage::Dense, 0, true, AttributeStorage::Compact)]
    #[case::empty_always_dense(0, 0, AttributeStorage::Dense, 0, true, AttributeStorage::Compact)]
    #[case::empty_one_way(32, 0, AttributeStorage::Dense, 0, true, AttributeStorage::Compact)]
    fn attribute_storage_uses_policy_and_current_form(
        #[case] max_compact: u16,
        #[case] min_dense: u16,
        #[case] current: AttributeStorage,
        #[case] count: usize,
        #[case] compact_feasible: bool,
        #[case] expected: AttributeStorage,
    ) {
        let policy = AttributePolicy::from_prefix(Some(AttributePhaseChange {
            max_compact,
            min_dense,
        }))
        .unwrap();
        let previous_count = match current {
            AttributeStorage::Compact => count.saturating_sub(1),
            AttributeStorage::Dense => count + 1,
        };
        assert_eq!(
            policy.select(current, previous_count, count, compact_feasible),
            expected
        );
    }

    #[rstest]
    #[case::compact_at_max(AttributeStorage::Compact, 8, AttributeStorage::Compact)]
    #[case::dense_above_max(AttributeStorage::Compact, 9, AttributeStorage::Dense)]
    #[case::dense_at_min(AttributeStorage::Dense, 6, AttributeStorage::Dense)]
    #[case::compact_below_min(AttributeStorage::Dense, 5, AttributeStorage::Compact)]
    fn omitted_attribute_policy_uses_default_thresholds(
        #[case] current: AttributeStorage,
        #[case] count: usize,
        #[case] expected: AttributeStorage,
    ) {
        assert_eq!(
            AttributePolicy::from_prefix(None).unwrap().select(
                current,
                match current {
                    AttributeStorage::Compact => count - 1,
                    AttributeStorage::Dense => count + 1,
                },
                count,
                true
            ),
            expected
        );
    }

    #[rstest]
    #[case::compact_above_max(AttributeStorage::Compact, 9, AttributeStorage::Compact)]
    #[case::dense_below_min(AttributeStorage::Dense, 5, AttributeStorage::Dense)]
    fn unchanged_count_preserves_storage(
        #[case] current: AttributeStorage,
        #[case] count: usize,
        #[case] expected: AttributeStorage,
    ) {
        assert_eq!(
            AttributePolicy::DEFAULT.select(current, count, count, true),
            expected
        );
    }

    #[rstest]
    #[case::zero_maximum(0, 1)]
    #[case::reversed(7, 8)]
    fn invalid_attribute_thresholds_preserve_the_stored_pair(
        #[case] max_compact: u16,
        #[case] min_dense: u16,
    ) {
        assert_eq!(
            AttributePolicy::from_prefix(Some(AttributePhaseChange {
                max_compact,
                min_dense,
            }))
            .unwrap_err(),
            FormatError::InvalidAttributePhaseChange {
                max_compact,
                min_dense,
            }
        );
    }
}
