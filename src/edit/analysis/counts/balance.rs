//! Checked count arithmetic for original references and prospective additions.

use core::num::NonZeroU32;

/// An original count remainder with deferred prospective additions.
///
/// The remainder is reduced by successful removals. Additions are retained at [`u64`] width until
/// [`Self::remaining`] checks the final net count against [`u32::MAX`]. Source consistency,
/// reference identity, retirement eligibility, and publication permission require separate evidence.
pub(in crate::edit) struct CountBalance {
    original: u32,
    remainder: u32,
    added: u64,
}

impl CountBalance {
    /// Starts with `original` and defers the prospective additions in `added`.
    pub(in crate::edit) fn new(original: u32, added: u64) -> Self {
        Self {
            original,
            remainder: original,
            added,
        }
    }

    /// Subtracts `removed` from the remaining original count.
    ///
    /// Prospective additions are excluded from this subtraction. On error, the current remainder
    /// is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CountArithmeticError::Underflow`] if cumulative original-reference removals would
    /// exceed the original count.
    pub(in crate::edit) fn remove(
        &mut self,
        removed: NonZeroU32,
    ) -> Result<(), CountArithmeticError> {
        let Self {
            original,
            remainder,
            added,
        } = self;
        *remainder =
            remainder
                .checked_sub(removed.get())
                .ok_or(CountArithmeticError::Underflow {
                    original: *original,
                    added: *added,
                    removed: u64::from(*original - *remainder) + u64::from(removed.get()),
                })?;
        Ok(())
    }

    /// Returns whether the original-count remainder and prospective additions are both zero.
    pub(in crate::edit) fn is_zero(&self) -> bool {
        self.remainder == 0 && self.added == 0
    }

    /// Returns the final net count after all successful removals.
    ///
    /// Adds the deferred prospective count to the original-count remainder.
    ///
    /// # Errors
    ///
    /// Returns [`CountArithmeticError::Overflow`] if the net count exceeds [`u32::MAX`].
    pub(in crate::edit) fn remaining(self) -> Result<u32, CountArithmeticError> {
        let Self {
            original,
            remainder,
            added,
        } = self;
        u32::try_from(added)
            .ok()
            .and_then(|added| remainder.checked_add(added))
            .ok_or(CountArithmeticError::Overflow {
                original,
                added,
                removed: u64::from(original - remainder),
            })
    }
}

/// An underflow during removal or an overflow in the net count.
#[derive(Debug, Eq, PartialEq)]
pub(in crate::edit) enum CountArithmeticError {
    /// The net count exceeds [`u32::MAX`].
    Overflow {
        /// The original input count.
        original: u32,
        /// The prospective addition count.
        added: u64,
        /// The cumulative number of successfully removed original references.
        removed: u64,
    },
    /// Attempted original-reference removals exceed the original count.
    Underflow {
        /// The original input count.
        original: u32,
        /// The prospective addition count.
        added: u64,
        /// The cumulative removal count, including the rejected batch.
        removed: u64,
    },
}

#[cfg(test)]
mod tests;
