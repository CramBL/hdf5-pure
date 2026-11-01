//! Pure placement and encoding of the managers for a persistent PAGE edit.

use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::{Extent, FreeClass, PagedTail, SessionSpace};

use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::free_space_manager::{EncodedManager, ManagerProjection};
use crate::width::FormatWidths;

/// Holds the encoded managers and the same final free-space state for adoption.
pub(super) struct PagedPersistencePlan {
    /// The complete reservation, final EOA, and free-space state to adopt.
    pub tail: PagedTail,
    /// The manager addresses to record in File Space Info, undefined for inactive managers.
    pub slots: [StoredAddress; NUM_FILE_FSM_MANAGERS],
    /// The encoded managers, with surplus capacity owned by the final section list.
    pub blocks: Vec<EncodedManager>,
}

impl PagedPersistencePlan {
    /// Plans placement using only durable free space or appended metadata pages.
    ///
    /// Bounds section-list capacity before placement, including the known metadata-page tail.
    /// Encodes surplus capacity as section-list padding, so allocation does not depend on exact
    /// serialized-size convergence. Returns `None` when `reuse_only` is set and no reservation fits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] or [`FormatError::ValueTooLargeForPlatform`] if
    /// capacity or address calculations overflow, [`FormatError::Internal`] if the strategy is
    /// not PAGE or the capacity does not contain the managers, or an encoding error if the
    /// managers cannot be serialized at `widths`.
    pub fn new(
        space: &SessionSpace,
        eoa: StoredAddress,
        freed: &[(Extent, FreeClass)],
        old_blocks: &[Extent],
        extension_len: u64,
        widths: FormatWidths,
        reuse_only: bool,
    ) -> Result<Option<Self>, FormatError> {
        space.page_size().ok_or_else(|| {
            FormatError::Internal("PAGE persistence requires PAGE session space".into())
        })?;
        if let Some(tail) = space.plan_paged_tail(eoa, freed, old_blocks, extension_len, false)? {
            if ManagerProjection::new(tail.post.admitted_snapshot(), widths)?.is_empty() {
                return Ok(Some(Self {
                    tail,
                    slots: [StoredAddress::undefined(widths.offsets); NUM_FILE_FSM_MANAGERS],
                    blocks: Vec::new(),
                }));
            }
        }
        let section_limit = space.paged_tail_section_limit(
            freed
                .iter()
                .map(|&(extent, _)| extent)
                .chain(old_blocks.iter().copied()),
        )?;
        let prefixes = format::section_info_capacity(widths, 0)? * (PAGED_MANAGERS - 1);
        let fixed = format::free_space_manager_header_len(widths) * PAGED_MANAGERS + prefixes;
        let fixed = fixed
            .checked_add(extension_len)
            .ok_or(FormatError::OffsetOverflow {
                offset: extension_len,
                length: fixed,
            })?;
        let mut capacity = format::section_info_capacity(widths, section_limit)?
            .checked_add(fixed)
            .ok_or(FormatError::OffsetOverflow {
                offset: fixed,
                length: section_limit as u64,
            })?;
        let reused = space
            .plan_paged_tail(eoa, freed, old_blocks, capacity, false)?
            .map(|tail| ProjectedTail::new(tail, widths))
            .transpose()?;
        let mut projected = match reused {
            Some(projected) if !projected.managers.is_empty() => projected,
            _ if reuse_only => return Ok(None),
            _ => ProjectedTail::new(
                space
                    .plan_paged_tail(eoa, freed, old_blocks, capacity, true)?
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "an appended PAGE tail requires no reusable extent".into(),
                        )
                    })?,
                widths,
            )?,
        };
        if projected.managers.is_empty() {
            capacity = capacity.checked_add(1).ok_or(FormatError::OffsetOverflow {
                offset: capacity,
                length: 1,
            })?;
            projected = ProjectedTail::new(
                space
                    .plan_paged_tail(eoa, freed, old_blocks, capacity, true)?
                    .ok_or_else(|| {
                        FormatError::Internal(
                            "an appended PAGE tail requires no reusable extent".into(),
                        )
                    })?,
                widths,
            )?;
        }
        let ProjectedTail { tail, managers } = projected;
        let measured = managers.encoded_len();
        let used = extension_len
            .checked_add(measured)
            .ok_or(FormatError::OffsetOverflow {
                offset: extension_len,
                length: measured,
            })?;
        if used > capacity || managers.is_empty() {
            return Err(FormatError::Internal(
                "the PAGE manager capacity does not contain the final manager set".into(),
            ));
        }
        let start = tail.allocation.start().checked_add(extension_len)?;
        let plan = managers.into_plan(start)?;
        let slots = plan.slots;
        let blocks = plan
            .encode(tail.allocation.end())?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            tail,
            slots,
            blocks,
        }))
    }
}

/// A proposed PAGE tail and the manager projection measured from its free state.
struct ProjectedTail {
    tail: PagedTail,
    managers: ManagerProjection,
}

impl ProjectedTail {
    fn new(tail: PagedTail, widths: FormatWidths) -> Result<Self, FormatError> {
        let managers = ManagerProjection::new(tail.post.admitted_snapshot(), widths)?;
        Ok(Self { tail, managers })
    }
}

const PAGED_MANAGERS: u64 = 3;

#[cfg(test)]
mod tests;
