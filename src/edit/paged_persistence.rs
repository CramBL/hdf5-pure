//! Pure placement and encoding of extension chunks and managers for a persistent PAGE edit.

use std::collections::BTreeSet;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::ObjectHeaderAllocationGeometry;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::{Extent, FreeClass, PagedTail, SessionSpace};

use self::layout::PageLayout;
use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::free_space_manager::{EncodedManager, ManagerProjection};
use crate::width::FormatWidths;

mod layout;

/// A proposed PAGE tail with encoded managers and complete allocation coverage.
///
/// Complete extension chunks, manager allocations, and admitted free gaps partition the
/// reservation. The proposed post-publication free-space state excludes every occupied block.
pub(super) struct PagedPersistencePlan {
    /// The complete reservation, final EOA, and free-space state to adopt.
    tail: PagedTail,
    /// Complete superblock-extension chunks in continuation order, including owned padding.
    extension: Vec<Extent>,
    /// The manager addresses to record in File Space Info, undefined for inactive managers.
    slots: [StoredAddress; NUM_FILE_FSM_MANAGERS],
    /// The manager headers and section lists at their measured lengths.
    blocks: Vec<EncodedManager>,
    /// The enclosing reservation and occupied allocation boundaries for retirement accounting.
    coverage: PublicationReservation,
}

impl PagedPersistencePlan {
    /// Plans placement using only durable free space or appended metadata pages.
    ///
    /// Places each complete extension chunk, manager header, and section list within one page
    /// when smaller than a page, or at a page boundary otherwise. Tries a contiguous layout
    /// before a layout with admitted alignment gaps. Retains the projected sections and their
    /// distinct metadata sizes for count-based measurement. Surplus owned capacity belongs to
    /// extension chunks. Returns `None` when `reuse_only` is set and no candidate layout fits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] or [`FormatError::ValueTooLargeForPlatform`] if
    /// capacity or address calculations overflow. Returns [`FormatError::Internal`] if the
    /// strategy is not PAGE, a layout cannot represent the header, or measured sections differ
    /// from the final projection or reservation partition. Returns encoding errors if managers
    /// cannot be serialized at the geometry's field widths.
    pub fn new(
        space: &SessionSpace,
        eoa: StoredAddress,
        freed: &[(Extent, FreeClass)],
        old_blocks: &[Extent],
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        reuse_only: bool,
    ) -> Result<Option<Self>, FormatError> {
        let widths = geometry.widths();
        let page = space.page_size().ok_or_else(|| {
            FormatError::Internal("PAGE persistence requires PAGE session space".into())
        })?;
        let extension_len = geometry
            .root_chunk_lengths(false)
            .map(|range| *range.start())
            .min()
            .ok_or_else(|| FormatError::Internal("a PAGE extension has no root encoding".into()))?;
        if let Some(tail) = space.plan_paged_tail(
            eoa,
            freed,
            old_blocks,
            ByteLength::new(extension_len),
            false,
        )? {
            if tail.post.admitted_snapshot().sections().next().is_none() {
                let managers = ManagerProjection::new(tail.post.admitted_snapshot(), widths)?;
                if let Some(extension) = contiguous_extension(&tail, &managers, geometry, page)? {
                    return Self::contiguous(tail, managers, extension).map(Some);
                }
            }
        }
        let section_limit = space.paged_tail_section_limit(
            freed
                .iter()
                .map(|&(extent, _)| extent)
                .chain(old_blocks.iter().copied()),
        )?;
        let capacity = ManagerCapacity::new(
            section_limit,
            RESERVATION_SECTION_LIMIT,
            extension_len,
            widths,
        )?;
        let rejected = if let Some(tail) =
            space.plan_paged_tail(eoa, freed, old_blocks, capacity.bytes.length(), false)?
        {
            let allocation = tail.allocation;
            if let Some(plan) = Self::from_reusable_tail(tail, geometry, page, widths)? {
                return Ok(Some(plan));
            }
            Some(allocation)
        } else {
            None
        };
        if rejected.is_some_and(|allocation| {
            capacity.bytes.length() < ByteLength::new(page.get())
                && allocation.start().get() / page.get()
                    != (allocation.end().get() - 1) / page.get()
        }) {
            if let Some(tail) =
                space.plan_paged_block_tail(eoa, freed, old_blocks, capacity.bytes)?
            {
                if let Some(plan) = Self::from_reusable_tail(tail, geometry, page, widths)? {
                    return Ok(Some(plan));
                }
            }
        }
        if reuse_only {
            return Ok(None);
        }
        let mut tail = space
            .plan_paged_tail(eoa, freed, old_blocks, capacity.bytes.length(), true)?
            .ok_or_else(|| {
                FormatError::Internal("an appended PAGE tail requires no reusable extent".into())
            })?;
        let mut managers = ManagerProjection::new(tail.post.admitted_snapshot(), widths)?;
        if managers.is_empty() {
            tail = space
                .plan_paged_tail(
                    eoa,
                    freed,
                    old_blocks,
                    capacity.with_spare_byte()?.bytes.length(),
                    true,
                )?
                .ok_or_else(|| {
                    FormatError::Internal(
                        "an appended PAGE tail requires no reusable extent".into(),
                    )
                })?;
            managers = ManagerProjection::new(tail.post.admitted_snapshot(), widths)?;
        }
        if let Some(extension) = contiguous_extension(&tail, &managers, geometry, page)? {
            return Self::contiguous(tail, managers, extension).map(Some);
        }
        drop(tail);
        drop(managers);
        let append = space.plan_paged_append(eoa, freed, old_blocks)?;
        let (base, sizes) = ManagerProjection::indexed(append.admitted_snapshot(), widths)?;
        let mut chosen = None;
        for controls in [false, true] {
            if let Some(layout) =
                PageLayout::new(geometry, page, eoa, None, &base, &sizes, controls)?
            {
                chosen = Some(layout);
                break;
            }
        }
        drop(base);
        drop(sizes);
        let layout = chosen.ok_or_else(|| {
            FormatError::Internal("an appended PAGE reservation has no layout".into())
        })?;
        let tail = append.reserve(ByteLength::new(
            layout.end.checked_distance_from(eoa).ok_or_else(|| {
                FormatError::Internal("a PAGE reservation ends before its start".into())
            })?,
        ))?;
        Self::placed(tail, layout, &[], widths, None).map(Some)
    }

    /// Returns the planned publication state and its complete reservation coverage.
    ///
    /// The tuple contains the proposed tail, extension chunks, File Space Info slots, encoded
    /// managers, and checked reservation coverage, in that order.
    pub(super) fn into_parts(
        self,
    ) -> (
        PagedTail,
        Vec<Extent>,
        [StoredAddress; NUM_FILE_FSM_MANAGERS],
        Vec<EncodedManager>,
        PublicationReservation,
    ) {
        let Self {
            tail,
            extension,
            slots,
            blocks,
            coverage,
        } = self;
        (tail, extension, slots, blocks, coverage)
    }

    fn from_reusable_tail(
        tail: PagedTail,
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        page: FileSpacePageSize,
        widths: FormatWidths,
    ) -> Result<Option<Self>, FormatError> {
        let (managers, sizes) = ManagerProjection::indexed(tail.post.admitted_snapshot(), widths)?;
        if managers.is_empty() {
            return Ok(None);
        }
        if let Some(extension) = contiguous_extension(&tail, &managers, geometry, page)? {
            return Self::contiguous(tail, managers, extension).map(Some);
        }
        Self::reused(tail, managers, sizes, geometry, page, widths)
    }

    fn contiguous(
        tail: PagedTail,
        managers: ManagerProjection,
        extension: Vec<Extent>,
    ) -> Result<Self, FormatError> {
        let start = extension
            .last()
            .ok_or_else(|| FormatError::Internal("a PAGE extension has no chunks".into()))?
            .end();
        let plan = managers.into_plan(start)?;
        let coverage = PublicationReservation::new(ReservationPartition::new(
            tail.allocation,
            extension.iter().copied().chain(plan.allocations()),
        )?)?;
        let slots = plan.slots();
        let blocks = plan.encode_allocations().collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            tail,
            extension,
            slots,
            blocks,
            coverage,
        })
    }

    fn reused(
        mut tail: PagedTail,
        mut base: ManagerProjection,
        mut sizes: BTreeSet<u64>,
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        page: FileSpacePageSize,
        widths: FormatWidths,
    ) -> Result<Option<Self>, FormatError> {
        let root = layout::root_allocation(geometry, page, tail.allocation.start())?;
        if !tail.allocation.range().contains(root.range()) {
            return Ok(None);
        }
        let mut boundary = Vec::new();
        if let Some(leading) = Extent::between(tail.allocation.start(), root.start()) {
            boundary.push(leading);
        }
        let mut end = tail.allocation.end();
        let offset = end.get() % page.get();
        if offset > 0 && offset < *geometry.continuation_chunk_lengths(false).start() {
            end = end.checked_sub(offset).ok_or(FormatError::OffsetOverflow {
                offset: end.get(),
                length: offset,
            })?;
            if let Some(trailing) = Extent::between(end, tail.allocation.end()) {
                boundary.push(trailing);
            }
        }
        if end <= root.end() {
            return Ok(None);
        }
        if boundary
            .iter()
            .any(|gap| !tail.allocation.range().contains(gap.range()))
        {
            return Ok(None);
        }
        if !boundary.is_empty() {
            drop(base);
            drop(sizes);
            tail = tail.with_metadata_padding(&boundary)?;
            (base, sizes) = ManagerProjection::indexed(tail.post.admitted_snapshot(), widths)?;
        }
        let mut chosen = None;
        for controls in [false, true] {
            if let Some(layout) = PageLayout::new(
                geometry,
                page,
                tail.allocation.start(),
                Some(end),
                &base,
                &sizes,
                controls,
            )? {
                chosen = Some(layout);
                break;
            }
        }
        chosen
            .map(|layout| Self::placed(tail, layout, &boundary, widths, Some((base, sizes))))
            .transpose()
    }

    fn placed(
        mut tail: PagedTail,
        layout: PageLayout,
        boundary: &[Extent],
        widths: FormatWidths,
        cached: Option<(ManagerProjection, BTreeSet<u64>)>,
    ) -> Result<Self, FormatError> {
        let PageLayout {
            extension,
            managers: placements,
            gaps,
            end: _,
            metadata_counts,
        } = layout;
        let internal = gaps
            .iter()
            .copied()
            .filter(|gap| !boundary.contains(gap))
            .collect::<Vec<_>>();
        let (managers, sizes) = if internal.is_empty() {
            match cached {
                Some(cached) => cached,
                None => ManagerProjection::indexed(tail.post.admitted_snapshot(), widths)?,
            }
        } else {
            drop(cached);
            tail = tail.with_metadata_padding(&internal)?;
            ManagerProjection::indexed(tail.post.admitted_snapshot(), widths)?
        };
        if (
            managers.sections(ManagerKind::PagedSmallMetadata).len() as u64,
            sizes.len() as u64,
        ) != metadata_counts
        {
            return Err(FormatError::Internal(
                "PAGE gap counts differ from the final manager projection".into(),
            ));
        }
        drop(sizes);
        let plan = managers.place_with_kind(|kind, lengths| {
            let allocations = placements
                .iter()
                .find(|(candidate, _)| *candidate == kind)
                .map(|(_, allocations)| *allocations)
                .ok_or_else(|| FormatError::Internal("a PAGE manager has no placement".into()))?;
            if [allocations[0].len(), allocations[1].len()] != lengths.map(ByteLength::new) {
                return Err(FormatError::Internal(
                    "PAGE manager lengths differ from their measured placement".into(),
                ));
            }
            Ok(allocations)
        })?;
        let partition = ReservationPartition::new(
            tail.allocation,
            extension.iter().copied().chain(plan.allocations()),
        )?;
        let mut expected = boundary.iter().copied().chain(internal).collect::<Vec<_>>();
        expected.sort_unstable();
        if partition.gaps != expected {
            return Err(FormatError::Internal(
                "PAGE reservation gaps differ from their admitted ranges".into(),
            ));
        }
        let coverage = PublicationReservation::new(partition)?;
        let slots = plan.slots();
        let blocks = plan.encode_allocations().collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            tail,
            extension,
            slots,
            blocks,
            coverage,
        })
    }
}

/// Complete allocations and unoccupied ranges within a stored reservation.
///
/// Allocations and gaps are ordered and disjoint, and together cover the reservation exactly.
/// Adjacent allocations retain their separate boundaries. Allocation ownership, admission, and
/// PAGE placement remain the caller's responsibility.
#[derive(Debug)]
pub(super) struct ReservationPartition {
    reservation: Extent,
    occupied: Vec<Extent>,
    gaps: Vec<Extent>,
}

impl ReservationPartition {
    /// Checks allocation containment and overlap, then records exact gaps in address order.
    ///
    /// With no allocations, the only gap is the complete reservation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if an allocation is outside `reservation` or allocations
    /// overlap.
    fn new(
        reservation: Extent,
        occupied: impl IntoIterator<Item = Extent>,
    ) -> Result<Self, FormatError> {
        let mut occupied = occupied.into_iter().collect::<Vec<_>>();
        occupied.sort_unstable();
        let mut gaps = Vec::new();
        let mut cursor = reservation.start();
        for &allocation in &occupied {
            if !reservation.range().contains(allocation.range()) {
                return Err(FormatError::Internal(
                    "a PAGE block is outside its reservation".into(),
                ));
            }
            if allocation.start() < cursor {
                return Err(FormatError::Internal(
                    "PAGE block allocations overlap".into(),
                ));
            }
            if let Some(gap) = Extent::between(cursor, allocation.start()) {
                gaps.push(gap);
            }
            cursor = allocation.end();
        }
        if let Some(gap) = Extent::between(cursor, reservation.end()) {
            gaps.push(gap);
        }
        Ok(Self {
            reservation,
            occupied,
            gaps,
        })
    }

    /// Requires complete allocation coverage and returns the occupied reservation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if any unoccupied range remains.
    fn into_occupied(self) -> Result<OccupiedReservation, FormatError> {
        let Self {
            reservation,
            occupied,
            gaps,
        } = self;
        if !gaps.is_empty() {
            return Err(FormatError::Internal(
                "a PAGE reservation has unoccupied bytes".into(),
            ));
        }
        debug_assert_eq!(
            occupied.first().map(|extent| extent.start()),
            Some(reservation.start())
        );
        debug_assert_eq!(
            occupied.last().map(|extent| extent.end()),
            Some(reservation.end())
        );
        Ok(OccupiedReservation::whole(reservation))
    }
}

/// A stored reservation completely covered by planned allocations.
///
/// The caller establishes allocation ownership and publication conditions before making the
/// storage reusable.
#[derive(Debug)]
pub(super) struct OccupiedReservation {
    reservation: Extent,
}

impl OccupiedReservation {
    /// Records coverage by one allocation occupying the whole reservation.
    fn whole(reservation: Extent) -> Self {
        Self { reservation }
    }

    /// Returns the complete stored extent reserved for the planned allocations.
    pub(super) fn reservation(&self) -> Extent {
        self.reservation
    }

    /// Returns the complete allocation boundary for retirement accounting.
    ///
    /// The caller establishes that references have been replaced and published before freeing
    /// this storage.
    pub(super) fn into_retirement(self) -> Vec<Extent> {
        vec![self.reservation]
    }
}

/// A complete reservation with separate occupied and admitted free ranges.
///
/// Retirement includes only occupied allocation boundaries. The caller establishes ownership
/// and successful reference publication before reusing those allocations.
pub(super) enum PublicationReservation {
    Occupied(OccupiedReservation),
    Partitioned(ReservationPartition),
}

impl PublicationReservation {
    fn new(partition: ReservationPartition) -> Result<Self, FormatError> {
        if partition.gaps.is_empty() {
            return partition.into_occupied().map(Self::Occupied);
        }
        Ok(Self::Partitioned(partition))
    }

    /// Returns the enclosing range reserved for publication, including its free gaps.
    pub(super) fn reservation(&self) -> Extent {
        match self {
            Self::Occupied(occupied) => occupied.reservation(),
            Self::Partitioned(partition) => partition.reservation,
        }
    }

    /// Returns occupied allocation boundaries, excluding gaps admitted during planning.
    pub(super) fn into_retirement(self) -> Vec<Extent> {
        match self {
            Self::Occupied(occupied) => occupied.into_retirement(),
            Self::Partitioned(partition) => partition.occupied,
        }
    }
}

fn contiguous_extension(
    tail: &PagedTail,
    managers: &ManagerProjection,
    geometry: &ObjectHeaderAllocationGeometry<'_>,
    page: FileSpacePageSize,
) -> Result<Option<Vec<Extent>>, FormatError> {
    let Some(len) = tail
        .allocation
        .len()
        .checked_sub(ByteLength::new(managers.encoded_len()))
        .filter(|len| !len.is_zero())
    else {
        return Ok(None);
    };
    let ranges = match geometry.contiguous_allocations(tail.allocation.start(), len.get()) {
        Ok(ranges) => ranges,
        Err(FormatError::NestingDepthExceeded | FormatError::ValueTooLargeForPlatform { .. }) => {
            return Ok(None);
        }
        Err(err) => return Err(err),
    };
    let extension = ranges
        .into_iter()
        .map(|range| {
            Extent::new(range.start(), ByteLength::new(range.length()))
                .ok_or_else(|| FormatError::Internal("a PAGE extension chunk is empty".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for &chunk in &extension {
        if layout::place(page, chunk.start(), chunk.len().get())? != chunk {
            return Ok(None);
        }
    }
    let mut cursor = extension
        .last()
        .ok_or_else(|| FormatError::Internal("a PAGE extension has no chunks".into()))?
        .end();
    for lengths in managers.block_lengths() {
        for len in lengths {
            let block = layout::place(page, cursor, len)?;
            if block.start() != cursor {
                return Ok(None);
            }
            cursor = block.end();
        }
    }
    Ok(Some(extension))
}

/// Holds a positive byte bound for PAGE managers and the superblock extension.
#[derive(Debug)]
struct ManagerCapacity {
    bytes: NonZeroByteLength,
}

impl ManagerCapacity {
    /// Combines semantic and reservation section bounds using the output field widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the combined count exceeds
    /// [`usize::MAX`], [`FormatError::OffsetOverflow`] if the byte bound exceeds [`u64::MAX`],
    /// or [`FormatError::Internal`] if the calculated capacity is zero.
    fn new(
        semantic_sections: usize,
        reservation_sections: usize,
        extension_len: u64,
        widths: FormatWidths,
    ) -> Result<Self, FormatError> {
        let section_limit = semantic_sections.checked_add(reservation_sections).ok_or(
            FormatError::ValueTooLargeForPlatform {
                value: (semantic_sections as u64).saturating_add(reservation_sections as u64),
                target: "PAGE persistence section count",
            },
        )?;
        let prefixes = format::section_info_capacity(widths, 0)? * (PAGED_MANAGERS - 1);
        let fixed = format::free_space_manager_header_len(widths) * PAGED_MANAGERS + prefixes;
        let fixed = fixed
            .checked_add(extension_len)
            .ok_or(FormatError::OffsetOverflow {
                offset: extension_len,
                length: fixed,
            })?;
        let bytes = format::section_info_capacity(widths, section_limit)?
            .checked_add(fixed)
            .ok_or(FormatError::OffsetOverflow {
                offset: fixed,
                length: section_limit as u64,
            })?;
        Ok(Self {
            bytes: NonZeroByteLength::new(bytes).ok_or_else(|| {
                FormatError::Internal("PAGE manager capacity contains no headers".into())
            })?,
        })
    }

    /// Adds one byte for the appended-tail fallback.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the capacity is [`u64::MAX`].
    fn with_spare_byte(self) -> Result<Self, FormatError> {
        Ok(Self {
            bytes: self
                .bytes
                .length()
                .checked_add(ByteLength::new(1))
                .and_then(ByteLength::nonzero)
                .ok_or(FormatError::OffsetOverflow {
                    offset: self.bytes.get(),
                    length: 1,
                })?,
        })
    }
}

const PAGED_MANAGERS: u64 = 3;
// Removing one reservation can add two partial edges and split one aligned interior.
// Appending instead restores at most one aligned run and adds one metadata-page tail.
const RESERVATION_SECTION_LIMIT: usize = 3;

#[cfg(test)]
mod tests;
