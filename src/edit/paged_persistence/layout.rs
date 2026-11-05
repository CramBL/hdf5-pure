//! PAGE layouts with measured manager storage and bounded adjustment of free-section counts.
//!
//! Extension chunks, manager allocations, and free gaps partition the proposed reservation.
//! The caller can adjust section-list lengths within one count-field width using one-byte gaps
//! between padding chunks, then admits the gaps and checks the final manager projection.

use std::collections::BTreeSet;

use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_format::__private::ObjectHeaderAllocationGeometry;
use hdf5_pure_format::__private::SectionInfoCounts;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::ReservationRequest;

use crate::address::StoredAddress;
use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;
use crate::free_space_manager::ManagerProjection;
use crate::width::FormatWidths;

/// Complete stored block placements and free gaps before encoding or session adoption.
pub(super) struct PageLayout {
    pub(super) extension: Vec<Extent>,
    pub(super) managers: Vec<(ManagerKind, [Extent; 2])>,
    pub(super) gaps: Vec<Extent>,
    pub(super) end: StoredAddress,
    /// The small metadata manager's section and distinct-size counts, in that order.
    pub(super) metadata_counts: (u64, u64),
}

impl PageLayout {
    /// Places blocks within a reusable reservation, or determines an appended reservation's end.
    ///
    /// `end` is present for reuse. With `controls`, reserves padding chunks separated by
    /// one-byte free gaps and searches a bounded count window for a representable suffix.
    /// Returns `None` if the selected reservation or count window cannot represent the layout.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if address or count calculations overflow, or
    /// [`FormatError::Internal`] if an appended start is unaligned or the calculated padding
    /// cannot be represented. Propagates format measurement errors.
    pub(super) fn new(
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        page: FileSpacePageSize,
        start: StoredAddress,
        end: Option<StoredAddress>,
        base: &ManagerProjection,
        sizes: &BTreeSet<u64>,
        controls: bool,
    ) -> Result<Option<Self>, FormatError> {
        if end.is_none() && start.get() % page.get() != 0 {
            return Err(FormatError::Internal(
                "an appended PAGE reservation is not aligned".into(),
            ));
        }
        let widths = geometry.widths();
        let continued = geometry.continuation_chunk_lengths(true);
        let terminal = geometry.continuation_chunk_lengths(false);
        let minimum = *continued.start();
        let step = u64::from(widths.offsets.get()) + 1;
        let window = (minimum - 1).div_ceil(step);
        let appended = end.is_none();
        let section_count = add(
            base.sections(ManagerKind::PagedSmallMetadata).len() as u64,
            u64::from(appended),
        )?;
        let control_count = if controls {
            control_count(section_count, window)?
        } else {
            0
        };
        let arena = Arena::new(geometry, page, start, minimum, control_count)?;
        let mut gaps = arena.leading.into_iter().collect::<Vec<_>>();
        let mut cursor = arena.end();
        let mut summary = SectionSummary {
            sections: base.sections(ManagerKind::PagedSmallMetadata).len() as u64,
            sizes,
            extra: Vec::new(),
        };
        let mut managers = Vec::new();
        for (kind, lengths) in base
            .measurements()
            .filter(|(kind, _)| *kind != ManagerKind::PagedSmallMetadata)
        {
            let header = place_after(page, &mut cursor, lengths[0], &mut gaps, &mut summary)?;
            let list = place_after(page, &mut cursor, lengths[1], &mut gaps, &mut summary)?;
            managers.push((kind, [header, list]));
        }
        let metadata = summary.sections > 0 || appended || controls;
        let mut metadata_header = None;
        let mut metadata_start = cursor;
        let mut upper_len = 0;
        if metadata {
            let header = place_after(
                page,
                &mut cursor,
                hdf5_pure_format::__private::free_space_manager_header_len(widths),
                &mut gaps,
                &mut summary,
            )?;
            metadata_header = Some(header);
            upper_len = summary.measure(widths, control_count, None, appended)?;
            let list = place(page, cursor, upper_len)?;
            metadata_start = list.start();
            if let Some(gap) = Extent::between(cursor, metadata_start) {
                summary.insert(gap.len())?;
                gaps.push(gap);
                upper_len = summary.measure(widths, control_count, None, appended)?;
            }
        }
        let upper_end = metadata_start.checked_add(upper_len)?;
        let end = match end {
            Some(end) => end,
            None => {
                let end = upper_end.checked_add(minimum)?;
                let offset = end.get() % page.get();
                end.checked_add(if offset == 0 {
                    *terminal.start()
                } else {
                    terminal.start().saturating_sub(offset)
                })?
            }
        };
        if upper_end.checked_add(minimum)? > end {
            return Ok(None);
        }
        let tail_size = appended.then(|| page.get() - end.get() % page.get());
        let first = if controls { control_count - window } else { 0 };
        for active in first..=control_count {
            let list_len = if metadata {
                summary.measure(widths, active, tail_size, false)?
            } else {
                0
            };
            let suffix_start = metadata_start.checked_add(list_len)?;
            let remaining = page.get() - suffix_start.get() % page.get();
            if suffix_start.get() % page.get() != 0
                && remaining < minimum
                && end > suffix_start.checked_add(remaining)?
            {
                continue;
            }
            let suffix = extent(suffix_start, distance(suffix_start, end)?)?;
            let (extension, control_gaps) = match arena.chunks(page, geometry, active, suffix) {
                Ok(chunks) => chunks,
                Err(FormatError::NestingDepthExceeded) => continue,
                Err(error) => return Err(error),
            };
            let mut selected_gaps = gaps.iter().copied().chain(control_gaps).collect::<Vec<_>>();
            if let Some(header) = metadata_header {
                managers.push((
                    ManagerKind::PagedSmallMetadata,
                    [header, extent(metadata_start, list_len)?],
                ));
            }
            selected_gaps.sort_unstable();
            let counts = summary.counts(active, tail_size, false)?;
            return Ok(Some(Self {
                extension,
                managers,
                gaps: selected_gaps,
                end,
                metadata_counts: counts,
            }));
        }
        if controls && appended {
            return Err(FormatError::Internal(
                "a bounded PAGE control window has no representable suffix".into(),
            ));
        }
        Ok(None)
    }
}

/// A root chunk followed by padding cells available for one-byte free gaps.
struct Arena {
    root: Extent,
    cells: Vec<Extent>,
    leading: Option<Extent>,
}

impl Arena {
    fn new(
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        page: FileSpacePageSize,
        start: StoredAddress,
        minimum: u64,
        count: u64,
    ) -> Result<Self, FormatError> {
        let mut root = root_allocation(geometry, page, start)?;
        let leading = Extent::between(start, root.start());
        let mut cells = Vec::<Extent>::new();
        let mut cursor = root.end();
        if count > 0 {
            let cell_len = minimum + 1;
            let first = place(page, cursor, cell_len)?;
            if first.start() != cursor {
                root = extent(
                    root.start(),
                    add(root.len(), distance(cursor, first.start())?)?,
                )?;
                cursor = root.end();
            }
            for _ in 0..=count {
                let cell = place(page, cursor, cell_len)?;
                if cell.start() != cursor {
                    let previous = cells.pop().ok_or_else(|| {
                        FormatError::Internal("a PAGE control arena has no preceding cell".into())
                    })?;
                    cells.push(extent(
                        previous.start(),
                        distance(previous.start(), cell.start())?,
                    )?);
                }
                cells.push(cell);
                cursor = cell.end();
            }
        }
        Ok(Self {
            root,
            cells,
            leading,
        })
    }

    fn end(&self) -> StoredAddress {
        let Self {
            root,
            cells,
            leading: _,
        } = self;
        cells.last().copied().unwrap_or(*root).end()
    }

    fn chunks(
        &self,
        page: FileSpacePageSize,
        geometry: &ObjectHeaderAllocationGeometry<'_>,
        active: u64,
        suffix: Extent,
    ) -> Result<(Vec<Extent>, Vec<Extent>), FormatError> {
        let Self {
            root,
            cells,
            leading: _,
        } = self;
        let mut extension = vec![*root];
        let mut gaps = Vec::new();
        let mut cursor = root.end();
        let active =
            usize::try_from(active).map_err(|_overflow| FormatError::ValueTooLargeForPlatform {
                value: active,
                target: "PAGE control count",
            })?;
        for cell in cells.iter().take(active) {
            let gap_start = cell
                .end()
                .checked_sub(1)
                .ok_or(FormatError::OffsetOverflow {
                    offset: cell.end().get(),
                    length: 1,
                })?;
            if let Some(owned) = Extent::between(cursor, gap_start) {
                padding_chunks(page, geometry, owned, false, &mut extension)?;
            }
            gaps.push(extent(gap_start, 1)?);
            cursor = cell.end();
        }
        if let Some(last) = cells.last() {
            if let Some(owned) = Extent::between(cursor, last.end()) {
                padding_chunks(page, geometry, owned, false, &mut extension)?;
            }
        }
        padding_chunks(page, geometry, suffix, true, &mut extension)?;
        Ok((extension, gaps))
    }
}

fn control_count(sections: u64, window: u64) -> Result<u64, FormatError> {
    let mut count = window + 1;
    loop {
        let first = add(sections, count - window)?;
        let last = add(add(sections, count)?, MAX_ALIGNMENT_GAPS)?;
        if SectionInfoCounts::new(first, 1)?.count_width()
            == SectionInfoCounts::new(last, 1)?.count_width()
        {
            return Ok(count);
        }
        count = add(count, 1)?;
        if count > 2 * window + MAX_ALIGNMENT_GAPS + 1 {
            return Err(FormatError::Internal(
                "PAGE control counts cross more than one width boundary".into(),
            ));
        }
    }
}

/// Chooses a PAGE chunk 0 placement with spare representable padding capacity.
///
/// Uses the smallest length in a root-size range that can grow by more than one minimum
/// continued chunk. Computes geometry only. The caller establishes ownership and availability.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if no root-size range has sufficient padding capacity, or
/// [`FormatError::OffsetOverflow`] if placement or the exclusive end exceeds [`u64::MAX`].
pub(super) fn root_allocation(
    geometry: &ObjectHeaderAllocationGeometry<'_>,
    page: FileSpacePageSize,
    start: StoredAddress,
) -> Result<Extent, FormatError> {
    let minimum = *geometry.continuation_chunk_lengths(true).start();
    let len = geometry
        .root_chunk_lengths(true)
        .find(|range| range.end().saturating_sub(*range.start()) > minimum)
        .map(|range| *range.start())
        .ok_or_else(|| FormatError::Internal("a PAGE root has no padding capacity".into()))?;
    place(page, start, len)
}

/// Metadata section counts and additional sizes relative to one retained projection.
struct SectionSummary<'a> {
    sections: u64,
    sizes: &'a BTreeSet<u64>,
    extra: Vec<u64>,
}

impl SectionSummary<'_> {
    fn insert(&mut self, size: u64) -> Result<(), FormatError> {
        let Self {
            sections,
            sizes,
            extra,
        } = self;
        *sections = add(*sections, 1)?;
        if !sizes.contains(&size) && !extra.contains(&size) {
            extra.push(size);
        }
        Ok(())
    }

    fn counts(
        &self,
        controls: u64,
        tail: Option<u64>,
        unknown_tail: bool,
    ) -> Result<(u64, u64), FormatError> {
        let Self {
            sections,
            sizes,
            extra,
        } = self;
        let sections = add(
            add(*sections, controls)?,
            u64::from(tail.is_some() || unknown_tail),
        )?;
        let mut groups = add(sizes.len() as u64, extra.len() as u64)?;
        let contains = |size| sizes.contains(&size) || extra.contains(&size);
        if controls > 0 && !contains(1) {
            groups = add(groups, 1)?;
        }
        if unknown_tail || tail.is_some_and(|size| !contains(size) && (controls == 0 || size != 1))
        {
            groups = add(groups, 1)?;
        }
        Ok((sections, groups))
    }

    fn measure(
        &self,
        widths: FormatWidths,
        controls: u64,
        tail: Option<u64>,
        unknown_tail: bool,
    ) -> Result<u64, FormatError> {
        let (sections, groups) = self.counts(controls, tail, unknown_tail)?;
        SectionInfoCounts::new(sections, groups)?.encoded_len(widths)
    }
}

fn place_after(
    page: FileSpacePageSize,
    cursor: &mut StoredAddress,
    len: u64,
    gaps: &mut Vec<Extent>,
    summary: &mut SectionSummary<'_>,
) -> Result<Extent, FormatError> {
    let allocation = place(page, *cursor, len)?;
    if let Some(gap) = Extent::between(*cursor, allocation.start()) {
        summary.insert(gap.len())?;
        gaps.push(gap);
    }
    *cursor = allocation.end();
    Ok(allocation)
}

fn padding_chunks(
    page: FileSpacePageSize,
    geometry: &ObjectHeaderAllocationGeometry<'_>,
    allocation: Extent,
    terminal: bool,
    chunks: &mut Vec<Extent>,
) -> Result<(), FormatError> {
    let continued = geometry.continuation_chunk_lengths(true);
    let last = geometry.continuation_chunk_lengths(false);
    let minimum = *continued.start();
    let maximum = (*continued.end()).min(*last.end());
    let mut cursor = allocation.start();
    while cursor < allocation.end() {
        if chunks.len() >= ObjectHeaderAllocationGeometry::MAX_CHUNKS {
            return Err(FormatError::NestingDepthExceeded);
        }
        let remaining = distance(cursor, allocation.end())?;
        let partial = page.get() - cursor.get() % page.get();
        let segment = if cursor.get() % page.get() != 0 {
            remaining.min(partial)
        } else if page.get() <= maximum {
            let whole = maximum / page.get() * page.get();
            if remaining > maximum {
                whole
            } else {
                remaining
            }
        } else {
            remaining.min(page.get())
        };
        let count = segment.div_ceil(maximum);
        if count > ObjectHeaderAllocationGeometry::MAX_CHUNKS.saturating_sub(chunks.len()) as u64 {
            return Err(FormatError::NestingDepthExceeded);
        }
        let final_segment = segment == remaining;
        let last_min = if terminal && final_segment {
            *last.start()
        } else {
            minimum
        };
        let required = add(
            (count - 1)
                .checked_mul(minimum)
                .ok_or(FormatError::OffsetOverflow {
                    offset: count - 1,
                    length: minimum,
                })?,
            last_min,
        )?;
        if required > segment {
            return Err(FormatError::Internal(
                "PAGE padding cannot fill its planned page fragment".into(),
            ));
        }
        let mut left = segment;
        for index in 0..count {
            let reserve = if index + 1 == count {
                0
            } else {
                (count - index - 2) * minimum + last_min
            };
            let len = maximum.min(left - reserve);
            let chunk = extent(cursor, len)?;
            chunks.push(chunk);
            cursor = chunk.end();
            left -= len;
        }
    }
    Ok(())
}

/// Computes the first legal PAGE block range at or after a stored address.
///
/// Blocks smaller than a page stay within one page. Blocks at least one page long start at a
/// page boundary.
/// The caller establishes ownership and availability before reserving or writing the range.
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if `len` is zero, or [`FormatError::OffsetOverflow`] if
/// alignment or the exclusive end exceeds [`u64::MAX`].
pub(super) fn place(
    page: FileSpacePageSize,
    start: StoredAddress,
    len: u64,
) -> Result<Extent, FormatError> {
    let size = NonZeroByteLength::new(len)
        .ok_or_else(|| FormatError::Internal("a PAGE block is empty".into()))?;
    Ok(ReservationRequest::new(size, AllocationClass::Metadata)
        .for_block()
        .place_after(start, Some(page))?
        .extent())
}

fn extent(start: StoredAddress, len: u64) -> Result<Extent, FormatError> {
    Extent::new(start, len).ok_or(FormatError::OffsetOverflow {
        offset: start.get(),
        length: len,
    })
}

fn distance(start: StoredAddress, end: StoredAddress) -> Result<u64, FormatError> {
    end.checked_distance_from(start)
        .ok_or_else(|| FormatError::Internal("a PAGE range ends before it starts".into()))
}

fn add(left: u64, right: u64) -> Result<u64, FormatError> {
    left.checked_add(right).ok_or(FormatError::OffsetOverflow {
        offset: left,
        length: right,
    })
}

const MAX_ALIGNMENT_GAPS: u64 = 6;

#[cfg(test)]
mod tests;
