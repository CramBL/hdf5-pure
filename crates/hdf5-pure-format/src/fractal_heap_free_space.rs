//! Fractal-heap client free-space records and reconstructed section geometry.
//!
//! Serialized single and first-row records describe free space in the heap's managed address
//! space. The parser reconstructs normal-row and indirect sections from first-row records and
//! the heap's doubling table. Section offsets are logical heap offsets.
//!
//! The section classes are defined in "Free-space Index" of the [format specification,
//! version 4.0][spec]. `H5HF_FSPACE_SECT_CLS_NORMAL_ROW` and `H5HF_FSPACE_SECT_CLS_INDIRECT`
//! declare ghost classes in `H5HFsection.c`, HDF5 2.2.0, with no serialization callbacks.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex

use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroU16;
use core::num::NonZeroU64;
use core::ops::Range;

use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_core::StoredAddress;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::FractalHeapLayout;
use crate::fractal_heap::FractalHeapLayoutError;
use crate::free_space_manager::FreeSpaceManagerHeaderFrame;
use crate::free_space_manager::FreeSpaceManagerSectionListRef;
use crate::width::FormatWidths;

/// A section class in the fractal heap's managed address space.
///
/// The parser accepts serialized [`Single`](Self::Single) and [`FirstRow`](Self::FirstRow)
/// records. It reconstructs [`NormalRow`](Self::NormalRow) and [`Indirect`](Self::Indirect)
/// geometry from first-row records and rejects those classes as serialized records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FractalHeapSectionClass {
    /// A free byte range within one direct block.
    Single = 0,
    /// The first direct row represented by an indirect section.
    FirstRow = 1,
    /// An additional direct row represented by an indirect section.
    NormalRow = 2,
    /// A span of doubling-table entries in an indirect block.
    Indirect = 3,
}

/// A free-space manager header interpreted with a fractal heap's doubling table.
///
/// Constructed by [`FreeSpaceManagerHeaderFrame::fractal_heap_header`]. Header validation
/// checks the fractal-heap client ID, section counts, and heap geometry. Section-list parsing
/// validates the serialized records and their reconstructed counts and accounting sizes.
#[derive(Clone, Debug)]
pub struct FractalHeapFreeSpaceManager<'a> {
    frame: FreeSpaceManagerHeaderFrame,
    heap: &'a FractalHeapHeader,
    layout: FractalHeapLayout,
    /// The direct-block prefix size, including an optional checksum.
    overhead: u64,
    /// Full-block section accounting indexed by indirect-block row count.
    summaries: Vec<SectionSummary>,
}

impl<'a> FractalHeapFreeSpaceManager<'a> {
    pub(crate) fn new(
        widths: FormatWidths,
        frame: FreeSpaceManagerHeaderFrame,
        heap: &'a FractalHeapHeader,
    ) -> Result<Self, FormatError> {
        if frame.client_id() != CLIENT_ID {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the client ID is {}, not {CLIENT_ID}, the ID of the fractal heap client",
                frame.client_id()
            )));
        }
        if frame.section_classes() != NUM_CLASSES
            || frame.addr_space_bits() != heap.max_heap_size
            || frame.max_section_size() != heap.max_direct_block_size
            || frame
                .serialized_sections()
                .checked_add(frame.unserialized_sections())
                != Some(frame.total_sections())
        {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the heap geometry and manager header fields disagree".into(),
            ));
        }
        if frame.serialized_sections() == 0 && frame.total_sections() != 0 {
            return Err(FormatError::InvalidFreeSpaceManager(
                "a heap manager with ghost sections has no serialized proxy".into(),
            ));
        }
        if frame.serialized_sections() > 0
            && frame.section_list_addr() == DecodedStoredAddress::Undefined
        {
            return Err(FormatError::InvalidFreeSpaceManager(
                "a serialized heap manager has no section-list address".into(),
            ));
        }
        let layout = FractalHeapLayout::new(heap, widths.offsets, widths.lengths)?;
        if heap.current_rows_in_root_indirect_block > layout.max_root_rows() {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the heap root row count exceeds its doubling table".into(),
            ));
        }
        // H5HF_MAN_ABS_DIRECT_OVERHEAD in H5HFpkg.h, HDF5 2.2.0, includes the optional checksum.
        let overhead = METADATA_PREFIX_LEN
            + u64::from(widths.offsets.get())
            + layout.block_offset_size().to_u64()
            + if heap.flags & FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED == 0 {
                0
            } else {
                CHECKSUM_LEN
            };
        if layout.block_size_for_row(0)? <= overhead {
            return Err(FormatError::InvalidFreeSpaceManager(
                "a direct block has no usable managed space".into(),
            ));
        }
        let mut manager = Self {
            frame,
            heap,
            layout,
            overhead,
            summaries: Vec::new(),
        };
        manager.summaries.push(SectionSummary::default());
        for rows in 1..=manager.layout.max_root_rows() {
            let summary = manager.summarize(rows, 0, u64::from(rows) * manager.width())?;
            manager.summaries.push(summary);
        }
        Ok(manager)
    }

    /// Creates a section-list reference at the resolved physical address.
    ///
    /// The caller resolves the header's stored section-list address to `address` and passes
    /// the stored address of the owning manager header as `header_addr` for owner checks.
    /// Source bounds and physical ownership are checked by the caller.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if the section-list address or
    /// `header_addr` is undefined, or the used length is shorter than the prefix and checksum.
    /// Returns [`FormatError::AddressTooLarge`] if `header_addr` exceeds the offset width, and
    /// [`FormatError::OffsetOverflow`] if either physical range ends past [`u64::MAX`].
    pub fn section_list_ref(
        &self,
        header_addr: StoredAddress,
        address: PhysicalAddress,
    ) -> Result<FractalHeapFreeSpaceSectionListRef<'_, 'a>, FormatError> {
        Ok(FractalHeapFreeSpaceSectionListRef {
            manager: self,
            reference: self.frame.section_list_ref(header_addr, address)?,
        })
    }

    /// Returns the common header framing used for client-specific validation.
    pub fn header(&self) -> &FreeSpaceManagerHeaderFrame {
        &self.frame
    }

    /// Reconstructs the direct-row section covered by the indirect section.
    ///
    /// Pass an `indirect` section obtained from this manager. The returned section describes
    /// consecutive direct-block entries in `row`, with a per-block usable size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if `row` is outside the section or
    /// contains indirect blocks. Returns [`FormatError::InvalidFractalHeapGeometry`] if the
    /// doubling-table geometry is invalid, and [`FormatError::OffsetOverflow`] if a derived
    /// heap offset overflows.
    ///
    /// # Examples
    ///
    /// See the [example] for obtaining a parsed first-row record and reconstructing a direct
    /// row and an indirect child.
    ///
    /// [example]: FreeSpaceManagerHeaderFrame::fractal_heap_header
    pub fn row_section(
        &self,
        indirect: FractalHeapIndirectSection,
        row: u16,
    ) -> Result<FractalHeapRowSection, FormatError> {
        let (start, end) = self.row_entries(indirect, row)?;
        if usize::from(row) >= self.layout.direct_rows() {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the requested row contains indirect blocks".into(),
            ));
        }
        let offset = add(
            indirect.block_offset,
            self.layout
                .slot_offset(usize::from(row), start.to_usize()?)?,
        )?;
        let first = self.first_row(indirect)?;
        Ok(FractalHeapRowSection {
            class: if indirect.first && offset == first.offset {
                FractalHeapSectionClass::FirstRow
            } else {
                FractalHeapSectionClass::NormalRow
            },
            offset,
            size: nonzero(self.usable_size(row)?)?,
            row,
            column: start.narrow_or_else(|| {
                FormatError::InvalidFreeSpaceManager("a column exceeds u16".into())
            })?,
            blocks: row_block_count(end - start)?,
        })
    }

    /// Reconstructs the indirect child covered by a doubling-table entry.
    ///
    /// Pass an `indirect` section obtained from this manager. The returned section covers all
    /// entries of the child at `row` and `column`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if the entry is outside the section.
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if `row` is a direct row or the child
    /// geometry is invalid, and [`FormatError::OffsetOverflow`] if a derived heap offset
    /// overflows.
    ///
    /// # Examples
    ///
    /// See the [example] for obtaining a parsed first-row record and reconstructing a direct
    /// row and an indirect child.
    ///
    /// [example]: FreeSpaceManagerHeaderFrame::fractal_heap_header
    pub fn child_section(
        &self,
        indirect: FractalHeapIndirectSection,
        row: u16,
        column: u16,
    ) -> Result<FractalHeapIndirectSection, FormatError> {
        let (start, end) = self.row_entries(indirect, row)?;
        if u64::from(column) < start || u64::from(column) >= end {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the child column is outside the indirect section".into(),
            ));
        }
        let rows = self.layout.child_indirect_rows(usize::from(row))?;
        Ok(FractalHeapIndirectSection {
            block_offset: add(
                indirect.block_offset,
                self.layout
                    .slot_offset(usize::from(row), usize::from(column))?,
            )?,
            rows,
            start_row: 0,
            start_column: 0,
            blocks: nonzero(u64::from(rows) * self.width())?,
            first: indirect.first && (row, column) == (indirect.start_row, indirect.start_column),
        })
    }

    fn decode(&self, records: &[u8]) -> Result<FractalHeapFreeSpace, FormatError> {
        let mut reader = RecordFields::new(records);
        let count_width = encoded_size(self.frame.serialized_sections());
        let size_width = encoded_size(self.frame.max_section_size());
        let mut sections = Vec::new();
        let mut spans = Vec::new();
        let mut summary = SectionSummary::default();
        let mut previous_size = 0;
        let mut decoded = 0;
        while decoded < self.frame.serialized_sections() {
            let count = reader.uint(count_width)?;
            let size = reader.uint(size_width)?;
            if count == 0
                || count > self.frame.serialized_sections() - decoded
                || size == 0
                || size >= self.frame.max_section_size()
                || size <= previous_size
            {
                return Err(FormatError::InvalidFreeSpaceManager(
                    "a record set has an invalid count or section size".into(),
                ));
            }
            previous_size = size;
            for _ in 0..count {
                let (section, span, represented) = self.decode_record(&mut reader, size)?;
                summary = summary.add(represented)?;
                if summary.count > self.frame.total_sections() {
                    return Err(FormatError::InvalidFreeSpaceManager(
                        "the reconstructed section count exceeds the header count".into(),
                    ));
                }
                sections.push(section);
                spans.push(span);
                decoded += 1;
            }
        }
        if reader.remaining().iter().any(|&byte| byte != 0) {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the section list has nonzero bytes after its records".into(),
            ));
        }
        if summary.count != self.frame.total_sections() || summary.space != self.frame.total_space()
        {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the reconstructed sections describe {} sections and {} bytes, not the header's {} sections and {} bytes",
                summary.count,
                summary.space,
                self.frame.total_sections(),
                self.frame.total_space()
            )));
        }
        spans.sort_unstable_by_key(|span| span.start);
        if spans.windows(2).any(|pair| pair[0].end > pair[1].start) {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the logical managed-space sections overlap".into(),
            ));
        }
        Ok(FractalHeapFreeSpace {
            sections,
            total_sections: summary.count,
            total_space: summary.space,
        })
    }

    fn decode_record(
        &self,
        reader: &mut RecordFields<'_>,
        size: u64,
    ) -> Result<(FractalHeapFreeSection, Range<u64>, SectionSummary), FormatError> {
        let offset = reader.uint(self.layout.block_offset_size())?;
        let class = reader.byte()?;
        Ok(match class {
            value if value == FractalHeapSectionClass::Single as u8 => {
                let span = self.single_span(offset, size)?;
                (
                    FractalHeapFreeSection::Single {
                        offset,
                        size: NonZeroU64::new(size).ok_or_else(|| {
                            FormatError::InvalidFreeSpaceManager("a single section is empty".into())
                        })?,
                    },
                    span,
                    SectionSummary {
                        count: 1,
                        space: size,
                    },
                )
            }
            value if value == FractalHeapSectionClass::FirstRow as u8 => {
                let block_offset = reader.uint(self.layout.block_offset_size())?;
                let row = reader.u16()?;
                let column = reader.u16()?;
                let blocks = reader.u16()?;
                let rows = self.indirect_rows(block_offset)?;
                let indirect = FractalHeapIndirectSection {
                block_offset,
                rows,
                start_row: row,
                start_column: column,
                blocks: NonZeroU64::new(u64::from(blocks)).ok_or_else(|| FormatError::InvalidFreeSpaceManager("the indirect section's row, column, or block count is outside its doubling table".into()))?,
                first: true,
            };
                self.entries(indirect)?;
                let first = self.first_row(indirect)?;
                if (offset, size) != (first.offset, first.size.get()) {
                    return Err(FormatError::InvalidFreeSpaceManager(
                        "the first-row record disagrees with its reconstructed first row".into(),
                    ));
                }
                let span = self.indirect_span(indirect)?;
                let represented =
                    self.summarize(rows, self.entries(indirect)?.start, u64::from(blocks))?;
                (
                    FractalHeapFreeSection::FirstRow {
                        row: first,
                        indirect,
                    },
                    span,
                    represented,
                )
            }
            class_id => {
                return Err(FormatError::UnsupportedSerializedSectionClass {
                    client_id: CLIENT_ID,
                    class_id,
                });
            }
        })
    }

    fn single_span(&self, offset: u64, size: u64) -> Result<Range<u64>, FormatError> {
        let end = add(offset, size)?;
        let (block_offset, block_size) = self.direct_block(offset)?;
        if offset < add(block_offset, self.overhead)?
            || end > add(block_offset, block_size)?
            || end > self.heap.managed_space
        {
            return Err(FormatError::InvalidFreeSpaceManager(
                "a single section crosses a direct block's usable space".into(),
            ));
        }
        Ok(offset..end)
    }

    fn indirect_rows(&self, offset: u64) -> Result<u16, FormatError> {
        if self.heap.current_rows_in_root_indirect_block == 0 || offset >= self.heap.managed_space {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the indirect section has no corresponding heap block".into(),
            ));
        }
        let mut block_offset = 0;
        let mut rows = self.heap.current_rows_in_root_indirect_block;
        while block_offset != offset {
            let (row, column) = self.slot_containing(block_offset, rows, offset)?;
            rows = self.layout.child_indirect_rows(usize::from(row))?;
            block_offset = add(
                block_offset,
                self.layout
                    .slot_offset(usize::from(row), column.to_usize()?)?,
            )?;
        }
        Ok(rows)
    }

    fn direct_block(&self, offset: u64) -> Result<(u64, u64), FormatError> {
        if offset >= self.heap.managed_space {
            return Err(FormatError::InvalidFreeSpaceManager(
                "a section offset is outside the managed heap".into(),
            ));
        }
        if self.heap.current_rows_in_root_indirect_block == 0 {
            return Ok((0, self.layout.block_size_for_row(0)?));
        }
        let mut block_offset = 0;
        let mut rows = self.heap.current_rows_in_root_indirect_block;
        loop {
            let (row, column) = self.slot_containing(block_offset, rows, offset)?;
            block_offset = add(
                block_offset,
                self.layout
                    .slot_offset(usize::from(row), column.to_usize()?)?,
            )?;
            if usize::from(row) < self.layout.direct_rows() {
                return Ok((
                    block_offset,
                    self.layout.block_size_for_row(usize::from(row))?,
                ));
            }
            rows = self.layout.child_indirect_rows(usize::from(row))?;
        }
    }

    fn slot_containing(
        &self,
        offset: u64,
        rows: u16,
        target: u64,
    ) -> Result<(u16, u64), FormatError> {
        let relative = target.checked_sub(offset).ok_or_else(|| {
            FormatError::InvalidFreeSpaceManager("the section precedes its indirect block".into())
        })?;
        for row in 0..rows {
            let start = self.layout.row_offset(usize::from(row))?;
            let size = self.layout.block_size_for_row(usize::from(row))?;
            if relative >= start && relative - start < multiply(size, self.width())? {
                return Ok((row, (relative - start) / size));
            }
        }
        Err(FormatError::InvalidFreeSpaceManager(
            "the section is outside the indirect block's doubling table".into(),
        ))
    }

    fn entries(&self, section: FractalHeapIndirectSection) -> Result<Range<u64>, FormatError> {
        let start = add(
            multiply(u64::from(section.start_row), self.width())?,
            u64::from(section.start_column),
        )?;
        let end = add(start, section.blocks.get())?;
        if u64::from(section.start_column) >= self.width()
            || section.start_row >= section.rows
            || end > multiply(u64::from(section.rows), self.width())?
        {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the indirect section's row, column, or block count is outside its doubling table"
                    .into(),
            ));
        }
        Ok(start..end)
    }

    fn row_entries(
        &self,
        section: FractalHeapIndirectSection,
        row: u16,
    ) -> Result<(u64, u64), FormatError> {
        let entries = self.entries(section)?;
        let start = multiply(u64::from(row), self.width())?;
        let end = add(start, self.width())?;
        let from = start.max(entries.start);
        let to = end.min(entries.end);
        if from >= to {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the requested row is outside the indirect section".into(),
            ));
        }
        Ok((from - start, to - start))
    }

    fn first_row(
        &self,
        mut section: FractalHeapIndirectSection,
    ) -> Result<FractalHeapRowSection, FormatError> {
        while usize::from(section.start_row) >= self.layout.direct_rows() {
            section = self.child_section(section, section.start_row, section.start_column)?;
        }
        let (start, end) = self.row_entries(section, section.start_row)?;
        Ok(FractalHeapRowSection {
            class: FractalHeapSectionClass::FirstRow,
            offset: add(
                section.block_offset,
                self.layout
                    .slot_offset(usize::from(section.start_row), start.to_usize()?)?,
            )?,
            size: nonzero(self.usable_size(section.start_row)?)?,
            row: section.start_row,
            column: section.start_column,
            blocks: row_block_count(end - start)?,
        })
    }

    fn indirect_span(
        &self,
        section: FractalHeapIndirectSection,
    ) -> Result<Range<u64>, FormatError> {
        let entries = self.entries(section)?;
        let last = entries.end - 1;
        let last_row = last / self.width();
        let last_column = last % self.width();
        let start = add(
            section.block_offset,
            self.layout.slot_offset(
                usize::from(section.start_row),
                usize::from(section.start_column),
            )?,
        )?;
        let end = add(
            add(
                section.block_offset,
                self.layout
                    .slot_offset(last_row.to_usize()?, last_column.to_usize()?)?,
            )?,
            self.layout.block_size_for_row(last_row.to_usize()?)?,
        )?;
        if end > self.heap.managed_space {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the indirect section extends beyond the managed heap".into(),
            ));
        }
        Ok(start..end)
    }

    // H5HF__sect_indirect_init_rows in H5HFsection.c and H5FS__sect_link in H5FSsection.c, HDF5 2.2.0,
    // count direct-row sections and their per-block usable sizes, excluding structural indirect nodes.
    fn summarize(&self, rows: u16, start: u64, blocks: u64) -> Result<SectionSummary, FormatError> {
        let end = add(start, blocks)?;
        if blocks == 0 || end > u64::from(rows) * self.width() {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the indirect summary exceeds its doubling table".into(),
            ));
        }
        let mut summary = SectionSummary::default();
        for row in 0..rows {
            let row_start = u64::from(row) * self.width();
            let from = row_start.max(start);
            let to = (row_start + self.width()).min(end);
            if from >= to {
                continue;
            }
            let represented = if usize::from(row) < self.layout.direct_rows() {
                SectionSummary {
                    count: 1,
                    space: self.usable_size(row)?,
                }
            } else {
                let child_rows = self.layout.child_indirect_rows(usize::from(row))?;
                let child = self.summaries.get(usize::from(child_rows)).ok_or_else(|| {
                    FormatError::InvalidFreeSpaceManager(
                        "an indirect row has no computed child summary".into(),
                    )
                })?;
                child.multiply(to - from)?
            };
            summary = summary.add(represented)?;
        }
        Ok(summary)
    }

    fn usable_size(&self, row: u16) -> Result<u64, FormatError> {
        self.layout
            .block_size_for_row(usize::from(row))?
            .checked_sub(self.overhead)
            .ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager("a direct block has no usable space".into())
            })
    }

    fn width(&self) -> u64 {
        self.layout.table_width().to_u64()
    }
}

/// A physical section-list reference with fractal-heap validation context.
///
/// The complete allocation is [`range`](Self::range). Parsing requires only
/// [`parse_range`](Self::parse_range), the used bytes recorded in the manager header.
#[derive(Clone, Debug)]
pub struct FractalHeapFreeSpaceSectionListRef<'manager, 'heap> {
    manager: &'manager FractalHeapFreeSpaceManager<'heap>,
    reference: FreeSpaceManagerSectionListRef,
}

impl FractalHeapFreeSpaceSectionListRef<'_, '_> {
    /// Parses serialized sections and validates their reconstructed heap geometry.
    ///
    /// Normal-row sections and indirect children can be reconstructed through
    /// [`FractalHeapFreeSpaceManager::row_section`] and
    /// [`FractalHeapFreeSpaceManager::child_section`]. Checksum verification follows the
    /// `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range). Returns
    /// [`FormatError::UnsupportedSerializedSectionClass`] for serialized classes other than
    /// single and first row. Returns [`FormatError::InvalidFreeSpaceManager`] for malformed
    /// framing, records, overlapping logical sections, or inconsistent reconstructed totals.
    /// Geometry, arithmetic, and checksum failures use their corresponding [`FormatError`]
    /// variants.
    pub fn parse(&self, region: FileRegion<'_>) -> Result<FractalHeapFreeSpace, FormatError> {
        self.manager.decode(self.reference.parse(region)?.records())
    }

    /// Parses sections with reconstructed geometry and mandatory checksum validation.
    ///
    /// Verifies the trailing checksum within [`parse_range`](Self::parse_range), regardless of the
    /// `checksum` feature. Reserved allocation capacity is excluded from parsing.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if the stored checksum differs from the
    /// computed one. Other failures are the errors of [`parse`](Self::parse).
    pub fn parse_checked(
        &self,
        region: FileRegion<'_>,
    ) -> Result<FractalHeapFreeSpace, FormatError> {
        self.manager
            .decode(self.reference.parse_checked(region)?.records())
    }

    /// Returns the complete physical allocation, including reserved capacity.
    pub const fn range(&self) -> FileRange {
        self.reference.range()
    }

    /// Returns the used physical bytes required to parse the section list.
    pub const fn parse_range(&self) -> FileRange {
        self.reference.parse_range()
    }
}

/// Serialized heap sections with validated reconstructed accounting totals.
///
/// The stored sections are single and first-row records. Totals include the normal-row
/// sections represented by first-row records. Structural indirect sections are excluded from
/// the section count and accounting size.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapFreeSpace {
    sections: Vec<FractalHeapFreeSection>,
    total_sections: u64,
    total_space: u64,
}

impl FractalHeapFreeSpace {
    /// Returns the serialized records in their encoded order.
    pub fn sections(&self) -> &[FractalHeapFreeSection] {
        &self.sections
    }

    /// Returns the number of single and direct-row sections, including ghost rows.
    pub const fn total_sections(&self) -> u64 {
        self.total_sections
    }

    /// Returns the manager's free-space accounting total in bytes.
    ///
    /// Each single section contributes its length. Each direct-row section contributes one
    /// per-block usable size, independent of the number of blocks covered by that row.
    pub const fn total_space(&self) -> u64 {
        self.total_space
    }
}

/// A serialized free-space record interpreted in the heap's managed address space.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapFreeSection {
    /// A free byte range within the usable area of a direct block.
    Single {
        /// The byte offset in the heap's managed address space.
        offset: u64,
        /// The nonzero byte length of the free range.
        size: NonZeroU64,
    },
    /// A first direct row and the indirect section represented by its wire record.
    FirstRow {
        /// The first direct row, including its per-block usable size.
        row: FractalHeapRowSection,
        /// The represented doubling-table entries, including any indirect children.
        indirect: FractalHeapIndirectSection,
    },
}

/// A span of consecutive direct-block entries within one doubling-table row.
///
/// The offset is in the heap's managed address space. The size is the usable byte count of
/// each block in the row, excluding its metadata prefix and optional checksum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapRowSection {
    class: FractalHeapSectionClass,
    offset: u64,
    size: NonZeroU64,
    row: u16,
    column: u16,
    blocks: NonZeroU16,
}

impl FractalHeapRowSection {
    /// Returns whether the row is the serialized first row or a reconstructed normal row.
    pub const fn class(&self) -> FractalHeapSectionClass {
        self.class
    }
    /// Returns the first entry's byte offset in the heap's managed address space.
    pub const fn offset(&self) -> u64 {
        self.offset
    }
    /// Returns the usable byte count of each direct block in the row.
    pub const fn size(&self) -> u64 {
        self.size.get()
    }
    /// Returns the zero-based doubling-table row.
    pub const fn row(&self) -> u16 {
        self.row
    }
    /// Returns the zero-based column of the first covered entry.
    pub const fn column(&self) -> u16 {
        self.column
    }
    /// Returns the nonzero number of consecutive entries covered in the row.
    pub const fn blocks(&self) -> u16 {
        self.blocks.get()
    }
}

/// A span of doubling-table entries in a logical indirect block.
///
/// Entries may describe direct blocks or indirect children. Reconstruct their geometry with
/// [`FractalHeapFreeSpaceManager::row_section`] and
/// [`FractalHeapFreeSpaceManager::child_section`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIndirectSection {
    block_offset: u64,
    rows: u16,
    start_row: u16,
    start_column: u16,
    blocks: NonZeroU64,
    first: bool,
}

impl FractalHeapIndirectSection {
    /// Returns the indirect semantic section class.
    pub const fn class(&self) -> FractalHeapSectionClass {
        FractalHeapSectionClass::Indirect
    }
    /// Returns the indirect block's byte offset in the heap's managed address space.
    pub const fn block_offset(&self) -> u64 {
        self.block_offset
    }
    /// Returns the row count of the indirect block's doubling table.
    pub const fn rows(&self) -> u16 {
        self.rows
    }
    /// Returns the zero-based row of the first covered entry.
    pub const fn start_row(&self) -> u16 {
        self.start_row
    }
    /// Returns the zero-based column of the first covered entry.
    pub const fn start_column(&self) -> u16 {
        self.start_column
    }
    /// Returns the nonzero number of covered entries, which may span rows.
    pub const fn blocks(&self) -> u64 {
        self.blocks.get()
    }
}

/// Manager accounting for single and direct-row sections, excluding indirect nodes.
#[derive(Clone, Copy, Debug, Default)]
struct SectionSummary {
    count: u64,
    space: u64,
}

impl SectionSummary {
    fn add(self, other: Self) -> Result<Self, FormatError> {
        Ok(Self {
            count: add(self.count, other.count)?,
            space: add(self.space, other.space)?,
        })
    }

    fn multiply(self, factor: u64) -> Result<Self, FormatError> {
        Ok(Self {
            count: multiply(self.count, factor)?,
            space: multiply(self.space, factor)?,
        })
    }
}

/// A cursor over little-endian fields in the used section-list record bytes.
struct RecordFields<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> RecordFields<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn byte(&mut self) -> Result<u8, FormatError> {
        self.uint(1)?
            .narrow_or_else(|| FormatError::InvalidFreeSpaceManager("a class exceeds u8".into()))
    }

    fn u16(&mut self) -> Result<u16, FormatError> {
        self.uint(2)?.narrow_or_else(|| {
            FormatError::InvalidFreeSpaceManager("a row field exceeds u16".into())
        })
    }

    fn uint(&mut self, width: usize) -> Result<u64, FormatError> {
        let end = self.position.checked_add(width).ok_or_else(|| {
            FormatError::InvalidFreeSpaceManager("a record field position overflows".into())
        })?;
        let data = self.bytes.get(self.position..end).ok_or_else(|| {
            FormatError::InvalidFreeSpaceManager(
                "a section record field runs past the used section list".into(),
            )
        })?;
        self.position = end;
        Ok(data
            .iter()
            .enumerate()
            .fold(0, |value, (i, &byte)| value | (u64::from(byte) << (8 * i))))
    }

    fn remaining(&self) -> &'a [u8] {
        self.bytes.get(self.position..).unwrap_or_default()
    }
}

fn row_block_count(value: u64) -> Result<NonZeroU16, FormatError> {
    NonZeroU16::new(value.narrow_or_else(|| {
        FormatError::InvalidFreeSpaceManager("a row block count exceeds u16".into())
    })?)
    .ok_or_else(|| FormatError::InvalidFreeSpaceManager("a row block count is zero".into()))
}

fn nonzero(value: u64) -> Result<NonZeroU64, FormatError> {
    NonZeroU64::new(value).ok_or_else(|| {
        FormatError::InvalidFreeSpaceManager(
            "a reconstructed section has zero size or block count".into(),
        )
    })
}

fn encoded_size(value: u64) -> usize {
    value
        .to_le_bytes()
        .iter()
        .rposition(|&byte| byte != 0)
        .map_or(1, |at| at + 1)
}

fn add(offset: u64, length: u64) -> Result<u64, FormatError> {
    offset
        .checked_add(length)
        .ok_or(FormatError::OffsetOverflow { offset, length })
}

fn multiply(value: u64, factor: u64) -> Result<u64, FormatError> {
    value.checked_mul(factor).ok_or_else(|| {
        FormatError::InvalidFreeSpaceManager(
            "the reconstructed heap section arithmetic overflows".into(),
        )
    })
}

impl From<FractalHeapLayoutError> for FormatError {
    fn from(_error: FractalHeapLayoutError) -> Self {
        Self::InvalidFractalHeapGeometry
    }
}

// The client and class count are defined in "Free-space Index", format specification version 4.0.
const CLIENT_ID: u8 = 0;
const NUM_CLASSES: u16 = 4;
// H5HF_MAN_ABS_DIRECT_OVERHEAD in H5HFpkg.h, HDF5 2.2.0, includes the metadata prefix and checksum.
const METADATA_PREFIX_LEN: u64 = 5;
const CHECKSUM_LEN: u64 = 4;

#[cfg(test)]
mod tests {
    use hdf5_pure_core::__private::StoredAddressExt;
    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use crate::checksum;
    use crate::free_space_manager::FreeSpaceManagerHeaderRef;

    use super::*;

    #[test]
    fn parses_c_written_single_and_first_row_sections_with_reconstructed_ghost_counts() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = FractalHeapHeader::parse_bytes(
            &fractal_heap::dense_link_free_space_header(),
            widths.offsets,
            widths.lengths,
        )
        .unwrap();
        let header_bytes = fractal_heap::dense_link_free_space_manager();
        let header_ref =
            FreeSpaceManagerHeaderRef::new(widths, PhysicalAddress::new(1403)).unwrap();
        let frame = header_ref
            .parse(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
            .unwrap();
        assert_eq!(
            frame.section_list_addr(),
            DecodedStoredAddress::Defined(StoredAddress::new(5069))
        );
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(1403), PhysicalAddress::new(5069))
            .unwrap();
        let bytes = fractal_heap::dense_link_free_space_sections();
        let parsed = reference
            .parse(FileRegion::new(reference.parse_range(), &bytes).unwrap())
            .unwrap();
        assert_eq!((parsed.total_sections(), parsed.total_space()), (10, 10933));
        assert_eq!(parsed.sections().len(), 7);
        let FractalHeapFreeSection::FirstRow { row, indirect } = parsed.sections()[4] else {
            panic!("expected a first-row section")
        };
        assert_eq!(
            row,
            FractalHeapRowSection {
                class: FractalHeapSectionClass::FirstRow,
                offset: 0,
                size: NonZeroU64::new(491).unwrap(),
                row: 0,
                column: 0,
                blocks: NonZeroU16::new(4).unwrap()
            }
        );
        assert_eq!(
            indirect,
            FractalHeapIndirectSection {
                block_offset: 0,
                rows: 8,
                start_row: 0,
                start_column: 0,
                blocks: NonZeroU64::new(16).unwrap(),
                first: true
            }
        );
        assert_eq!(indirect.class(), FractalHeapSectionClass::Indirect);
        assert_eq!(
            manager.row_section(indirect, 3).unwrap(),
            FractalHeapRowSection {
                class: FractalHeapSectionClass::NormalRow,
                offset: 8192,
                size: NonZeroU64::new(2027).unwrap(),
                row: 3,
                column: 0,
                blocks: NonZeroU16::new(4).unwrap()
            }
        );
    }

    #[rstest]
    #[case::two_byte(2, 2)]
    #[case::four_byte(4, 4)]
    #[case::eight_byte(8, 8)]
    #[case::mixed_widths(8, 4)]
    fn parses_single_sections_in_a_direct_root(#[case] offsets: u8, #[case] lengths: u8) {
        let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
        let mut heap = heap(widths);
        heap.current_rows_in_root_indirect_block = 0;
        heap.managed_space = heap.starting_block_size;
        let overhead = METADATA_PREFIX_LEN + u64::from(offsets) + 2;
        let records = single_record(FractalHeapSectionClass::Single as u8, overhead, 10);
        let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let parsed = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap();
        assert_eq!(
            parsed.sections(),
            &[FractalHeapFreeSection::Single {
                offset: overhead,
                size: NonZeroU64::new(10).unwrap()
            }]
        );
        assert_eq!((parsed.total_sections(), parsed.total_space()), (1, 10));
    }

    #[test]
    fn reconstructs_normal_rows_and_indirect_children_with_ghost_totals_and_spare_capacity() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let mut heap = heap(widths);
        heap.current_rows_in_root_indirect_block = 6;
        heap.managed_space = 16384;
        let records = first_row_record(1, 113, 0, 0, 0, 24);
        let (header, list) = encoded_manager(widths, &records, 17, 1, 16, 3841, 64);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        assert_eq!(
            reference.range().length(),
            reference.parse_range().length() + 64
        );
        let parsed = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap();
        assert_eq!((parsed.total_sections(), parsed.total_space()), (17, 3841));
        let FractalHeapFreeSection::FirstRow { indirect, .. } = parsed.sections()[0] else {
            panic!("expected first row")
        };
        let child = manager.child_section(indirect, 5, 0).unwrap();
        assert_eq!(
            (
                child.class(),
                child.block_offset(),
                child.rows(),
                child.blocks()
            ),
            (FractalHeapSectionClass::Indirect, 8192, 3, 12)
        );
        assert_eq!(
            manager.row_section(child, 0).unwrap().class(),
            FractalHeapSectionClass::NormalRow
        );
    }

    #[rstest]
    #[case::normal_row(FractalHeapSectionClass::NormalRow as u8)]
    #[case::indirect(FractalHeapSectionClass::Indirect as u8)]
    #[case::reserved(4)]
    #[case::reserved_high(255)]
    fn reports_unsupported_serialized_classes(#[case] class_id: u8) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let records = single_record(class_id, 15, 10);
        let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let err = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::UnsupportedSerializedSectionClass {
                client_id: CLIENT_ID,
                class_id
            }
        );
    }

    #[rstest]
    #[case::zero_blocks(
        0,
        0,
        0,
        0,
        0,
        113,
        "the indirect section's row, column, or block count is outside its doubling table"
    )]
    #[case::row_outside(
        0,
        4,
        0,
        1,
        0,
        113,
        "the indirect section's row, column, or block count is outside its doubling table"
    )]
    #[case::column_outside(
        0,
        0,
        4,
        1,
        0,
        113,
        "the indirect section's row, column, or block count is outside its doubling table"
    )]
    #[case::blocks_outside(
        0,
        3,
        3,
        2,
        0,
        113,
        "the indirect section's row, column, or block count is outside its doubling table"
    )]
    #[case::offset_disagrees(
        0,
        0,
        0,
        1,
        1,
        113,
        "the first-row record disagrees with its reconstructed first row"
    )]
    #[case::size_disagrees(
        0,
        0,
        0,
        1,
        0,
        112,
        "the first-row record disagrees with its reconstructed first row"
    )]
    fn rejects_first_row_geometry(
        #[case] block_offset: u64,
        #[case] row: u16,
        #[case] column: u16,
        #[case] blocks: u16,
        #[case] offset: u64,
        #[case] size: u64,
        #[case] reason: &str,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let mut records = first_row_record(1, size, offset, row, column, blocks);
        records[6..8].copy_from_slice(&u16::try_from(block_offset).unwrap().to_le_bytes());
        let (header, list) = encoded_manager(widths, &records, 1, 1, 0, size, 0);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let err = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err();
        assert_eq!(err, FormatError::InvalidFreeSpaceManager(reason.into()));
    }

    #[test]
    fn rejects_a_short_supplied_region_separately_from_truncated_records() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let (header, list) = encoded_manager(
            widths,
            &first_row_record(1, 113, 0, 0, 0, 1),
            1,
            1,
            0,
            113,
            0,
        );
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let short = FileRange::new(PhysicalAddress::new(4096), list.len().to_u64() - 1).unwrap();
        let err = reference
            .parse(FileRegion::new(short, &list[..list.len() - 1]).unwrap())
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::InsufficientRegion {
                required_start: 4096,
                required_length: reference.parse_range().length(),
                provided_start: 4096,
                provided_length: short.length()
            }
        );
        let (header, list) = encoded_manager(
            widths,
            &first_row_record(1, 113, 0, 0, 0, 1)[..12],
            1,
            1,
            0,
            113,
            0,
        );
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let err = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::InvalidFreeSpaceManager(
                "a section record field runs past the used section list".into()
            )
        );
    }

    #[rstest]
    #[case::file_client(5, 1, "the client ID is 1, not 0, the ID of the fractal heap client")]
    #[case::reserved_client(5, 2, "the client ID is 2, not 0, the ID of the fractal heap client")]
    #[case::count_sum(14, 2, "the heap geometry and manager header fields disagree")]
    #[case::class_count(38, 3, "the heap geometry and manager header fields disagree")]
    #[case::heap_width(44, 24, "the heap geometry and manager header fields disagree")]
    #[case::maximum_section_size(47, 2, "the heap geometry and manager header fields disagree")]
    fn rejects_client_specific_header_fields(
        #[case] at: usize,
        #[case] value: u8,
        #[case] reason: &str,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let (mut header, _) = encoded_manager(
            widths,
            &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
            1,
            1,
            0,
            10,
            0,
        );
        header[at] = value;
        refresh_checksum(&mut header);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        assert_eq!(
            frame.fractal_heap_header(&heap).unwrap_err(),
            FormatError::InvalidFreeSpaceManager(reason.into())
        );
    }

    #[rstest]
    #[case::ghost_count(
        5,
        1,
        4,
        964,
        "the reconstructed sections describe 4 sections and 964 bytes, not the header's 5 sections and 964 bytes"
    )]
    #[case::space_total(
        4,
        1,
        3,
        965,
        "the reconstructed sections describe 4 sections and 964 bytes, not the header's 4 sections and 965 bytes"
    )]
    #[case::count_excess(
        3,
        1,
        2,
        964,
        "the reconstructed section count exceeds the header count"
    )]
    fn rejects_reconstructed_count_and_space_disagreement(
        #[case] total: u64,
        #[case] serial: u64,
        #[case] ghost: u64,
        #[case] space: u64,
        #[case] reason: &str,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let (header, list) = encoded_manager(
            widths,
            &first_row_record(1, 113, 0, 0, 0, 16),
            total,
            serial,
            ghost,
            space,
            0,
        );
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let err = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err();
        assert_eq!(err, FormatError::InvalidFreeSpaceManager(reason.into()));
    }

    #[rstest]
    #[case::single_overlap(false)]
    #[case::first_row_overlap(true)]
    fn rejects_overlapping_logical_sections(#[case] first_row: bool) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let records = if first_row {
            let mut records = single_record(FractalHeapSectionClass::Single as u8, 15, 10);
            records.extend(first_row_record(1, 113, 0, 0, 0, 1));
            records
        } else {
            let mut records = single_record(FractalHeapSectionClass::Single as u8, 15, 10);
            records[0] = 2;
            records.extend_from_slice(&[20, 0, 0]);
            records
        };
        let (header, list) = encoded_manager(
            widths,
            &records,
            2,
            2,
            0,
            if first_row { 123 } else { 20 },
            0,
        );
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let err = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err();
        assert_eq!(
            err,
            FormatError::InvalidFreeSpaceManager(
                "the logical managed-space sections overlap".into()
            )
        );
    }

    #[rstest]
    #[case::zero_count(0, 0, "a record set has an invalid count or section size")]
    #[case::zero_size(1, 0, "a record set has an invalid count or section size")]
    #[case::count_excess(0, 2, "a record set has an invalid count or section size")]
    #[case::nonzero_padding(6, 1, "the section list has nonzero bytes after its records")]
    fn rejects_record_sets_and_padding(#[case] at: usize, #[case] value: u8, #[case] reason: &str) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let mut records = single_record(FractalHeapSectionClass::Single as u8, 15, 10);
        records.push(0);
        records[at] = value;
        let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        assert_eq!(
            reference
                .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
                .unwrap_err(),
            FormatError::InvalidFreeSpaceManager(reason.into())
        );
    }

    #[rstest]
    #[case::single_header(0, 10)]
    #[case::single_block_crossing(120, 10)]
    fn rejects_single_sections_outside_direct_block_data(#[case] offset: u64, #[case] size: u64) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let (header, list) = encoded_manager(
            widths,
            &single_record(FractalHeapSectionClass::Single as u8, offset, size),
            1,
            1,
            0,
            size,
            0,
        );
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        assert_eq!(
            reference
                .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
                .unwrap_err(),
            FormatError::InvalidFreeSpaceManager(
                "a single section crosses a direct block's usable space".into()
            )
        );
    }

    #[rstest]
    #[case::backlink(5, 33)]
    #[case::signature(0, b'X')]
    #[case::version(4, 1)]
    fn rejects_common_section_list_framing(#[case] at: usize, #[case] value: u8) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let (header, mut list) = encoded_manager(
            widths,
            &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
            1,
            1,
            0,
            10,
            0,
        );
        list[at] = value;
        refresh_checksum(&mut list);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let err = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err();
        let expected = match at {
            0 => FormatError::InvalidFreeSpaceManager(
                "the section list signature is b\"XSSE\", not b\"FSSE\"".into(),
            ),
            4 => FormatError::UnsupportedVersion(1),
            _ => FormatError::InvalidFreeSpaceManager(
                "the section list belongs to the header at 33, not to the header at 32".into(),
            ),
        };
        assert_eq!(err, expected);
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    #[case::header(true)]
    #[case::section_list(false)]
    fn rejects_checksum_corruption(#[case] header_corrupt: bool) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let (mut header, mut list) = encoded_manager(
            widths,
            &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
            1,
            1,
            0,
            10,
            0,
        );
        let block = if header_corrupt {
            &mut header
        } else {
            &mut list
        };
        let end = block.len() - 4;
        let expected = u32::from_le_bytes(block[end..].try_into().unwrap());
        block[6] ^= 1;
        let computed = checksum::jenkins_lookup3(&block[..end]);
        let err = if header_corrupt {
            FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap_err()
        } else {
            let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
            let manager = frame.fractal_heap_header(&heap).unwrap();
            let reference = manager
                .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
                .unwrap();
            reference
                .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
                .unwrap_err()
        };
        assert_eq!(err, FormatError::ChecksumMismatch { expected, computed });
    }

    fn refresh_checksum(block: &mut [u8]) {
        let at = block.len() - 4;
        let checksum = checksum::jenkins_lookup3(&block[..at]);
        block[at..].copy_from_slice(&checksum.to_le_bytes());
    }

    #[rstest]
    #[case::child_block(8192, 0, 0, 12, 8192, 3, 467)]
    #[case::indirect_row(0, 5, 1, 2, 10240, 6, 934)]
    fn reconstructs_first_rows_under_indirect_entries(
        #[case] block_offset: u16,
        #[case] row: u16,
        #[case] column: u16,
        #[case] blocks: u16,
        #[case] offset: u64,
        #[case] count: u64,
        #[case] space: u64,
    ) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let mut heap = heap(widths);
        heap.current_rows_in_root_indirect_block = 6;
        heap.managed_space = 16384;
        let mut records = first_row_record(1, 113, offset, row, column, blocks);
        records[6..8].copy_from_slice(&block_offset.to_le_bytes());
        let (header, list) = encoded_manager(widths, &records, count, 1, count - 1, space, 0);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let parsed = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap();
        assert_eq!(
            (parsed.total_sections(), parsed.total_space()),
            (count, space)
        );
        let FractalHeapFreeSection::FirstRow { row: first, .. } = parsed.sections()[0] else {
            panic!("expected first row")
        };
        assert_eq!(
            (first.offset(), first.size(), first.class()),
            (offset, 113, FractalHeapSectionClass::FirstRow)
        );
    }

    #[rstest]
    #[case::inside_direct_block(1)]
    #[case::direct_block_start(128)]
    #[case::outside_heap(4096)]
    fn rejects_indirect_block_offsets_without_corresponding_geometry(#[case] offset: u16) {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let mut records = first_row_record(1, 113, 0, 0, 0, 1);
        records[6..8].copy_from_slice(&offset.to_le_bytes());
        let (header, list) = encoded_manager(widths, &records, 1, 1, 0, 113, 0);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let err = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap_err();
        assert_eq!(
            err,
            if offset == 4096 {
                FormatError::InvalidFreeSpaceManager(
                    "the indirect section has no corresponding heap block".into(),
                )
            } else {
                FormatError::InvalidFractalHeapGeometry
            }
        );
    }

    #[test]
    fn parses_64_bit_logical_offsets_from_small_local_regions() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let mut heap = heap(widths);
        heap.max_heap_size = 64;
        heap.current_rows_in_root_indirect_block = 55;
        heap.managed_space = 1 << 63;
        let offset = (1u64 << 40) + 21;
        let mut records = vec![1, 10, 0];
        records.extend_from_slice(&offset.to_le_bytes());
        records.push(FractalHeapSectionClass::Single as u8);
        let (mut header, list) = encoded_manager(widths, &records, 1, 1, 0, 10, 0);
        header[44] = 64;
        refresh_checksum(&mut header);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        let manager = frame.fractal_heap_header(&heap).unwrap();
        let reference = manager
            .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(4096))
            .unwrap();
        let parsed = reference
            .parse(FileRegion::new(reference.parse_range(), &list).unwrap())
            .unwrap();
        assert_eq!(
            parsed.sections(),
            &[FractalHeapFreeSection::Single {
                offset,
                size: NonZeroU64::new(10).unwrap()
            }]
        );
    }

    #[test]
    fn distinguishes_heap_and_file_client_header_validation() {
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let heap = heap(widths);
        let (mut header, _) = encoded_manager(
            widths,
            &single_record(FractalHeapSectionClass::Single as u8, 15, 10),
            1,
            1,
            0,
            10,
            0,
        );
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        assert_eq!(
            frame.file_header().unwrap_err(),
            FormatError::InvalidFreeSpaceManager(
                "the client ID is 0, not 1, the ID of the file client".into()
            )
        );
        header[5] = 1;
        header[38] = 3;
        refresh_checksum(&mut header);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header).unwrap();
        assert_eq!(frame.file_header().unwrap().total_sections(), 1);
        assert_eq!(
            frame.fractal_heap_header(&heap).unwrap_err(),
            FormatError::InvalidFreeSpaceManager(
                "the client ID is 1, not 0, the ID of the fractal heap client".into()
            )
        );
    }

    fn heap(widths: FormatWidths) -> FractalHeapHeader {
        let mut heap = FractalHeapHeader::parse_bytes(
            &fractal_heap::Header::new(256).build(Widths {
                offset: usize::from(widths.offsets.get()),
                length: usize::from(widths.lengths.get()),
            }),
            widths.offsets,
            widths.lengths,
        )
        .unwrap();
        heap.managed_space = 4096;
        heap.current_rows_in_root_indirect_block = 4;
        heap
    }

    fn single_record(class: u8, offset: u64, size: u64) -> Vec<u8> {
        let mut record = vec![1];
        record.extend_from_slice(&u16::try_from(size).unwrap().to_le_bytes());
        record.extend_from_slice(&u16::try_from(offset).unwrap().to_le_bytes());
        record.push(class);
        record
    }

    fn first_row_record(
        count: u8,
        size: u64,
        offset: u64,
        row: u16,
        column: u16,
        blocks: u16,
    ) -> Vec<u8> {
        let mut record = single_record(FractalHeapSectionClass::FirstRow as u8, offset, size);
        record[0] = count;
        record.extend_from_slice(&0u16.to_le_bytes());
        record.extend_from_slice(&row.to_le_bytes());
        record.extend_from_slice(&column.to_le_bytes());
        record.extend_from_slice(&blocks.to_le_bytes());
        record
    }

    fn encoded_manager(
        widths: FormatWidths,
        records: &[u8],
        total: u64,
        serialized: u64,
        ghost: u64,
        space: u64,
        spare: u64,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut list = FSM_SECTION_LIST_SIGNATURE.to_vec();
        list.push(FSM_VERSION);
        crate::bytes::write_offset(&mut list, 32, widths.offsets);
        list.extend_from_slice(records);
        list.extend_from_slice(&checksum::jenkins_lookup3(&list).to_le_bytes());
        let mut header = FSM_HEADER_SIGNATURE.to_vec();
        header.extend_from_slice(&[FSM_VERSION, CLIENT_ID]);
        for value in [space, total, serialized, ghost] {
            crate::bytes::write_length(&mut header, value, widths.lengths);
        }
        for value in [NUM_CLASSES, 80, 120, 16] {
            header.extend_from_slice(&value.to_le_bytes());
        }
        crate::bytes::write_length(&mut header, 1024, widths.lengths);
        crate::bytes::write_offset(&mut header, 4096, widths.offsets);
        crate::bytes::write_length(&mut header, list.len().to_u64(), widths.lengths);
        crate::bytes::write_length(&mut header, list.len().to_u64() + spare, widths.lengths);
        header.extend_from_slice(&checksum::jenkins_lookup3(&header).to_le_bytes());
        (header, list)
    }
    // The signatures and version are defined in "Free-space Index", format specification version 4.0.
    const FSM_HEADER_SIGNATURE: &[u8; 4] = b"FSHD";
    const FSM_SECTION_LIST_SIGNATURE: &[u8; 4] = b"FSSE";
    const FSM_VERSION: u8 = 0;
}
