//! Adapts persistent free-space managers between storage, semantic manager sets, and the wire
//! codec.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeSet;
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::BTreeSet;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_format::__private::FreeSection;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_format::__private::SECTION_CLASS_LARGE;
use hdf5_pure_format::__private::SECTION_CLASS_SIMPLE;
use hdf5_pure_format::__private::SECTION_CLASS_SMALL;
use hdf5_pure_format::__private::SectionInfoCounts;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::{AdmittedFreeSnapshot, ManagerKind, ManagerSections};

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::file_space_info::FileSpaceStrategy;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::source::Source;
use crate::width::FormatWidths;

/// Reads the free sections of the managers at `manager_addrs` from the file `data`.
///
/// Adds `base`, the file's base address, to every address the file stores. Skips an undefined
/// manager address, and a manager whose section list address is undefined.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if `base` and a stored address sum past `u64::MAX`,
/// [`FormatError::ValueTooLargeForPlatform`] if a position in the file or the length of a section
/// list does not fit a `usize`, [`FormatError::InvalidFreeSpaceManager`] if a header or any part
/// of a section list's allocation is past the end of `data`, and the errors of
/// [`FreeSpaceManagerHeader::parse`] and
/// [`parse_section_info`](hdf5_pure_format::__private::parse_section_info) for each block it reads.
#[cfg(test)]
fn read_persisted_sections(
    data: &[u8],
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[DecodedStoredAddress],
) -> Result<Vec<FreeSection>, FormatError> {
    let mut sections = Vec::new();
    for &addr in manager_addrs {
        let DecodedStoredAddress::Defined(addr) = addr else {
            continue;
        };
        let a = base.resolve(addr)?.get().to_usize()?;
        let header = FreeSpaceManagerHeader::parse(
            widths,
            data.get(a..).ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(format!(
                    "the header at {a} is past the end of the file, {} bytes long",
                    data.len()
                ))
            })?,
        )?;
        if header.section_list_addr().is_undefined(widths.offsets) {
            continue;
        }
        let fa = base.resolve(header.section_list_addr())?.get().to_usize()?;
        let used = header.section_list_used().to_usize()?;
        let allocated = header.section_list_allocated().to_usize()?;
        fa.checked_add(allocated)
            .filter(|&end| end <= data.len())
            .ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(format!(
                    "the section list allocation at {fa}, {allocated} bytes long, runs past the \
                     end of the file, {} bytes long",
                    data.len()
                ))
            })?;
        let block = fa
            .checked_add(used)
            .and_then(|end| data.get(fa..end))
            .ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(format!(
                    "the section list at {fa}, {used} bytes long, runs past the end of the file, \
                     {} bytes long",
                    data.len()
                ))
            })?;
        sections.extend(hdf5_pure_format::__private::parse_section_info(
            widths, block, addr, &header,
        )?);
    }
    Ok(sections)
}

/// Recovered free sections and checked physical ranges of their manager header and section-list
/// allocations.
pub(crate) type PersistedSections = (Vec<FreeSection>, Vec<FileRange>);

/// Recovers free sections and complete manager allocations through a random-access
/// [`Source`].
///
/// Adds `base` to stored addresses and skips undefined manager or section list addresses.
/// Reads each section list's used bytes and records its allocated length for reclamation.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if an address conversion or allocation end overflows,
/// [`FormatError::InvalidFreeSpaceManager`] if a section list allocation exceeds the source
/// length, and [`FormatError::ValueTooLargeForPlatform`] if a required read length exceeds
/// [`usize::MAX`]. Returns source read errors and the errors of [`FreeSpaceManagerHeader::parse`]
/// and [`parse_section_info`](hdf5_pure_format::__private::parse_section_info) for each block.
pub(crate) fn read_persisted_sections_source<S: Source>(
    src: &S,
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[DecodedStoredAddress],
) -> Result<PersistedSections, FormatError> {
    let mut sections = Vec::new();
    let blocks = read_persisted_with_sections(
        src,
        widths,
        base,
        manager_addrs,
        |address, header, bytes| {
            sections.extend(hdf5_pure_format::__private::parse_section_info(
                widths, bytes, address, header,
            )?);
            Ok(())
        },
    )?;
    Ok((sections, blocks))
}

/// Recovers complete physical manager allocations without retaining free-section records.
///
/// Skips undefined manager addresses. Includes each manager header even when its section-list
/// address is undefined. Checks each section list's complete allocation against the source length,
/// reads its used bytes, and checks its free-section records.
///
/// # Errors
///
/// Returns the address-conversion, source-read, and parsing errors of
/// [`read_persisted_sections_source`].
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "allocation-only recovery has no production consumer"
    )
)]
pub(crate) fn read_persisted_allocations_source(
    src: &impl Source,
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[DecodedStoredAddress],
) -> Result<Vec<FileRange>, FormatError> {
    read_persisted_with_sections(
        src,
        widths,
        base,
        manager_addrs,
        |address, header, bytes| {
            hdf5_pure_format::__private::visit_section_info(
                widths,
                bytes,
                address,
                header,
                |_section| {},
            )
        },
    )
}

/// Reads manager headers and used section-list bytes while recording complete physical ranges.
///
/// Calls `read_sections` with each defined section list's stored header address, parsed header,
/// and used bytes. Checks the full allocation against the source length before reading the list.
///
/// # Errors
///
/// Returns the callback's error or the address-conversion, source-read, and header-parsing errors
/// of [`read_persisted_sections_source`].
fn read_persisted_with_sections(
    src: &impl Source,
    widths: FormatWidths,
    base: BaseAddress,
    manager_addrs: &[DecodedStoredAddress],
    mut read_sections: impl FnMut(
        StoredAddress,
        &FreeSpaceManagerHeader,
        &[u8],
    ) -> Result<(), FormatError>,
) -> Result<Vec<FileRange>, FormatError> {
    let mut blocks = Vec::new();
    let hdr_len = hdf5_pure_format::__private::free_space_manager_header_len(widths);
    for &addr in manager_addrs {
        let DecodedStoredAddress::Defined(addr) = addr else {
            continue;
        };
        let a = base.resolve(addr)?;
        let fshd = src.read_exact_at(a.get(), hdr_len.to_usize()?)?;
        let header = FreeSpaceManagerHeader::parse(widths, &fshd)?;
        blocks.push(FileRange::new(a, hdr_len)?);
        if header.section_list_addr().is_undefined(widths.offsets) {
            continue;
        }
        let fa = base.resolve(header.section_list_addr())?;
        let used = header.section_list_used();
        let allocated = header.section_list_allocated();
        let range = FileRange::new(fa, allocated)?;
        if range.end().get() > src.len() {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the section list allocation at {}, {allocated} bytes long, runs past the end of \
                 the file, {} bytes long",
                fa.get(),
                src.len()
            )));
        }
        let block = src.read_exact_at(fa.get(), used.to_usize()?)?;
        read_sections(addr, &header, &block)?;
        blocks.push(range);
    }
    Ok(blocks)
}

/// Returns the semantic kind represented by File Space Info manager `slot` for `strategy`.
pub(crate) fn manager_kind(strategy: FileSpaceStrategy, slot: usize) -> Option<ManagerKind> {
    match strategy {
        FileSpaceStrategy::FsmAggr => Some(ManagerKind::Flat),
        FileSpaceStrategy::Page => match slot {
            0 => Some(ManagerKind::PagedSmallMetadata),
            2 => Some(ManagerKind::PagedSmallRaw),
            6 => Some(ManagerKind::PagedLargeGeneric),
            _ => None,
        },
        FileSpaceStrategy::Aggr | FileSpaceStrategy::None => None,
    }
}

/// A semantic manager's complete stored allocations, section class, and wire sections.
pub(crate) struct ManagerBlock {
    kind: ManagerKind,
    header: Extent,
    section_list: Extent,
    class: u8,
    sections: Vec<FreeSection>,
    /// The encoded section-list length before any surplus allocation padding.
    section_list_len: u64,
}

impl ManagerBlock {
    /// Returns the semantic kind used to assign the File Space Info slot.
    pub(crate) fn kind(&self) -> ManagerKind {
        self.kind
    }

    /// Returns the complete header allocation in stored coordinates.
    pub(crate) fn header(&self) -> Extent {
        self.header
    }
}

/// A checked manager layout in stored coordinates.
///
/// Header allocations have their measured encoding lengths. Section-list allocations include
/// their encodings and any supplied surplus capacity. All allocations are disjoint from each other
/// and from the projected free sections. Allocation ownership and PAGE placement remain the
/// caller's responsibility.
pub(crate) struct ManagerPlan {
    widths: FormatWidths,
    /// Manager address per File Space Info slot, the undefined address at the file's offset width
    /// where inactive.
    slots: [StoredAddress; NUM_FILE_FSM_MANAGERS],
    /// The active managers in ascending File Space Info slot order.
    blocks: Vec<ManagerBlock>,
    /// The largest exclusive allocation end, or the supplied start for an empty contiguous plan.
    end_of_managers: StoredAddress,
}

impl ManagerPlan {
    /// Checks complete manager allocations against their encodings and projected free sections.
    ///
    /// Uses `empty_end` as the end address when `blocks` is empty.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a header length differs from its measured length,
    /// a section list is too short, or allocations overlap each other or projected free sections.
    /// Returns [`FormatError::DefinedAddressIsUndefined`], [`FormatError::AddressTooLarge`], or
    /// [`FormatError::LengthTooLarge`] if an allocation cannot be encoded at `widths`.
    fn new(
        widths: FormatWidths,
        blocks: Vec<ManagerBlock>,
        empty_end: StoredAddress,
    ) -> Result<Self, FormatError> {
        let mut slots = [StoredAddress::undefined(widths.offsets); NUM_FILE_FSM_MANAGERS];
        let mut end_of_managers = empty_end;
        let max_length = u64::MAX >> (u64::BITS - u32::from(widths.lengths.get()) * u8::BITS);
        for (index, block) in blocks.iter().enumerate() {
            let ManagerBlock {
                kind,
                header,
                section_list,
                class: _,
                sections: _,
                section_list_len,
            } = block;
            if header.len()
                != ByteLength::new(hdf5_pure_format::__private::free_space_manager_header_len(
                    widths,
                ))
                || section_list.len() < ByteLength::new(*section_list_len)
            {
                return Err(FormatError::Internal(
                    "manager allocations differ from their measured lengths".into(),
                ));
            }
            if section_list.len() > ByteLength::new(max_length) {
                return Err(FormatError::LengthTooLarge {
                    length: section_list.len().get(),
                    length_size: widths.lengths.get(),
                });
            }
            for extent in [*header, *section_list] {
                if extent.start().is_undefined(widths.offsets) {
                    return Err(FormatError::DefinedAddressIsUndefined {
                        address: extent.start().get(),
                        offset_size: widths.offsets.get(),
                    });
                }
                if extent.end().get() > StoredAddress::undefined(widths.offsets).get() {
                    return Err(FormatError::AddressTooLarge {
                        address: extent.end().get(),
                        offset_size: widths.offsets.get(),
                    });
                }
                if blocks.iter().any(|manager| {
                    manager.sections.iter().any(|section| {
                        extent.start().get() < section.addr.get().saturating_add(section.size)
                            && section.addr < extent.end()
                    })
                }) {
                    return Err(FormatError::Internal(
                        "a manager allocation overlaps a projected free section".into(),
                    ));
                }
                if blocks[..index].iter().any(|other| {
                    [other.header, other.section_list]
                        .iter()
                        .any(|other| extent.range().overlaps(other.range()))
                }) {
                    return Err(FormatError::Internal("manager allocations overlap".into()));
                }
            }
            if header.range().overlaps(section_list.range()) {
                return Err(FormatError::Internal("manager allocations overlap".into()));
            }
            slots[manager_wire(*kind).0] = header.start();
            end_of_managers = end_of_managers.max(header.end()).max(section_list.end());
        }
        Ok(Self {
            widths,
            slots,
            blocks,
            end_of_managers,
        })
    }

    /// Serializes managers with surplus allocation capacity assigned to the final section list.
    ///
    /// The caller supplies the complete stored allocation end, including any owned padding.
    /// Extends the final section list only when it ends at the plan's largest allocation end.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `allocation_end` precedes the planned end, the final
    /// section list cannot be extended, or the added capacity overlaps a projected free section.
    /// Returns [`FormatError::OffsetOverflow`] if padded section-list arithmetic overflows, or
    /// [`FormatError::AddressTooLarge`] or [`FormatError::LengthTooLarge`] if the padded allocation
    /// cannot be encoded at the file's widths. Serialization errors from
    /// [`serialize_free_space_manager_with_allocation`][serialize] are returned in iterator items.
    ///
    /// [serialize]: hdf5_pure_format::__private::serialize_free_space_manager_with_allocation
    pub(crate) fn encode(
        self,
        allocation_end: StoredAddress,
    ) -> Result<impl Iterator<Item = Result<EncodedManager, FormatError>>, FormatError> {
        let padding = allocation_end
            .checked_distance_from(self.end())
            .ok_or_else(|| {
                FormatError::Internal("the manager allocation does not contain its encoding".into())
            })?;
        if padding == 0 {
            return Ok(self.encode_allocations());
        }
        let Self {
            widths,
            slots: _,
            mut blocks,
            end_of_managers,
        } = self;
        if let Some(block) = blocks.last_mut() {
            if block.section_list.end() != end_of_managers {
                return Err(FormatError::Internal(
                    "the final manager section list does not end its allocation".into(),
                ));
            }
            block.section_list = manager_extent(
                block.section_list.start(),
                block
                    .section_list
                    .len()
                    .checked_add(ByteLength::new(padding))
                    .ok_or(FormatError::OffsetOverflow {
                        offset: block.section_list.len().get(),
                        length: padding,
                    })?
                    .get(),
            )?;
        }
        Ok(Self::new(widths, blocks, allocation_end)?.encode_allocations())
    }

    /// Returns header addresses by File Space Info slot, with undefined inactive slots.
    pub(crate) fn slots(&self) -> [StoredAddress; NUM_FILE_FSM_MANAGERS] {
        self.slots
    }

    /// Returns active managers in File Space Info slot order.
    pub(crate) fn blocks(&self) -> &[ManagerBlock] {
        &self.blocks
    }

    /// Returns complete stored allocations in File Space Info slot order.
    ///
    /// For each manager, yields the header allocation and then the section-list allocation,
    /// including any surplus capacity.
    pub(crate) fn allocations(&self) -> impl Iterator<Item = Extent> + '_ {
        self.blocks
            .iter()
            .flat_map(|block| [block.header, block.section_list])
    }

    /// Returns the largest exclusive allocation end.
    ///
    /// For an empty contiguous plan, returns the start passed to [`ManagerProjection::into_plan`].
    /// For an empty plan from [`ManagerProjection::place`], returns zero.
    pub(crate) fn end(&self) -> StoredAddress {
        self.end_of_managers
    }

    /// Serializes managers using their complete planned allocations.
    ///
    /// Returns one header and section-list pair per manager, in File Space Info slot order.
    /// Each section-list buffer includes its complete allocated capacity.
    ///
    /// # Errors
    ///
    /// Serialization errors from [`serialize_free_space_manager_with_allocation`][serialize]
    /// are returned in iterator items.
    ///
    /// [serialize]: hdf5_pure_format::__private::serialize_free_space_manager_with_allocation
    pub(crate) fn encode_allocations(
        self,
    ) -> impl Iterator<Item = Result<EncodedManager, FormatError>> {
        let Self {
            widths,
            slots: _,
            blocks,
            end_of_managers: _,
        } = self;
        blocks.into_iter().map(move |block| {
            let ManagerBlock {
                kind: _,
                header,
                section_list,
                class,
                sections,
                section_list_len: _,
            } = block;
            let (header_bytes, section_list_bytes) =
                hdf5_pure_format::__private::serialize_free_space_manager_with_allocation(
                    widths,
                    &sections,
                    header.start(),
                    section_list.start(),
                    class,
                    Some(section_list.len().get()),
                )?;
            #[cfg(all(test, feature = "std"))]
            crate::edit::observations::record_manager_encoding(
                header_bytes.len() + section_list_bytes.len(),
            );
            Ok(EncodedManager {
                header_address: header.start(),
                header: header_bytes,
                section_list_address: section_list.start(),
                section_list: section_list_bytes,
            })
        })
    }
}

/// Serialized manager metadata and its stored emission addresses.
///
/// The section-list buffer includes its complete planned allocation capacity.
pub(crate) struct EncodedManager {
    pub(crate) header_address: StoredAddress,
    pub(crate) header: Vec<u8>,
    pub(crate) section_list_address: StoredAddress,
    pub(crate) section_list: Vec<u8>,
}

/// Manager sections and their measured wire lengths, retained before address assignment.
pub(crate) struct ManagerProjection {
    blocks: Vec<ProjectedManager>,
    widths: FormatWidths,
    encoded_len: u64,
}

impl ManagerProjection {
    /// Projects admitted sections and measures their manager encodings at `widths`.
    ///
    /// Retains the projected sections for placement and encoding without repeating projection.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the combined manager length exceeds [`u64::MAX`].
    pub(crate) fn new(
        snapshot: AdmittedFreeSnapshot<'_>,
        widths: FormatWidths,
    ) -> Result<Self, FormatError> {
        Self::from_managers(&ManagerSections::from_snapshot(snapshot), widths)
    }

    /// Projects admitted sections and returns the small metadata manager's distinct sizes.
    ///
    /// Retains sections and measured lengths for placement and encoding. The size index supports
    /// measurement of additional padding sections without rebuilding the base size groups.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the combined manager length exceeds [`u64::MAX`].
    pub(crate) fn indexed(
        snapshot: AdmittedFreeSnapshot<'_>,
        widths: FormatWidths,
    ) -> Result<(Self, BTreeSet<u64>), FormatError> {
        let mut sizes = BTreeSet::new();
        let projection = Self::from_managers_indexed(
            &ManagerSections::from_snapshot(snapshot),
            widths,
            Some(&mut sizes),
        )?;
        Ok((projection, sizes))
    }

    /// Borrows the projected sections of `kind`, or an empty slice for an inactive manager.
    pub(crate) fn sections(&self, kind: ManagerKind) -> &[FreeSection] {
        self.blocks
            .iter()
            .find(|block| block.kind == kind)
            .map_or(&[], |block| block.sections.as_slice())
    }

    /// Returns manager kinds and measured header/list lengths in File Space Info slot order.
    pub(crate) fn measurements(&self) -> impl Iterator<Item = (ManagerKind, [u64; 2])> + '_ {
        self.blocks
            .iter()
            .zip(self.block_lengths())
            .map(|(block, lengths)| (block.kind, lengths))
    }

    fn from_managers(
        managers: &[ManagerSections],
        widths: FormatWidths,
    ) -> Result<Self, FormatError> {
        Self::from_managers_indexed(managers, widths, None)
    }

    fn from_managers_indexed(
        managers: &[ManagerSections],
        widths: FormatWidths,
        mut metadata_sizes: Option<&mut BTreeSet<u64>>,
    ) -> Result<Self, FormatError> {
        #[cfg(all(test, feature = "std"))]
        crate::edit::observations::record_manager_projection(
            managers.len(),
            managers
                .iter()
                .map(|manager| manager.sections().len())
                .sum(),
        );
        let mut blocks = Vec::with_capacity(managers.len());
        for manager in managers {
            let sections: Vec<_> = manager
                .sections()
                .iter()
                .map(|extent| FreeSection {
                    addr: extent.start(),
                    size: extent.len().get(),
                })
                .collect();
            let section_list_len = if let Some(index) = metadata_sizes
                .as_deref_mut()
                .filter(|_| manager.kind() == ManagerKind::PagedSmallMetadata)
            {
                index.extend(sections.iter().map(|section| section.size));
                SectionInfoCounts::new(sections.len() as u64, index.len() as u64)?
                    .encoded_len(widths)?
            } else {
                let sizes: Vec<_> = sections.iter().map(|section| section.size).collect();
                hdf5_pure_format::__private::section_info_len(widths, &sizes)
            };
            let (slot, class) = manager_wire(manager.kind());
            blocks.push(ProjectedManager {
                kind: manager.kind(),
                slot,
                class,
                sections,
                section_list_len,
            });
        }
        blocks.sort_unstable_by_key(|block| block.slot);
        debug_assert!(
            blocks.windows(2).all(|pair| pair[0].slot != pair[1].slot),
            "two semantic free-space managers map to the same File Space Info slot"
        );
        let mut projection = Self {
            blocks,
            widths,
            encoded_len: 0,
        };
        let encoded_len =
            projection
                .block_lengths()
                .try_fold(0_u64, |length, [header, section_list]| {
                    length
                        .checked_add(header)
                        .and_then(|sum| sum.checked_add(section_list))
                        .ok_or(FormatError::OffsetOverflow {
                            offset: length,
                            length: header.saturating_add(section_list),
                        })
                })?;
        projection.encoded_len = encoded_len;
        Ok(projection)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Returns projected manager counts, section capacity, and measured encoding length.
    #[cfg(test)]
    pub(crate) fn work(&self) -> ManagerWork {
        let Self {
            blocks,
            widths: _,
            encoded_len,
        } = self;
        ManagerWork {
            managers: blocks.len() as u64,
            sections: blocks.iter().map(|block| block.sections.len() as u64).sum(),
            section_capacity: blocks
                .iter()
                .map(|block| block.sections.capacity() as u64)
                .sum(),
            encoded_bytes: *encoded_len,
        }
    }

    /// Returns the combined header and section-list length before allocation padding.
    pub(crate) fn encoded_len(&self) -> u64 {
        self.encoded_len
    }

    /// Places managers contiguously from `start`, in ascending File Space Info slot order.
    ///
    /// All block addresses use stored coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a manager's exclusive end exceeds [`u64::MAX`],
    /// or the placement errors described by [`Self::place`].
    pub(crate) fn into_plan(self, start: StoredAddress) -> Result<ManagerPlan, FormatError> {
        let mut cursor = start;
        let mut plan = self.place(|[header_len, section_list_len]| {
            let header = manager_extent(cursor, header_len)?;
            let section_list = manager_extent(header.end(), section_list_len)?;
            cursor = section_list.end();
            Ok([header, section_list])
        })?;
        if plan.blocks.is_empty() {
            plan.end_of_managers = start;
        }
        Ok(plan)
    }

    /// Returns measured header and section-list lengths for each active manager.
    ///
    /// Pairs are in File Space Info slot order. Lengths are in bytes and exclude surplus capacity.
    pub(crate) fn block_lengths(&self) -> impl ExactSizeIterator<Item = [u64; 2]> + '_ {
        let Self {
            blocks,
            widths,
            encoded_len: _,
        } = self;
        blocks.iter().map(|block| {
            [
                hdf5_pure_format::__private::free_space_manager_header_len(*widths),
                block.section_list_len,
            ]
        })
    }

    /// Constructs a checked manager layout from the caller's allocations.
    ///
    /// Calls `place` once per active manager, in File Space Info slot order, with the measured
    /// header and section-list lengths. The caller returns their complete allocations.
    /// Header lengths must match exactly, and section lists may include surplus capacity.
    /// Allocation ownership and PAGE placement remain the caller's responsibility.
    ///
    /// # Errors
    ///
    /// Returns the caller's error, [`FormatError::Internal`] if a header length differs, a section
    /// list is too short, or allocations overlap each other or projected free sections, or
    /// [`FormatError::DefinedAddressIsUndefined`], [`FormatError::AddressTooLarge`], or
    /// [`FormatError::LengthTooLarge`] if an allocation cannot be encoded at the file's widths.
    pub(crate) fn place(
        self,
        mut place: impl FnMut([u64; 2]) -> Result<[Extent; 2], FormatError>,
    ) -> Result<ManagerPlan, FormatError> {
        self.place_with_kind(|_, lengths| place(lengths))
    }

    /// Constructs a checked layout with manager identity passed to the placement callback.
    ///
    /// Uses the same allocation requirements and checks as [`Self::place`].
    ///
    /// # Errors
    ///
    /// Returns the placement and encoding errors described by [`Self::place`].
    pub(crate) fn place_with_kind(
        self,
        mut place: impl FnMut(ManagerKind, [u64; 2]) -> Result<[Extent; 2], FormatError>,
    ) -> Result<ManagerPlan, FormatError> {
        let Self {
            blocks,
            widths,
            encoded_len: _,
        } = self;
        let blocks = blocks
            .into_iter()
            .map(
                |ProjectedManager {
                     kind,
                     slot: _,
                     class,
                     sections,
                     section_list_len,
                 }| {
                    let [header, section_list] = place(
                        kind,
                        [
                            hdf5_pure_format::__private::free_space_manager_header_len(widths),
                            section_list_len,
                        ],
                    )?;
                    Ok(ManagerBlock {
                        kind,
                        header,
                        section_list,
                        class,
                        sections,
                        section_list_len,
                    })
                },
            )
            .collect::<Result<Vec<_>, FormatError>>()?;
        ManagerPlan::new(widths, blocks, StoredAddress::ZERO)
    }
}

/// Constructs a nonempty stored allocation with a checked exclusive end.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if `length` is zero or the exclusive end overflows.
fn manager_extent(start: StoredAddress, length: u64) -> Result<Extent, FormatError> {
    Extent::new(start, ByteLength::new(length)).ok_or(FormatError::OffsetOverflow {
        offset: start.get(),
        length,
    })
}

/// Manager projection counts and measured encoding length.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ManagerWork {
    /// Projected semantic managers.
    pub(crate) managers: u64,
    /// Wire sections in the projected managers.
    pub(crate) sections: u64,
    /// Combined section-vector capacity in elements.
    pub(crate) section_capacity: u64,
    /// Combined header and section-list length before allocation padding, in bytes.
    pub(crate) encoded_bytes: u64,
}

/// One manager's semantic sections, wire class, slot, and measured section-list length.
struct ProjectedManager {
    kind: ManagerKind,
    slot: usize,
    class: u8,
    sections: Vec<FreeSection>,
    section_list_len: u64,
}

/// Returns the encoded length of `managers` without assigning them file positions.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the combined length exceeds [`u64::MAX`].
#[cfg(test)]
pub(crate) fn manager_blocks_len(
    managers: &[ManagerSections],
    widths: FormatWidths,
) -> Result<u64, FormatError> {
    Ok(ManagerProjection::from_managers(managers, widths)?.encoded_len())
}

/// Returns the File Space Info slot and section class for `kind`.
// The writer writes flat sections in DRAW to preserve unused space through native raw appends.
// The C library persists non-self-referential managers in `H5MF_settle_raw_data_fsm`
// (`H5MF.c`, HDF5 2.2.0).
pub(crate) fn manager_wire(kind: ManagerKind) -> (usize, u8) {
    match kind {
        ManagerKind::Flat => (DRAW_MANAGER_SLOT, SECTION_CLASS_SIMPLE),
        ManagerKind::PagedLargeGeneric => (6, SECTION_CLASS_LARGE),
        ManagerKind::PagedSmallMetadata => (0, SECTION_CLASS_SMALL),
        ManagerKind::PagedSmallRaw => (DRAW_MANAGER_SLOT, SECTION_CLASS_SMALL),
    }
}

/// The zero-based slot for the small DRAW manager in a version 1 File Space Info message.
///
/// Defined in "The File Space Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo
const DRAW_MANAGER_SLOT: usize = 2;

/// Places `managers` contiguously from `start` using their HDF5 wire slots and section classes.
///
/// Uses each manager's kind to select its File Space Info slot and `FSSE` section class.
/// The caller passes stored-coordinate extents, and the returned block addresses use the same
/// coordinate system.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if a manager's exclusive end exceeds [`u64::MAX`],
/// or the placement errors described by [`ManagerProjection::place`].
#[cfg(test)]
pub(crate) fn plan_managers(
    managers: &[ManagerSections],
    start: StoredAddress,
    widths: FormatWidths,
) -> Result<ManagerPlan, FormatError> {
    ManagerProjection::from_managers(managers, widths)?.into_plan(start)
}

#[cfg(test)]
mod tests {
    use core::cell::RefCell;

    use hdf5_pure_core::__private::ByteLength;
    use hdf5_pure_core::PhysicalAddress;
    use hdf5_pure_space::__private::{Extent, FreeList, SessionSpace};
    use rstest::rstest;
    use test_util::free_space;

    use crate::address::StoredAddressExt;
    use crate::file_space_info::FileSpacePageSize;
    use crate::width::OffsetWidth;

    use super::*;

    mod placement;
    mod visiting;

    /// A free section at `addr` spanning `size` bytes.
    fn section(addr: u64, size: u64) -> FreeSection {
        FreeSection {
            addr: StoredAddress::new(addr),
            size,
        }
    }

    fn widths(offset_size: u8, length_size: u8) -> FormatWidths {
        FormatWidths::from_sizes(offset_size, length_size).unwrap()
    }

    #[rstest]
    #[case::header(
        736,
        1000,
        "the header at 1000 is past the end of the file, 736 bytes long"
    )]
    #[case::section_list(
        735,
        619,
        "the section list allocation at 701, 35 bytes long, runs past the end of the file, 735 bytes long"
    )]
    fn a_manager_past_the_end_of_the_file_fails_to_read(
        #[case] file_len: usize,
        #[case] manager_addr: u64,
        #[case] reason: &str,
    ) {
        // The header is at 619 and its 35-byte section list at 701, so the file ends at 736.
        let fshd = free_space::single_section_header();
        let fsse = free_space::single_section_info();
        let mut buf = vec![0u8; 736];
        buf[619..619 + fshd.len()].copy_from_slice(&fshd);
        buf[701..701 + fsse.len()].copy_from_slice(&fsse);
        buf.truncate(file_len);

        assert_eq!(
            read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(
                    manager_addr
                ))],
            ),
            Err(FormatError::InvalidFreeSpaceManager(reason.to_owned()))
        );
    }

    #[rstest]
    #[case::tight(0, 35)]
    #[case::spare_capacity(0, 51)]
    #[case::userblock(512, 51)]
    #[case::large_capacity(0, 1 << 33)]
    fn source_recovery_reads_used_bytes_and_records_allocated_bytes(
        #[case] base: u64,
        #[case] allocated: u64,
    ) {
        let source = manager_source(base, allocated, base + 701 + allocated);
        let (sections, blocks) = read_persisted_sections_source(
            &source,
            widths(8, 8),
            BaseAddress::new(base),
            &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
        )
        .unwrap();

        assert_eq!(sections, vec![section(2848, 1600)]);
        assert_eq!(
            blocks,
            vec![
                FileRange::new(PhysicalAddress::new(base + 619), 82).unwrap(),
                FileRange::new(PhysicalAddress::new(base + 701), allocated).unwrap()
            ]
        );
        assert_eq!(
            *source.reads.borrow(),
            vec![(base + 619, 82), (base + 701, 35)]
        );
    }

    #[rstest]
    #[case::past_source(
        0,
        51,
        751,
        FormatError::InvalidFreeSpaceManager(
            "the section list allocation at 701, 51 bytes long, runs past the end of the file, 751 bytes long".to_owned()
        )
    )]
    #[case::allocation_overflow(
        u64::MAX - 1000,
        300,
        u64::MAX,
        FormatError::OffsetOverflow { offset: u64::MAX - 299, length: 300 }
    )]
    fn source_recovery_rejects_invalid_allocation_geometry(
        #[case] base: u64,
        #[case] allocated: u64,
        #[case] file_len: u64,
        #[case] expected: FormatError,
    ) {
        let source = manager_source(base, allocated, file_len);
        assert_eq!(
            read_persisted_sections_source(
                &source,
                widths(8, 8),
                BaseAddress::new(base),
                &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(*source.reads.borrow(), vec![(base + 619, 82)]);
    }

    #[test]
    fn both_read_paths_accept_checksummed_unused_section_list_bytes() {
        let header = free_space::oversized_section_list_header();
        let mut list = free_space::oversized_section_list();
        list[185] = 0xd0;
        let len = list.len();
        test_util::checksum::restamp(&mut list, 0, len);
        let mut file = vec![0; 645 + list.len()];
        file[48..48 + header.len()].copy_from_slice(&header);
        file[645..].copy_from_slice(&list);
        let managers = [DecodedStoredAddress::Defined(StoredAddress::new(48))];
        let expected = vec![
            section(842, 10),
            section(5375, 10),
            section(474, 24),
            section(2444, 27),
            section(1595, 47),
            section(130, 74),
            section(2618, 194),
            section(3793, 254),
            section(1122, 326),
            section(4385, 393),
        ];

        assert_eq!(
            read_persisted_sections(&file, widths(8, 8), BaseAddress::ZERO, &managers),
            Ok(expected.clone()),
        );
        assert_eq!(
            read_persisted_sections_source(
                &crate::source::BytesSource::new(&file),
                widths(8, 8),
                BaseAddress::ZERO,
                &managers,
            ),
            Ok((
                expected,
                vec![
                    FileRange::new(PhysicalAddress::new(48), 82).unwrap(),
                    FileRange::new(PhysicalAddress::new(645), 197).unwrap()
                ]
            )),
        );
    }

    #[test]
    fn read_persisted_sections_follows_managers() {
        // Place the single-section FSHD@619 + FSSE@701 fixtures in a buffer at
        // their real offsets and read them through the manager-address indirection.
        let fshd = free_space::single_section_header();
        let fsse = free_space::single_section_info();
        let mut buf = vec![0u8; 701 + fsse.len()];
        buf[619..619 + fshd.len()].copy_from_slice(&fshd);
        buf[701..701 + fsse.len()].copy_from_slice(&fsse);

        let got = read_persisted_sections(
            &buf,
            widths(8, 8),
            BaseAddress::ZERO,
            &[
                DecodedStoredAddress::Defined(StoredAddress::new(619)),
                DecodedStoredAddress::Undefined,
                DecodedStoredAddress::Undefined,
            ],
        )
        .unwrap();
        assert_eq!(got, vec![section(2848, 1600)]);
        // No defined managers -> no sections.
        assert!(
            read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Undefined]
            )
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn an_undefined_manager_slot_is_skipped_in_a_four_byte_offset_file() {
        assert!(
            read_persisted_sections(
                &[],
                widths(4, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Undefined]
            )
            .unwrap()
            .is_empty(),
            "an all-0xFF manager address is the sentinel for an unused slot"
        );
    }

    #[test]
    fn a_manager_with_no_section_info_is_skipped_in_a_four_byte_offset_file() {
        let (fshd, _fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths(4, 8),
            &[],
            StoredAddress::new(0),
            StoredAddress::new(0xFFFF_FFFF),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();

        assert!(
            read_persisted_sections(
                &fshd,
                widths(4, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(0))]
            )
            .unwrap()
            .is_empty(),
            "a manager tracking nothing records no section info"
        );

        let (sections, blocks) = read_persisted_sections_source(
            &crate::source::BytesSource::new(&fshd),
            widths(4, 8),
            BaseAddress::ZERO,
            &[DecodedStoredAddress::Defined(StoredAddress::new(0))],
        )
        .unwrap();
        assert!(sections.is_empty());
        assert_eq!(
            blocks,
            vec![
                FileRange::new(
                    PhysicalAddress::new(0),
                    hdf5_pure_format::__private::free_space_manager_header_len(widths(4, 8))
                )
                .unwrap()
            ],
            "only the header block was read"
        );
    }

    #[rstest]
    #[case::exact(0)]
    #[case::padded(16)]
    fn projected_manager_retains_measured_geometry_through_encoding(#[case] padding: u64) {
        let free =
            FreeList::from_sorted([
                Extent::new(StoredAddress::new(2848), ByteLength::new(1600)).unwrap()
            ])
            .unwrap();
        let projection =
            ManagerProjection::new(AdmittedFreeSnapshot::flat(&free), widths(8, 8)).unwrap();
        let expected_length = u64::try_from(
            free_space::single_section_header().len() + free_space::single_section_info().len(),
        )
        .unwrap();
        assert_eq!(projection.encoded_len(), expected_length);
        let plan = projection.into_plan(StoredAddress::new(619)).unwrap();
        assert_eq!(plan.end_of_managers, StoredAddress::new(736));
        assert_eq!(plan.slots[2], StoredAddress::new(619));
        let encoded = plan
            .encode(StoredAddress::new(736 + padding))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0].header_address, StoredAddress::new(619));
        assert_eq!(encoded[0].section_list_address, StoredAddress::new(701));
        assert_eq!(encoded[0].header.len(), 82);
        assert_eq!(
            u64::try_from(encoded[0].section_list.len()).unwrap(),
            35 + padding
        );
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &encoded[0].header).unwrap();
        assert_eq!(header.section_list_used(), 35 + padding);
        assert_eq!(header.section_list_allocated(), 35 + padding);
        assert_eq!(header.total_space(), 1600);
        assert_eq!(header.total_sections(), 1);
        assert_eq!(
            hdf5_pure_format::__private::parse_section_info(
                widths(8, 8),
                &encoded[0].section_list,
                StoredAddress::new(619),
                &header,
            )
            .unwrap(),
            [section(2848, 1600)]
        );
    }

    #[test]
    fn manager_encoding_rejects_capacity_shorter_than_its_projection() {
        let free =
            FreeList::from_sorted([
                Extent::new(StoredAddress::new(2848), ByteLength::new(1600)).unwrap()
            ])
            .unwrap();
        let projection =
            ManagerProjection::new(AdmittedFreeSnapshot::flat(&free), widths(8, 8)).unwrap();
        let plan = projection.into_plan(StoredAddress::new(619)).unwrap();
        let Err(error) = plan.encode(StoredAddress::new(735)) else {
            panic!("expected insufficient manager capacity");
        };
        assert_eq!(
            error,
            FormatError::Internal("the manager allocation does not contain its encoding".into())
        );
    }

    #[test]
    fn serialize_roundtrips_through_parse() {
        let sections = [
            section(4096, 512),
            section(9000, 512),
            section(20000, 70000),
        ];
        let (fshd, _fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths(8, 8),
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_sections(), 3);
        assert_eq!(header.total_space(), 512 + 512 + 70000);
        assert_eq!(header.section_list_addr(), StoredAddress::new(1100));

        // Place both blocks in a buffer and read them back through the manager
        // indirection; the recovered sections match (order-independent).
        let (fshd, fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths(8, 8),
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let mut buf = vec![0u8; 1100 + fsse.len()];
        buf[1000..1000 + fshd.len()].copy_from_slice(&fshd);
        buf[1100..1100 + fsse.len()].copy_from_slice(&fsse);
        let mut got = read_persisted_sections(
            &buf,
            widths(8, 8),
            BaseAddress::ZERO,
            &[DecodedStoredAddress::Defined(StoredAddress::new(1000))],
        )
        .unwrap();
        got.sort_by_key(|s| s.addr);
        let mut want = sections.to_vec();
        want.sort_by_key(|s| s.addr);
        assert_eq!(got, want);
    }

    #[test]
    fn an_allocated_section_list_past_eof_is_rejected_by_both_read_paths() {
        let allocated = 44;
        let source = manager_source(0, allocated, 736);
        let mut buf = vec![0u8; 736];
        buf[619..701].copy_from_slice(&source.header);
        buf[701..].copy_from_slice(&source.sections);
        let expected = FormatError::InvalidFreeSpaceManager(format!(
            "the section list allocation at 701, {allocated} bytes long, runs past the \
             end of the file, {} bytes long",
            buf.len()
        ));

        assert_eq!(
            read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(
            read_persisted_sections_source(
                &source,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(619))],
            )
            .unwrap_err(),
            expected
        );
    }

    #[test]
    fn class_id_is_emitted_and_ignored_on_read() {
        // The class-id byte is written per section and round-trips (the reader
        // ignores its value, recovering the same offsets/sizes for any class).
        let sections = [section(2000, 12768)];
        for class in [
            SECTION_CLASS_SIMPLE,
            SECTION_CLASS_SMALL,
            SECTION_CLASS_LARGE,
        ] {
            let (fshd, fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
                widths(8, 8),
                &sections,
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                class,
            )
            .unwrap();
            // The last section record's class byte precedes the 4-byte checksum.
            assert_eq!(fsse[fsse.len() - 5], class, "class byte written");
            let mut buf = vec![0u8; 1100 + fsse.len()];
            buf[1000..1000 + fshd.len()].copy_from_slice(&fshd);
            buf[1100..1100 + fsse.len()].copy_from_slice(&fsse);
            let got = read_persisted_sections(
                &buf,
                widths(8, 8),
                BaseAddress::ZERO,
                &[DecodedStoredAddress::Defined(StoredAddress::new(1000))],
            )
            .unwrap();
            assert_eq!(got, sections, "class {class} round-trips");
        }
    }

    #[test]
    fn semantic_paged_managers_keep_the_existing_wire_slots_and_classes() {
        let page_size = FileSpacePageSize::DEFAULT;
        let mut space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            ByteLength::new(0),
            page_size,
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallMetadata),
            Extent::new(StoredAddress::new(100), ByteLength::new(200)).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedSmallRaw),
            Extent::new(StoredAddress::new(5000), ByteLength::new(200)).unwrap(),
        );
        space.seed_persisted(
            Some(ManagerKind::PagedLargeGeneric),
            Extent::new(StoredAddress::new(8192), ByteLength::new(page_size.get())).unwrap(),
        );
        let managers = ManagerSections::paged(
            &space.paged_sections().expect("the strategy is PAGE"),
            page_size,
        );
        let plan = plan_managers(&managers, StoredAddress::new(1000), widths(8, 8)).unwrap();

        assert_eq!(plan.blocks.len(), 3);
        assert_eq!(plan.slots[0], plan.blocks[0].header.start());
        assert_eq!(plan.slots[2], plan.blocks[1].header.start());
        assert_eq!(plan.slots[6], plan.blocks[2].header.start());
        assert_eq!(plan.blocks[0].kind, ManagerKind::PagedSmallMetadata);
        assert_eq!(plan.blocks[1].kind, ManagerKind::PagedSmallRaw);
        assert_eq!(plan.blocks[2].kind, ManagerKind::PagedLargeGeneric);
        assert_eq!(plan.blocks[0].class, SECTION_CLASS_SMALL);
        assert_eq!(plan.blocks[1].class, SECTION_CLASS_SMALL);
        assert_eq!(plan.blocks[2].class, SECTION_CLASS_LARGE);
        assert_eq!(plan.blocks[0].sections, vec![section(100, 200)]);
        assert_eq!(plan.blocks[1].sections, vec![section(5000, 200)]);
        assert_eq!(
            plan.blocks[2].sections,
            vec![section(8192, page_size.get())]
        );
        assert!(
            plan.slots
                .iter()
                .enumerate()
                .all(|(slot, addr)| [0, 2, 6].contains(&slot)
                    || addr.is_undefined(OffsetWidth::Eight))
        );

        for block in &plan.blocks {
            let (fshd, fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
                widths(8, 8),
                &block.sections,
                block.header.start(),
                block.section_list.start(),
                block.class,
            )
            .unwrap();
            assert_eq!(
                block.section_list.start(),
                block.header.start().checked_add(fshd.len() as u64).unwrap(),
                "FSSE follows its FSHD"
            );
            assert_eq!(
                block
                    .section_list
                    .start()
                    .checked_add(fsse.len() as u64)
                    .unwrap(),
                if block.kind == ManagerKind::PagedLargeGeneric {
                    plan.end_of_managers
                } else {
                    plan.blocks
                        .iter()
                        .find(|candidate| candidate.header.start() > block.header.start())
                        .unwrap()
                        .header
                        .start()
                },
                "manager blocks remain contiguous"
            );
        }
    }

    #[test]
    fn manager_slots_translate_to_semantic_kinds_only_at_the_storage_boundary() {
        assert_eq!(
            manager_kind(FileSpaceStrategy::FsmAggr, 7),
            Some(ManagerKind::Flat)
        );
        assert_eq!(
            manager_kind(FileSpaceStrategy::Page, 0),
            Some(ManagerKind::PagedSmallMetadata)
        );
        assert_eq!(
            manager_kind(FileSpaceStrategy::Page, 2),
            Some(ManagerKind::PagedSmallRaw)
        );
        assert_eq!(
            manager_kind(FileSpaceStrategy::Page, 6),
            Some(ManagerKind::PagedLargeGeneric)
        );
        assert_eq!(manager_kind(FileSpaceStrategy::Page, 7), None);
    }

    struct ManagerSource {
        base: u64,
        file_len: u64,
        header: Vec<u8>,
        sections: Vec<u8>,
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl Source for ManagerSource {
        fn len(&self) -> u64 {
            self.file_len
        }

        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.reads.borrow_mut().push((offset, buf.len()));
            let data = match offset.checked_sub(self.base) {
                Some(619) => &self.header,
                Some(701) => &self.sections,
                _ => {
                    return Err(FormatError::UnexpectedEof {
                        expected: buf.len(),
                        available: 0,
                    });
                }
            };
            buf.copy_from_slice(data.get(..buf.len()).ok_or(FormatError::UnexpectedEof {
                expected: buf.len(),
                available: data.len(),
            })?);
            Ok(())
        }
    }

    fn manager_source(base: u64, allocated: u64, file_len: u64) -> ManagerSource {
        let mut header = free_space::single_section_header();
        header[70..78].copy_from_slice(&allocated.to_le_bytes());
        let len = header.len();
        test_util::checksum::restamp(&mut header, 0, len);
        ManagerSource {
            base,
            file_len,
            header,
            sections: free_space::single_section_info(),
            reads: RefCell::new(Vec::new()),
        }
    }
    #[rstest]
    #[case::header(u64::MAX, 82, u64::MAX)]
    #[case::section_list(u64::MAX - 82, 35, u64::MAX)]
    fn manager_placement_rejects_overflow(
        #[case] start: u64,
        #[case] length: u64,
        #[case] offset: u64,
    ) {
        let managers =
            ManagerSections::flat([
                Extent::new(StoredAddress::new(16), ByteLength::new(8)).unwrap()
            ]);
        let error = plan_managers(&managers, StoredAddress::new(start), widths(8, 8))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(error, FormatError::OffsetOverflow { offset, length });
    }

    #[test]
    fn manager_placement_preserves_the_maximum_exclusive_end() {
        let managers =
            ManagerSections::flat([
                Extent::new(StoredAddress::new(16), ByteLength::new(8)).unwrap()
            ]);
        let length = manager_blocks_len(&managers, widths(8, 8)).unwrap();
        let start = StoredAddress::new(u64::MAX - length);
        let plan = plan_managers(&managers, start, widths(8, 8)).unwrap();
        assert_eq!(plan.blocks[0].header.start(), start);
        assert_eq!(
            plan.blocks[0].section_list.start(),
            start.checked_add(82).unwrap()
        );
        assert_eq!(plan.end_of_managers, StoredAddress::new(u64::MAX));
    }
}
