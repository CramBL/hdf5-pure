//! File framing and recorded capacity for one source phase.
//!
//! Local ranges and problems are retained independently of topology and heap-reference facts.
//! The caller establishes source freshness, allocation ownership, and retirement permission
//! separately.

use std::sync::Arc;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{AllocationClass, Extent, FreeClass, RetainedKind};

use super::AnalysisPhase;
use super::ReconstructionSource;
use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_space_info::FileSpaceInfo;
use crate::free_space_manager;
use crate::message_type::MessageType;
use crate::object_header::MessageFilter;
use crate::object_header::ObjectHeader;
use crate::source::SourceMetadata;
use crate::superblock;

pub(in crate::edit) mod claims;

impl AnalysisPhase<'_> {
    /// Collects local file framing and recorded capacity once in this source phase.
    ///
    /// Requires a reconstruction source from the same phase. Records read and support failures
    /// in [`PhysicalStorage::problems`]. Source freshness remains the caller's responsibility.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `source` belongs to another phase.
    pub(in crate::edit) fn physical_storage(
        &mut self,
        source: &ReconstructionSource,
    ) -> Result<&PhysicalStorage, FormatError> {
        if !self
            .identity
            .as_ref()
            .is_some_and(|identity| Arc::ptr_eq(identity, &source.identity))
        {
            return Err(FormatError::Internal(
                "physical storage query belongs to another source phase".into(),
            ));
        }
        let engine = self.engine;
        Ok(self
            .physical
            .get_or_insert_with(|| PhysicalStorage::read(engine)))
    }
}

/// Local file and allocator observations for one source phase.
///
/// Retains observed extension allocations when later message parsing fails. Topology completeness,
/// reference completeness, allocation ownership, and retirement permission require separate proofs.
pub(in crate::edit) struct PhysicalStorage {
    /// The source image length when these facts were collected.
    physical_size: u64,
    /// The end address recorded in the session's superblock.
    logical_eoa: u64,
    file: Vec<FileStorageRange>,
    /// Producer-recorded metadata capacity, resolved to physical coordinates.
    metadata_capacity: Vec<FileRange>,
    /// Reusable sections, retained PAGE ranges, and unused aggregate capacity.
    non_live: Vec<CapacityRange>,
    problems: Vec<StorageProblem>,
}

impl PhysicalStorage {
    /// Collects file observations, metadata capacity, and allocator ranges independently.
    fn read(engine: &WriteEngine) -> Self {
        let mut storage = Self {
            physical_size: engine.image().len(),
            logical_eoa: engine.superblock().eof_address,
            file: Vec::new(),
            metadata_capacity: Vec::new(),
            non_live: Vec::new(),
            problems: Vec::new(),
        };
        storage.read_file(engine);
        storage.read_metadata_capacity(engine);
        storage.read_non_live(engine);
        storage
    }

    /// Returns the source image length observed when these facts were collected.
    pub(in crate::edit) fn physical_size(&self) -> u64 {
        self.physical_size
    }
    /// Returns the end address recorded in the session's superblock.
    pub(in crate::edit) fn logical_eoa(&self) -> u64 {
        self.logical_eoa
    }
    /// Returns observed file ranges, including ranges retained before a local failure.
    pub(in crate::edit) fn file(&self) -> &[FileStorageRange] {
        &self.file
    }
    /// Returns producer-recorded metadata capacity in physical coordinates.
    pub(in crate::edit) fn metadata_capacity(&self) -> &[FileRange] {
        &self.metadata_capacity
    }
    /// Borrows recorded allocator ranges with their capacity categories.
    ///
    /// Reusable sections precede retained PAGE ranges, followed by aggregate spare. Preserves
    /// original boundaries without sorting or coalescing.
    pub(in crate::edit) fn non_live(&self) -> &[CapacityRange] {
        &self.non_live
    }

    /// Returns local failures and unsupported storage observations.
    pub(in crate::edit) fn problems(&self) -> &[StorageProblem] {
        &self.problems
    }

    /// Reads framing and manager allocations, retaining earlier ranges on a later failure.
    fn read_file(&mut self, engine: &WriteEngine) {
        let (located, range) = match superblock::read_with_range(&engine.image()) {
            Ok(parsed) => parsed,
            Err(error) => {
                self.problems.push(StorageProblem::Framing {
                    kind: FileStorageKind::Superblock,
                    error,
                });
                return;
            }
        };
        if located.address.get() > 0 {
            match FileRange::new(PhysicalAddress::new(0), located.address.get()) {
                Ok(range) => self.file.push(FileStorageRange {
                    range,
                    kind: FileStorageKind::Userblock,
                }),
                Err(error) => self.problems.push(StorageProblem::Address(error)),
            }
        }
        self.file.push(FileStorageRange {
            range,
            kind: FileStorageKind::Superblock,
        });
        let current = engine.superblock();
        if located.address != engine.context.signature()
            || located.widths != engine.widths()
            || located.superblock.version() != current.version()
            || located.superblock.base_address != current.base_address
            || located.superblock.eof_address != current.eof_address
            || located.superblock.root_group_address != current.root_group_address
            || located.superblock.superblock_extension_address
                != current.superblock_extension_address
        {
            self.problems.push(StorageProblem::Context);
            return;
        }
        for address in [
            located.superblock.free_space_address,
            located.superblock.driver_info_address,
        ]
        .into_iter()
        .flatten()
        {
            if !address.is_undefined(engine.widths().offsets) {
                self.problems
                    .push(StorageProblem::LegacyAllocation(address));
            }
        }
        let Some(extension) = current
            .superblock_extension_address
            .filter(|address| !address.is_undefined(engine.widths().offsets))
        else {
            return;
        };
        let extension = match engine.addresses().resolve(extension) {
            Ok(address) => address,
            Err(error) => {
                self.problems.push(StorageProblem::Address(error));
                return;
            }
        };
        let mut chunks = Vec::new();
        let parsed = ObjectHeader::parse_from_source_with_chunk_observer(
            &SourceMetadata(&engine.image()),
            AccessMode::ReadWrite,
            extension,
            engine.widths(),
            engine.addresses().effective_base(),
            MessageFilter::All,
            &mut |chunk| chunks.push(chunk.allocation()),
        );
        self.file
            .extend(chunks.into_iter().map(|range| FileStorageRange {
                range,
                kind: FileStorageKind::Extension,
            }));
        let header = match parsed {
            Ok(header) => header,
            Err(error) => {
                self.problems.push(StorageProblem::Framing {
                    kind: FileStorageKind::Extension,
                    error,
                });
                return;
            }
        };
        let mut info = None;
        for message in header.messages {
            if message.flags.is_shared() {
                self.problems
                    .push(StorageProblem::ExtensionMessage(message.msg_type));
                continue;
            }
            match message.msg_type {
                MessageType::FILE_SPACE_INFO if info.is_none() => {
                    info = Some(hdf5_pure_format::__private::parse_file_space_info(
                        engine.widths(),
                        &message.data,
                    ));
                }
                MessageType::NIL
                | MessageType::OBJECT_HEADER_CONTINUATION
                | MessageType::B_TREE_K_VALUES => {}
                kind => self.problems.push(StorageProblem::ExtensionMessage(kind)),
            }
        }
        if let Some(info) = info {
            match info {
                Ok(info) => self.read_managers(engine, info),
                Err(error) => self.problems.push(StorageProblem::Framing {
                    kind: FileStorageKind::Extension,
                    error,
                }),
            }
        }
    }

    /// Reads complete persistent-manager allocations and records any read or framing failure.
    fn read_managers(&mut self, engine: &WriteEngine, info: FileSpaceInfo) {
        if !info.persist {
            return;
        }
        match free_space_manager::read_persisted_allocations_source(
            &engine.image(),
            engine.widths(),
            engine.addresses().effective_base(),
            &info.manager_addrs,
        ) {
            Ok(ranges) => self
                .file
                .extend(ranges.into_iter().map(|range| FileStorageRange {
                    range,
                    kind: FileStorageKind::Manager,
                })),
            Err(error) => self.problems.push(StorageProblem::Framing {
                kind: FileStorageKind::Manager,
                error,
            }),
        }
    }

    /// Resolves recorded metadata capacity and checks allocation-publication availability.
    fn read_metadata_capacity(&mut self, engine: &WriteEngine) {
        if let Some(persist) = &engine.persist {
            for extent in &persist.old_blocks {
                match Self::resolve(engine, *extent) {
                    Ok(range) => self.metadata_capacity.push(range),
                    Err(error) => self.problems.push(StorageProblem::Address(error)),
                }
            }
        }
        if let Err(error) = engine.context.allocations.available() {
            self.problems.push(StorageProblem::Publication(error));
        }
    }

    /// Records reusable sections, retained PAGE extents, and aggregate spare.
    ///
    /// Traverses allocator records without source reads or page expansion. Records address
    /// conversion failures and continues with the other ranges.
    fn read_non_live(&mut self, engine: &WriteEngine) {
        let page = engine.space.is_paged();
        for (extent, class) in engine.space.admitted_snapshot().sections() {
            if page && class == FreeClass::Dead {
                continue;
            }
            self.record_capacity(engine, extent, CapacityKind::Reusable(class));
        }
        for retained in engine.space.retained_extents() {
            self.record_capacity(
                engine,
                retained.extent(),
                CapacityKind::Retained(retained.kind()),
            );
        }
        if let Some(aggregate) = engine.aggregation {
            for (extent, class) in aggregate.unused_with_class() {
                self.record_capacity(engine, extent, CapacityKind::Aggregate(class));
            }
        }
    }

    /// Records one physical range with its category or an address-conversion problem.
    fn record_capacity(&mut self, engine: &WriteEngine, extent: Extent, kind: CapacityKind) {
        match Self::resolve(engine, extent) {
            Ok(range) => self.non_live.push(CapacityRange { range, kind }),
            Err(error) => self.problems.push(StorageProblem::Address(error)),
        }
    }

    /// Converts one complete stored extent to its checked physical range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if address conversion or the physical end overflows.
    fn resolve(engine: &WriteEngine, extent: Extent) -> Result<FileRange, FormatError> {
        FileRange::new(
            engine.addresses().resolve(extent.start())?,
            extent.len().get(),
        )
    }
}

/// A physical file range classified by its framing or allocation kind.
///
/// Extension and manager ranges are complete allocations. Superblock ranges are parser framing,
/// and userblock ranges are the bytes before the signature. Ownership requires separate evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) struct FileStorageRange {
    pub(in crate::edit) range: FileRange,
    pub(in crate::edit) kind: FileStorageKind,
}

/// The file structure or prefix associated with an observed physical range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum FileStorageKind {
    /// A complete superblock-extension header chunk or continuation allocation.
    Extension,
    /// A complete manager header or section-list allocation, including unused capacity.
    Manager,
    /// The physical bytes required for the superblock parse.
    Superblock,
    /// The bytes before the superblock signature.
    Userblock,
}

/// A recorded allocator extent in physical coordinates with its capacity category.
///
/// Original boundaries and the [`CapacityKind`] association are preserved. Source freshness,
/// allocation ownership, and permission to reuse storage require separate proofs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) struct CapacityRange {
    range: FileRange,
    kind: CapacityKind,
}

impl CapacityRange {
    /// Returns the complete recorded extent in physical coordinates.
    pub(in crate::edit) fn range(self) -> FileRange {
        self.range
    }

    /// Returns the recorded capacity category.
    pub(in crate::edit) fn kind(self) -> CapacityKind {
        self.kind
    }
}

/// The allocator category associated with a recorded capacity range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::edit) enum CapacityKind {
    /// Spare within the metadata or raw-data aggregate.
    Aggregate(AllocationClass),
    /// A retained PAGE extent with its recorded kind.
    Retained(RetainedKind),
    /// An admitted reusable section with its recorded free class.
    Reusable(FreeClass),
}

/// A failed or unsupported local storage observation.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the state retains original observations for failure and qualification diagnostics"
    )
)]
pub(in crate::edit) enum StorageProblem {
    /// A failure in address conversion or range arithmetic.
    Address(FormatError),
    /// A parsed superblock inconsistent with the session's source context.
    Context,
    /// A shared or unsupported extension message, including a repeated File Space Info message.
    ExtensionMessage(MessageType),
    /// A failure in reading or parsing one kind of file storage.
    Framing {
        kind: FileStorageKind,
        error: FormatError,
    },
    /// A legacy free-space or driver-info address without collected allocation geometry.
    LegacyAllocation(StoredAddress),
    /// An unavailable allocation-publication state.
    Publication(Error),
}

#[cfg(test)]
mod tests;
