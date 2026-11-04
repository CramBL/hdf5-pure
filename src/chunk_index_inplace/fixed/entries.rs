//! Entry-window patches, checked checksums, and batched page initialization references.

use core::iter::Peekable;
use core::num::NonZeroU8;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FixedArrayDataBlockRef;
use hdf5_pure_format::__private::FixedArrayEntryWindow;
use hdf5_pure_format::__private::FixedArrayPageRef;
use hdf5_pure_format::__private::JenkinsLookup3;

use crate::convert::Narrow;
use crate::edit::retirement::ChunkAllocation;
use crate::error::Error;
use crate::error::FormatError;

use super::super::FiniteGeometry;
use super::super::RetiredChunk;
use super::super::Store;
use super::super::metadata;

/// A retained source window or an uninitialized page, with predecessor allocation facts.
pub(super) struct EntryPatch {
    first: u64,
    window: FixedArrayEntryWindow,
    target: Target,
    original: Vec<u8>,
    predecessors: Vec<ChunkAllocation>,
}

impl EntryPatch {
    /// Publishes new page bits after one shared content barrier, followed by a reference barrier.
    ///
    /// Computes original and replacement bitmap checksums in one bounded pass, then writes the
    /// affected bitmap bytes and one checksum. The caller emits all page bodies first.
    ///
    /// # Errors
    ///
    /// Propagates stale-bitmap, checksum, range, publication, and synchronization errors.
    pub(super) fn publish_initializations(
        file: &mut impl Store,
        patches: &[Self],
    ) -> Result<(), Error> {
        let bitmaps = patches.iter().filter_map(|patch| match patch.target {
            Target::Initialize { bitmap, .. } => Some(bitmap),
            _ => None,
        });
        let Some(first) = bitmaps.clone().next() else {
            return Ok(());
        };
        file.sync()?;
        first.prefix.patch_bits(file, bitmaps)?;
        file.sync()
    }

    /// Reads affected entry windows and captures existing predecessor ranges.
    ///
    /// Verifies checksummed bodies and retains original bytes only for initialized windows.
    /// The caller establishes complete source ownership before using captured retirements.
    ///
    /// # Errors
    ///
    /// Propagates source, range, checksum, entry, and allocation-capture errors.
    pub(super) fn read(
        file: &impl Store,
        block: FixedArrayDataBlockRef,
        geometry: &FiniteGeometry,
        first: u64,
        end: u64,
    ) -> Result<(Vec<Self>, Option<ChunkRecord>), Error> {
        metadata::require_range(file, block.range())?;
        let prefix = block.prefix_range()?;
        let bytes = file.read_metadata_at(prefix.start().get(), prefix.length().to_usize()?)?;
        block.parse_prefix(FileRegion::new(prefix, &bytes)?)?;
        let checksum =
            ChecksumTarget::read(file, block.checksummed_range()?, block.checksum_range()?)?;
        let unpaged = block
            .unpaged_prefix_range()?
            .map(|_| block.parse_unpaged_prefix(FileRegion::new(prefix, &bytes)?))
            .transpose()?;
        let mut patches = Vec::new();
        let mut slot = first;
        let mut tail = None;
        while slot < end {
            let maximum = NonZeroUsize::new((end - slot).to_usize()?)
                .ok_or(FormatError::InvalidFixedArrayGeometry)?;
            let (window, target, initialized) = match unpaged {
                Some(unpaged) => (
                    unpaged
                        .window(slot, maximum)?
                        .ok_or(FormatError::InvalidFixedArrayGeometry)?,
                    Target::Existing {
                        checksum,
                        bitmap: None,
                    },
                    true,
                ),
                None => {
                    let page = block
                        .page_for_slot(slot)?
                        .ok_or(FormatError::InvalidFixedArrayGeometry)?;
                    metadata::require_range(file, page.parse_range())?;
                    let (bitmap, mask) = block
                        .page_bitmap_byte(slot)?
                        .ok_or(FormatError::InvalidFixedArrayGeometry)?;
                    let mut value = [0];
                    file.read_at(bitmap.start().get(), &mut value)?;
                    let initialized = value[0] & mask.get() != 0;
                    let bitmap = PageBitmap {
                        prefix: checksum,
                        field: bitmap,
                        mask,
                    };
                    let target = if initialized {
                        Target::Existing {
                            checksum: ChecksumTarget::read(
                                file,
                                page.checksummed_range()?,
                                page.checksum_range()?,
                            )?,
                            bitmap: Some(bitmap),
                        }
                    } else {
                        Target::Initialize { page, bitmap }
                    };
                    (
                        page.window(slot, maximum)?
                            .ok_or(FormatError::InvalidFixedArrayGeometry)?,
                        target,
                        initialized,
                    )
                }
            };
            let mut predecessors = Vec::new();
            let original = if initialized {
                let range = window.parse_range();
                let bytes = file.read_exact_at(range.start().get(), range.length().to_usize()?)?;
                for entry in window
                    .parse(FileRegion::new(range, &bytes)?)?
                    .records(geometry.chunk_bytes.to_u64())
                {
                    let (entry, record) = entry?;
                    if entry == first {
                        tail = Some(record);
                    }
                    let allocation = ChunkAllocation::capture(
                        file.addresses(),
                        &super::chunk_info(record, entry, geometry)?,
                    )?;
                    metadata::require_range(file, allocation.range())?;
                    predecessors.push(allocation);
                }
                bytes
            } else {
                Vec::new()
            };
            patches.push(Self {
                first: slot,
                window,
                target,
                original,
                predecessors,
            });
            slot = window.end_slot();
        }
        Ok((patches, tail))
    }

    /// Checks the retained checksum, bitmap state, and original window bytes before mutation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for changed metadata. Propagates source and range
    /// errors.
    pub(super) fn require_current(&self, file: &impl Store) -> Result<(), Error> {
        match self.target {
            Target::Existing { checksum, bitmap } => {
                if let Some(bitmap) = bitmap {
                    bitmap.require_current(file, true)?;
                }
                checksum.require_current(file)?;
                let range = self.window.parse_range();
                let mut scratch = [0; IO_WINDOW_BYTES];
                let mut offset = 0;
                while offset < self.original.len() {
                    let length = (self.original.len() - offset).min(scratch.len());
                    file.read_at(
                        range.start().checked_add(offset.to_u64())?.get(),
                        &mut scratch[..length],
                    )?;
                    if scratch[..length] != self.original[offset..offset + length] {
                        return Err(Error::AppendUnsupported(
                            "fixed-array entries changed after append planning",
                        ));
                    }
                    offset += length;
                }
            }
            Target::Initialize { bitmap, .. } => bitmap.require_current(file, false)?,
        }
        Ok(())
    }

    /// Writes a checked entry patch or complete page and retires confirmed predecessors.
    ///
    /// Existing entries are synchronized before retirement. New pages have no captured
    /// predecessors, and their initialization bits are published by
    /// [`Self::publish_initializations`].
    ///
    /// # Errors
    ///
    /// Propagates encoding, source, checksum, publication, and synchronization errors.
    pub(super) fn publish(
        &mut self,
        file: &mut impl Store,
        records: &[ChunkRecord],
    ) -> Result<(), Error> {
        let Self {
            first: _,
            window,
            target,
            original,
            predecessors,
        } = self;
        let records = records.iter().copied().map(Some).collect::<Vec<_>>();
        match target {
            Target::Existing { checksum, .. } => {
                window.encode_records(original, &records)?;
                checksum.patch(file, window.parse_range(), original)?;
                file.sync()?;
            }
            Target::Initialize { page, .. } => {
                let mut position = page.parse_range().start();
                page.emit_records(window, &records, |bytes| {
                    file.write_at(position.get(), bytes)?;
                    position = position.checked_add(bytes.len().to_u64())?;
                    Ok::<(), Error>(())
                })?;
                if position != page.parse_range().end() {
                    return Err(FormatError::Internal(
                        "fixed-array page emission differs from its range".into(),
                    )
                    .into());
                }
            }
        }
        for allocation in predecessors.drain(..) {
            file.retire(RetiredChunk { allocation });
        }
        Ok(())
    }

    /// Returns the first logical slot covered by the patch.
    pub(super) fn first_slot(&self) -> u64 {
        self.first
    }

    /// Returns the entry count as a bounded host length.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the count exceeds [`usize::MAX`].
    pub(super) fn entries(&self) -> Result<usize, FormatError> {
        (self.window.end_slot() - self.first).to_usize()
    }
}

#[derive(Clone, Copy)]
/// Existing checked bytes or a page body whose initialization bit is clear.
enum Target {
    Existing {
        checksum: ChecksumTarget,
        bitmap: Option<PageBitmap>,
    },
    Initialize {
        page: FixedArrayPageRef,
        bitmap: PageBitmap,
    },
}

#[derive(Clone, Copy)]
/// One page bit within a shared checksummed data-block prefix.
struct PageBitmap {
    prefix: ChecksumTarget,
    field: FileRange,
    mask: NonZeroU8,
}

impl PageBitmap {
    fn apply(
        pending: &mut Peekable<impl Iterator<Item = Self>>,
        position: PhysicalAddress,
        bytes: &mut [u8],
    ) -> Result<(), Error> {
        let end = position.checked_add(bytes.len().to_u64())?;
        while let Some(bitmap) = pending.peek() {
            if bitmap.field.start() >= end {
                break;
            }
            let offset = bitmap
                .field
                .start()
                .get()
                .checked_sub(position.get())
                .ok_or(FormatError::InvalidFixedArrayGeometry)?
                .to_usize()?;
            let value = bytes
                .get_mut(offset)
                .ok_or(FormatError::InvalidFixedArrayGeometry)?;
            if *value & bitmap.mask.get() != 0 {
                return Err(Error::AppendUnsupported(
                    "fixed-array page was initialized before its publication",
                ));
            }
            *value |= bitmap.mask.get();
            pending.next();
        }
        Ok(())
    }

    fn require_current(&self, file: &impl Store, initialized: bool) -> Result<(), Error> {
        self.prefix.require_current(file)?;
        let mut value = [0];
        file.read_at(self.field.start().get(), &mut value)?;
        if (value[0] & self.mask.get() != 0) != initialized {
            return Err(Error::AppendUnsupported(
                "fixed-array page initialization changed after append planning",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
/// Checked body and checksum ranges with the checksum observed during planning.
struct ChecksumTarget {
    body: FileRange,
    checksum: FileRange,
    expected: u32,
}

impl ChecksumTarget {
    fn read(file: &impl Store, body: FileRange, checksum: FileRange) -> Result<Self, Error> {
        if body.end() != checksum.start()
            || checksum.length() != core::mem::size_of::<u32>().to_u64()
        {
            return Err(FormatError::InvalidFixedArrayGeometry.into());
        }
        let range = FileRange::new(
            body.start(),
            body.length()
                .checked_add(checksum.length())
                .ok_or(FormatError::InvalidFixedArrayGeometry)?,
        )?;
        metadata::verify_range(file, range)?;
        let mut bytes = [0; core::mem::size_of::<u32>()];
        file.read_at(checksum.start().get(), &mut bytes)?;
        Ok(Self {
            body,
            checksum,
            expected: u32::from_le_bytes(bytes),
        })
    }

    fn require_current(&self, file: &impl Store) -> Result<(), Error> {
        let mut bytes = [0; core::mem::size_of::<u32>()];
        file.read_at(self.checksum.start().get(), &mut bytes)?;
        if u32::from_le_bytes(bytes) != self.expected {
            return Err(Error::AppendUnsupported(
                "fixed-array checksum changed after append planning",
            ));
        }
        Ok(())
    }

    /// Checks and updates ordered page bits with one checksum pass and bounded write windows.
    ///
    /// Verifies the original checksum before changing bytes. Bits may share a bitmap byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] for incompatible or unordered bitmap
    /// locations, [`Error::AppendUnsupported`] for stale state, or
    /// [`FormatError::ChecksumMismatch`] for corrupted source bytes. Propagates I/O and range
    /// errors.
    fn patch_bits(
        &self,
        file: &mut impl Store,
        bitmaps: impl Iterator<Item = PageBitmap> + Clone,
    ) -> Result<(), Error> {
        let mut first = None;
        let mut last = None;
        for bitmap in bitmaps.clone() {
            if bitmap.prefix.body != self.body
                || bitmap.prefix.checksum != self.checksum
                || bitmap.prefix.expected != self.expected
                || bitmap.field.start() < self.body.start()
                || bitmap.field.end() > self.body.end()
                || bitmap.field.length() != 1
                || last.is_some_and(|last| bitmap.field.start() < last)
            {
                return Err(FormatError::InvalidFixedArrayGeometry.into());
            }
            first.get_or_insert(bitmap.field.start());
            last = Some(bitmap.field.start());
        }
        let (Some(first), Some(last)) = (first, last) else {
            return Ok(());
        };
        self.require_current(file)?;
        let mut original = JenkinsLookup3::new(self.body.length());
        let mut replacement = JenkinsLookup3::new(self.body.length());
        let mut scratch = [0; IO_WINDOW_BYTES];
        let mut position = self.body.start();
        let mut pending = bitmaps.clone().peekable();
        while position < self.body.end() {
            let length = (self.body.end().get() - position.get())
                .min(scratch.len().to_u64())
                .to_usize()?;
            file.read_at(position.get(), &mut scratch[..length])?;
            original.update(&scratch[..length])?;
            PageBitmap::apply(&mut pending, position, &mut scratch[..length])?;
            replacement.update(&scratch[..length])?;
            position = position.checked_add(length.to_u64())?;
        }
        let computed = original.finish()?;
        if computed != self.expected {
            return Err(FormatError::ChecksumMismatch {
                expected: self.expected,
                computed,
            }
            .into());
        }
        let checksum = replacement.finish()?;
        let end = last.checked_add(1)?;
        let mut position = first;
        let mut pending = bitmaps.peekable();
        while position < end {
            let length = (end.get() - position.get())
                .min(scratch.len().to_u64())
                .to_usize()?;
            file.read_at(position.get(), &mut scratch[..length])?;
            PageBitmap::apply(&mut pending, position, &mut scratch[..length])?;
            file.write_at(position.get(), &scratch[..length])?;
            position = position.checked_add(length.to_u64())?;
        }
        file.write_at(self.checksum.start().get(), &checksum.to_le_bytes())
    }

    /// Checks the original body and publishes replacement field bytes and their checksum.
    ///
    /// Uses bounded read windows, including when the checksummed body exceeds the field range.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for an invalid field range or changed checksum, or
    /// [`FormatError::ChecksumMismatch`] for corrupted source bytes. Propagates I/O and range
    /// errors.
    fn patch(&self, file: &mut impl Store, field: FileRange, bytes: &[u8]) -> Result<(), Error> {
        if field.start() < self.body.start()
            || field.end() > self.body.end()
            || field.length() != bytes.len().to_u64()
        {
            return Err(Error::AppendUnsupported(
                "a fixed-array field lies outside its checksummed body",
            ));
        }
        self.require_current(file)?;
        let mut original = JenkinsLookup3::new(self.body.length());
        let mut replacement = JenkinsLookup3::new(self.body.length());
        let mut scratch = [0; IO_WINDOW_BYTES];
        let mut position = self.body.start();
        while position < self.body.end() {
            let length = (self.body.end().get() - position.get())
                .min(scratch.len().to_u64())
                .to_usize()?;
            let end = position.checked_add(length.to_u64())?;
            file.read_at(position.get(), &mut scratch[..length])?;
            original.update(&scratch[..length])?;
            let start = position.max(field.start());
            let stop = end.min(field.end());
            if start < stop {
                let from = (start.get() - position.get()).to_usize()?;
                let source = (start.get() - field.start().get()).to_usize()?;
                let count = (stop.get() - start.get()).to_usize()?;
                scratch
                    .get_mut(from..from + count)
                    .ok_or(FormatError::InvalidFixedArrayGeometry)?
                    .copy_from_slice(
                        bytes
                            .get(source..source + count)
                            .ok_or(FormatError::InvalidFixedArrayGeometry)?,
                    );
            }
            replacement.update(&scratch[..length])?;
            position = end;
        }
        let computed = original.finish()?;
        if computed != self.expected {
            return Err(FormatError::ChecksumMismatch {
                expected: self.expected,
                computed,
            }
            .into());
        }
        let checksum = replacement.finish()?;
        file.write_at(field.start().get(), bytes)?;
        file.write_at(self.checksum.start().get(), &checksum.to_le_bytes())
    }
}

const IO_WINDOW_BYTES: usize = 4_096;
