//! Checked entry updates and bounded data-block and page emission for version 0 Fixed Arrays.

use core::num::NonZeroUsize;

use super::FixedArrayPlacement;
use super::FixedArrayPlan;
use super::data_block;
use super::data_block::DATA_BLOCK_SIGNATURE;
use super::framing::CHECKSUM_LEN;
use super::framing::VERSION;
use super::geometry;
use super::geometry::Geometry;
use super::header;
use crate::bytes;
use crate::checksum::JenkinsLookup3;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::ElementEncoding;
use crate::convert::Narrow;
use crate::error::FormatError;

impl FixedArrayPlan {
    /// Selects the page threshold as a base 2 exponent and recomputes complete block lengths.
    ///
    /// Paging starts when the entry count exceeds `2^page_bits`. See [`Self::encode_record`] for
    /// an example combining custom paging with independent field widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if `page_bits` is 64 or more, or a block
    /// length or their sum overflows.
    pub fn with_page_bits(mut self, page_bits: u8) -> Result<Self, FormatError> {
        self.geometry =
            Geometry::new(self.geometry.encoding(), self.geometry.entries(), page_bits)?;
        self.page_bits = page_bits;
        self.data_block_length = self.geometry.data_block_len()?;
        self.total_length =
            geometry::sum([header::header_len(self.widths), self.data_block_length])?;
        Ok(self)
    }

    /// Encodes one entry at the plan's retained address and filtered-size widths.
    ///
    /// Writes `None` as an undefined address with zero filtered fields. Returns the encoded byte
    /// count, preserving trailing bytes and all output on error. Filtered sizes use the entry
    /// encoding, independently of the file's length width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is too short. Returns the address and
    /// size errors of [`super::FixedArrayEntryWindow::encode_records`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let plan = format::FixedArrayPlan::new(
    ///     9, 16, format::FormatWidths::from_sizes(2, 4)?, true,
    /// )?.with_page_bits(3)?;
    /// assert_eq!(plan.page_bits(), 3);
    /// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [18, 93]);
    /// let mut output = [0xcc; 10];
    /// let length = plan.encode_record(&mut output, Some(format::ChunkRecord {
    ///     address: StoredAddress::new(0x1234), stored_size: 0x56, filter_mask: 3,
    /// }))?;
    /// assert_eq!(length, 8);
    /// assert_eq!(output, [0x34, 0x12, 0x56, 0, 3, 0, 0, 0, 0xcc, 0xcc]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn encode_record(
        &self,
        output: &mut [u8],
        record: Option<ChunkRecord>,
    ) -> Result<usize, FormatError> {
        let element = self.geometry.encoding().encode(record)?;
        let bytes = element.as_bytes();
        bytes::ensure_len(output, 0, bytes.len())?;
        output[..bytes.len()].copy_from_slice(bytes);
        Ok(bytes.len())
    }

    /// Returns the base 2 logarithm of the maximum entries per page.
    pub const fn page_bits(&self) -> u8 {
        self.page_bits
    }
}

impl FixedArrayPlacement {
    /// Writes an empty data block to successive sink windows.
    ///
    /// Writes undefined entries for an unpaged block. For a paged block, writes a zero
    /// initialization bitmap and zero bytes across the reserved page ranges. The caller initializes
    /// a page before setting its bitmap bit. Computes the prefix or whole-block checksum
    /// regardless of the `checksum` feature.
    ///
    /// Uses a 4 KiB stack buffer and a bounded prefix allocation, independently of the entry
    /// count. The caller controls retention of the emitted bytes and publication of references.
    /// The caller obtains the header bytes from [`Self::serialize_header`].
    ///
    /// # Errors
    ///
    /// Returns a sink error unchanged, including errors after partial output. Converts
    /// [`FormatError::AllocationFailed`] if the prefix reservation fails. Geometry or length
    /// failures are reported through [`FormatError`] before the affected sink window.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let plan = format::FixedArrayPlan::new(
    ///     3, 16, format::FormatWidths::from_sizes(2, 4)?, false,
    /// )?;
    /// let expected = plan.block_lengths().nth(1).unwrap();
    /// let placed = plan.place(StoredAddress::new(64), StoredAddress::new(256))?;
    /// let mut bytes_written = 0_u64;
    /// placed.emit_empty_data_block::<FormatError>(|bytes| {
    ///     bytes_written += bytes.len() as u64;
    ///     Ok(())
    /// })?;
    /// assert_eq!(bytes_written, expected);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn emit_empty_data_block<E: From<FormatError>>(
        &self,
        mut write: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        let Self {
            plan,
            header,
            data_block,
        } = *self;
        let FixedArrayPlan {
            widths,
            geometry,
            encoding: _,
            entry_size: _,
            page_bits: _,
            data_block_length: _,
            total_length: _,
        } = plan;
        let mut prefix = super::output_buffer(data_block::data_block_prefix_len(widths.offsets))?;
        prefix.extend_from_slice(&DATA_BLOCK_SIGNATURE);
        prefix.push(VERSION);
        prefix.push(geometry.encoding().client().id());
        bytes::try_write_defined_offset(&mut prefix, header.start().get(), widths.offsets)?;
        let body_length = match geometry.paging() {
            Some(paging) => geometry::sum([
                prefix.len().to_u64(),
                paging.pages().div_ceil(u64::from(u8::BITS)),
            ])?,
            None => data_block
                .length()
                .checked_sub(CHECKSUM_LEN)
                .ok_or(FormatError::InvalidFixedArrayGeometry)?,
        };
        let mut checksum = JenkinsLookup3::new(body_length);
        checksum.update(&prefix)?;
        write(&prefix)?;
        match geometry.paging() {
            Some(paging) => emit_repeated(
                &mut write,
                Some(&mut checksum),
                &[0],
                paging.pages().div_ceil(u64::from(u8::BITS)),
            )?,
            None => {
                let element = geometry.encoding().encode(None)?;
                emit_repeated(
                    &mut write,
                    Some(&mut checksum),
                    element.as_bytes(),
                    geometry::product(
                        geometry.entries(),
                        u64::from(geometry.encoding().stride().get()),
                    )?,
                )?;
            }
        }
        write(&checksum.finish()?.to_le_bytes())?;
        let tail = data_block
            .length()
            .checked_sub(geometry::sum([body_length, CHECKSUM_LEN])?)
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        emit_repeated(&mut write, None, &[0], tail)
    }
}

pub(super) fn emit_records<E: From<FormatError>>(
    write: &mut impl FnMut(&[u8]) -> Result<(), E>,
    checksum: &mut JenkinsLookup3,
    encoding: ElementEncoding,
    records: &[Option<ChunkRecord>],
) -> Result<(), E> {
    let stride = usize::from(encoding.stride().get());
    let mut scratch = [0; EMISSION_SCRATCH_LEN];
    let maximum = scratch.len() / stride;
    if maximum == 0 {
        return Err(FormatError::InvalidFixedArrayGeometry.into());
    }
    for records in records.chunks(maximum) {
        let length = records.len() * stride;
        for (bytes, &record) in scratch[..length].chunks_exact_mut(stride).zip(records) {
            bytes.copy_from_slice(encoding.encode(record)?.as_bytes());
        }
        checksum.update(&scratch[..length])?;
        write(&scratch[..length])?;
    }
    Ok(())
}

pub(super) fn emit_repeated<E: From<FormatError>>(
    write: &mut impl FnMut(&[u8]) -> Result<(), E>,
    mut checksum: Option<&mut JenkinsLookup3>,
    pattern: &[u8],
    mut length: u64,
) -> Result<(), E> {
    let Some(stride) = NonZeroUsize::new(pattern.len()) else {
        return Err(FormatError::Internal("an emitted byte pattern is empty".into()).into());
    };
    let mut scratch = [0; EMISSION_SCRATCH_LEN];
    let used = scratch.len() / stride.get() * stride.get();
    if used == 0 || length % stride.get().to_u64() != 0 {
        return Err(FormatError::InvalidFixedArrayGeometry.into());
    }
    for element in scratch[..used].chunks_exact_mut(stride.get()) {
        element.copy_from_slice(pattern);
    }
    while length != 0 {
        let count = length.min(used.to_u64()).to_usize()?;
        let bytes = &scratch[..count];
        if let Some(checksum) = checksum.as_mut() {
            checksum.update(bytes)?;
        }
        write(bytes)?;
        length -= count.to_u64();
    }
    Ok(())
}

const EMISSION_SCRATCH_LEN: usize = 4_096;
