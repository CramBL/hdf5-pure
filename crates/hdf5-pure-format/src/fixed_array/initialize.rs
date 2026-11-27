use super::FixedArrayPlacement;
use crate::bytes;
use crate::checksum::JenkinsLookup3;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fixed_array::data_block;
use crate::fixed_array::data_block::DATA_BLOCK_SIGNATURE;
use crate::fixed_array::emit;
use crate::fixed_array::framing::CHECKSUM_LEN;
use crate::fixed_array::framing::VERSION;
use crate::fixed_array::geometry;

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
        let [header, data_block] = self.ranges();
        let geometry = self.plan().geometry();
        let offsets = geometry.encoding().offsets();
        let mut prefix = emit::output_buffer(data_block::data_block_prefix_len(offsets))?;
        prefix.extend_from_slice(&DATA_BLOCK_SIGNATURE);
        prefix.push(VERSION);
        prefix.push(geometry.encoding().client().id());
        bytes::try_write_defined_offset(&mut prefix, header.start().get(), offsets)?;
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
            Some(paging) => emit::emit_repeated(
                &mut write,
                Some(&mut checksum),
                &[0],
                paging.pages().div_ceil(u64::from(u8::BITS)),
            )?,
            None => {
                let element = geometry.encoding().encode(None)?;
                emit::emit_repeated(
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
        emit::emit_repeated(&mut write, None, &[0], tail)
    }
}
