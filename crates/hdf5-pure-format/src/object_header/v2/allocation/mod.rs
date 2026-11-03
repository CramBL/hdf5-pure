//! Chunk planning and encoding for version 2 object-header allocations.

use alloc::format;
use alloc::vec::Vec;
use core::ops::{Range, RangeInclusive};

use hdf5_pure_core::AddressRange;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::{FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::width::{self, FormatWidths, UintWidth};

use super::worklist;
use super::{CHECKSUM_LEN, OCHK_SIGNATURE, ObjectHeaderPrefix};

/// Holds checked chunk geometry and stored addresses for a version 2 object header.
///
/// The plan borrows the source message records and retains their ranges and the complete chunk
/// geometry. Non-NIL records remain in chunk 0. Additional chunks contain padding, continuation
/// records, or both. The chunks are disjoint and ordered by their continuation links.
///
/// [`Self::encode_chunks`] serializes one complete chunk per iteration. [`Self::encode`]
/// concatenates those encodings in continuation order.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_format::__private::{
///     FormatWidths, ObjectHeaderAllocation, ObjectHeaderPrefix,
/// };
///
/// let address = StoredAddress::new(128);
/// let plan = ObjectHeaderAllocation::new(
///     ObjectHeaderPrefix::PLAIN,
///     FormatWidths::from_sizes(8, 8)?,
///     address,
///     512,
///     &[],
/// )?;
/// assert_eq!(plan.chunks().collect::<Vec<_>>(), [(address, 512)]);
/// assert_eq!(plan.encode().len(), 512);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Debug)]
pub struct ObjectHeaderAllocation<'a> {
    geometry: ObjectHeaderAllocationGeometry<'a>,
    root_width: UintWidth,
    chunks: Vec<Chunk>,
    allocation_len: usize,
}

impl<'a> ObjectHeaderAllocation<'a> {
    /// Plans chunks that cover exactly `allocation_len` bytes beginning at `address`.
    ///
    /// The caller establishes ownership and availability of the allocation and passes a stored,
    /// base-relative address. `messages` contains records in `prefix.layout`, with source
    /// continuations already flattened. Discards source NIL records and trailing bytes shorter
    /// than a record prefix. Retains the other records, including their flags and creation indices.
    ///
    /// Empty input may require a wider Size of Chunk #0 field with a zero-byte body. With
    /// [`ObjectHeaderPrefix::PLAIN`], the three smallest supported allocation lengths are 11, 12,
    /// and 14 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the allocation cannot be materialized
    /// on this host, [`FormatError::OffsetOverflow`] if its exclusive end exceeds [`u64::MAX`],
    /// [`FormatError::AddressTooLarge`] or [`FormatError::DefinedAddressIsUndefined`] if either
    /// endpoint is not a defined address at the output width, and [`FormatError::UnexpectedEof`]
    /// if a message body extends beyond `messages`.
    ///
    /// Returns [`FormatError::Internal`] if `messages` contains a continuation record or the
    /// allocation cannot represent the header, and [`FormatError::NestingDepthExceeded`] if more
    /// than 256 chunks, including chunk 0, are required.
    pub fn new(
        prefix: ObjectHeaderPrefix,
        widths: FormatWidths,
        address: StoredAddress,
        allocation_len: u64,
        messages: &'a [u8],
    ) -> Result<Self, FormatError> {
        let allocation_len =
            ObjectHeaderAllocationGeometry::allocation_len(widths, address, allocation_len)?;
        ObjectHeaderAllocationGeometry::new(prefix, widths, messages)?
            .plan_allocation(address, allocation_len)
    }

    /// Returns the complete chunk encodings concatenated in continuation order.
    ///
    /// The buffer length is the sum of the complete chunk lengths. Use [`Self::encode_chunks`]
    /// to write chunks at their independently assigned addresses.
    ///
    /// Each chunk contains at most one NIL record. Nonempty padding shorter than a record prefix
    /// is a trailing gap. Checksums are emitted regardless of the reader's checksum feature.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.allocation_len);
        for (index, chunk) in self.chunks.iter().enumerate() {
            self.encode_chunk(index, chunk, &mut out);
        }
        assert_eq!(
            out.len(),
            self.allocation_len,
            "header chunks cover the complete allocation"
        );
        out
    }

    /// Returns each stored chunk address with its complete encoding.
    ///
    /// Serializes chunks in continuation order, beginning with chunk 0. Each iteration
    /// materializes one complete chunk, including padding and a checksum. The caller writes each
    /// buffer at its returned stored address.
    pub fn encode_chunks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, Vec<u8>)> + '_ {
        self.chunks.iter().enumerate().map(|(index, chunk)| {
            let mut out = Vec::with_capacity(chunk.len);
            self.encode_chunk(index, chunk, &mut out);
            (chunk.address, out)
        })
    }

    /// Returns chunk addresses and complete lengths in continuation order, beginning with chunk 0.
    ///
    /// Addresses are stored and base-relative. Lengths include signatures, record prefixes, and
    /// checksums.
    pub fn chunks(&self) -> impl ExactSizeIterator<Item = (StoredAddress, u64)> + '_ {
        self.chunks
            .iter()
            .map(|chunk| (chunk.address, chunk.len.to_u64()))
    }

    /// Appends one complete chunk encoding and its checksum to `out`.
    fn encode_chunk(&self, index: usize, chunk: &Chunk, out: &mut Vec<u8>) {
        let start = out.len();
        if index == 0 {
            let fixed = super::OHDR_SIGNATURE.len()
                + 2
                + self.geometry.prefix.optional_len()
                + usize::from(self.root_width.get())
                + CHECKSUM_LEN;
            self.geometry
                .prefix
                .write_prefix(out, self.root_width, chunk.len - fixed);
            for record in &self.geometry.records {
                out.extend_from_slice(&self.geometry.messages[record.clone()]);
            }
        } else {
            out.extend_from_slice(&OCHK_SIGNATURE);
        }
        let trailing_gap =
            chunk.next.is_some() && chunk.padding < self.geometry.prefix.layout.prefix_len();
        if !trailing_gap {
            self.write_padding(out, chunk.padding);
        }
        if let Some((address, len)) = chunk.next {
            self.geometry.prefix.layout.write_record_prefix(
                out,
                MessageType::OBJECT_HEADER_CONTINUATION
                    .to_u16()
                    .to_le_bytes()[0],
                MessageFlags::NONE,
                0,
                u16::from(self.geometry.widths.offsets.get())
                    + u16::from(self.geometry.widths.lengths.get()),
            );
            bytes::write_offset(out, address.get(), self.geometry.widths.offsets);
            bytes::write_length(out, len.to_u64(), self.geometry.widths.lengths);
        }
        if trailing_gap {
            self.write_padding(out, chunk.padding);
        }
        let checksum = checksum::jenkins_lookup3(&out[start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
        assert_eq!(
            out.len() - start,
            chunk.len,
            "a planned header chunk has its exact owned length"
        );
    }

    // H5O__chunk_deserialize in H5Ocache.c, HDF5 2.2.0, rejects gaps after NIL
    // records and coalesces adjacent NIL records on read-write opening within each chunk.
    fn write_padding(&self, out: &mut Vec<u8>, len: usize) {
        let prefix = self.geometry.prefix.layout.prefix_len();
        if len >= prefix {
            let body = len - prefix;
            // Allocation constructors bound every NIL body to its two-byte field.
            let body = u16::try_from(body).expect("a planned NIL body fits its size field");
            self.geometry.prefix.layout.write_record_prefix(
                out,
                MessageType::NIL.to_u16().to_le_bytes()[0],
                MessageFlags::NONE,
                0,
                body,
            );
            out.resize(out.len() + usize::from(body), 0);
        } else {
            out.resize(out.len() + len, 0);
        }
    }
}

/// Holds checked message records for planning version 2 object-header chunks.
///
/// Borrows the source records and retains the ranges of the non-NIL records. Planning keeps
/// those records in chunk 0, in source order, with their flags and creation indices.
/// Use [`Self::contiguous_allocations`] to inspect chunk ranges before consuming the geometry.
#[derive(Debug)]
pub struct ObjectHeaderAllocationGeometry<'a> {
    prefix: ObjectHeaderPrefix,
    widths: FormatWidths,
    messages: &'a [u8],
    records: Vec<Range<usize>>,
    records_len: usize,
}

impl<'a> ObjectHeaderAllocationGeometry<'a> {
    /// Parses the records once for subsequent allocation planning.
    ///
    /// The caller passes records in `prefix.layout`, with source continuations already flattened.
    /// Discards NIL records and trailing bytes shorter than a record prefix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a message body extends beyond `messages`, and
    /// [`FormatError::Internal`] if `messages` contains a continuation record.
    pub fn new(
        prefix: ObjectHeaderPrefix,
        widths: FormatWidths,
        messages: &'a [u8],
    ) -> Result<Self, FormatError> {
        let mut records = Vec::new();
        let mut records_len = 0;
        let mut start = 0;
        while let Some(record) = prefix.layout.next_message(messages, start)? {
            let end = record.body_range.end;
            if record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                return Err(FormatError::Internal(
                    "an object-header allocation requires flattened message records".into(),
                ));
            }
            if record.msg_type != MessageType::NIL {
                records.push(start..end);
                records_len += end - start;
            }
            start = end;
        }
        Ok(Self {
            prefix,
            widths,
            messages,
            records,
            records_len,
        })
    }

    /// Plans contiguous chunks that cover the complete supplied allocation.
    ///
    /// The caller establishes ownership and availability of the allocation and passes a stored,
    /// base-relative `address`. Equivalent to [`ObjectHeaderAllocation::new`] with the records
    /// parsed by [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the allocation cannot be materialized
    /// on this host, [`FormatError::OffsetOverflow`] if its exclusive end exceeds [`u64::MAX`],
    /// and [`FormatError::AddressTooLarge`] or [`FormatError::DefinedAddressIsUndefined`] if either
    /// endpoint is not a defined address at the output width.
    ///
    /// Returns [`FormatError::Internal`] if the allocation cannot represent the header, and
    /// [`FormatError::NestingDepthExceeded`] if more than 256 chunks, including chunk 0, are required.
    pub fn into_allocation(
        self,
        address: StoredAddress,
        allocation_len: u64,
    ) -> Result<ObjectHeaderAllocation<'a>, FormatError> {
        let allocation_len = Self::allocation_len(self.widths, address, allocation_len)?;
        self.plan_allocation(address, allocation_len)
    }

    /// Plans contiguous chunks using the caller's complete chunk lengths.
    ///
    /// `lengths` begins with chunk 0 and includes every signature, prefix, record, and checksum.
    /// Its sum must equal `allocation_len`. The caller establishes ownership and availability of
    /// the allocation and passes a stored, base-relative `address`. All retained records remain
    /// in chunk 0. Additional chunks contain padding, a continuation record, or both.
    ///
    /// Choose lengths from [`Self::root_chunk_lengths`] and [`Self::continuation_chunk_lengths`].
    /// The caller chooses any page alignment or placement policy.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the allocation cannot be materialized
    /// on this host, [`FormatError::OffsetOverflow`] if its exclusive end or the sum of `lengths`
    /// exceeds [`u64::MAX`], and [`FormatError::AddressTooLarge`] or
    /// [`FormatError::DefinedAddressIsUndefined`] if either endpoint is not a defined address at
    /// the output width.
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if `lengths` contains more than 256 chunks,
    /// [`FormatError::ObjectHeaderAllocationLengthMismatch`] if its sum differs from
    /// `allocation_len`, and [`FormatError::InvalidObjectHeaderChunkLength`] if `lengths` is empty
    /// or a chunk cannot represent its records in the supplied length.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{
    ///     FormatWidths, ObjectHeaderAllocationGeometry, ObjectHeaderPrefix,
    /// };
    ///
    /// let geometry = ObjectHeaderAllocationGeometry::new(
    ///     ObjectHeaderPrefix::PLAIN,
    ///     FormatWidths::from_sizes(8, 8)?,
    ///     &[],
    /// )?;
    /// let lengths = [3_968, 128];
    /// assert!(geometry.root_chunk_lengths(true).any(|range| range.contains(&lengths[0])));
    /// assert!(geometry.continuation_chunk_lengths(false).contains(&lengths[1]));
    /// let plan = geometry.into_chunked_allocation(StoredAddress::new(128), 4_096, &lengths)?;
    /// assert_eq!(plan.chunks().collect::<Vec<_>>(), [
    ///     (StoredAddress::new(128), 3_968),
    ///     (StoredAddress::new(4_096), 128),
    /// ]);
    /// assert_eq!(plan.encode().len(), 4_096);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn into_chunked_allocation(
        self,
        mut address: StoredAddress,
        allocation_len: u64,
        lengths: &[u64],
    ) -> Result<ObjectHeaderAllocation<'a>, FormatError> {
        let allocation_len = Self::allocation_len(self.widths, address, allocation_len)?;
        if lengths.len() > worklist::MAX_CONTINUATIONS {
            return Err(FormatError::NestingDepthExceeded);
        }
        let actual = lengths.iter().try_fold(0_u64, |total, &length| {
            total
                .checked_add(length)
                .ok_or(FormatError::OffsetOverflow {
                    offset: total,
                    length,
                })
        })?;
        if actual != allocation_len.to_u64() {
            return Err(FormatError::ObjectHeaderAllocationLengthMismatch {
                expected: allocation_len.to_u64(),
                actual,
            });
        }
        let Some((&root_len, children)) = lengths.split_first() else {
            return Err(FormatError::InvalidObjectHeaderChunkLength {
                index: 0,
                length: 0,
            });
        };
        let root_width = self.root_width(root_len, !children.is_empty())?;
        let mut chunks = Vec::with_capacity(lengths.len());
        for (index, &length) in lengths.iter().enumerate() {
            let next_len = lengths.get(index + 1).copied();
            let next_address = address.checked_add(length)?;
            let next = next_len
                .map(|length| length.to_usize().map(|len| (next_address, len)))
                .transpose()?;
            chunks.push(self.chunk(index, root_width, address, length, next)?);
            address = next_address;
        }
        Ok(ObjectHeaderAllocation {
            geometry: self,
            root_width,
            chunks,
            allocation_len,
        })
    }

    /// Plans complete chunks at the caller's stored allocation ranges.
    ///
    /// `allocations` begins with chunk 0, followed by continuation chunks in link order. The
    /// ranges may be physically separated or in decreasing address order. Each range includes
    /// all signatures, prefixes, records, padding, and checksums for its chunk. Retained records
    /// remain in chunk 0, in source order, with their flags and creation indices.
    ///
    /// Choose lengths from [`Self::root_chunk_lengths`] and [`Self::continuation_chunk_lengths`].
    /// The caller establishes allocation ownership, availability, and any PAGE placement policy.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if more than 256 chunks are supplied, and
    /// [`FormatError::InvalidObjectHeaderChunkLength`] if `allocations` is empty or a chunk
    /// cannot represent its records in the supplied length.
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the combined encoding length cannot
    /// be materialized on this host, [`FormatError::AddressTooLarge`] or
    /// [`FormatError::DefinedAddressIsUndefined`] if an endpoint is not a defined address at the
    /// output width, and [`FormatError::ObjectHeaderChunkOverlap`] if complete chunk ranges
    /// overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::AddressRange;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::FormatWidths;
    /// use hdf5_pure_format::__private::ObjectHeaderAllocationGeometry;
    /// use hdf5_pure_format::__private::ObjectHeaderPrefix;
    ///
    /// let geometry = ObjectHeaderAllocationGeometry::new(
    ///     ObjectHeaderPrefix::PLAIN,
    ///     FormatWidths::from_sizes(8, 8)?,
    ///     &[],
    /// )?;
    /// let allocations = [
    ///     AddressRange::new(StoredAddress::new(128), 64)?,
    ///     AddressRange::new(StoredAddress::new(4_096), 32)?,
    ///     AddressRange::new(StoredAddress::new(2_048), 16)?,
    /// ];
    /// let plan = geometry.into_placed_allocation(allocations.iter().copied())?;
    /// for (allocation, (address, bytes)) in allocations.into_iter().zip(plan.encode_chunks()) {
    ///     assert_eq!(address, allocation.start());
    ///     assert_eq!(bytes.len() as u64, allocation.length());
    /// }
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn into_placed_allocation(
        self,
        allocations: impl ExactSizeIterator<Item = AddressRange<StoredAddress>>,
    ) -> Result<ObjectHeaderAllocation<'a>, FormatError> {
        if allocations.len() > worklist::MAX_CONTINUATIONS {
            return Err(FormatError::NestingDepthExceeded);
        }
        let mut allocations = allocations.peekable();
        let root_len = allocations.peek().map_or(0, |range| range.length());
        let root_width = self.root_width(root_len, allocations.len() > 1)?;
        let mut chunks = Vec::with_capacity(allocations.len());
        let mut allocation_len = 0_usize;
        while let Some(range) = allocations.next() {
            let len = Self::allocation_len(self.widths, range.start(), range.length())?;
            allocation_len = allocation_len
                .checked_add(len)
                .filter(|&len| len <= isize::MAX as usize)
                .ok_or(FormatError::ValueTooLargeForPlatform {
                    value: allocation_len.to_u64().saturating_add(range.length()),
                    target: "object header allocation",
                })?;
            let next = allocations
                .peek()
                .map(|range| range.length().to_usize().map(|len| (range.start(), len)))
                .transpose()?;
            chunks.push(self.chunk(
                chunks.len(),
                root_width,
                range.start(),
                range.length(),
                next,
            )?);
        }
        let mut order = (0..chunks.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&index| (chunks[index].address, index));
        for pair in order.windows(2) {
            let first = pair[0];
            let second = pair[1];
            if chunks[first]
                .address
                .checked_add(chunks[first].len.to_u64())?
                > chunks[second].address
            {
                return Err(FormatError::ObjectHeaderChunkOverlap { first, second });
            }
        }
        Ok(ObjectHeaderAllocation {
            geometry: self,
            root_width,
            chunks,
            allocation_len,
        })
    }

    /// Returns the address and length widths used for chunk planning and encoding.
    pub fn widths(&self) -> FormatWidths {
        self.widths
    }

    /// Returns complete contiguous chunk ranges without consuming the parsed records.
    ///
    /// The ranges cover `allocation_len` bytes beginning at the stored, base-relative `address`.
    /// Uses the same chunk geometry as [`Self::into_allocation`]. The caller establishes
    /// allocation ownership, availability, and any PAGE placement policy.
    ///
    /// # Errors
    ///
    /// Returns the allocation-planning errors described by [`Self::into_allocation`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{
    ///     FormatWidths, ObjectHeaderAllocationGeometry, ObjectHeaderPrefix,
    /// };
    ///
    /// let geometry = ObjectHeaderAllocationGeometry::new(
    ///     ObjectHeaderPrefix::PLAIN,
    ///     FormatWidths::from_sizes(8, 8)?,
    ///     &[],
    /// )?;
    /// let address = StoredAddress::new(128);
    /// let ranges = geometry.contiguous_allocations(address, 512)?;
    /// let plan = geometry.into_allocation(address, 512)?;
    /// assert_eq!(
    ///     ranges.iter().map(|range| (range.start(), range.length())).collect::<Vec<_>>(),
    ///     plan.chunks().collect::<Vec<_>>(),
    /// );
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn contiguous_allocations(
        &self,
        address: StoredAddress,
        allocation_len: u64,
    ) -> Result<Vec<AddressRange<StoredAddress>>, FormatError> {
        let allocation_len = Self::allocation_len(self.widths, address, allocation_len)?;
        let (_, chunks) = self.plan_chunks(address, allocation_len)?;
        chunks
            .into_iter()
            .map(|chunk| AddressRange::new(chunk.address, chunk.len.to_u64()))
            .collect()
    }

    /// Returns the complete chunk 0 length ranges that can represent the retained records.
    ///
    /// With `continued`, includes space for a continuation record. The ranges include the header
    /// prefix, its optional fields, and the checksum. They are ordered by increasing Size of
    /// Chunk #0 field width and may overlap. For empty input without a continuation, they include
    /// single-length ranges for empty chunk bodies.
    pub fn root_chunk_lengths(
        &self,
        continued: bool,
    ) -> impl Iterator<Item = RangeInclusive<u64>> + '_ {
        self.root_geometries(continued).map(|(_, range)| range)
    }

    /// Returns the complete continuation chunk lengths that fit the output length field.
    ///
    /// Includes the signature, checksum, and space for at most one NIL record. With `continued`,
    /// also includes a continuation record. A terminal chunk contains at least one NIL record.
    pub fn continuation_chunk_lengths(&self, continued: bool) -> RangeInclusive<u64> {
        let fixed = (OCHK_SIGNATURE.len() + CHECKSUM_LEN).to_u64();
        let occupied = if continued {
            self.continuation_len()
        } else {
            0
        }
        .to_u64();
        let prefix = self.prefix.layout.prefix_len().to_u64();
        let minimum = fixed + if continued { occupied } else { prefix };
        let maximum = (fixed + occupied + prefix + OBJECT_HEADER_MESSAGE_MAX.to_u64())
            .min(width::length_width_max(self.widths.lengths));
        minimum..=maximum
    }

    fn plan_allocation(
        self,
        address: StoredAddress,
        allocation_len: usize,
    ) -> Result<ObjectHeaderAllocation<'a>, FormatError> {
        let (root_width, chunks) = self.plan_chunks(address, allocation_len)?;
        Ok(ObjectHeaderAllocation {
            geometry: self,
            root_width,
            chunks,
            allocation_len,
        })
    }

    fn plan_chunks(
        &self,
        address: StoredAddress,
        allocation_len: usize,
    ) -> Result<(UintWidth, Vec<Chunk>), FormatError> {
        let prefix = self.prefix;
        let widths = self.widths;
        let records_len = self.records_len;
        for root_width in [
            UintWidth::One,
            UintWidth::Two,
            UintWidth::Four,
            UintWidth::Eight,
        ] {
            let fixed = super::OHDR_SIGNATURE.len()
                + 2
                + prefix.optional_len()
                + usize::from(root_width.get())
                + CHECKSUM_LEN;
            let Some(padding) = allocation_len.checked_sub(fixed + records_len) else {
                continue;
            };
            let first = Chunk::new(prefix, widths, address, fixed + records_len, padding, true)?;
            let root_body = first.len - fixed;
            if (root_body > 0 && root_body < prefix.layout.prefix_len())
                || UintWidth::smallest_for_len(root_body).get() > root_width.get()
            {
                continue;
            }
            let mut chunks = Vec::new();
            let mut chunk = first;
            while let Some(remaining) = chunk.remaining {
                if chunks.len() + 1 >= worklist::MAX_CONTINUATIONS {
                    return Err(FormatError::NestingDepthExceeded);
                }
                let next_address = chunk.address.checked_add(chunk.len.to_u64())?;
                let next = Chunk::new(
                    prefix,
                    widths,
                    next_address,
                    OCHK_SIGNATURE.len() + CHECKSUM_LEN,
                    remaining - OCHK_SIGNATURE.len() - CHECKSUM_LEN,
                    false,
                )?;
                chunk.next = Some((next.address, next.len));
                chunks.push(chunk);
                chunk = next;
            }
            chunks.push(chunk);
            return Ok((root_width, chunks));
        }
        Err(FormatError::Internal(format!(
            "a {allocation_len}-byte allocation cannot represent an object header with {records_len} message bytes"
        )))
    }

    fn root_width(&self, length: u64, continued: bool) -> Result<UintWidth, FormatError> {
        self.root_geometries(continued)
            .find_map(|(width, range)| range.contains(&length).then_some(width))
            .ok_or(FormatError::InvalidObjectHeaderChunkLength { index: 0, length })
    }

    fn chunk(
        &self,
        index: usize,
        root_width: UintWidth,
        address: StoredAddress,
        length: u64,
        next: Option<(StoredAddress, usize)>,
    ) -> Result<Chunk, FormatError> {
        if index != 0
            && !self
                .continuation_chunk_lengths(next.is_some())
                .contains(&length)
        {
            return Err(FormatError::InvalidObjectHeaderChunkLength { index, length });
        }
        let fixed = if index == 0 {
            self.root_fixed(root_width) + self.records_len
        } else {
            OCHK_SIGNATURE.len() + CHECKSUM_LEN
        };
        let len = length.to_usize()?;
        Ok(Chunk {
            address,
            len,
            padding: len
                - fixed
                - if next.is_some() {
                    self.continuation_len()
                } else {
                    0
                },
            remaining: None,
            next,
        })
    }

    fn root_geometries(
        &self,
        continued: bool,
    ) -> impl Iterator<Item = (UintWidth, RangeInclusive<u64>)> + '_ {
        [
            UintWidth::One,
            UintWidth::Two,
            UintWidth::Four,
            UintWidth::Eight,
        ]
        .into_iter()
        .flat_map(move |width| {
            let fixed = self.root_fixed(width).to_u64();
            let occupied = self.records_len.to_u64()
                + if continued {
                    self.continuation_len().to_u64()
                } else {
                    0
                };
            let prefix = self.prefix.layout.prefix_len().to_u64();
            let minimum = fixed + if occupied == 0 { prefix } else { occupied };
            let body_limit = u64::MAX >> (64 - 8 * width.get());
            let maximum =
                fixed + (occupied + prefix + OBJECT_HEADER_MESSAGE_MAX.to_u64()).min(body_limit);
            [
                (occupied == 0).then_some((width, fixed..=fixed)),
                (minimum <= maximum).then_some((width, minimum..=maximum)),
            ]
            .into_iter()
            .flatten()
        })
    }

    fn root_fixed(&self, width: UintWidth) -> usize {
        super::OHDR_SIGNATURE.len()
            + 2
            + self.prefix.optional_len()
            + usize::from(width.get())
            + CHECKSUM_LEN
    }

    fn continuation_len(&self) -> usize {
        self.prefix.layout.prefix_len()
            + usize::from(self.widths.offsets.get())
            + usize::from(self.widths.lengths.get())
    }

    fn allocation_len(
        widths: FormatWidths,
        address: StoredAddress,
        allocation_len: u64,
    ) -> Result<usize, FormatError> {
        if allocation_len > isize::MAX as u64 {
            return Err(FormatError::ValueTooLargeForPlatform {
                value: allocation_len,
                target: "object header allocation",
            });
        }
        let end = address.checked_add(allocation_len)?;
        Self::require_address(widths, address)?;
        Self::require_address(widths, end)?;
        allocation_len.to_usize()
    }

    fn require_address(widths: FormatWidths, address: StoredAddress) -> Result<(), FormatError> {
        if address.get() > width::offset_width_max(widths.offsets) {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: widths.offsets.get(),
            });
        }
        if address.is_undefined(widths.offsets) {
            return Err(FormatError::DefinedAddressIsUndefined {
                address: address.get(),
                offset_size: widths.offsets.get(),
            });
        }
        Ok(())
    }
    /// The maximum number of complete chunks in one allocation, including chunk 0.
    pub const MAX_CHUNKS: usize = worklist::MAX_CONTINUATIONS;
}

/// Complete chunk geometry and its planned continuation target.
#[derive(Debug)]
struct Chunk {
    address: StoredAddress,
    len: usize,
    padding: usize,
    /// The complete length reserved for subsequent chunks, in bytes.
    remaining: Option<usize>,
    /// The address and complete length of the next chunk, once planned.
    next: Option<(StoredAddress, usize)>,
}

impl Chunk {
    fn new(
        prefix: ObjectHeaderPrefix,
        widths: FormatWidths,
        address: StoredAddress,
        fixed: usize,
        available: usize,
        root: bool,
    ) -> Result<Self, FormatError> {
        let maximum = prefix.layout.prefix_len() + OBJECT_HEADER_MESSAGE_MAX;
        let limit = if root {
            usize::MAX
        } else {
            width::length_width_max(widths.lengths)
                .min(usize::MAX as u64)
                .to_usize()?
        };
        if available <= maximum && fixed + available <= limit {
            return Ok(Self {
                address,
                len: fixed + available,
                padding: available,
                remaining: None,
                next: None,
            });
        }
        let continuation = prefix.layout.prefix_len()
            + usize::from(widths.offsets.get())
            + usize::from(widths.lengths.get());
        let minimum = OCHK_SIGNATURE.len() + prefix.layout.prefix_len() + CHECKSUM_LEN;
        let padding = maximum
            .min(limit.saturating_sub(fixed + continuation))
            .min(available - continuation - minimum);
        let len = fixed + padding + continuation;
        let remaining = available - padding - continuation;
        Ok(Self {
            address,
            len,
            padding,
            remaining: Some(remaining),
            next: None,
        })
    }
}

#[cfg(test)]
mod tests;
