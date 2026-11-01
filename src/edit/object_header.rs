//! Object-header traversal and reconstruction for staged edits.

use std::collections::VecDeque;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::LayoutVersion;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::OBJECT_HEADER_PREFIX_MAX_LEN;
use hdf5_pure_format::__private::ObjectHeaderAllocation;
use hdf5_pure_format::__private::ObjectHeaderChunk;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::RecordSplice;
use hdf5_pure_space::__private::Extent;

use crate::address::AddressContext;
use crate::convert::Narrow;
use crate::data_layout::{COMPACT_DATA_OFFSET, DataLayout, LAYOUT_CLASS_COMPACT};
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_space_info::FileSpaceInfo;
use crate::file_writer::{LENGTH_SIZE, OFFSET_SIZE};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::Source;
use crate::width::FormatWidths;

use super::WriteEngine;

impl WriteEngine {
    /// Reads the extension header and replaces its File Space Info message bodies.
    ///
    /// `ext_addr` is an absolute file offset. The replacement is a persistent version 1 message
    /// encoded at the file's widths.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the header cannot be traversed or has no File
    /// Space Info message. Propagates parsing, source-read, and encoding errors.
    pub(super) fn rewrite_extension_region(
        &self,
        ext_addr: u64,
        info: &FileSpaceInfo,
    ) -> Result<OhRegion, Error> {
        let widths = self.context.widths();
        let region = Self::gather_oh_messages(&self.image(), ext_addr, self.context.addresses())?;
        rewrite_extension_region_bytes(&region, widths, info)
    }

    /// Collects a version 2 header's records into one region for reconstruction.
    ///
    /// Reads each chunk into a bounded buffer and omits continuation records and trailing gaps.
    /// Preserves the chunk 0 prefix properties for [`build_v2_object_header`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the header cannot be traversed. Propagates
    /// parsing and source-read errors from [`visit_oh_chunks`].
    pub(super) fn gather_oh_messages<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        base: AddressContext,
    ) -> Result<OhRegion, Error> {
        gather_records(src, addr, base, HeaderRecords::All)
    }
}

/// Occupied extension records retained under an immutable source borrow.
///
/// NIL and continuation records are omitted. Record flags, creation indexes, and chunk 0 prefix
/// properties are preserved. The caller borrows the source immutably while sizing and encoding
/// replacement File Space Info bodies.
pub(crate) struct ExtensionReconstruction<'a, S: Source + ?Sized> {
    _source: &'a S,
    region: OhRegion,
}

impl<'a, S: Source + ?Sized> ExtensionReconstruction<'a, S> {
    /// Reads occupied version 2 extension records from the physical address `addr`.
    ///
    /// Reads each chunk into a temporary buffer and retains its occupied records in traversal
    /// order. Continuation addresses are resolved through `base`.
    ///
    /// # Errors
    ///
    /// Returns the traversal errors from [`visit_oh_chunks`].
    pub(crate) fn read(src: &'a S, addr: u64, base: AddressContext) -> Result<Self, Error> {
        Ok(Self {
            _source: src,
            region: gather_records(src, addr, base, HeaderRecords::Occupied)?,
        })
    }

    /// Replaces every retained File Space Info body without reading the source.
    ///
    /// # Errors
    ///
    /// Returns the encoding and missing-message errors from [`rewrite_extension_region_bytes`].
    pub(crate) fn rewrite(
        &self,
        widths: FormatWidths,
        info: &FileSpaceInfo,
    ) -> Result<OhRegion, Error> {
        rewrite_extension_region_bytes(&self.region, widths, info)
    }
}

fn gather_records<S: Source + ?Sized>(
    src: &S,
    addr: u64,
    base: AddressContext,
    records: HeaderRecords,
) -> Result<OhRegion, Error> {
    let mut out = None;
    visit_oh_chunks(src, addr, base, |chunk| {
        let out = out.get_or_insert_with(|| OhRegion::empty(chunk.props()));
        let layout = chunk.layout();
        let (region, mut p) = chunk.message_region();
        while let Some((msg_type, _body, body_end)) = next_record(layout, region, p)? {
            if msg_type != MessageType::OBJECT_HEADER_CONTINUATION
                && (records == HeaderRecords::All || msg_type != MessageType::NIL)
            {
                out.push_bytes(&region[p..body_end]);
            }
            p = body_end;
        }
        Ok(())
    })?;
    out.ok_or_else(|| FormatError::Internal("object header has no root chunk".into()).into())
}

#[derive(Clone, Copy, PartialEq)]
enum HeaderRecords {
    All,
    Occupied,
}

/// Returns a copy of `region` whose first Data Layout message holds `raw` as its compact data.
///
/// The rewritten record keeps its flags, its creation index, and its layout version. Every
/// other record is copied unchanged, a later Data Layout message included. The compact fields
/// of layout versions 3 and 4 are defined in "The Data Layout Message" of the [format
/// specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if `region` lacks a Data Layout message, if the body of a
/// record it reads runs past the end of `region`, if the byte after the version of the first
/// Data Layout message, the layout class in versions 3 and 4, is not the compact class, or if
/// `raw` is longer than [`OBJECT_HEADER_MESSAGE_MAX`] less [`COMPACT_DATA_OFFSET`]. Returns
/// [`Error::Format`] if that byte is the compact class and the version is neither 3 nor 4.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
pub(super) fn rebuild_compact_layout_region(
    region: &OhRegion,
    raw: &[u8],
) -> Result<OhRegion, Error> {
    // This guard applies the bound of `encode_compact` to the whole message body
    // the layout becomes (version, class, the 2-byte inline size and the data),
    // and rejects a body that does not fit with `EditUnsupported`.
    if raw.len() > OBJECT_HEADER_MESSAGE_MAX - COMPACT_DATA_OFFSET {
        return Err(Error::EditUnsupported(
            "compact dataset data is too large to overwrite in place",
        ));
    }
    // libhdf5 reads the first Data Layout message of a header (`H5O_msg_read_oh`
    // in `H5Omessage.c`, HDF5 2.2.0).
    let splice = region
        .splice_first(MessageType::DATA_LAYOUT)?
        .ok_or(Error::EditUnsupported(
            "compact dataset header has no data-layout message",
        ))?;
    let &[version, LAYOUT_CLASS_COMPACT, ..] = splice.record().body else {
        return Err(Error::EditUnsupported(
            "compact-layout overwrite found a non-compact data layout",
        ));
    };
    let layout = DataLayout::encode_compact(LayoutVersion::try_from(version)?, raw)?;
    Ok(region.with_bytes(splice.replace_body(&layout)?))
}

/// Replaces every File Space Info body with a persistent version 1 encoding of `info`.
///
/// Preserves the original record flags, creation indexes, prefix properties, and unrelated
/// bytes. Both commit paths use this operation to prepare the extension region.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or no File Space Info
/// message is present. Returns [`Error::Format`] if `info` cannot be encoded.
pub(crate) fn rewrite_extension_region_bytes(
    region: &OhRegion,
    widths: FormatWidths,
    info: &FileSpaceInfo,
) -> Result<OhRegion, Error> {
    let new_body = hdf5_pure_format::__private::serialize_file_space_info_v1(widths, info)?;
    HeaderRewrite::parse(region)?
        .replace_bodies(MessageType::FILE_SPACE_INFO, &new_body)?
        .ok_or(Error::EditUnsupported(
            "a persisting file's superblock extension has no File Space Info message",
        ))
}

/// Extension chunk geometry collected under the editor's record and traversal rules.
///
/// Retains the first error while the source parser completes the header. The caller must finish
/// that parse before calling [`finish`](Self::finish). Recorded ranges use physical coordinates.
pub(super) struct ExtensionAllocations {
    ranges: ExtensionRanges,
    count: usize,
    error: Option<Error>,
}

impl ExtensionAllocations {
    /// Creates an empty collector with the root counted toward the traversal budget.
    pub(super) fn new() -> Self {
        Self {
            ranges: ExtensionRanges::Empty,
            count: 1,
            error: None,
        }
    }

    /// Records a chunk's complete physical range or the first geometry error.
    ///
    /// Later observations are ignored after an error. Stored continuation addresses are resolved
    /// through `base`, with their complete allocations bounded by `file_len`.
    pub(super) fn observe(
        &mut self,
        chunk: ObjectHeaderChunk<'_>,
        base: AddressContext,
        file_len: u64,
    ) {
        if self.error.is_none() {
            self.error = self.ingest(chunk, base, file_len).err();
        }
    }

    /// Materializes the observed ranges for a completed persistent consumer.
    ///
    /// # Errors
    ///
    /// Returns the first error recorded by [`observe`](Self::observe).
    pub(super) fn finish(self) -> Result<Vec<FileRange>, Error> {
        if let Some(error) = self.error {
            return Err(error);
        }
        Ok(self.ranges.to_vec())
    }

    /// Checks one observed chunk and appends its complete range on success.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for a version 1 header or an exhausted chunk budget.
    /// Returns errors from [`next_record`], [`ContinuationChunk::parse_span`], and
    /// [`ExtensionRanges::push`].
    fn ingest(
        &mut self,
        chunk: ObjectHeaderChunk<'_>,
        base: AddressContext,
        file_len: u64,
    ) -> Result<(), Error> {
        let layout = chunk.v2_layout().ok_or(Error::EditUnsupported(
            "an object does not use a version 2 object header",
        ))?;
        let region = chunk.records().bytes();
        let mut start = 0;
        while let Some((kind, body, end)) = next_record(layout, region, start)? {
            if kind == MessageType::OBJECT_HEADER_CONTINUATION {
                ContinuationChunk::parse_span(&region[body..end], base, file_len)?;
                self.count += 1;
                if self.count > MAX_OH_CHUNKS {
                    return Err(Error::EditUnsupported(
                        "object header has too many continuation chunks",
                    ));
                }
            }
            start = end;
        }
        self.ranges.push(chunk.allocation())
    }
}

/// Inline chunk ranges with capacity reserved in every variant.
///
/// The fixed array is initialized when the second range is stored. A persistent consumer
/// materializes only the observed ranges as a vector.
#[expect(
    clippy::large_enum_variant,
    variant_size_differences,
    reason = "continuation geometry stays inline within the editor chunk limit until persistence requires a vector"
)]
enum ExtensionRanges {
    Empty,
    Many {
        slots: [FileRange; MAX_OH_CHUNKS],
        len: usize,
    },
    Single(FileRange),
}

impl ExtensionRanges {
    /// Appends one range within the editor's chunk limit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the inline array is full.
    fn push(&mut self, range: FileRange) -> Result<(), Error> {
        match self {
            Self::Empty => *self = Self::Single(range),
            Self::Single(first) => {
                let mut slots = [*first; MAX_OH_CHUNKS];
                slots[1] = range;
                *self = Self::Many { slots, len: 2 };
            }
            Self::Many { slots, len } => {
                *slots.get_mut(*len).ok_or(Error::EditUnsupported(
                    "object header has too many continuation chunks",
                ))? = range;
                *len += 1;
            }
        }
        Ok(())
    }

    /// Copies the observed ranges into an owned vector.
    fn to_vec(&self) -> Vec<FileRange> {
        match self {
            Self::Empty => Vec::new(),
            Self::Many { slots, len } => slots.iter().take(*len).copied().collect(),
            Self::Single(range) => vec![*range],
        }
    }
}

/// A source region whose record boundaries have been checked before rewriting.
///
/// The immutable borrow retains the bytes and prefix properties used during that check.
struct HeaderRewrite<'a> {
    region: &'a OhRegion,
}

impl<'a> HeaderRewrite<'a> {
    /// Checks every record boundary, including records after a possible replacement.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a record extends past the region.
    fn parse(region: &'a OhRegion) -> Result<Self, Error> {
        let mut msg_start = 0;
        while let Some((_, _, body_end)) = region.next_message(msg_start)? {
            msg_start = body_end;
        }
        Ok(Self { region })
    }

    /// Replaces every body of `msg_type`, preserving record metadata and unrelated bytes.
    ///
    /// Returns `None` if the region contains no record of `msg_type`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if `body` exceeds the message size limit.
    fn replace_bodies(
        &self,
        msg_type: MessageType,
        body: &[u8],
    ) -> Result<Option<OhRegion>, Error> {
        let region = self.region;
        let mut out = Vec::with_capacity(region.len());
        let mut msg_start = 0;
        let mut replaced = false;
        while let Some((record_type, _, body_end)) = region.next_message(msg_start)? {
            let bytes = &region[msg_start..body_end];
            if record_type == msg_type {
                let splice = RecordSplice::first(region.layout(), bytes, msg_type)
                    .map_err(record_error)?
                    .ok_or_else(|| {
                        Error::Format(FormatError::Internal(
                            "a parsed message record is absent from its source region".into(),
                        ))
                    })?;
                splice.replace_body_into(&mut out, body)?;
                replaced = true;
            } else {
                out.extend_from_slice(bytes);
            }
            msg_start = body_end;
        }
        out.extend_from_slice(&region[msg_start..]);
        Ok(replaced.then(|| region.with_bytes(out)))
    }
}

/// Returns the absolute chunk 0 message range and parsed prefix properties.
///
/// `prefix` starts at `addr` and contains up to [`OBJECT_HEADER_PREFIX_MAX_LEN`] bytes.
/// `file_len` bounds the message region and its trailing checksum. The version 2 prefix and
/// chunk fields are defined in "Version 2 Data Object Header Prefix" of the [format
/// specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the header is not version 2, the prefix is truncated,
/// or the chunk or its checksum extends past `file_len`. Returns [`Error::Format`] for other
/// prefix errors.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub(super) fn oh_region_at(
    prefix: &[u8],
    addr: u64,
    file_len: u64,
) -> Result<(u64, u64, ObjectHeaderPrefix), Error> {
    let parsed = ObjectHeaderPrefix::parse(prefix).map_err(|err| match err {
        FormatError::InvalidObjectHeaderSignature | FormatError::InvalidObjectHeaderVersion(_) => {
            Error::EditUnsupported("an object does not use a version 2 object header")
        }
        FormatError::UnexpectedEof { .. } => Error::EditUnsupported("truncated object header"),
        err => Error::Format(err),
    })?;
    let region_start = addr
        .checked_add(parsed.len.to_u64())
        .ok_or(Error::EditUnsupported("truncated object header"))?;
    // The region is followed by a 4-byte checksum, which must also be present.
    let region_end = region_start
        .checked_add(parsed.chunk0_size)
        .filter(|e| e.checked_add(4).is_some_and(|end| end <= file_len))
        .ok_or(Error::EditUnsupported("truncated object header"))?;
    Ok((region_start, region_end, parsed.prefix))
}

/// A bounded version 2 header chunk and its full on-disk extent.
///
/// The buffer excludes the trailing checksum. The extent includes it for storage reclamation.
pub(crate) struct OhChunk {
    /// The complete physical range, including the trailing checksum.
    pub(crate) span: FileRange,
    /// Bytes from the start of [`span`](Self::span) through the end of the message region.
    buf: Vec<u8>,
    /// Offset of the first message within [`buf`](Self::buf).
    messages_start: usize,
    /// The prefix properties shared by every chunk of the header.
    props: ObjectHeaderPrefix,
}

impl OhChunk {
    /// Returns the chunk bytes and the offset at which its records start.
    ///
    /// The slice ends before the checksum.
    pub(crate) fn message_region(&self) -> (&[u8], usize) {
        (&self.buf, self.messages_start)
    }

    /// Returns the record layout parsed from the header's chunk 0 prefix.
    pub(crate) fn layout(&self) -> MessageRecordLayout {
        self.props.layout
    }

    /// Returns the prefix properties shared by every chunk of this header.
    pub(crate) fn props(&self) -> ObjectHeaderPrefix {
        self.props
    }
}

/// Reads chunk 0 with its checksum excluded from the message buffer.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] for an unsupported or truncated header. Propagates
/// parsing and source-read errors.
fn read_oh_chunk0<S: Source + ?Sized>(src: &S, addr: u64) -> Result<OhChunk, Error> {
    let file_len = src.len();
    let window = file_len
        .saturating_sub(addr)
        .min(OBJECT_HEADER_PREFIX_MAX_LEN as u64)
        .to_usize()?;
    let prefix = src.read_metadata_at(addr, window)?;
    let (rs, re, props) = oh_region_at(&prefix, addr, file_len)?;
    // `re >= rs > addr`, so both differences are non-negative, and `oh_region_at`
    // has checked that the 4-byte checksum past `re` is present.
    let len = (re - addr).to_usize()?;
    Ok(OhChunk {
        span: FileRange::new(PhysicalAddress::new(addr), len as u64 + 4)?,
        buf: src.read_metadata_at(addr, len)?,
        messages_start: (rs - addr).to_usize()?,
        props,
    })
}

/// Reads every chunk of a version 2 header, following continuation records.
///
/// Returns chunk 0 first, followed by continuation chunks in breadth-first order. Retains every
/// chunk buffer and its complete physical range.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the header is not version 2, a chunk or record extends
/// past its source bounds, or the traversal exceeds [`MAX_OH_CHUNKS`]. Returns errors from
/// [`read_oh_chunk0`] and [`ContinuationChunk::read`].
pub(crate) fn read_oh_chunks<S: Source + ?Sized>(
    src: &S,
    addr: u64,
    base: AddressContext,
) -> Result<Vec<OhChunk>, Error> {
    let mut chunks = Vec::with_capacity(1);
    visit_oh_chunks(src, addr, base, |chunk| {
        chunks.push(chunk);
        Ok(())
    })?;
    Ok(chunks)
}

/// Visits version 2 header chunks in breadth-first continuation order.
///
/// Reads the next chunk after the visitor consumes the current chunk. Pending continuations
/// contain complete physical ranges and inherited prefix properties.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a chunk or record is unsupported or truncated, or if
/// the traversal exceeds [`MAX_OH_CHUNKS`]. Propagates source-read, address-conversion, parsing,
/// and visitor errors.
fn visit_oh_chunks<S: Source + ?Sized>(
    src: &S,
    addr: u64,
    base: AddressContext,
    mut visit: impl FnMut(OhChunk) -> Result<(), Error>,
) -> Result<(), Error> {
    let mut chunk = read_oh_chunk0(src, addr)?;
    let mut pending = VecDeque::new();
    let mut count = 1;
    loop {
        let props = chunk.props();
        let layout = props.layout;
        let (region, mut p) = chunk.message_region();
        while let Some((msg_type, body, body_end)) = next_record(layout, region, p)? {
            if msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
                pending.push_back(ContinuationChunk::parse(
                    &region[body..body_end],
                    base,
                    props,
                    src.len(),
                )?);
                count += 1;
                if count > MAX_OH_CHUNKS {
                    return Err(Error::EditUnsupported(
                        "object header has too many continuation chunks",
                    ));
                }
            }
            p = body_end;
        }
        visit(chunk)?;
        let Some(next) = pending.pop_front() else {
            return Ok(());
        };
        chunk = next.read(src)?;
    }
}

/// A continuation's complete physical range and inherited prefix properties.
///
/// The range is resolved from the stored address before reading. Version 2 block framing is
/// defined in "The Object Header Continuation Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_continuation
struct ContinuationChunk {
    span: FileRange,
    props: ObjectHeaderPrefix,
}

impl ContinuationChunk {
    /// Resolves a continuation's complete allocation within `file_len`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `body` is truncated or the block is out of bounds.
    /// Returns [`Error::Format`] for an undefined address or address-conversion overflow.
    fn parse(
        body: &[u8],
        base: AddressContext,
        props: ObjectHeaderPrefix,
        file_len: u64,
    ) -> Result<Self, Error> {
        Ok(Self {
            span: Self::parse_span(body, base, file_len)?,
            props,
        })
    }

    /// Resolves a continuation's complete physical range within `file_len`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `body` is truncated or the block is out of bounds.
    /// Returns [`Error::Format`] for an undefined address or address-conversion overflow.
    fn parse_span(body: &[u8], base: AddressContext, file_len: u64) -> Result<FileRange, Error> {
        let continuation = match ObjectHeaderContinuation::parse(body, OFFSET_SIZE, LENGTH_SIZE) {
            Ok(continuation) => continuation,
            Err(FormatError::UnexpectedEof { .. }) => {
                return Err(Error::EditUnsupported("malformed continuation message"));
            }
            Err(err) => return Err(Error::Format(err)),
        };
        let len = continuation.length();
        let off = base.resolve(continuation.address())?;
        off.get()
            .checked_add(len)
            .filter(|&end| end <= file_len && len >= 8)
            .ok_or(Error::EditUnsupported("continuation block out of bounds"))?;
        FileRange::new(off, len).map_err(Error::from)
    }

    /// Reads the complete block and excludes its checksum from the message buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the block signature is invalid. Propagates
    /// source-read, size-conversion, and parsing errors.
    fn read<S: Source + ?Sized>(self, src: &S) -> Result<OhChunk, Error> {
        let mut buf =
            src.read_metadata_at(self.span.start().get(), self.span.length().to_usize()?)?;
        let messages = match hdf5_pure_format::__private::continuation_block_messages(&buf) {
            Ok(messages) => messages,
            Err(FormatError::InvalidObjectHeaderSignature) => {
                return Err(Error::EditUnsupported(
                    "invalid continuation block signature",
                ));
            }
            Err(err) => return Err(Error::Format(err)),
        };
        buf.truncate(messages.end);
        Ok(OhChunk {
            span: self.span,
            buf,
            messages_start: messages.start,
            props: self.props,
        })
    }
}

/// Returns the type and the body range of the message record at `msg_start` in `region`,
/// as [`MessageRecordLayout::next_message`] reads it.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if the body of the record runs past the end
/// of `region`.
pub(crate) fn next_record(
    layout: MessageRecordLayout,
    region: &[u8],
    msg_start: usize,
) -> Result<Option<(MessageType, usize, usize)>, Error> {
    Ok(layout
        .next_message(region, msg_start)
        .map_err(record_error)?
        .map(|record| {
            (
                record.msg_type,
                record.body_range.start,
                record.body_range.end,
            )
        }))
}

/// Maps a truncated record to [`Error::EditUnsupported`] and other failures to [`Error::Format`].
fn record_error(err: FormatError) -> Error {
    match err {
        FormatError::UnexpectedEof { .. } => {
            Error::EditUnsupported("malformed object header message")
        }
        err => Error::Format(err),
    }
}

/// A version 2 message region and its header-prefix properties.
///
/// [`WriteEngine::gather_oh_messages`] collects records into this region, and
/// [`build_v2_object_header`] serializes it. The record layout and optional prefix fields are
/// retained when the bytes are replaced.
#[derive(Clone, Debug, Default)]
pub(crate) struct OhRegion {
    bytes: Vec<u8>,
    props: ObjectHeaderPrefix,
}

impl OhRegion {
    /// Creates a region with the supplied record bytes and prefix properties.
    pub(crate) fn new(bytes: Vec<u8>, props: ObjectHeaderPrefix) -> Self {
        Self { bytes, props }
    }

    /// Creates an empty region with the supplied prefix properties.
    pub(super) fn empty(props: ObjectHeaderPrefix) -> Self {
        Self::new(Vec::new(), props)
    }

    /// Creates a region with replacement bytes and the same prefix properties.
    pub(super) fn with_bytes(&self, bytes: Vec<u8>) -> Self {
        Self::new(bytes, self.props)
    }

    /// Returns a copy containing complete non-NIL records and the same prefix properties.
    ///
    /// Omits any trailing gap after the last complete record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a record extends past the region.
    #[cfg(test)]
    pub(crate) fn without_nil(&self) -> Result<Self, Error> {
        let mut bytes = Vec::new();
        let mut start = 0;
        while let Some((kind, _, end)) = self.next_message(start)? {
            if kind != MessageType::NIL {
                bytes.extend_from_slice(&self[start..end]);
            }
            start = end;
        }
        Ok(self.with_bytes(bytes))
    }

    /// Returns the layout used to parse and serialize this region.
    pub(crate) fn layout(&self) -> MessageRecordLayout {
        self.props.layout
    }

    /// Returns the prefix properties used to rebuild the header.
    pub(crate) fn props(&self) -> ObjectHeaderPrefix {
        self.props
    }

    /// Returns the type and body range of the record at `msg_start`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the record extends past the region.
    pub(crate) fn next_message(
        &self,
        msg_start: usize,
    ) -> Result<Option<(MessageType, usize, usize)>, Error> {
        next_record(self.props.layout, &self.bytes, msg_start)
    }

    /// Selects the first record of `msg_type` for replacement, or returns `None` if absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the body of a record it reads runs past the end of
    /// the region.
    pub(super) fn splice_first(
        &self,
        msg_type: MessageType,
    ) -> Result<Option<RecordSplice<'_>>, Error> {
        RecordSplice::first(self.layout(), &self.bytes, msg_type).map_err(record_error)
    }

    /// Returns a record's attribute creation index, or `None` if it is not stored.
    pub(super) fn creation_index(&self, msg_start: usize) -> Option<u16> {
        self.props.layout.creation_index(&self.bytes, msg_start)
    }

    /// Appends records already encoded in this region's layout.
    pub(super) fn push_bytes(&mut self, record: &[u8]) {
        self.bytes.extend_from_slice(record);
    }

    /// Returns the record bytes for a patch that preserves record lengths.
    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Appends a message record with flags and creation index set to zero.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte type field or `body` is longer than
    /// [`OBJECT_HEADER_MESSAGE_MAX`], as [`MessageRecordLayout::record`] does.
    pub(super) fn push(&mut self, msg_type: MessageType, body: &[u8]) {
        let record = self.props.layout.record(msg_type, body);
        self.push_bytes(&record);
    }

    /// Appends a shared-message reference with the shared flag set.
    ///
    /// # Panics
    ///
    /// Panics if `msg_type` does not fit the 1-byte type field or `reference` is longer than
    /// [`OBJECT_HEADER_MESSAGE_MAX`], as [`MessageRecordLayout::record_with_creation_index`] does.
    pub(super) fn push_shared(&mut self, msg_type: MessageType, reference: &[u8]) {
        let record = self.props.layout.record_with_creation_index(
            msg_type,
            MessageFlags::SHARED,
            0,
            reference,
        );
        self.push_bytes(&record);
    }
}

impl core::ops::Deref for OhRegion {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}

/// Re-encodes a shared-message reference for a version 2 object header.
///
/// Preserves the referenced object address or shared-message heap ID.
///
/// # Errors
///
/// Returns [`Error::Format`] if the reference is truncated or has an unsupported version.
pub(super) fn modernize_shared_reference(
    body: &[u8],
    widths: FormatWidths,
) -> Result<Vec<u8>, Error> {
    let reference = crate::shared_message::parse_shared_ref_with_widths(body, widths)?;
    Ok(match reference.location {
        crate::shared_message::SharedLocation::ObjectHeader(addr) => {
            crate::shared_message::encode_committed_ref(addr, widths.offsets)
        }
        crate::shared_message::SharedLocation::SohmHeap(id) => {
            crate::shared_message::encode_sohm_ref(&id)
        }
    })
}

/// Builds a single-chunk version 2 header with normalized attribute information.
///
/// Calls [`super::attributes::ensure_attribute_info`] before serializing the prefix, records,
/// and checksum.
/// Preserves the prefix's optional fields and updates existing modification and change times
/// when the system clock is available.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region. Propagates other
/// errors from [`super::attributes::ensure_attribute_info`].
pub(crate) fn build_v2_object_header(region: &OhRegion) -> Result<Vec<u8>, Error> {
    Ok(build_v2_object_header_verbatim(&normalized_header(region)?))
}

/// Builds a normalized version 2 header that fills `allocation`.
///
/// The allocation uses stored, base-relative coordinates and includes all chunk framing and
/// padding. Uses [`ObjectHeaderAllocation`] to encode its checksummed chunks. Preserves optional
/// prefix fields and updates existing modification and change times when the system clock is
/// available.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if compact attribute metadata cannot be normalized, or
/// [`Error::Format`] for an allocation-planning error from [`ObjectHeaderAllocation::new`].
pub(crate) fn build_v2_object_header_allocation(
    region: &OhRegion,
    widths: FormatWidths,
    allocation: Extent,
) -> Result<Vec<u8>, Error> {
    let owned = normalized_header(region)?;
    Ok(ObjectHeaderAllocation::new(
        stamped_prefix(&owned),
        widths,
        allocation.start(),
        allocation.len(),
        &owned,
    )?
    .encode())
}

/// Returns a copy with compact Attribute Info metadata present where required.
///
/// Preserves existing Attribute Info messages, including those naming dense storage.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if a record extends past the region or the next attribute
/// creation index cannot be represented.
fn normalized_header(region: &OhRegion) -> Result<OhRegion, Error> {
    let mut owned = region.clone();
    super::attributes::ensure_attribute_info(&mut owned)?;
    Ok(owned)
}

/// Serializes a normalized region and updates existing modification and change times.
///
/// Preserves access and birth times, and the optional attribute phase-change thresholds.
/// Preserves all timestamps if the system clock is unavailable.
fn build_v2_object_header_verbatim(region: &OhRegion) -> Vec<u8> {
    stamped_prefix(region).encode_header(region)
}

/// Returns prefix properties with existing modification and change times updated.
///
/// Preserves access and birth times, and all timestamps if the system clock is unavailable.
fn stamped_prefix(region: &OhRegion) -> ObjectHeaderPrefix {
    let prefix = region.props();
    ObjectHeaderPrefix {
        times: prefix.times.map(|times| match unix_time_now() {
            Some(now) => times.touched(now),
            None => times,
        }),
        ..prefix
    }
}

/// Returns the current Unix time, saturated to the timestamp field's maximum.
///
/// Returns `None` if the system clock is before the Unix epoch.
#[cfg(feature = "std")]
pub(super) fn unix_time_now() -> Option<u32> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| u32::try_from(d.as_secs()).unwrap_or(u32::MAX))
}

/// Returns `None` when the system clock is unavailable.
#[cfg(not(feature = "std"))]
pub(super) fn unix_time_now() -> Option<u32> {
    None
}

/// The traversal limit for header chunks, including cyclic continuation chains.
const MAX_OH_CHUNKS: usize = 256;

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::AttributePhaseChange;
    use hdf5_pure_format::__private::ObjectTimes;
    use rstest::rstest;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    use test_util::allocation;
    use test_util::file_space_info as info_bytes;
    use test_util::object_header::v2::HeaderFlags;
    use test_util::widths::Widths;

    use crate::address::{BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt};
    use crate::attribute_request::AttributeName;
    use crate::file_space_info::{self, FileSpacePageSize, FileSpaceStrategy};
    use crate::file_writer::WIDTHS;
    use crate::source::BytesSource;
    use crate::type_builders::AttrValue;

    use super::*;

    mod reconstruction;

    #[rstest]
    #[case::one_message(false, vec![1, 2, 3, 4])]
    #[case::repeated_message(true, vec![1, 2, 3, 4])]
    #[case::version_0(false, info_bytes::version_0(
        Widths::EIGHT,
        info_bytes::ALL_PERSIST,
        1,
        &[u64::MAX; info_bytes::VERSION_0_MANAGERS],
    ))]
    fn an_extension_rewrite_preserves_record_metadata_and_unrelated_bytes(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
        #[case] repeated: bool,
        #[case] source_body: Vec<u8>,
    ) {
        let info = file_space_info::persistent_placeholder(
            FileSpaceStrategy::Page,
            7,
            FileSpacePageSize::DEFAULT,
        );
        let body =
            hdf5_pure_format::__private::serialize_file_space_info_v1(WIDTHS, &info).unwrap();
        let props = ObjectHeaderPrefix {
            layout,
            times: Some(ObjectTimes {
                access: 1,
                modification: 2,
                change: 3,
                birth: 4,
            }),
            attribute_phase_change: Some(AttributePhaseChange {
                max_compact: 12,
                min_dense: 5,
            }),
        };
        let region_bytes = |body: &[u8]| {
            [
                layout.record_with_creation_index(
                    MessageType::DATATYPE,
                    MessageFlags::SHARED,
                    9,
                    &[0xAB; 7],
                ),
                layout.record_with_creation_index(
                    MessageType::FILE_SPACE_INFO,
                    MessageFlags::FORBID_SHARING | MessageFlags::MARK_IF_UNKNOWN,
                    3,
                    body,
                ),
                layout.record(MessageType::GROUP_INFO, &[0; 2]),
                if repeated {
                    layout.record_with_creation_index(
                        MessageType::FILE_SPACE_INFO,
                        MessageFlags::CONSTANT,
                        11,
                        body,
                    )
                } else {
                    Vec::new()
                },
                vec![0x11, 0x22, 0x33],
            ]
            .concat()
        };
        let source = OhRegion::new(region_bytes(&source_body), props);

        let rewritten = rewrite_extension_region_bytes(&source, WIDTHS, &info).unwrap();

        assert_eq!(rewritten.props(), props);
        assert_eq!(&rewritten[..], &region_bytes(&body));

        let allocation = Extent::new(StoredAddress::new(1_024), 131_123).unwrap();
        let encoded = build_v2_object_header_allocation(&rewritten, WIDTHS, allocation).unwrap();
        let addresses = AddressContext::from_effective_base(BaseAddress::new(4_096));
        let physical = addresses.resolve(allocation.start()).unwrap();
        let mut image = vec![0; usize::try_from(physical.get()).unwrap()];
        image.extend(encoded);
        let src = BytesSource::new(&image);
        let chunks = read_oh_chunks(&src, physical.get(), addresses).unwrap();
        assert!(chunks.len() > 1);
        assert_eq!(
            chunks.iter().map(|chunk| chunk.span.length()).sum::<u64>(),
            allocation.len()
        );
        let actual = WriteEngine::gather_oh_messages(&src, physical.get(), addresses)
            .unwrap()
            .without_nil()
            .unwrap();
        assert_eq!(&actual[..], &rewritten.without_nil().unwrap()[..]);
        assert_eq!(actual.props().layout, props.layout);
        assert_eq!(
            actual.props().attribute_phase_change,
            props.attribute_phase_change
        );
        let times = actual.props().times.unwrap();
        assert_eq!((times.access, times.birth), (1, 4));
        assert!(times.modification >= 2);
        assert!(times.change >= 3);
    }

    #[rstest]
    #[case::plain(MessageRecordLayout::PLAIN)]
    #[case::tracked(TRACKED)]
    fn allocated_headers_keep_compact_attribute_normalization(#[case] layout: MessageRecordLayout) {
        let body = AttributeName::try_from("audit_counter")
            .unwrap()
            .build_value(&AttrValue::U32(7))
            .unwrap()
            .serialize(WIDTHS.offsets, WIDTHS.lengths.get())
            .unwrap();
        let region = OhRegion::new(
            layout.record_with_creation_index(
                MessageType::ATTRIBUTE,
                MessageFlags::NONE,
                17,
                &body,
            ),
            ObjectHeaderPrefix::with_layout(layout),
        );
        assert!(
            region
                .splice_first(MessageType::ATTRIBUTE_INFO)
                .unwrap()
                .is_none()
        );
        let addresses = AddressContext::from_effective_base(BaseAddress::ZERO);
        let expected = build_v2_object_header(&region).unwrap();
        let expected =
            WriteEngine::gather_oh_messages(&BytesSource::new(expected), 0, addresses).unwrap();
        let bytes = build_v2_object_header_allocation(
            &region,
            WIDTHS,
            Extent::new(StoredAddress::ZERO, 8_192).unwrap(),
        )
        .unwrap();
        let actual = WriteEngine::gather_oh_messages(&BytesSource::new(bytes), 0, addresses)
            .unwrap()
            .without_nil()
            .unwrap();
        assert_eq!(&actual[..], &expected[..]);
        assert!(
            actual
                .splice_first(MessageType::ATTRIBUTE_INFO)
                .unwrap()
                .is_some()
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[rstest]
    #[case::plain(MessageRecordLayout::PLAIN, "plain_extension_growth")]
    #[case::tracked(TRACKED, "tracked_extension_growth")]
    fn a_growing_extension_rewrite_keeps_the_allocation_count(
        #[case] layout: MessageRecordLayout,
        #[case] measurement: &'static str,
    ) {
        let old_body = info_bytes::version_0(
            Widths::EIGHT,
            info_bytes::ALL_PERSIST,
            1,
            &[u64::MAX; info_bytes::VERSION_0_MANAGERS],
        );
        let source = OhRegion::new(
            [
                layout.record(MessageType::FILE_SPACE_INFO, &old_body),
                layout.record(MessageType::NIL, &[0; 2]),
            ]
            .concat(),
            ObjectHeaderPrefix::with_layout(layout),
        );
        let info = file_space_info::persistent_placeholder(
            FileSpaceStrategy::FsmAggr,
            1,
            FileSpacePageSize::DEFAULT,
        );

        let (rewritten, measured) = allocation::measure(measurement, || {
            rewrite_extension_region_bytes(&source, WIDTHS, &info).unwrap()
        });

        assert_eq!(measured.blocks, 3, "{measured}");
        let new_body =
            hdf5_pure_format::__private::serialize_file_space_info_v1(WIDTHS, &info).unwrap();
        assert_eq!(
            &rewritten[..],
            &[
                layout.record(MessageType::FILE_SPACE_INFO, &new_body),
                layout.record(MessageType::NIL, &[0; 2]),
            ]
            .concat(),
        );
    }

    #[rstest]
    #[case::before_replacement(false)]
    #[case::after_replacement(true)]
    fn an_extension_rewrite_rejects_a_record_past_the_source_region(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
        #[case] after_replacement: bool,
    ) {
        let info = file_space_info::persistent_placeholder(
            FileSpaceStrategy::Page,
            7,
            FileSpacePageSize::DEFAULT,
        );
        let mut bytes = if after_replacement {
            layout.record(MessageType::FILE_SPACE_INFO, &[1, 2])
        } else {
            Vec::new()
        };
        let mut malformed = layout.record(MessageType::GROUP_INFO, &[0; 2]);
        malformed.pop();
        bytes.extend_from_slice(&malformed);
        let source = OhRegion::new(bytes, ObjectHeaderPrefix::with_layout(layout));

        let err = rewrite_extension_region_bytes(&source, WIDTHS, &info).unwrap_err();

        let Error::EditUnsupported(reason) = err else {
            panic!("expected EditUnsupported, got {err:?}");
        };
        assert_eq!(reason, "malformed object header message");
    }

    #[rstest]
    fn an_extension_without_file_space_info_is_rejected(
        #[values(MessageRecordLayout::PLAIN, TRACKED)] layout: MessageRecordLayout,
    ) {
        let info = file_space_info::persistent_placeholder(
            FileSpaceStrategy::Page,
            7,
            FileSpacePageSize::DEFAULT,
        );
        let source = OhRegion::new(
            layout.record(MessageType::GROUP_INFO, &[0; 2]),
            ObjectHeaderPrefix::with_layout(layout),
        );

        let err = rewrite_extension_region_bytes(&source, WIDTHS, &info).unwrap_err();

        let Error::EditUnsupported(reason) = err else {
            panic!("expected EditUnsupported, got {err:?}");
        };
        assert_eq!(
            reason,
            "a persisting file's superblock extension has no File Space Info message"
        );
    }

    #[rstest]
    #[case::paged_index(test_util::paged_index::paged_index())]
    #[case::generic_large(test_util::generic_large::generic_large())]
    fn an_extension_keeps_its_unmodified_record_bytes(#[case] image: Vec<u8>) {
        let source = BytesSource::new(image);
        crate::edit::test_util::assert_extension_records(&source);
    }

    const TRACKED: MessageRecordLayout =
        MessageRecordLayout::from_header_flags(HeaderFlags::TRACKS_CREATION_ORDER.0);
}
