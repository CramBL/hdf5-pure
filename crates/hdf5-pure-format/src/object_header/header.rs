use alloc::vec::Vec;

use hdf5_pure_core::PhysicalAddress;

use super::context::ParseContext;
use super::{MessageFilter, ObjectHeaderChunk, ObjectHeaderPrefix};
use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::bytes;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::metadata_source::MetadataSource;
use crate::width::FormatWidths;

/// OHDR signature for v2 object headers.
pub(super) const OHDR_SIGNATURE: [u8; 4] = *b"OHDR";

/// A single parsed header message.
///
/// Retains the message body, record flags, and optional creation index.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct HeaderMessage {
    /// The message type.
    pub msg_type: MessageType,
    /// Size of the message data in bytes.
    pub size: usize,
    /// The record's flags. `data` holds a reference to the message when
    /// [`MessageFlags::SHARED`] is set.
    pub flags: MessageFlags,
    /// Creation order (v2 only, when tracking is enabled).
    pub creation_order: Option<u16>,
    /// Raw message data bytes.
    pub data: Vec<u8>,
}

/// A parsed HDF5 object header.
///
/// Contains the non-NIL messages selected by the filter. For version 2, [`prefix`](Self::prefix)
/// holds the record layout and optional timestamps and attribute thresholds, including when
/// message filtering removes every record.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Header version (1 or 2).
    pub version: u8,
    /// The version 2 record layout and optional prefix fields, or `None` for version 1.
    pub prefix: Option<ObjectHeaderPrefix>,
    /// The messages retained from the initial chunk and its continuations.
    pub messages: Vec<HeaderMessage>,
    /// Object reference count (v1 only).
    pub reference_count: Option<u32>,
    /// Contains the v2 object header flags. Version 1 uses zero.
    pub flags: u8,
    /// Access time, when present in a version 2 prefix.
    pub access_time: Option<u32>,
    /// Modification time, when present in a version 2 prefix.
    pub modification_time: Option<u32>,
    /// Change time, when present in a version 2 prefix.
    pub change_time: Option<u32>,
    /// Birth time, when present in a version 2 prefix.
    pub birth_time: Option<u32>,
}

impl ObjectHeader {
    /// Parses the object header at `offset` in `data`.
    ///
    /// `offset_size` and `length_size` come from the superblock, and `access_mode` is
    /// the mode the caller reads under.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a
    /// width is not 2, 4, or 8, and otherwise the errors
    /// [`parse_with_widths`](Self::parse_with_widths) returns.
    pub fn parse(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_with_widths(
            data,
            access_mode,
            offset,
            FormatWidths::from_sizes(offset_size, length_size)?,
        )
    }

    /// Uses file-format widths validated by the superblock parser.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedMessage`] if a message record holds a type
    /// this parser cannot name and [`MessageFlags::must_be_understood`] returns
    /// `true` for `access_mode`.
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a continuation body is too short to hold the
    /// address and length fields at the supplied widths, after checking record framing. In this
    /// case, `expected` is the combined field width and `available` is the body length.
    ///
    /// Also returns [`FormatError::UnexpectedEof`] if a version 2 record has a complete prefix
    /// and its body extends into the chunk checksum or beyond it.
    ///
    /// Returns [`FormatError::UndefinedContinuationAddress`] if a complete continuation body
    /// stores an undefined address, in either header version.
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if a version 1 header exceeds 256 followed
    /// continuation references or 32 continuation hops after the root, or if a version 2 header
    /// exceeds 256 discovered continuation references. The parser counts repeated and zero-length
    /// targets separately, including when filtering removes every message.
    pub fn parse_with_widths(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        widths: FormatWidths,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_with_base_and_widths(data, access_mode, offset, widths, BaseAddress::ZERO)
    }

    /// Parses an object header, applying `base_address` to continuation offsets.
    ///
    /// Pass the absolute file position of the object header as `offset`. For either header
    /// version, the parser adds `base_address` to each stored continuation address before
    /// reading the chunk. It checks each message against `access_mode`, as in [`Self::parse`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a
    /// width is not 2, 4, or 8, and otherwise the errors [`parse`](Self::parse) returns.
    pub fn parse_with_base(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_with_base_and_widths(
            data,
            access_mode,
            offset,
            FormatWidths::from_sizes(offset_size, length_size)?,
            base_address,
        )
    }

    /// Applies `base_address` to continuation offsets using validated file-format widths.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse_with_widths`](Self::parse_with_widths) returns.
    pub fn parse_with_base_and_widths(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        widths: FormatWidths,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_filtered_with_widths(
            data,
            access_mode,
            offset,
            widths,
            base_address,
            MessageFilter::All,
        )
    }

    /// Parses an object header, keeping only the messages `filter` names.
    ///
    /// The header uses the same validation, checksum checks, and continuation
    /// traversal as [`parse_with_base`](Self::parse_with_base). The result differs
    /// only in which messages it carries. See [`MessageFilter`] for targeted
    /// parsing.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a
    /// width is not 2, 4, or 8, and otherwise the errors [`parse`](Self::parse) returns.
    pub fn parse_filtered(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
        filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_filtered_with_widths(
            data,
            access_mode,
            offset,
            FormatWidths::from_sizes(offset_size, length_size)?,
            base_address,
            filter,
        )
    }

    /// Keeps only the messages `filter` names while using validated file-format widths.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse_with_widths`](Self::parse_with_widths) returns.
    pub fn parse_filtered_with_widths(
        data: &[u8],
        access_mode: AccessMode,
        offset: usize,
        widths: FormatWidths,
        base_address: BaseAddress,
        mut filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        let context = ParseContext {
            access_mode,
            widths,
            base_address,
        };
        bytes::ensure_len(data, offset, 4)?;
        if data[offset..offset + 4] == OHDR_SIGNATURE {
            Self::parse_v2(data, context, offset, filter)
        } else {
            Self::parse_v1(data, context, offset, &mut filter)
        }
    }

    /// Parses an object header from a [`MetadataSource`] using bounded reads for each
    /// header and continuation chunk.
    ///
    /// Continuation offsets are resolved through `base_address` as in
    /// [`Self::parse_with_base`]. The parser reads individual chunks. Version 1 retains recursive
    /// ancestor buffers, and version 2 retains chunk 0 alongside the current continuation.
    /// Every message record is tested against `access_mode` as in [`parse`](Self::parse).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a
    /// width is not 2, 4, or 8, and otherwise the errors [`parse`](Self::parse) returns and the
    /// error `source` returns if a read fails.
    pub fn parse_from_source<S: MetadataSource + ?Sized>(
        source: &S,
        access_mode: AccessMode,
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_from_source_with_widths(
            source,
            access_mode,
            address,
            FormatWidths::from_sizes(offset_size, length_size)?,
            base_address,
        )
    }

    /// Reads an object header through a [`MetadataSource`] using validated file-format widths.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse_with_widths`](Self::parse_with_widths) returns and the error
    /// `source` returns if a read fails.
    pub fn parse_from_source_with_widths<S: MetadataSource + ?Sized>(
        source: &S,
        access_mode: AccessMode,
        address: u64,
        widths: FormatWidths,
        base_address: BaseAddress,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_from_source_filtered_with_widths(
            source,
            access_mode,
            address,
            widths,
            base_address,
            MessageFilter::All,
        )
    }

    /// Parses an object header from a [`MetadataSource`], keeping only the messages `filter` names.
    ///
    /// The streaming counterpart of [`parse_filtered`](Self::parse_filtered).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a
    /// width is not 2, 4, or 8, and otherwise the errors [`parse`](Self::parse) returns and the
    /// error `source` returns if a read fails.
    pub fn parse_from_source_filtered<S: MetadataSource + ?Sized>(
        source: &S,
        access_mode: AccessMode,
        address: u64,
        offset_size: u8,
        length_size: u8,
        base_address: BaseAddress,
        filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_from_source_filtered_with_widths(
            source,
            access_mode,
            address,
            FormatWidths::from_sizes(offset_size, length_size)?,
            base_address,
            filter,
        )
    }

    /// Filters an object header read through a [`MetadataSource`] using validated format widths.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse_with_widths`](Self::parse_with_widths) returns and the error
    /// `source` returns if a read fails.
    pub fn parse_from_source_filtered_with_widths<S: MetadataSource + ?Sized>(
        source: &S,
        access_mode: AccessMode,
        address: u64,
        widths: FormatWidths,
        base_address: BaseAddress,
        filter: MessageFilter<'_>,
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_from_source_observed(
            source,
            ParseContext {
                access_mode,
                widths,
                base_address,
            },
            address,
            filter,
            &mut None,
        )
    }

    /// Parses an object header and observes each chunk's borrowed message region.
    ///
    /// `source` positions and `address` use physical file coordinates. Stored continuation
    /// addresses are resolved through `base_address`. Calls `observer` after acquiring each
    /// chunk and before parsing its message records. Version 1 continuations are visited
    /// depth-first, and version 2 continuations breadth-first.
    ///
    /// The observer borrows bytes from the active source buffer. The returned header retains
    /// only the messages selected by `filter`.
    /// The observer may have been called if parsing returns an error. The caller establishes
    /// allocation ownership and source freshness separately.
    ///
    /// # Errors
    ///
    /// Returns the parsing and source-read errors from
    /// [`parse_from_source_filtered_with_widths`](Self::parse_from_source_filtered_with_widths),
    /// and [`FormatError::OffsetOverflow`] if an observed physical range overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::PhysicalAddress;
    /// use hdf5_pure_core::__private::BaseAddressExt;
    /// use hdf5_pure_format::BaseAddress;
    /// use hdf5_pure_format::__private::{
    ///     AccessMode, FormatWidths, MessageFilter, ObjectHeader, ObjectHeaderPrefix,
    /// };
    ///
    /// let bytes = ObjectHeaderPrefix::PLAIN.encode_header(&[]);
    /// let mut allocations = Vec::new();
    /// let header = ObjectHeader::parse_from_source_with_chunk_observer(
    ///     bytes.as_slice(),
    ///     AccessMode::ReadOnly,
    ///     PhysicalAddress::new(0),
    ///     FormatWidths::from_sizes(8, 8)?,
    ///     BaseAddress::ZERO,
    ///     MessageFilter::All,
    ///     &mut |chunk| {
    ///         assert!(chunk.records().bytes().is_empty());
    ///         allocations.push(chunk.allocation());
    ///     },
    /// )?;
    /// assert_eq!(header.version, 2);
    /// assert!(header.messages.is_empty());
    /// assert_eq!(allocations.len(), 1);
    /// assert_eq!(allocations[0].start(), PhysicalAddress::new(0));
    /// assert_eq!(allocations[0].length(), u64::try_from(bytes.len())?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn parse_from_source_with_chunk_observer<S: MetadataSource + ?Sized>(
        source: &S,
        access_mode: AccessMode,
        address: PhysicalAddress,
        widths: FormatWidths,
        base_address: BaseAddress,
        filter: MessageFilter<'_>,
        observer: &mut dyn FnMut(ObjectHeaderChunk<'_>),
    ) -> Result<ObjectHeader, FormatError> {
        Self::parse_from_source_observed(
            source,
            ParseContext {
                access_mode,
                widths,
                base_address,
            },
            address.get(),
            filter,
            &mut Some(observer),
        )
    }

    /// Dispatches the source parser with optional borrowed chunk observations.
    ///
    /// # Errors
    ///
    /// Returns source-read errors from [`MetadataSource::read_at`] and the errors from
    /// [`Self::parse_v1_from_source`] or [`Self::parse_v2_from_source`].
    fn parse_from_source_observed<S: MetadataSource + ?Sized>(
        source: &S,
        context: ParseContext,
        address: u64,
        mut filter: MessageFilter<'_>,
        observer: &mut Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    ) -> Result<ObjectHeader, FormatError> {
        let mut sig = [0u8; 4];
        source.read_at(address, &mut sig)?;
        if sig == OHDR_SIGNATURE {
            Self::parse_v2_from_source(source, context, address, filter, observer)
        } else {
            Self::parse_v1_from_source(source, context, address, &mut filter, observer)
        }
    }
}
