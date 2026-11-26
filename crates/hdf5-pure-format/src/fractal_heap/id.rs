use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU64;

use super::header::FractalHeapFiltering;
use super::header::FractalHeapHeader;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2::HugeObjectDirectRecord;
use crate::bytes;
use crate::codec::Encode;
use crate::error::FormatError;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The type of a fractal heap ID: where the heap stores the object the ID refers to.
///
/// Bits 4 and 5 of the first byte of a heap ID hold the type. The heap IDs are defined in "Fractal
/// Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FractalHeapIdType {
    /// The object is in a direct block of the heap.
    Managed,
    /// The object is in the file outside the blocks of the heap, and the heap ID holds its address
    /// and length or its key in the huge-object B-tree of the heap.
    Huge,
    /// The heap ID holds the object.
    Tiny,
}

impl FractalHeapIdType {
    /// Reads the type bits of a version 0 heap ID.
    ///
    /// The caller supplies the first byte of a complete ID. Bits 6 and 7 hold the version, and
    /// bits 4 and 5 hold the type. The low four bits hold part or all of a tiny object's length
    /// encoding. In managed and huge IDs, those bits are reserved.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFractalHeapIdVersion`] if the version is not 0,
    /// [`FormatError::InvalidHeapIdType`] if the type bits hold 3, or
    /// [`FormatError::InvalidFractalHeapIdFlags`] if a managed or huge ID has set reserved bits.
    /// Checks these conditions in that order.
    fn from_first_byte(byte0: u8) -> Result<Self, FormatError> {
        let version = byte0 >> HEAP_ID_VERSION_SHIFT;
        if version != HEAP_ID_VERSION {
            return Err(FormatError::UnsupportedFractalHeapIdVersion(version));
        }
        let kind = match (byte0 >> HEAP_ID_TYPE_SHIFT) & HEAP_ID_TYPE_MASK {
            HEAP_ID_TYPE_MANAGED => Self::Managed,
            HEAP_ID_TYPE_HUGE => Self::Huge,
            HEAP_ID_TYPE_TINY => return Ok(Self::Tiny),
            other => return Err(FormatError::InvalidHeapIdType(other)),
        };
        let flags = byte0 & HEAP_ID_RESERVED_MASK;
        if flags != 0 {
            return Err(FormatError::InvalidFractalHeapIdFlags(flags));
        }
        Ok(kind)
    }

    /// Returns the first byte of a heap ID of this type, whose version bits are 0.
    const fn first_byte(self) -> u8 {
        let type_bits = match self {
            Self::Managed => HEAP_ID_TYPE_MANAGED,
            Self::Huge => HEAP_ID_TYPE_HUGE,
            Self::Tiny => HEAP_ID_TYPE_TINY,
        };
        type_bits << HEAP_ID_TYPE_SHIFT
    }
}

/// The location of a huge object, as its heap ID gives it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HugeObjectReference {
    /// The heap ID holds this key, which the huge-object B-tree of the heap maps to the address
    /// and the length of the object.
    Indexed(HugeIndirectKey),
    /// The heap ID holds a key in the heap's type-2 filtered huge-object tracker.
    FilteredIndexed(HugeIndirectKey),
    /// The heap ID holds the address and the length of the object.
    Inline {
        /// The address of the object.
        addr: StoredAddress,
        /// The length of the object in bytes.
        len: u64,
    },
    /// The heap ID holds the filtered object's location, sizes, and filter mask.
    FilteredInline(FilteredHugeObjectLocation),
}

/// The stored address, encoded and decoded sizes, and filter mask of a filtered huge object.
///
/// Both sizes are positive. The reader resolves the base-relative address before reading
/// [`encoded_size`](Self::encoded_size) bytes, then reverses the active filters and checks the
/// result against [`decoded_size`](Self::decoded_size).
/// If every filter is skipped, the encoded and decoded sizes are equal.
///
/// Obtained from [`FractalHeapIdLayout::parse`] or
/// [`HugeObjectRecordDecoder::parse`](crate::btree::v2::HugeObjectRecordDecoder::parse).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilteredHugeObjectLocation {
    address: StoredAddress,
    encoded_size: NonZeroU64,
    filter_mask: u32,
    decoded_size: NonZeroU64,
}

impl FilteredHugeObjectLocation {
    pub(crate) fn parse(
        bytes: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
        filter_count: usize,
    ) -> Result<Self, FormatError> {
        let mut cursor = bytes::ByteCursor::new(bytes);
        let address = cursor.address(offsets)?;
        let encoded_size = cursor.length(lengths)?;
        let filter_mask = cursor.u32()?;
        let decoded_size = cursor.length(lengths)?;
        if address.get() >= StoredAddress::undefined(offsets).get()
            || filter_count > u32::BITS as usize
            || (filter_count < u32::BITS as usize && filter_mask >> filter_count != 0)
        {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let encoded_size =
            NonZeroU64::new(encoded_size).ok_or(FormatError::InvalidFractalHeapGeometry)?;
        let decoded_size =
            NonZeroU64::new(decoded_size).ok_or(FormatError::InvalidFractalHeapGeometry)?;
        address
            .get()
            .checked_add(encoded_size.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: address.get(),
                length: encoded_size.get(),
            })?;
        let skipped = filter_mask.count_ones() as usize == filter_count;
        if skipped && encoded_size != decoded_size {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(Self {
            address,
            encoded_size,
            filter_mask,
            decoded_size,
        })
    }

    /// Returns the base-relative address of the encoded payload.
    pub const fn address(self) -> StoredAddress {
        self.address
    }
    /// Returns the encoded physical allocation length in bytes.
    pub const fn encoded_size(self) -> u64 {
        self.encoded_size.get()
    }
    /// Returns the skipped filters as set bits at their pipeline indices.
    pub const fn filter_mask(self) -> u32 {
        self.filter_mask
    }
    /// Returns the expected payload length after reversing its active filters.
    pub const fn decoded_size(self) -> u64 {
        self.decoded_size.get()
    }
}

/// The encoded width of a numeric key in an indirect huge-object heap ID.
///
/// The tracker record's key uses the file's length width in both dialects. Callers compare
/// decoded numeric values when the heap ID and tracker record use different widths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HugeIndirectKeyEncoding {
    /// Uses the smaller of the heap ID payload length and eight bytes.
    ///
    /// `H5HF__huge_init` applies this rule in `H5HFhuge.c` (HDF5 2.2.0).
    Libhdf5,
    /// Uses the file's length width.
    ///
    /// The key is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    PublishedSpec,
}

/// A numeric huge-object tracker key decoded under an explicit width dialect.
///
/// The value fits the tracker's length-width field. Obtained from
/// [`FractalHeapIdLayout::parse`] for an indirect huge-object heap ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeIndirectKey {
    value: u64,
    encoding: HugeIndirectKeyEncoding,
}

impl HugeIndirectKey {
    /// Returns the numeric key for comparison with the tracker record's object ID.
    pub const fn value(self) -> u64 {
        self.value
    }
    /// Returns the width dialect used to decode the heap ID's key.
    pub const fn encoding(self) -> HugeIndirectKeyEncoding {
        self.encoding
    }
}

impl FractalHeapHeader {
    /// Returns the heap ID of the managed object of `length` bytes at heap offset `heap_offset`.
    ///
    /// After its first byte, the ID stores the offset and length in separate little-endian byte
    /// fields. The offset width is [`max_heap_size`](Self::max_heap_size) rounded up to whole
    /// bytes. The length width is derived from the direct-block offset width and
    /// [`max_managed_object_size`](Self::max_managed_object_size), as `H5HF_hdr_finish_init_phase1`
    /// in `H5HFhdr.c` computes it (HDF5 1.10.11). Trailing ID padding is zeroed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the header cannot describe the
    /// managed-ID fields. Returns [`FormatError::Internal`] if `heap_offset` requires more bits
    /// than [`max_heap_size`](Self::max_heap_size) or `length` does not fit its encoded field.
    pub fn encode_managed_id(&self, heap_offset: u64, length: u64) -> Result<Vec<u8>, FormatError> {
        ManagedHeapIdLayout::new(self)
            .ok_or(FormatError::InvalidFractalHeapGeometry)?
            .encode(heap_offset, length)
    }

    /// Returns the heap ID of the huge object whose key in the huge-object B-tree is `huge_id`.
    ///
    /// The huge IDs of a heap store a key where they are too short for an address of
    /// `offset_width` and a length of `length_width`, as [`FractalHeapIdLayout::parse`]
    /// reads them.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters its objects,
    /// and [`FormatError::Internal`] if the huge IDs of the heap store an address and a length, or
    /// if `huge_id` does not fit the ID.
    pub fn encode_huge_id(
        &self,
        huge_id: u64,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<Vec<u8>, FormatError> {
        if self.filtering.is_filtered() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if self.huge_ids_direct_with_widths(FormatWidths {
            offsets: offset_width,
            lengths: length_width,
        }) {
            return Err(FormatError::Internal(
                "a key was encoded into a heap ID that holds its object's address".into(),
            ));
        }
        self.encode_id(FractalHeapIdType::Huge, Some(huge_id), "huge object key")
    }

    /// Encodes an unfiltered huge object's address and length directly into its heap ID.
    ///
    /// Direct IDs are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] for a filtered heap, or
    /// [`FormatError::Internal`] if the ID cannot hold a direct address and length. Propagates
    /// defined-address and length errors from [`HugeObjectDirectRecord::encode`].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    pub(crate) fn encode_huge_direct_id(
        &self,
        address: StoredAddress,
        length: u64,
        widths: FormatWidths,
    ) -> Result<Vec<u8>, FormatError> {
        if self.filtering.is_filtered() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if !self.huge_ids_direct_with_widths(widths) {
            return Err(FormatError::Internal(
                "an address was encoded into a heap ID that holds a B-tree key".into(),
            ));
        }
        let mut id = Vec::with_capacity(usize::from(self.heap_id_length));
        id.push(FractalHeapIdType::Huge.first_byte());
        HugeObjectDirectRecord { address, length }.encode(
            &mut id,
            widths.offsets,
            widths.lengths,
        )?;
        id.resize(usize::from(self.heap_id_length), 0);
        Ok(id)
    }

    /// Returns a heap ID of `id_type` whose bytes after the first store `payload` little-endian.
    ///
    /// A `payload` of `None` is one too wide for the ID, and the error names it as `what`.
    fn encode_id(
        &self,
        id_type: FractalHeapIdType,
        payload: Option<u64>,
        what: &str,
    ) -> Result<Vec<u8>, FormatError> {
        let mut id = vec![0; usize::from(self.heap_id_length)];
        let Some((first, rest)) = id.split_first_mut() else {
            return Err(FormatError::Internal(
                "a heap ID was encoded for a heap whose IDs are empty".into(),
            ));
        };
        let too_wide = || {
            FormatError::Internal(format!(
                "a {what} does not fit a {}-byte heap ID",
                self.heap_id_length
            ))
        };
        let payload = payload.ok_or_else(too_wide)?.to_le_bytes();
        let (low, high) = payload.split_at(rest.len().min(payload.len()));
        if high.iter().any(|&byte| byte != 0) {
            return Err(too_wide());
        }
        *first = id_type.first_byte();
        for (byte, &value) in rest.iter_mut().zip(low) {
            *byte = value;
        }
        Ok(id)
    }

    /// Returns `true` if a huge object ID has room for the object's location after its prefix.
    ///
    /// An unfiltered location requires an address and a length. A filtered location also
    /// includes a four-byte filter mask and the decoded length. Returns `false` if the
    /// location does not fit or either field width is unsupported.
    pub fn huge_ids_direct(&self, offset_size: u8, length_size: u8) -> bool {
        let Ok(widths) = FormatWidths::from_sizes(offset_size, length_size) else {
            return false;
        };
        self.huge_ids_direct_with_widths(widths)
    }

    /// Returns whether huge heap IDs store object addresses and lengths at `widths`.
    pub fn huge_ids_direct_with_widths(&self, widths: FormatWidths) -> bool {
        if self.filtering.is_filtered() {
            let avail = usize::from(self.heap_id_length).saturating_sub(1);
            let offset_size = usize::from(widths.offsets.get());
            let length_size = usize::from(widths.lengths.get());
            avail >= offset_size + length_size + 4 + length_size
        } else {
            Self::unfiltered_huge_ids_direct(self.heap_id_length, widths)
        }
    }

    /// Returns `true` if an unfiltered heap ID can hold its prefix, address, and length.
    ///
    /// The fields are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    pub(crate) fn unfiltered_huge_ids_direct(heap_id_length: u16, widths: FormatWidths) -> bool {
        usize::from(heap_id_length).saturating_sub(HEAP_ID_PREFIX_BYTES)
            >= usize::from(widths.offsets.get()) + usize::from(widths.lengths.get())
    }
}

/// The immutable decoding parameters for the IDs of a fractal heap.
///
/// The parameters are derived from a [`FractalHeapHeader`] and the file's address and length
/// widths. [`parse`](Self::parse) decodes supplied bytes without reading heap blocks or the
/// huge-object B-tree. The heap IDs are defined in "Fractal Heap" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug)]
pub struct FractalHeapIdLayout {
    encoded_len: usize,
    managed: Option<ManagedHeapIdLayout>,
    huge: HugeIdLayout,
}

impl FractalHeapIdLayout {
    /// Derives decoding parameters with the [`HugeIndirectKeyEncoding::Libhdf5`] key width.
    ///
    /// `offsets` and `lengths` are the file's address and length widths. See
    /// [`with_huge_key_encoding`](Self::with_huge_key_encoding) to select another key dialect.
    pub fn new(header: &FractalHeapHeader, offsets: OffsetWidth, lengths: LengthWidth) -> Self {
        Self::with_huge_key_encoding(header, offsets, lengths, HugeIndirectKeyEncoding::Libhdf5)
    }

    /// Derives decoding parameters with an explicit indirect huge-key width dialect.
    ///
    /// `offsets` and `lengths` are the file's address and length widths. The dialect applies to
    /// filtered and unfiltered indirect IDs. Direct IDs store their object's location instead.
    /// Invalid huge-object geometry is reported when [`parse`](Self::parse) reads a huge ID.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_core::StoredAddress;
    /// # use hdf5_pure_core::__private::StoredAddressExt;
    /// # use hdf5_pure_format::__private::{
    /// #     FractalHeapHeader, FractalHeapIdKind, FractalHeapIdLayout, HugeIndirectKeyEncoding,
    /// #     HugeObjectReference, LengthWidth, OffsetWidth,
    /// # };
    /// # use hdf5_pure_format::FormatError;
    /// # use test_util::{fractal_heap, widths::Widths};
    /// # let mut heap = FractalHeapHeader::parse_bytes(
    /// #     &fractal_heap::Header::new(256).build(Widths::EIGHT),
    /// #     OffsetWidth::Eight,
    /// #     LengthWidth::Eight,
    /// # )?;
    /// # heap.btree_huge_objects_address = StoredAddress::new(512);
    /// let ids = FractalHeapIdLayout::with_huge_key_encoding(
    ///     &heap,
    ///     OffsetWidth::Eight,
    ///     LengthWidth::Two,
    ///     HugeIndirectKeyEncoding::PublishedSpec,
    /// );
    /// let bytes = [0x10, 5, 1, 0, 0, 0, 0];
    /// let FractalHeapIdKind::Huge(HugeObjectReference::Indexed(key)) = ids.parse(&bytes)?.kind()
    /// else {
    ///     unreachable!();
    /// };
    /// assert_eq!(key.value(), 261);
    /// assert_eq!(key.encoding(), HugeIndirectKeyEncoding::PublishedSpec);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn with_huge_key_encoding(
        header: &FractalHeapHeader,
        offsets: OffsetWidth,
        lengths: LengthWidth,
        encoding: HugeIndirectKeyEncoding,
    ) -> Self {
        let encoded_len = usize::from(header.heap_id_length);
        let huge = HugeIdLayout {
            direct: header.huge_ids_direct_with_widths(FormatWidths { offsets, lengths }),
            offsets,
            lengths,
            tracker: header.btree_huge_objects_address,
            filter_count: match &header.filtering {
                FractalHeapFiltering::Unfiltered => None,
                FractalHeapFiltering::Filtered { pipeline, .. } => Some(pipeline.filters().len()),
            },
            encoding,
            key_width: match encoding {
                HugeIndirectKeyEncoding::PublishedSpec => usize::from(lengths.get()),
                // H5HF__huge_init in H5HFhuge.c, HDF5 2.2.0, bounds the indirect key by
                // the ID payload length and the eight-byte native huge ID.
                HugeIndirectKeyEncoding::Libhdf5 => encoded_len.saturating_sub(1).min(8),
            },
        };
        Self {
            encoded_len,
            managed: ManagedHeapIdLayout::new(header),
            huge,
        }
    }

    /// Parses one version 0 heap ID at the width declared by the heap header.
    ///
    /// Checks the ID's length, version, type, and any reserved prefix bits before reading its
    /// type-specific fields. Reads a managed object's offset and length as byte-aligned fields
    /// and ignores any padding after them.
    ///
    /// Borrows a tiny object's bytes from `bytes`. The object starts after a one-byte prefix
    /// for ID widths of at most 18 bytes, and a two-byte prefix for larger widths. Ignores any
    /// bytes after the object.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapIdLength`] if `bytes` has a different length
    /// from [`encoded_len`](Self::encoded_len), before reading its contents. Returns
    /// [`FormatError::UnexpectedEof`] if the supplied and declared ID lengths are both zero.
    ///
    /// Returns [`FormatError::UnsupportedFractalHeapIdVersion`] if the version is not 0,
    /// [`FormatError::InvalidHeapIdType`] if the type bits hold 3, or
    /// [`FormatError::InvalidFractalHeapIdFlags`] if a managed or huge ID has set reserved bits.
    /// Checks these conditions in that order.
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a tiny object's length extends past the ID.
    /// The error reports the required object end from the start of `bytes` and the supplied
    /// ID length.
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] for a managed ID if its header cannot
    /// describe the required fields, or its offset requires more bits than the heap permits.
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] for a huge ID if the tracker address
    /// is undefined or exceeds the address width, a required size is zero, the filter count or
    /// mask is invalid, or the encoded and decoded sizes differ with all filters skipped. Also
    /// returns this error if an indirect key's field exceeds the declared ID payload.
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a huge object's encoded range exceeds
    /// [`u64::MAX`], and [`FormatError::ValueTooLargeForRepresentation`] if an indirect key
    /// exceeds the tracker's numeric domain.
    pub fn parse<'a>(&self, bytes: &'a [u8]) -> Result<FractalHeapIdView<'a>, FormatError> {
        if bytes.len() != self.encoded_len {
            return Err(FormatError::InvalidFractalHeapIdLength {
                expected: self.encoded_len,
                actual: bytes.len(),
            });
        }
        let mut cursor = bytes::ByteCursor::new(bytes);
        let first = cursor.u8()?;
        Ok(FractalHeapIdView {
            kind: match FractalHeapIdType::from_first_byte(first)? {
                FractalHeapIdType::Managed => self
                    .managed
                    .ok_or(FormatError::InvalidFractalHeapGeometry)?
                    .parse(cursor.remaining())?,
                FractalHeapIdType::Huge => {
                    FractalHeapIdKind::Huge(self.parse_huge(cursor.remaining())?)
                }
                FractalHeapIdType::Tiny => FractalHeapIdKind::Tiny {
                    bytes: TinyIdForm::for_width(self.encoded_len).parse(first, &mut cursor)?,
                },
            },
        })
    }

    /// Returns the exact ID length in bytes declared by the heap header.
    pub const fn encoded_len(&self) -> usize {
        self.encoded_len
    }

    fn parse_huge(&self, payload: &[u8]) -> Result<HugeObjectReference, FormatError> {
        let HugeIdLayout {
            direct,
            offsets,
            lengths,
            tracker,
            filter_count,
            encoding,
            key_width,
        } = self.huge;
        if tracker.get() >= StoredAddress::undefined(offsets).get() {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(if direct {
            if let Some(filter_count) = filter_count {
                HugeObjectReference::FilteredInline(FilteredHugeObjectLocation::parse(
                    payload,
                    offsets,
                    lengths,
                    filter_count,
                )?)
            } else {
                let addr = StoredAddress::new(bytes::read_offset_width(payload, 0, offsets)?);
                let len = bytes::read_length_width(payload, usize::from(offsets.get()), lengths)?;
                if addr.get() >= StoredAddress::undefined(offsets).get() || len == 0 {
                    return Err(FormatError::InvalidFractalHeapGeometry);
                }
                addr.get()
                    .checked_add(len)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: addr.get(),
                        length: len,
                    })?;
                HugeObjectReference::Inline { addr, len }
            }
        } else {
            if key_width == 0 || key_width > payload.len() {
                return Err(FormatError::InvalidFractalHeapGeometry);
            }
            let key = bytes::ByteCursor::new(payload).uint(key_width)?;
            if !crate::width::length_width_holds(lengths, key) {
                return Err(FormatError::ValueTooLargeForRepresentation {
                    value: key,
                    target: "huge-object tracker key",
                });
            }
            let key = HugeIndirectKey {
                value: key,
                encoding,
            };
            if filter_count.is_some() {
                HugeObjectReference::FilteredIndexed(key)
            } else {
                HugeObjectReference::Indexed(key)
            }
        })
    }
}

/// A checked version 0 tiny fractal-heap ID encoding that borrows the object's bytes.
///
/// Both forms store the object length minus one. At ID widths up to 18 bytes, the normal
/// form stores it in the first byte's low four bits. At larger ID widths, the extended
/// form stores the high four bits there and the low eight bits in the second byte.
/// The object immediately follows the prefix.
///
/// The forms are defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug)]
pub struct TinyHeapIdEncoding<'a> {
    payload: &'a [u8],
    id_width: u16,
}

impl<'a> TinyHeapIdEncoding<'a> {
    /// Prepares a tiny ID at the heap's declared ID width using a borrowed payload.
    ///
    /// The normal form can represent up to 16 payload bytes. The extended form can
    /// represent up to 4096. The payload must also fit the bytes after the prefix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TinyHeapIdWidthTooSmall`] if `id_width` is 0 or 1,
    /// [`FormatError::EmptyTinyHeapIdPayload`] if `payload` is empty,
    /// [`FormatError::TinyHeapIdPayloadLengthTooLarge`] if the length exceeds the form's
    /// length field limit, or [`FormatError::TinyHeapIdPayloadExceedsCapacity`] if the
    /// payload exceeds the space after the prefix. Checks these conditions in that order.
    pub fn new(id_width: u16, payload: &'a [u8]) -> Result<Self, FormatError> {
        let width = usize::from(id_width);
        let form = TinyIdForm::for_width(width);
        let capacity = width.saturating_sub(form.prefix_len());
        if capacity == 0 {
            return Err(FormatError::TinyHeapIdWidthTooSmall(id_width));
        }
        let length = payload.len();
        if length == 0 {
            return Err(FormatError::EmptyTinyHeapIdPayload);
        }
        let maximum = form.max_payload_len();
        if length > maximum {
            return Err(FormatError::TinyHeapIdPayloadLengthTooLarge { length, maximum });
        }
        if length > capacity {
            return Err(FormatError::TinyHeapIdPayloadExceedsCapacity { length, capacity });
        }
        Ok(Self { payload, id_width })
    }

    /// Returns the declared ID width in bytes, including padding after the object.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes one complete tiny ID to the start of `output` and returns its byte count.
    ///
    /// Writes zeros after the payload through the declared ID width and leaves any
    /// output suffix unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] with the required ID width and available
    /// bytes if `output` is too short. Leaves all of `output` unchanged on error.
    ///
    /// # Examples
    ///
    /// A 19-byte ID uses the extended form for a two-byte payload.
    ///
    /// ```
    /// use hdf5_pure_format::__private::TinyHeapIdEncoding;
    ///
    /// let id = TinyHeapIdEncoding::new(19, b"hi")?;
    /// let mut output = [0xa5; 20];
    /// assert_eq!(id.encoded_size(), 19);
    /// assert_eq!(id.write_into(&mut output)?, 19);
    /// assert_eq!(&output[..4], &[0x20, 1, b'h', b'i']);
    /// assert_eq!(&output[4..19], &[0; 15]);
    /// assert_eq!(output[19], 0xa5);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }
}

impl Encode for TinyHeapIdEncoding<'_> {
    type Error = bytes::UnexpectedEof;

    fn encoded_len(&self) -> usize {
        usize::from(self.id_width)
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let width = self.encoded_len();
        let available = output.len();
        let id = output
            .get_mut(..width)
            .ok_or_else(|| bytes::UnexpectedEof::new(width, available))?;
        let form = TinyIdForm::for_width(width);
        let (prefix, body) = id.split_at_mut(form.prefix_len());
        let (payload, padding) = body.split_at_mut(self.payload.len());
        form.write_prefix(prefix, self.payload.len());
        payload.copy_from_slice(self.payload);
        // `H5HF__tiny_insert` in `H5HFtiny.c`, HDF5 2.2.0, fills the remaining ID bytes with zeros.
        padding.fill(0);
        Ok(width)
    }
}

enum TinyIdForm {
    Extended,
    Normal,
}

impl TinyIdForm {
    fn for_width(width: usize) -> Self {
        if width <= TINY_NORMAL_ID_MAX_LEN {
            Self::Normal
        } else {
            Self::Extended
        }
    }

    fn prefix_len(&self) -> usize {
        match self {
            Self::Normal => 1,
            Self::Extended => 2,
        }
    }

    fn max_payload_len(&self) -> usize {
        match self {
            Self::Normal => TINY_NORMAL_PAYLOAD_MAX_LEN,
            Self::Extended => TINY_EXTENDED_PAYLOAD_MAX_LEN,
        }
    }

    /// Reads the object with `cursor` positioned immediately after `first` in the ID.
    ///
    /// # Errors
    ///
    /// Returns [`bytes::UnexpectedEof`] if the extended length byte or the object is incomplete.
    fn parse<'a>(
        &self,
        first: u8,
        cursor: &mut bytes::ByteCursor<'a>,
    ) -> Result<&'a [u8], bytes::UnexpectedEof> {
        let length_bits = usize::from(first & TINY_LENGTH_MASK);
        let encoded_length = match self {
            Self::Normal => length_bits,
            Self::Extended => (length_bits << u8::BITS) | usize::from(cursor.u8()?),
        };
        cursor.take(encoded_length + 1)
    }

    /// Writes the version, type, and length into exactly [`Self::prefix_len`] bytes.
    ///
    /// The caller checks that `payload_len` is positive and fits the form's length field.
    fn write_prefix(&self, prefix: &mut [u8], payload_len: usize) {
        let [low, high, ..] = (payload_len - 1).to_le_bytes();
        let tag = FractalHeapIdType::Tiny.first_byte();
        match self {
            Self::Normal => prefix.copy_from_slice(&[tag | low]),
            Self::Extended => prefix.copy_from_slice(&[tag | high, low]),
        }
    }
}

/// The encoded byte widths and offset bit count of a managed heap ID.
///
/// The length field follows the whole-byte offset field. libhdf5 computes these widths in
/// `H5HF_hdr_finish_init_phase1` in `H5HFhdr.c` (HDF5 1.10.11).
#[derive(Clone, Copy, Debug)]
struct ManagedHeapIdLayout {
    encoded_len: usize,
    offset_bits: u32,
    offset_bytes: usize,
    length_bytes: usize,
}

impl ManagedHeapIdLayout {
    fn new(header: &FractalHeapHeader) -> Option<Self> {
        if !(1..=64).contains(&header.max_heap_size)
            || !header.max_direct_block_size.is_power_of_two()
            || header.max_managed_object_size == 0
        {
            return None;
        }
        let offset_bits = header.max_heap_size;
        let offset_bytes = usize::from(offset_bits).div_ceil(8);
        // H5HF_hdr_finish_init_phase1 in H5HFhdr.c, HDF5 1.10.11, bounds the
        // length field by both the direct-block offset width and managed-object limit.
        let direct_bits = header.max_direct_block_size.trailing_zeros();
        let managed_bits = u32::BITS - header.max_managed_object_size.leading_zeros();
        let length_bytes =
            usize::try_from(direct_bits.div_ceil(8).min(managed_bits.div_ceil(8))).ok()?;
        let encoded_len = usize::from(header.heap_id_length);
        if length_bytes == 0
            || encoded_len < offset_bytes.checked_add(length_bytes)?.checked_add(1)?
        {
            return None;
        }
        Some(Self {
            encoded_len,
            offset_bits: u32::from(offset_bits),
            offset_bytes,
            length_bytes,
        })
    }

    fn encode(&self, heap_offset: u64, length: u64) -> Result<Vec<u8>, FormatError> {
        if !fits_bits(heap_offset, self.offset_bits) {
            return Err(FormatError::Internal(format!(
                "heap offset {heap_offset} does not fit the heap's {}-bit offsets",
                self.offset_bits
            )));
        }
        let length_bytes = self.length_bytes;
        if length_bytes < 8
            && !fits_bits(
                length,
                u32::try_from(length_bytes * 8)
                    .map_err(|_conversion_error| FormatError::InvalidFractalHeapGeometry)?,
            )
        {
            return Err(FormatError::Internal(format!(
                "a managed object length does not fit a {}-byte heap ID",
                self.encoded_len
            )));
        }
        let mut bytes = Vec::with_capacity(self.encoded_len);
        bytes.push(FractalHeapIdType::Managed.first_byte());
        bytes.extend(
            heap_offset
                .to_le_bytes()
                .into_iter()
                .take(self.offset_bytes),
        );
        bytes.extend(length.to_le_bytes().into_iter().take(length_bytes));
        bytes.resize(self.encoded_len, 0);
        Ok(bytes)
    }

    fn parse(&self, payload: &[u8]) -> Result<FractalHeapIdKind<'static>, FormatError> {
        let mut cursor = bytes::ByteCursor::new(payload);
        let heap_offset = cursor.uint(self.offset_bytes)?;
        if !fits_bits(heap_offset, self.offset_bits) {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(FractalHeapIdKind::Managed {
            heap_offset,
            object_length: cursor.uint(self.length_bytes)?,
        })
    }
}

/// A decoded version 0 heap ID.
///
/// [`FractalHeapIdLayout::parse`] checks the encoded length and prefix before reading the
/// type-specific fields. Only a tiny object's bytes are borrowed from the encoded ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIdView<'a> {
    kind: FractalHeapIdKind<'a>,
}

impl<'a> FractalHeapIdView<'a> {
    /// Returns the decoded object location or the borrowed bytes of a tiny object.
    pub const fn kind(self) -> FractalHeapIdKind<'a> {
        self.kind
    }
}

/// The decoded fields of a heap ID, returned by [`FractalHeapIdView::kind`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapIdKind<'a> {
    /// The object is stored in the heap's managed blocks.
    Managed {
        /// The object's offset in the heap's address space.
        heap_offset: u64,
        /// The object's length in bytes.
        object_length: u64,
    },
    /// The object is stored outside the heap's managed blocks.
    Huge(HugeObjectReference),
    /// The encoded ID holds the object itself.
    Tiny {
        /// The object's bytes, borrowed from the encoded ID.
        bytes: &'a [u8],
    },
}

#[derive(Clone, Copy, Debug)]
struct HugeIdLayout {
    direct: bool,
    offsets: OffsetWidth,
    lengths: LengthWidth,
    tracker: StoredAddress,
    filter_count: Option<usize>,
    encoding: HugeIndirectKeyEncoding,
    key_width: usize,
}

fn fits_bits(value: u64, bits: u32) -> bool {
    value.checked_shr(bits).is_none_or(|high| high == 0)
}

/// The position of the type bits, bits 4 and 5, in the first byte of a heap ID, from the same
/// section as [`FractalHeapHeader`].
const HEAP_ID_TYPE_SHIFT: u8 = 4;

/// The position of the heap-ID version bits, from the same section as [`FractalHeapHeader`].
const HEAP_ID_VERSION_SHIFT: u8 = 6;
/// The heap-ID version defined in the same section as [`FractalHeapHeader`].
const HEAP_ID_VERSION: u8 = 0;
/// The reserved low four bits of managed and huge IDs, from the same section as
/// [`FractalHeapHeader`].
const HEAP_ID_RESERVED_MASK: u8 = 0x0F;

/// The mask of the two type bits once shifted down by [`HEAP_ID_TYPE_SHIFT`].
const HEAP_ID_TYPE_MASK: u8 = 0x03;

/// The type of the heap ID of a managed object, from the same section as
/// [`FractalHeapHeader`].
const HEAP_ID_TYPE_MANAGED: u8 = 0;

/// The type of the heap ID of a huge object, from the same section as [`FractalHeapHeader`].
const HEAP_ID_TYPE_HUGE: u8 = 1;

/// The type of the heap ID of a tiny object, from the same section as [`FractalHeapHeader`].
const HEAP_ID_TYPE_TINY: u8 = 2;

/// The largest heap ID width in bytes that uses the normal tiny ID form.
///
/// The normal form represents objects of up to 16 bytes, as defined in "Fractal Heap" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const TINY_NORMAL_ID_MAX_LEN: usize = 18;

// The low four bits hold the length less one in the normal form and its high four bits in the
// extended form ("Fractal Heap", specification 4.0).
const TINY_LENGTH_MASK: u8 = 0x0F;
const TINY_NORMAL_PAYLOAD_MAX_LEN: usize = TINY_LENGTH_MASK as usize + 1;
const TINY_EXTENDED_PAYLOAD_MAX_LEN: usize = TINY_NORMAL_PAYLOAD_MAX_LEN << u8::BITS;
/// The version-and-type byte of a heap ID, defined in "Fractal Heap" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const HEAP_ID_PREFIX_BYTES: usize = 1;

#[cfg(test)]
#[path = "tests/id.rs"]
mod tests;
