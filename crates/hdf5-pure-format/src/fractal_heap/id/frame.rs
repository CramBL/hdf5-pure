use super::huge::{HugeIdLayout, HugeIndirectKeyEncoding, HugeObjectReference};
use super::managed::ManagedHeapIdLayout;
use super::tiny::TinyIdForm;
use crate::bytes;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::width::{LengthWidth, OffsetWidth};

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
    /// #     &fractal_heap::Header::new(256).build(Widths::new(8, 2)),
    /// #     OffsetWidth::Eight,
    /// #     LengthWidth::Two,
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
        Self {
            encoded_len,
            managed: ManagedHeapIdLayout::new(header),
            huge: HugeIdLayout::new(header, offsets, lengths, encoding),
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
    /// is undefined or exceeds the address width. An indirect ID returns this error if its
    /// key field is empty or exceeds the ID payload. An inline location returns it if the object
    /// address is undefined or a size is zero. For a filtered inline location, the filter count
    /// must fit the mask, each set bit must identify a filter, and the encoded and decoded sizes
    /// must agree when all filters are skipped.
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
                    FractalHeapIdKind::Huge(self.huge.parse(cursor.remaining())?)
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

/// The type of a fractal heap ID: where the heap stores the object the ID refers to.
///
/// Bits 4 and 5 of the first byte of a heap ID hold the type. The heap IDs are defined in "Fractal
/// Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FractalHeapIdType {
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
    pub(super) const fn first_byte(self) -> u8 {
        let type_bits = match self {
            Self::Managed => HEAP_ID_TYPE_MANAGED,
            Self::Huge => HEAP_ID_TYPE_HUGE,
            Self::Tiny => HEAP_ID_TYPE_TINY,
        };
        type_bits << HEAP_ID_TYPE_SHIFT
    }
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

#[cfg(test)]
#[path = "tests/frame.rs"]
mod tests;
