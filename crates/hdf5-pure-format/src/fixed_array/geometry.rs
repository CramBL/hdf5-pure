use super::data_block;
use super::framing::CHECKSUM_LEN;
use crate::chunk_record::ElementEncoding;
use crate::chunk_record::InvalidElementEncoding;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The element encoding, the entry count, and the paging of a Fixed Array, which its header fixes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Geometry {
    encoding: ElementEncoding,
    /// The number of elements in the array, the maximum number of entries the header stores.
    entries: u64,
    /// `None` if the data block holds the elements itself.
    paging: Option<Paging>,
}

impl Geometry {
    pub(super) fn new(
        encoding: ElementEncoding,
        entries: u64,
        page_bits: u8,
    ) -> Result<Self, FormatError> {
        let page_entries = 1u64
            .checked_shl(u32::from(page_bits))
            .ok_or(FormatError::InvalidFixedArrayGeometry)?;
        Ok(Self {
            encoding,
            entries,
            paging: (entries > page_entries).then(|| Paging {
                page_entries,
                pages: entries.div_ceil(page_entries),
            }),
        })
    }

    pub(super) const fn encoding(&self) -> ElementEncoding {
        self.encoding
    }

    pub(super) const fn entries(&self) -> u64 {
        self.entries
    }

    pub(super) const fn paging(&self) -> Option<Paging> {
        self.paging
    }

    pub(super) fn data_block_len(&self) -> Result<u64, FormatError> {
        let Self {
            encoding,
            entries,
            paging,
        } = *self;
        let elements = product(entries, u64::from(encoding.stride().get()))?;
        match paging {
            Some(paging) => sum([
                paging.pages_offset(encoding.offsets())?,
                elements,
                product(paging.pages, CHECKSUM_LEN)?,
            ]),
            None => sum([
                data_block::data_block_prefix_len(encoding.offsets()),
                elements,
                CHECKSUM_LEN,
            ]),
        }
    }
}

pub(super) fn element_encoding(
    offsets: OffsetWidth,
    client: u8,
    entry_size: u8,
) -> Result<ElementEncoding, FormatError> {
    ElementEncoding::new(offsets, client, entry_size).map_err(|error| match error {
        InvalidElementEncoding::Client(client) => FormatError::InvalidFixedArrayClient(client),
        InvalidElementEncoding::EntrySize => FormatError::InvalidFixedArrayGeometry,
    })
}

/// The pages of a paged Fixed Array data block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Paging {
    /// The number of elements in a page, 2 to the power of the header's page bits. The last page
    /// holds the remaining elements, and may hold fewer.
    page_entries: u64,
    pages: u64,
}

impl Paging {
    pub(super) const fn pages(self) -> u64 {
        self.pages
    }

    pub(super) const fn page_entries(self) -> u64 {
        self.page_entries
    }

    /// Returns the offset of the first page from the start of the data block, past the prefix,
    /// the page-init bitmap, and its checksum.
    pub(super) fn pages_offset(self, offsets: OffsetWidth) -> Result<u64, FormatError> {
        sum([
            data_block::data_block_prefix_len(offsets),
            self.pages.div_ceil(8),
            CHECKSUM_LEN,
        ])
    }
}

/// Returns `a * b`, or [`FormatError::InvalidFixedArrayGeometry`] if the product overflows.
pub(super) fn product(a: u64, b: u64) -> Result<u64, FormatError> {
    a.checked_mul(b)
        .ok_or(FormatError::InvalidFixedArrayGeometry)
}

/// Returns the sum of `terms`, or [`FormatError::InvalidFixedArrayGeometry`] if it overflows.
pub(super) fn sum<const N: usize>(terms: [u64; N]) -> Result<u64, FormatError> {
    terms
        .into_iter()
        .try_fold(0, u64::checked_add)
        .ok_or(FormatError::InvalidFixedArrayGeometry)
}
