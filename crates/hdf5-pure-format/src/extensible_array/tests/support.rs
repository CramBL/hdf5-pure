use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use crate::address::{BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt};
use crate::chunk_record::{self, ChunkRecord, IndexSlots};
use crate::error::FormatError;
use crate::extensible_array::{
    self, ExtensibleArray, ExtensibleArrayDataBlock, ExtensibleArrayDataBlockRef,
    ExtensibleArrayHeaderLocator, ExtensibleArrayIndexBlock, ExtensibleArrayPageRef,
    ExtensibleArraySuperBlockRef,
};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

/// Returns the length of a header: 12 fixed bytes, six length-sized statistics, the index
/// block address, and the checksum.
pub(super) const fn eahd_len(os: u8, ls: u8) -> usize {
    12 + 6 * ls as usize + os as usize + 4
}

/// Returns the records of the Extensible Array whose header is at `header_offset` in
/// `file_data`.
pub(super) fn records_at(
    file_data: &[u8],
    header_offset: usize,
    chunk_byte_size: u64,
) -> Vec<(u64, ChunkRecord)> {
    let image = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), file_data.len() as u64).unwrap(),
        file_data,
    )
    .unwrap();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let locator = ExtensibleArrayHeaderLocator::new(
        widths,
        BaseAddress::ZERO,
        StoredAddress::new(header_offset as u64),
    );
    records_in(image, locator, chunk_byte_size).unwrap()
}

/// Returns the records of unfiltered chunks of `chunk_byte_size` bytes stored back to back from
/// `base_addr`, one per slot of `slots`.
pub(super) fn back_to_back(
    slots: core::ops::Range<u64>,
    base_addr: u64,
    chunk_byte_size: u64,
) -> Vec<(u64, ChunkRecord)> {
    slots
        .map(|slot| {
            let record = ChunkRecord {
                address: StoredAddress::new(base_addr + slot * chunk_byte_size),
                stored_size: chunk_byte_size,
                filter_mask: 0,
            };
            (slot, record)
        })
        .collect()
}

pub(super) const WIDTHS: FormatWidths = FormatWidths {
    offsets: OffsetWidth::Eight,
    lengths: LengthWidth::Eight,
};

pub(super) const ARRAY_AT: u64 = 0x80;

pub(super) const FIRST_PAGED_SLOT: u64 = 131_060;

// "The Extensible Array Index" in format specification, version 4.0, defines the version 0
// header field order and the four-byte filter mask for version 4 chunked layouts.
pub(super) const HEADER_MINIMUM_POINTERS: usize = 10;
pub(super) const FILTER_MASK_BYTES: u8 = 4;

/// The Extensible Array the writer builds at `ARRAY_AT`, in an image whose first byte is at
/// physical address `base`.
pub(super) struct Fixture {
    pub(super) base: u64,
    widths: FormatWidths,
    pub(super) bytes: Vec<u8>,
}

impl Fixture {
    pub(super) fn new(n: u64, has_filters: bool) -> Self {
        Self::at(0, WIDTHS, 8, n, has_filters)
    }

    pub(super) fn at(
        base: u64,
        widths: FormatWidths,
        chunk_bytes: u64,
        n: u64,
        has_filters: bool,
    ) -> Self {
        Self::with_slots(
            base,
            widths,
            chunk_bytes,
            &IndexSlots::dense(&chunk_record::chunks(chunk_bytes, n, has_filters)),
            has_filters,
        )
    }

    pub(super) fn with_slots(
        base: u64,
        widths: FormatWidths,
        chunk_bytes: u64,
        slots: &IndexSlots<'_>,
        has_filters: bool,
    ) -> Self {
        let array = extensible_array::build_extensible_array_at(
            slots,
            chunk_bytes,
            widths.offsets,
            widths.lengths,
            has_filters,
            StoredAddress::new(ARRAY_AT),
        )
        .unwrap();
        let mut bytes = vec![0; usize::try_from(ARRAY_AT).unwrap()];
        bytes.extend_from_slice(&array);
        Self {
            base,
            widths,
            bytes,
        }
    }

    pub(super) fn locator(&self) -> ExtensibleArrayHeaderLocator {
        ExtensibleArrayHeaderLocator::new(
            self.widths,
            BaseAddress::new(self.base),
            StoredAddress::new(ARRAY_AT),
        )
    }

    fn image(&self) -> FileRegion<'_> {
        FileRegion::new(
            FileRange::new(PhysicalAddress::new(self.base), self.bytes.len() as u64).unwrap(),
            &self.bytes,
        )
        .unwrap()
    }

    pub(super) fn read(&self, range: FileRange) -> Result<FileRegion<'_>, FormatError> {
        self.image().subregion(range)
    }

    pub(super) fn header(&self) -> Result<ExtensibleArray, FormatError> {
        let locator = self.locator();
        locator.parse(self.read(locator.parse_range()?)?)
    }

    pub(super) fn index_block(&self) -> ExtensibleArrayIndexBlock<'_> {
        let block = self.header().unwrap().index_block().unwrap();
        block
            .parse(self.read(block.parse_range()).unwrap())
            .unwrap()
    }

    pub(super) fn super_blocks(&self) -> Vec<ExtensibleArraySuperBlockRef> {
        self.index_block()
            .super_blocks()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    pub(super) fn data_blocks(&self) -> Vec<ExtensibleArrayDataBlockRef<'_>> {
        let mut blocks: Vec<ExtensibleArrayDataBlockRef<'_>> = self
            .index_block()
            .data_blocks()
            .collect::<Result<_, _>>()
            .unwrap();
        for super_block in self.super_blocks() {
            let super_block = super_block
                .parse(self.read(super_block.parse_range()).unwrap())
                .unwrap();
            for block in super_block.data_blocks() {
                blocks.push(block.unwrap());
            }
        }
        blocks
    }

    pub(super) fn paged_data_blocks(&self) -> Vec<ExtensibleArrayDataBlockRef<'_>> {
        self.data_blocks()
            .into_iter()
            .filter(|block| block.parse_range() != block.range())
            .collect()
    }

    pub(super) fn pages(
        &self,
        block: ExtensibleArrayDataBlockRef<'_>,
    ) -> Vec<ExtensibleArrayPageRef> {
        let ExtensibleArrayDataBlock::Pages(pages) = block
            .parse(self.read(block.parse_range()).unwrap())
            .unwrap()
        else {
            panic!("the data block {block:?} is not paged");
        };
        pages.pages().collect::<Result<_, _>>().unwrap()
    }

    pub(super) fn local(&self, at: PhysicalAddress) -> usize {
        usize::try_from(at.get() - self.base).unwrap()
    }

    pub(super) fn restamp(&mut self, range: FileRange) {
        let at = self.local(range.start());
        let len = usize::try_from(range.length()).unwrap();
        test_util::checksum::restamp(&mut self.bytes, at, len);
    }

    pub(super) fn records(
        &self,
        chunk_byte_size: u64,
    ) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        records_in(self.image(), self.locator(), chunk_byte_size)
    }
}

/// Returns the records of the index block and of each block and page that holds a slot below
/// the header's "Max Index Set", each parsed from the range its reference returns.
pub(in crate::extensible_array) fn records_in(
    image: FileRegion<'_>,
    locator: ExtensibleArrayHeaderLocator,
    chunk_byte_size: u64,
) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
    let read = |range: FileRange| image.subregion(range);
    let Some(index_block) = locator.parse(read(locator.parse_range()?)?)?.index_block() else {
        return Ok(Vec::new());
    };
    let index_block = index_block.parse(read(index_block.parse_range())?)?;
    let mut records: Vec<(u64, ChunkRecord)> = index_block
        .elements()
        .records(chunk_byte_size)
        .collect::<Result<_, _>>()?;
    let mut visit = |block: ExtensibleArrayDataBlockRef<'_>| {
        match block.parse(read(block.parse_range())?)? {
            ExtensibleArrayDataBlock::Elements(elements) => {
                for record in elements.records(chunk_byte_size) {
                    records.push(record?);
                }
            }
            ExtensibleArrayDataBlock::Pages(pages) => {
                for page in pages.pages() {
                    let page = page?;
                    for record in page
                        .parse(read(page.parse_range())?)?
                        .records(chunk_byte_size)
                    {
                        records.push(record?);
                    }
                }
            }
        }
        Ok::<_, FormatError>(())
    };
    for block in index_block.data_blocks() {
        let block = block?;
        if block.holds_set_slots() {
            visit(block)?;
        }
    }
    for super_block in index_block.super_blocks() {
        let super_block = super_block?;
        if !super_block.holds_set_slots() {
            continue;
        }
        for block in super_block
            .parse(read(super_block.parse_range())?)?
            .data_blocks()
        {
            let block = block?;
            if block.holds_set_slots() {
                visit(block)?;
            }
        }
    }
    Ok(records)
}
