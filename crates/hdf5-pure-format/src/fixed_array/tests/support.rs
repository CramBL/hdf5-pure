use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use test_util::checksum;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_record;
use crate::chunk_record::ChunkRecord;
use crate::chunk_record::IndexSlots;
use crate::error::FormatError;
use crate::fixed_array;
use crate::fixed_array::FixedArrayDataBlock;
use crate::fixed_array::FixedArrayDataBlockRef;
use crate::fixed_array::FixedArrayHeader;
use crate::fixed_array::FixedArrayHeaderLocator;
use crate::fixed_array::FixedArrayPageRef;
use crate::fixed_array::header;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

pub(super) const WIDTHS: FormatWidths = FormatWidths {
    offsets: OffsetWidth::Eight,
    lengths: LengthWidth::Eight,
};

pub(super) const ARRAY_AT: u64 = 0x80;

/// The Fixed Array the writer builds at `ARRAY_AT`, in an image whose first byte is at
/// physical address `base`.
pub(super) struct Fixture {
    base: u64,
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
        let array = fixed_array::build_fixed_array_at(
            &IndexSlots::dense(&chunk_record::chunks(chunk_bytes, n, has_filters)),
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

    pub(super) fn locator(&self) -> FixedArrayHeaderLocator {
        FixedArrayHeaderLocator::new(
            self.widths,
            BaseAddress::new(self.base),
            StoredAddress::new(ARRAY_AT),
        )
    }

    pub(super) fn image(&self) -> FileRegion<'_> {
        FileRegion::new(
            FileRange::new(PhysicalAddress::new(self.base), self.bytes.len() as u64).unwrap(),
            &self.bytes,
        )
        .unwrap()
    }

    pub(super) fn header(&self) -> Result<FixedArrayHeader, FormatError> {
        let locator = self.locator();
        locator.parse(self.image().subregion(locator.parse_range()?)?)
    }

    pub(super) fn data_block(&self) -> FixedArrayDataBlockRef {
        self.header().unwrap().data_block().unwrap()
    }

    pub(super) fn pages(&self) -> Vec<FixedArrayPageRef> {
        let block = self.data_block();
        let region = self
            .image()
            .subregion(block.parse_range().unwrap())
            .unwrap();
        let FixedArrayDataBlock::PageBitmap(bitmap) = block.parse(region).unwrap() else {
            panic!("the data block of {block:?} is not paged");
        };
        bitmap.pages().collect::<Result<_, _>>().unwrap()
    }

    pub(super) fn local(&self, at: PhysicalAddress) -> usize {
        usize::try_from(at.get() - self.base).unwrap()
    }

    pub(super) fn header_at(&self) -> usize {
        usize::try_from(ARRAY_AT).unwrap()
    }

    pub(super) fn data_block_at(&self) -> usize {
        self.local(self.data_block().range().start())
    }

    pub(super) fn restamp_header(&mut self) {
        let at = self.header_at();
        let len = usize::try_from(header::header_len(self.widths)).unwrap();
        checksum::restamp(&mut self.bytes, at, len);
    }

    pub(super) fn restamp_prefix(&mut self) {
        let block = self.data_block();
        let at = self.local(block.range().start());
        let len = usize::try_from(block.parse_range().unwrap().length()).unwrap();
        checksum::restamp(&mut self.bytes, at, len);
    }

    pub(super) fn records(
        &self,
        chunk_byte_size: u64,
    ) -> Result<Vec<(u64, ChunkRecord)>, FormatError> {
        let image = self.image();
        let Some(block) = self.header()?.data_block() else {
            return Ok(Vec::new());
        };
        match block.parse(image.subregion(block.parse_range()?)?)? {
            FixedArrayDataBlock::Elements(elements) => elements.records(chunk_byte_size).collect(),
            FixedArrayDataBlock::PageBitmap(bitmap) => {
                let mut records = Vec::new();
                for page in bitmap.pages() {
                    let page = page?;
                    for record in page
                        .parse(image.subregion(page.parse_range())?)?
                        .records(chunk_byte_size)
                    {
                        records.push(record?);
                    }
                }
                Ok(records)
            }
        }
    }
}
