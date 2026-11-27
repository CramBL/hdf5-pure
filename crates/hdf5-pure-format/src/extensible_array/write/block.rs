//! Bounded encoding for Extensible Array blocks and pages.

use hdf5_pure_core::AddressRange;

use super::encode::Pointer;
use super::stored_block_range;
use crate::address::{DecodedStoredAddress, StoredAddress};
use crate::bytes::UnexpectedEof;
use crate::checksum;
use crate::chunk_record::{self, ArrayClient, ElementEncoding, IndexSlots, InvalidElementEncoding};
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::extensible_array::{
    self, DataBlock, ExtensibleArrayCreationParameters, Layout, SuperBlock,
};
use crate::width::OffsetWidth;

/// The creation geometry, element layout, and owning header address of an Extensible Array.
///
/// Prepares secondary blocks and data blocks for caller-provided storage. "The Extensible Array
/// Index" of the [format specification, version 4.0][spec] defines the version 0 block fields.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy)]
pub struct ExtensibleArrayBlockWriter {
    layout: Layout,
    owner: StoredAddress,
}

impl ExtensibleArrayBlockWriter {
    /// Uses the stored address of the owning header and the element layout for `client`.
    ///
    /// An unfiltered element contains one address. A filtered element also contains a stored size
    /// of one to eight bytes and a four-byte filter mask.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayClient`] if `client` is unrecognized, or
    /// [`FormatError::InvalidExtensibleArrayGeometry`] if `element_size` does not match the
    /// client and address width. Returns [`FormatError::AddressTooLarge`] if `owner` exceeds
    /// that width or [`FormatError::DefinedAddressIsUndefined`] if it equals the undefined
    /// address marker.
    pub fn new(
        creation: ExtensibleArrayCreationParameters,
        offsets: OffsetWidth,
        client: u8,
        element_size: u8,
        owner: StoredAddress,
    ) -> Result<Self, FormatError> {
        let encoding =
            ElementEncoding::new(offsets, client, element_size).map_err(|error| match error {
                InvalidElementEncoding::Client(client) => {
                    FormatError::InvalidExtensibleArrayClient(client)
                }
                InvalidElementEncoding::EntrySize => FormatError::InvalidExtensibleArrayGeometry,
            })?;
        if element_size != encoding.stride().get()
            || matches!(
                encoding.client(),
                ArrayClient::FilteredChunks { chunk_size_len: 0 }
            )
        {
            return Err(FormatError::InvalidExtensibleArrayGeometry);
        }
        let owner = DecodedStoredAddress::Defined(owner).to_stored(offsets)?;
        Ok(Self {
            layout: creation.layout(encoding),
            owner,
        })
    }

    /// Prepares the secondary block at its zero-based index in the index block's secondary table.
    ///
    /// Pass the full page initialization bitmap and child address table, or pass `None` for
    /// either to write zero bits or undefined addresses, respectively. Each initialized page
    /// requires a defined child address. The writer preserves unused bitmap bits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if the bitmap or child address table has the
    /// wrong length, or [`FormatError::InvalidExtensibleArrayGeometry`] if the index is outside
    /// the secondary table, a page is initialized without a child address, or geometry arithmetic
    /// overflows. Returns [`FormatError::ValueTooLargeForRepresentation`] if a defined child's
    /// logical offset exceeds the block offset width, or [`FormatError::ValueTooLargeForPlatform`]
    /// if the encoded block length does not fit [`usize`].
    ///
    /// Returns [`FormatError::DefinedAddressIsUndefined`] if a defined child address equals the
    /// undefined marker, [`FormatError::AddressTooLarge`] if a child's address or exclusive end
    /// exceeds the address width, or [`FormatError::OffsetOverflow`] if calculating the end
    /// overflows [`u64`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{OffsetWidth, StoredAddress};
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::__private::{
    ///     EA_CLIENT_ID_UNFILTERED, ExtensibleArrayBlockWriter, ExtensibleArrayCreationParameters,
    /// };
    ///
    /// let writer = ExtensibleArrayBlockWriter::new(
    ///     ExtensibleArrayCreationParameters::canonical(),
    ///     OffsetWidth::Four,
    ///     EA_CLIENT_ID_UNFILTERED,
    ///     4,
    ///     StoredAddress::new(1_024),
    /// )?;
    /// let block = writer.prepare_secondary_block(0, None, None)?;
    /// let mut bytes = vec![0; block.encoded_size()];
    /// assert_eq!(block.write_into(&mut bytes)?, bytes.len());
    /// assert_eq!(&bytes[..4], b"EASB");
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn prepare_secondary_block<'a>(
        self,
        index: u32,
        bitmap: Option<&'a [u8]>,
        children: Option<&'a [Option<StoredAddress>]>,
    ) -> Result<ExtensibleArrayBlockEncoding<'a>, FormatError> {
        let layout = self.layout;
        let index = layout
            .first_super_block()
            .checked_add(index)
            .filter(|&index| index < layout.super_blocks())
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        let block = layout.super_block(index)?;
        let length = layout.super_block_len(block)?.to_usize()?;
        let bitmap_len = layout.bitmap_len(block)?.to_usize()?;
        let count = block.blocks.to_usize()?;
        if let Some(bitmap) = bitmap {
            require_len(bitmap_len, bitmap.len())?;
        }
        if let Some(children) = children {
            require_len(count, children.len())?;
            for (local, address) in children.iter().enumerate() {
                if let Some(address) = address {
                    let data = block.data_block(local.to_u64())?;
                    self.require_block_offset(data.offset)?;
                    let (length, _) = self.data_extent(data)?;
                    stored_block_range(layout.encoding().offsets(), *address, length)?;
                }
            }
        }
        if let (Some(bitmap), Some(pages)) = (bitmap, layout.pages(block.block_elements)) {
            require_page_owners(bitmap, children.unwrap_or_default(), block.blocks, pages)?;
        }
        Ok(self.secondary(
            length,
            block,
            bitmap_len,
            bitmap.map_or(Bitmap::Zero, Bitmap::Provided),
            children.map_or(Addresses::Empty, Addresses::Provided),
        ))
    }

    /// Prepares a data block at `offset` relative to the end of the index block's inline slots.
    ///
    /// Reads records from `slots` using their array-wide slot indices. The writer encodes
    /// unoccupied slots and slots beyond the supplied span as undefined addresses. It ignores
    /// records outside the block's range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if `offset` exceeds the block
    /// offset width, or [`FormatError::InvalidExtensibleArrayGeometry`] if `offset` does not
    /// start a data block or geometry arithmetic overflows. Returns
    /// [`FormatError::ValueTooLargeForPlatform`] if the encoded block length does not fit
    /// [`usize`]. Returns [`FormatError::AddressTooLarge`],
    /// [`FormatError::DefinedAddressIsUndefined`], or [`FormatError::LengthTooLarge`] if a
    /// chunk record in the block does not fit its element fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{OffsetWidth, StoredAddress};
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_format::__private::{
    ///     EA_CLIENT_ID_UNFILTERED, ExtensibleArrayBlockWriter, ExtensibleArrayCreationParameters,
    ///     IndexSlots,
    /// };
    ///
    /// let writer = ExtensibleArrayBlockWriter::new(
    ///     ExtensibleArrayCreationParameters::canonical(),
    ///     OffsetWidth::Four,
    ///     EA_CLIENT_ID_UNFILTERED,
    ///     4,
    ///     StoredAddress::new(1_024),
    /// )?;
    /// let slots = IndexSlots::dense(&[]);
    /// let block = writer.prepare_data_block(0, &slots)?;
    /// let mut bytes = vec![0; block.encoded_size()];
    /// assert_eq!(block.write_into(&mut bytes)?, bytes.len());
    /// assert_eq!(&bytes[..4], b"EADB");
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn prepare_data_block<'a>(
        self,
        offset: u64,
        slots: &'a IndexSlots<'a>,
    ) -> Result<ExtensibleArrayBlockEncoding<'a>, FormatError> {
        let block = self.data_geometry(offset)?;
        let layout = self.layout;
        let (length, range) = self.data_extent(block)?;
        let length = length.to_usize()?;
        let mut next = slots.next_occupied(range.start);
        while let Some(slot) = next.filter(|&slot| slot < range.end) {
            if let Some(record) = slots.at(slot.to_usize()?) {
                layout.encoding().require_record_fits(record)?;
            }
            next = slots.next_occupied(slot + 1);
        }
        Ok(self.data(slots, range, offset, length))
    }

    fn data_geometry(self, offset: u64) -> Result<DataBlock, FormatError> {
        self.require_block_offset(offset)?;
        let layout = self.layout;
        for index in 0..layout.super_blocks() {
            let block = layout.super_block(index)?;
            let Some(relative) = offset.checked_sub(block.offset) else {
                break;
            };
            let local = relative / block.block_elements;
            if local < block.blocks && relative % block.block_elements == 0 {
                return block.data_block(local);
            }
        }
        Err(FormatError::InvalidExtensibleArrayGeometry)
    }

    fn require_block_offset(self, offset: u64) -> Result<(), FormatError> {
        if offset.to_le_bytes()[usize::from(self.layout.block_offset_len())..]
            .iter()
            .any(|&byte| byte != 0)
        {
            return Err(FormatError::ValueTooLargeForRepresentation {
                value: offset,
                target: "Extensible Array block offset",
            });
        }
        Ok(())
    }

    fn data_extent(self, block: DataBlock) -> Result<(u64, core::ops::Range<u64>), FormatError> {
        let length = self.layout.data_block_len(block)?;
        Ok((length, self.layout.slots(block.offset, block.elements)?))
    }

    /// Uses the layout and checked header address of a placement.
    pub(super) fn placed(layout: Layout, owner: StoredAddress) -> Self {
        Self { layout, owner }
    }

    /// Prepares an index block from the placement's checked inline records and child addresses.
    ///
    /// The caller passes the complete encoded length, including the checksum. Each inline record
    /// fits the element layout. Each child address fits its field and has a slot in the direct
    /// data block or secondary block address table.
    pub(super) fn index<'a>(
        self,
        length: usize,
        slots: &'a IndexSlots<'a>,
        addresses: Addresses<'a>,
    ) -> ExtensibleArrayBlockEncoding<'a> {
        let layout = self.layout;
        ExtensibleArrayBlockEncoding {
            writer: self,
            length,
            body: Body::Index {
                elements_len: usize::from(layout.inline_elements())
                    * usize::from(layout.encoding().stride().get()),
                slots,
                addresses,
            },
        }
    }

    /// Prepares a secondary block from checked geometry and child addresses.
    ///
    /// The caller passes the complete encoded length, including the checksum, and the exact bitmap
    /// length. The page initialization bits correspond to present children. Their addresses and
    /// offsets fit the selected widths.
    pub(super) fn secondary<'a>(
        self,
        length: usize,
        block: SuperBlock,
        bitmap_len: usize,
        bitmap: Bitmap<'a>,
        addresses: Addresses<'a>,
    ) -> ExtensibleArrayBlockEncoding<'a> {
        ExtensibleArrayBlockEncoding {
            writer: self,
            length,
            body: Body::Secondary {
                block,
                bitmap_len,
                bitmap,
                addresses,
            },
        }
    }

    /// Prepares a data block from a checked slot range, offset, and records.
    ///
    /// The caller passes a range that covers the block's elements and the complete encoded length.
    /// For a paged block, the length includes the checksummed prefix and every checksummed page.
    pub(super) fn data<'a>(
        self,
        slots: &'a IndexSlots<'a>,
        range: core::ops::Range<u64>,
        offset: u64,
        length: usize,
    ) -> ExtensibleArrayBlockEncoding<'a> {
        let page_elements = self.layout.page_nelmts().get();
        let count = range.end - range.start;
        let elements = Elements {
            slots,
            encoding: self.layout.encoding(),
            first: range.start,
        };
        ExtensibleArrayBlockEncoding {
            writer: self,
            length,
            body: match self.layout.pages(count) {
                Some(pages) => {
                    #[cfg_attr(
                        target_pointer_width = "32",
                        expect(
                            clippy::cast_possible_truncation,
                            reason = "the checked block length fits usize and includes at least one byte per page"
                        )
                    )]
                    let count = pages as usize;
                    let prefix_len = self.layout.pages_offset_usize();
                    Body::PagedData {
                        offset,
                        pages: Pages {
                            first: ExtensibleArrayPageEncoding {
                                elements,
                                length: (length - prefix_len) / count,
                            },
                            count,
                            page_elements,
                        },
                        prefix_len,
                    }
                }
                None => Body::Data { offset, elements },
            },
        }
    }

    fn write_prefix(self, output: &mut [u8], signature: [u8; 4], offset: Option<u64>) -> &mut [u8] {
        let (prefix, body) = output.split_at_mut(self.layout.prefix_len_usize());
        let (signature_field, fields) = prefix.split_at_mut(signature.len());
        signature_field.copy_from_slice(&signature);
        let (properties, address) = fields.split_at_mut(extensible_array::BLOCK_BYTE_FIELDS);
        properties.copy_from_slice(&[
            extensible_array::VERSION,
            self.layout.encoding().client().id(),
        ]);
        write_address(address, self.owner);
        if let Some(offset) = offset {
            let (field, body) = body.split_at_mut(usize::from(self.layout.block_offset_len()));
            field.copy_from_slice(&offset.to_le_bytes()[..field.len()]);
            body
        } else {
            body
        }
    }
}

/// A complete version 0 index, secondary, or data block prepared for encoding.
///
/// Borrows any chunk records, child addresses, and bitmap bytes from preparation. A paged data
/// block includes its prefix and all pages. The caller provides storage for the encoding.
pub struct ExtensibleArrayBlockEncoding<'a> {
    writer: ExtensibleArrayBlockWriter,
    length: usize,
    body: Body<'a>,
}

impl<'a> ExtensibleArrayBlockEncoding<'a> {
    /// Returns the exact number of bytes written, including every checksum.
    pub fn encoded_size(&self) -> usize {
        Encode::encoded_len(self)
    }

    /// Writes the complete block at the start of `output` and returns its length in bytes.
    ///
    /// Writes checksums regardless of the `checksum` feature and preserves any output suffix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is too short, leaving it unchanged.
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }

    /// Returns a page within this data block for separate encoding.
    ///
    /// `index` is zero-based within the block. Returns `None` if the block is unpaged, is
    /// not a data block, or has no page at that index.
    pub fn page(&self, index: usize) -> Option<ExtensibleArrayPageEncoding<'a>> {
        let Body::PagedData { pages, .. } = &self.body else {
            return None;
        };
        (index < pages.count).then(|| pages.at(index))
    }

    /// Writes a block into a slice of exactly its encoded size.
    ///
    /// # Panics
    ///
    /// Can panic if `output` has the wrong length.
    pub(super) fn write_block(&self, output: &mut [u8]) {
        let Self {
            writer,
            length: _,
            body,
        } = self;
        match body {
            Body::Index {
                elements_len,
                slots,
                addresses,
            } => {
                write_checksummed(output, |body| {
                    let body =
                        writer.write_prefix(body, extensible_array::INDEX_BLOCK_SIGNATURE, None);
                    let (elements, pointers) = body.split_at_mut(*elements_len);
                    Elements {
                        slots,
                        encoding: writer.layout.encoding(),
                        first: 0,
                    }
                    .write(elements);
                    addresses.write(pointers, writer.layout.encoding().offsets());
                });
            }
            Body::Secondary {
                block,
                bitmap_len,
                bitmap,
                addresses,
            } => {
                write_checksummed(output, |body| {
                    let body = writer.write_prefix(
                        body,
                        extensible_array::SUPER_BLOCK_SIGNATURE,
                        Some(block.offset),
                    );
                    let (bits, pointers) = body.split_at_mut(*bitmap_len);
                    bitmap.write(bits, writer.layout, *block);
                    addresses.write(pointers, writer.layout.encoding().offsets());
                });
            }
            Body::Data { offset, elements } => {
                write_checksummed(output, |body| {
                    let body = writer.write_prefix(
                        body,
                        extensible_array::DATA_BLOCK_SIGNATURE,
                        Some(*offset),
                    );
                    elements.write(body);
                });
            }
            Body::PagedData {
                offset,
                pages,
                prefix_len,
            } => {
                let (prefix, payload) = output.split_at_mut(*prefix_len);
                write_checksummed(prefix, |body| {
                    writer.write_prefix(
                        body,
                        extensible_array::DATA_BLOCK_SIGNATURE,
                        Some(*offset),
                    );
                });
                for (index, page) in payload.chunks_exact_mut(pages.first.length).enumerate() {
                    pages.at(index).write_page(page);
                }
            }
        }
    }
}

impl Encode for ExtensibleArrayBlockEncoding<'_> {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        self.length
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        self.write_block(destination(output, self.length)?);
        Ok(self.length)
    }
}

/// A data block page with borrowed chunk records and a checked encoded size.
#[derive(Clone, Copy)]
pub struct ExtensibleArrayPageEncoding<'a> {
    elements: Elements<'a>,
    length: usize,
}

impl ExtensibleArrayPageEncoding<'_> {
    /// Returns the exact length in bytes of the page, including its checksum.
    pub fn encoded_size(&self) -> usize {
        Encode::encoded_len(self)
    }

    /// Writes the page and its checksum at the start of `output` and returns the byte count.
    ///
    /// Writes the checksum regardless of the `checksum` feature and preserves any output suffix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is too short, leaving it unchanged.
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }

    /// Writes the page into a slice of exactly its encoded size.
    ///
    /// # Panics
    ///
    /// Can panic if `output` has the wrong length.
    fn write_page(self, output: &mut [u8]) {
        write_checksummed(output, |body| self.elements.write(body));
    }
}

impl Encode for ExtensibleArrayPageEncoding<'_> {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        self.length
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        self.write_page(destination(output, self.length)?);
        Ok(self.length)
    }
}

enum Body<'a> {
    Data {
        offset: u64,
        elements: Elements<'a>,
    },
    Index {
        elements_len: usize,
        slots: &'a IndexSlots<'a>,
        addresses: Addresses<'a>,
    },
    PagedData {
        offset: u64,
        pages: Pages<'a>,
        prefix_len: usize,
    },
    Secondary {
        block: SuperBlock,
        bitmap_len: usize,
        bitmap: Bitmap<'a>,
        addresses: Addresses<'a>,
    },
}

struct Pages<'a> {
    first: ExtensibleArrayPageEncoding<'a>,
    count: usize,
    page_elements: u64,
}

impl<'a> Pages<'a> {
    fn at(&self, index: usize) -> ExtensibleArrayPageEncoding<'a> {
        // The page lies in the checked total element range of its block.
        let mut page = self.first;
        page.elements.first += index.to_u64() * self.page_elements;
        page
    }
}

#[derive(Clone, Copy)]
struct Elements<'a> {
    slots: &'a IndexSlots<'a>,
    encoding: ElementEncoding,
    first: u64,
}

impl Elements<'_> {
    fn write(self, output: &mut [u8]) {
        let stride = usize::from(self.encoding.stride().get());
        for (element, slot) in output.chunks_exact_mut(stride).zip(self.first..) {
            let record = usize::try_from(slot)
                .ok()
                .and_then(|slot| self.slots.at(slot));
            self.encoding.write_element(element, record);
        }
    }
}

/// The source of the block's child address table.
pub(super) enum Addresses<'a> {
    Empty,
    Placed {
        children: &'a [Pointer],
        ranges: &'a [AddressRange<StoredAddress>],
    },
    Provided(&'a [Option<StoredAddress>]),
}

impl Addresses<'_> {
    fn write(&self, output: &mut [u8], width: OffsetWidth) {
        let undefined = StoredAddress::undefined(width);
        let fields = output.chunks_exact_mut(usize::from(width.get()));
        match self {
            Self::Empty => output.fill(undefined.get().to_le_bytes()[0]),
            Self::Placed { children, ranges } => {
                let mut children = children.iter().peekable();
                for (slot, field) in fields.enumerate() {
                    let address = children
                        .next_if(|child| child.slot == slot.to_u64())
                        .map_or(undefined, |child| ranges[child.descriptor].start());
                    write_address(field, address);
                }
            }
            Self::Provided(children) => {
                for (field, address) in fields.zip(*children) {
                    write_address(field, address.unwrap_or(undefined));
                }
            }
        }
    }
}

/// The source of a secondary block's page initialization bits.
pub(super) enum Bitmap<'a> {
    Occupied {
        slots: &'a IndexSlots<'a>,
        children: &'a [Pointer],
        first_slot: u64,
    },
    Provided(&'a [u8]),
    Zero,
}

impl Bitmap<'_> {
    fn write(&self, output: &mut [u8], layout: Layout, block: SuperBlock) {
        match self {
            Self::Occupied {
                slots,
                children,
                first_slot,
            } => {
                output.fill(0);
                if let Some(pages) = layout.pages(block.block_elements) {
                    write_page_bits(
                        output,
                        slots,
                        children,
                        *first_slot,
                        block.block_elements,
                        layout.page_nelmts().get(),
                        pages,
                    );
                }
            }
            Self::Provided(bitmap) => output.copy_from_slice(bitmap),
            Self::Zero => output.fill(0),
        }
    }
}

fn write_page_bits(
    output: &mut [u8],
    slots: &IndexSlots<'_>,
    children: &[Pointer],
    first_slot: u64,
    block_elements: u64,
    page_elements: u64,
    pages: u64,
) {
    for child in children {
        let start = first_slot + child.slot * block_elements;
        let end = start + block_elements;
        let mut next = slots.next_occupied(start);
        while let Some(slot) = next.filter(|&slot| slot < end) {
            let page = (slot - start) / page_elements;
            let bit = child.slot * pages + page;
            #[expect(
                clippy::expect_used,
                reason = "the checked bitmap length includes the bits of every child and page"
            )]
            let byte = usize::try_from(bit / 8).expect("page bit lies within the bitmap");
            output[byte] |= chunk_record::page_bit_mask(bit);
            next = slots.next_occupied(start + (page + 1) * page_elements);
        }
    }
}

fn require_len(expected: usize, actual: usize) -> Result<(), FormatError> {
    if expected != actual {
        return Err(FormatError::DataSizeMismatch { expected, actual });
    }
    Ok(())
}

fn require_page_owners(
    bitmap: &[u8],
    children: &[Option<StoredAddress>],
    child_count: u64,
    pages_per_child: u64,
) -> Result<(), FormatError> {
    let page_count = extensible_array::product(child_count, pages_per_child)?;
    let bits_per_byte = u64::from(u8::BITS);
    let page_bytes = page_count.div_ceil(bits_per_byte).to_usize()?;
    // Preparation acquires the bitmap length rounded up separately for each child.
    for (index, &byte) in bitmap[..page_bytes].iter().enumerate() {
        let first_page = index.to_u64() * bits_per_byte;
        let mut set = byte;
        while set != 0 {
            let bit = u64::from(set.leading_zeros());
            let page = first_page + bit;
            if page >= page_count {
                break;
            }
            let child = (page / pages_per_child).to_usize()?;
            if children.get(child).copied().flatten().is_none() {
                return Err(FormatError::InvalidExtensibleArrayGeometry);
            }
            set ^= chunk_record::page_bit_mask(bit);
        }
    }
    Ok(())
}

/// Borrows the complete output span before the caller writes any bytes.
///
/// # Errors
///
/// Returns [`UnexpectedEof`] if `output` is shorter than `length`.
pub(super) fn destination(output: &mut [u8], length: usize) -> Result<&mut [u8], UnexpectedEof> {
    let available = output.len();
    output
        .get_mut(..length)
        .ok_or_else(|| UnexpectedEof::new(length, available))
}

/// Passes the body to `write`, then stores its checksum in the final four bytes.
///
/// # Panics
///
/// Panics if `output` has fewer than four bytes.
pub(super) fn write_checksummed(output: &mut [u8], write: impl FnOnce(&mut [u8])) {
    let (body, stored) = output.split_at_mut(output.len() - extensible_array::CHECKSUM_LEN);
    write(body);
    stored.copy_from_slice(&checksum::jenkins_lookup3(body).to_le_bytes());
}

fn write_address(field: &mut [u8], address: StoredAddress) {
    field.copy_from_slice(&address.get().to_le_bytes()[..field.len()]);
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use rstest::rstest;
    use test_util::bytes;
    use test_util::checksum;
    use test_util::extensible_array as fixture;

    use super::*;
    use crate::address::StoredAddressExt;
    use crate::chunk_record::ChunkRecord;
    use crate::extensible_array::SlotOccupancy;
    use crate::extensible_array::write::ExtensibleArrayPlan;
    use crate::width::FormatWidths;

    #[rstest]
    #[case::unfiltered(
        2,
        4,
        fixture::Width::Two,
        fixture::Width::Four,
        fixture::Client::Unfiltered
    )]
    #[case::filtered(
        4,
        2,
        fixture::Width::Four,
        fixture::Width::Two,
        fixture::Client::Filtered(fixture::Width::Two)
    )]
    fn block_writers_match_independent_bytes_and_preserve_output_bounds(
        #[case] offsets: u8,
        #[case] lengths: u8,
        #[case] offset_fixture: fixture::Width,
        #[case] length_fixture: fixture::Width,
        #[case] client: fixture::Client,
    ) {
        let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
        let filtered = matches!(client, fixture::Client::Filtered(_));
        let creation = creation();
        let offsets = usize::from(offsets);
        let records = [(201, 5, 0x90), (202, 13, 0x80), (203, 17, 0x0102_0304)].map(
            |(address, stored_size, filter_mask)| ChunkRecord {
                address: StoredAddress::new(address),
                stored_size,
                filter_mask,
            },
        );
        let slots = IndexSlots::new(&records, &[1, 2, 12], 13).unwrap();
        let addresses = [128, 1024, 256, 512, 768].map(StoredAddress::new);
        let [header, index, direct, paged, secondary] = addresses.map(StoredAddress::get);

        let direct_bytes =
            fixture_block(DATA_SIGNATURE, client, header, offsets, Some(0), |bytes| {
                fixture_element(bytes, Some(records[1]), client, offsets)
            });
        let page_bytes = [[None, None], [None, Some(records[2])]].map(|slots| {
            fixture_page(|bytes| {
                for slot in slots {
                    fixture_element(bytes, slot, client, offsets);
                }
            })
        });
        let mut paged_bytes =
            fixture_block(DATA_SIGNATURE, client, header, offsets, Some(7), |_| {});
        paged_bytes.extend(page_bytes.iter().flatten().copied());
        let secondary_bytes = fixture_block(
            SECONDARY_SIGNATURE,
            client,
            header,
            offsets,
            Some(7),
            |bytes| {
                bytes.extend_from_slice(&[0x40, 0]);
                for address in [Some(paged), None] {
                    bytes::push_address(bytes, address, offsets);
                }
            },
        );
        let index_bytes = fixture_block(
            fixture::INDEX_SIGNATURE,
            client,
            header,
            offsets,
            None,
            |bytes| {
                for record in [None, Some(records[0])] {
                    fixture_element(bytes, record, client, offsets);
                }
                for address in [
                    Some(direct),
                    None,
                    None,
                    Some(secondary),
                    None,
                    None,
                    None,
                    None,
                    None,
                ] {
                    bytes::push_address(bytes, address, offsets);
                }
            },
        );
        let header_bytes = fixture::Header::new(
            offset_fixture,
            length_fixture,
            client,
            fixture::Creation::SmallPages,
        )
        .with_index(
            index,
            fixture::HeaderStats {
                secondary_block_count: 1,
                secondary_block_size: secondary_bytes.len().to_u64(),
                data_block_count: 2,
                data_block_size: (direct_bytes.len() + paged_bytes.len()).to_u64(),
                max_index_set: 13,
                element_count: 7,
            },
        )
        .unwrap()
        .build();

        let writer = ExtensibleArrayBlockWriter::new(
            creation,
            widths.offsets,
            client.id(),
            client.element_size(offset_fixture),
            addresses[0],
        )
        .unwrap();
        let children = [Some(addresses[3]), None];
        let block = writer
            .prepare_secondary_block(1, Some(&[0x40, 0]), Some(&children))
            .unwrap();
        assert_encoding(
            &block,
            block.encoded_size(),
            |output| block.write_into(output),
            &secondary_bytes,
        );
        let block = writer.prepare_data_block(7, &slots).unwrap();
        assert_encoding(
            &block,
            block.encoded_size(),
            |output| block.write_into(output),
            &paged_bytes,
        );
        for (index, expected) in page_bytes.iter().enumerate() {
            let page = block.page(index).unwrap();
            assert_encoding(
                &page,
                page.encoded_size(),
                |output| page.write_into(output),
                expected,
            );
        }
        assert!(block.page(page_bytes.len()).is_none());

        let plan = ExtensibleArrayPlan::new(
            creation,
            SlotOccupancy::Sparse(&slots),
            13,
            100,
            widths,
            filtered,
        )
        .unwrap();
        let placement = plan.place(&addresses).unwrap();
        let encoding = placement.encoding(&slots).unwrap();
        let expected = [
            header_bytes,
            index_bytes,
            direct_bytes,
            paged_bytes,
            secondary_bytes,
        ];
        assert_eq!(encoding.blocks().len(), expected.len());
        for ((block, address), expected) in encoding.blocks().zip(addresses).zip(expected) {
            assert_eq!(block.address(), address);
            assert_encoding(
                &block,
                block.encoded_size(),
                |output| block.write_into(output),
                &expected,
            );
        }
    }

    #[rstest]
    #[case::unpaged(0, 3, 2, &[], None)]
    #[case::paged(1, 7, 2, &[0, 0], None)]
    #[case::trailing_unencodable_children(6, 255, 16, &[0; 16], None)]
    #[case::unused_bits(1, 7, 2, &[0x0F, 0xFF], Some(&[0x0F, 0xFF][..]))]
    fn secondary_block_writer_encodes_undefined_children(
        #[case] index: u32,
        #[case] offset: u8,
        #[case] pointers: usize,
        #[case] expected_bitmap: &[u8],
        #[case] supplied_bitmap: Option<&[u8]>,
    ) {
        let block = filtered_writer()
            .prepare_secondary_block(index, supplied_bitmap, None)
            .unwrap();
        let client = fixture::Client::Filtered(fixture::Width::Two);
        let expected = fixture_block(SECONDARY_SIGNATURE, client, 128, 2, Some(offset), |bytes| {
            bytes.extend_from_slice(expected_bitmap);
            for _ in 0..pointers {
                bytes::push_address(bytes, None, 2);
            }
        });
        assert_encoding(
            &block,
            block.encoded_size(),
            |output| block.write_into(output),
            &expected,
        );
    }

    #[rstest]
    #[case::address(65_536, 17, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
    #[case::stored_size(203, 65_536, FormatError::LengthTooLarge { length: 65_536, length_size: 2 })]
    fn data_block_preparation_rejects_an_unencodable_record_in_the_last_page(
        #[case] address: u64,
        #[case] stored_size: u64,
        #[case] error: FormatError,
    ) {
        let records = [
            ChunkRecord {
                address: StoredAddress::new(201),
                stored_size: 5,
                filter_mask: 0,
            },
            ChunkRecord {
                address: StoredAddress::new(address),
                stored_size,
                filter_mask: 0,
            },
        ];
        let slots = IndexSlots::new(&records, &[9, 12], 13).unwrap();
        assert_eq!(
            filtered_writer().prepare_data_block(7, &slots).err(),
            Some(error)
        );
    }

    #[rstest]
    #[case::bitmap(Some(&[0x40][..]), None, FormatError::DataSizeMismatch { expected: 2, actual: 1 })]
    #[case::children(None, Some(&[None][..]), FormatError::DataSizeMismatch { expected: 2, actual: 1 })]
    #[case::address(None, Some(&[None, Some(StoredAddress::new(65_536))][..]), FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
    #[case::missing_children(Some(&[0x40, 0][..]), None, FormatError::InvalidExtensibleArrayGeometry)]
    #[case::missing_second_child(Some(&[0x20, 0][..]), Some(&[Some(StoredAddress::new(512)), None][..]), FormatError::InvalidExtensibleArrayGeometry)]
    fn secondary_block_preparation_rejects_invalid_body_fields(
        #[case] bitmap: Option<&[u8]>,
        #[case] addresses: Option<&[Option<StoredAddress>]>,
        #[case] error: FormatError,
    ) {
        assert_eq!(
            filtered_writer()
                .prepare_secondary_block(1, bitmap, addresses)
                .err(),
            Some(error)
        );
    }

    #[test]
    fn secondary_block_preparation_rejects_a_defined_child_with_an_unencodable_offset() {
        let mut children = [None; 16];
        children[..2]
            .copy_from_slice(&[Some(StoredAddress::new(512)), Some(StoredAddress::new(768))]);
        assert_eq!(
            filtered_writer()
                .prepare_secondary_block(6, None, Some(&children))
                .err(),
            Some(FormatError::ValueTooLargeForRepresentation {
                value: 271,
                target: "Extensible Array block offset",
            })
        );
    }

    #[test]
    fn secondary_block_preparation_rejects_a_child_end_beyond_the_address_width() {
        let client = fixture::Client::Unfiltered;
        let writer = ExtensibleArrayBlockWriter::new(
            creation(),
            OffsetWidth::Two,
            client.id(),
            client.element_size(fixture::Width::Two),
            StoredAddress::new(128),
        )
        .unwrap();
        let children = [Some(StoredAddress::new(65_534)), None];
        assert_eq!(
            writer
                .prepare_secondary_block(1, None, Some(&children))
                .err(),
            Some(FormatError::AddressTooLarge {
                address: 65_563,
                offset_size: 2,
            })
        );
    }

    fn assert_encoding(
        block: &impl Encode<Error = UnexpectedEof>,
        length: usize,
        write: impl Fn(&mut [u8]) -> Result<usize, FormatError>,
        expected: &[u8],
    ) {
        assert_eq!(length, expected.len());
        assert_eq!(block.encoded_len(), length);
        assert_output(length, write, expected);
        assert_output(
            length,
            |output| block.encode_into(output).map_err(FormatError::from),
            expected,
        );
    }

    fn assert_output(
        length: usize,
        write: impl Fn(&mut [u8]) -> Result<usize, FormatError>,
        expected: &[u8],
    ) {
        let mut short = vec![0xA5; length - 1];
        assert_eq!(
            write(&mut short),
            Err(FormatError::UnexpectedEof {
                expected: length,
                available: length - 1,
            })
        );
        assert_eq!(short, vec![0xA5; length - 1]);

        let mut output = vec![0xA5; length + 3];
        assert_eq!(write(&mut output), Ok(length));
        assert_eq!(&output[..length], expected);
        assert_eq!(output[length..], [0xA5; 3]);
    }

    fn filtered_writer() -> ExtensibleArrayBlockWriter {
        let client = fixture::Client::Filtered(fixture::Width::Two);
        ExtensibleArrayBlockWriter::new(
            creation(),
            OffsetWidth::Two,
            client.id(),
            client.element_size(fixture::Width::Two),
            StoredAddress::new(128),
        )
        .unwrap()
    }

    fn creation() -> ExtensibleArrayCreationParameters {
        ExtensibleArrayCreationParameters::from_fields(
            fixture::Creation::SmallPages.layout_fields(),
        )
        .unwrap()
    }

    fn fixture_block(
        signature: &[u8; 4],
        client: fixture::Client,
        owner: u64,
        offsets: usize,
        offset: Option<u8>,
        body: impl FnOnce(&mut Vec<u8>),
    ) -> Vec<u8> {
        let mut bytes = signature.to_vec();
        bytes.extend_from_slice(&[BLOCK_VERSION, client.id()]);
        bytes::push_address(&mut bytes, Some(owner), offsets);
        bytes.extend(offset);
        body(&mut bytes);
        checksum::append(&mut bytes);
        bytes
    }

    fn fixture_page(body: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
        let mut bytes = Vec::new();
        body(&mut bytes);
        checksum::append(&mut bytes);
        bytes
    }

    fn fixture_element(
        output: &mut Vec<u8>,
        record: Option<ChunkRecord>,
        client: fixture::Client,
        offsets: usize,
    ) {
        bytes::push_address(output, record.map(|record| record.address.get()), offsets);
        if let fixture::Client::Filtered(width) = client {
            bytes::push_uint(
                output,
                record.map_or(0, |record| record.stored_size),
                usize::from(width.get()),
            );
            output.extend_from_slice(&record.map_or(0, |record| record.filter_mask).to_le_bytes());
        }
    }

    // "The Extensible Array Index" in the format specification, version 4.0, defines the
    // EADB and EASB signatures and version zero.
    const DATA_SIGNATURE: &[u8; 4] = b"EADB";
    const SECONDARY_SIGNATURE: &[u8; 4] = b"EASB";
    const BLOCK_VERSION: u8 = 0;
}
