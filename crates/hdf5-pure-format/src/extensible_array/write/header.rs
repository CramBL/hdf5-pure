use alloc::vec::Vec;

use super::{ExtensibleArrayPlacement, INDEX_BLOCK};
use crate::address::StoredAddress;
use crate::bytes::UnexpectedEof;
use crate::checksum;
use crate::chunk_record::{ArrayClient, ElementEncoding, InvalidElementEncoding};
use crate::codec::Encode;
use crate::error::FormatError;
use crate::extensible_array::{
    CHECKSUM_LEN, ExtensibleArrayCreationParameters, ExtensibleArrayHeader, ExtensibleArrayStats,
    HEADER_BYTE_FIELDS, HEADER_PREFIX_LEN, HEADER_SIGNATURE, VERSION,
};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

/// A prepared version 0 Extensible Array header with an undefined index-block address.
///
/// The writer preserves the supplied creation fields and writes six zero statistics, even
/// when the creation parameters reserve inline entries for an index block.
/// "The Extensible Array Index" of the [format specification, version 4.0][spec]
/// defines the header fields.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayHeaderWithoutIndexBlock {
    fields: HeaderFields,
}

impl ExtensibleArrayHeaderWithoutIndexBlock {
    /// Prepares a header from checked creation parameters and a supplied element encoding.
    ///
    /// For unfiltered chunks, pass client ID 0 and the address width as `element_size`.
    /// For filtered chunks, pass client ID 1 and the combined size of the address, a
    /// chunk-size field of 2 to 8 bytes, and a four-byte filter mask.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayClient`] for any other client ID, or
    /// [`FormatError::InvalidExtensibleArrayGeometry`] for an element size outside the
    /// supported sizes for that client.
    pub fn new(
        creation: ExtensibleArrayCreationParameters,
        widths: FormatWidths,
        client_id: u8,
        element_size: u8,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            fields: HeaderFields {
                elements: check_elements(widths.offsets, client_id, element_size)?,
                creation,
                lengths: widths.lengths,
            },
        })
    }

    /// Returns the exact header length, including the checksum.
    ///
    /// The length is 16 bytes plus six times the length width and one address width.
    pub fn encoded_size(&self) -> usize {
        self.fields.encoded_size()
    }

    /// Writes the complete Extensible Array header to `output` and returns its byte count.
    ///
    /// Writes an all-ones index-block address at the selected offset width, followed by
    /// the checksum over the preceding header bytes. Leaves bytes beyond the header
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] with the required and available lengths if
    /// `output` is shorter than [`Self::encoded_size`]. Leaves `output` unchanged on error.
    ///
    /// # Examples
    ///
    /// Writes a header for filtered chunks with four-byte addresses and two-byte lengths.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{
    ///     ExtensibleArrayCreationParameters, ExtensibleArrayHeaderWithoutIndexBlock, FormatWidths,
    /// };
    ///
    /// let creation = ExtensibleArrayCreationParameters::canonical();
    /// let widths = FormatWidths::from_sizes(4, 2)?;
    /// let header = ExtensibleArrayHeaderWithoutIndexBlock::new(creation, widths, 1, 10)?;
    /// let mut output = [0xa5; 33];
    /// assert_eq!(header.encoded_size(), 32);
    /// assert_eq!(header.write_into(&mut output)?, 32);
    /// assert_eq!(&output[..4], b"EAHD");
    /// assert_eq!(&output[12..24], &[0; 12]);
    /// assert_eq!(&output[24..28], &[0xff; 4]);
    /// assert_eq!(output[32], 0xa5);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(HeaderEncoding {
            fields: self.fields,
            index_block: None,
        }
        .encode_into(output)?)
    }
}

impl ExtensibleArrayPlacement<'_> {
    /// Prepares the header using the placed index address and a borrow of the checked statistics.
    pub(super) fn header_encoding(&self) -> HeaderEncoding<'_> {
        let plan = self.plan;
        HeaderEncoding {
            fields: HeaderFields {
                elements: plan.layout.encoding,
                creation: plan.creation,
                lengths: plan.widths.lengths,
            },
            // Planning checks the statistics and placement checks the index address at these widths.
            index_block: Some(AllocatedIndexBlock {
                stats: &plan.stats,
                address: self.ranges[INDEX_BLOCK].start(),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HeaderFields {
    elements: ElementEncoding,
    creation: ExtensibleArrayCreationParameters,
    lengths: LengthWidth,
}

impl HeaderFields {
    fn encoded_size(&self) -> usize {
        ExtensibleArrayHeader::serialized_size_with_widths(FormatWidths {
            offsets: self.elements.offsets(),
            lengths: self.lengths,
        })
    }
}

/// The fields to encode for a header with or without a placed index block.
pub(super) struct HeaderEncoding<'a> {
    fields: HeaderFields,
    index_block: Option<AllocatedIndexBlock<'a>>,
}

impl HeaderEncoding<'_> {
    /// Appends the complete header to a buffer with capacity for it.
    pub(super) fn append_into(&self, output: &mut Vec<u8>) {
        let start = output.len();
        output.resize(start + self.encoded_len(), 0);
        self.write_header(&mut output[start..]);
    }

    /// Writes the header and checksum into a destination of exactly the encoded length.
    fn write_header(&self, output: &mut [u8]) {
        let (content, checksum_field) = output.split_at_mut(output.len() - CHECKSUM_LEN);
        let (prefix, fields) = content.split_at_mut(HEADER_PREFIX_LEN);
        let (signature, properties) = prefix.split_at_mut(HEADER_SIGNATURE.len());
        signature.copy_from_slice(&HEADER_SIGNATURE);
        let [max_bits, inline, min_blocks, min_elements, page_bits] = self.fields.creation.encode();
        let encoded_properties: [u8; HEADER_BYTE_FIELDS] = [
            VERSION,
            self.fields.elements.client().id(),
            self.fields.elements.stride().get(),
            max_bits,
            inline,
            min_elements,
            min_blocks,
            page_bits,
        ];
        properties.copy_from_slice(&encoded_properties);

        let lengths = self.fields.lengths;
        let offsets = self.fields.elements.offsets();
        let (statistics, index_address) =
            fields.split_at_mut(ExtensibleArrayStats::fields_len(lengths));
        let address = match &self.index_block {
            Some(index) => {
                index.stats.write_fields(statistics, lengths);
                index.address
            }
            None => {
                // H5EA__hdr_create in H5EAhdr.c, HDF5 2.2.0, starts with zero statistics and an
                // undefined index block address.
                statistics.fill(0);
                StoredAddress::undefined(offsets)
            }
        };
        index_address.copy_from_slice(&address.get().to_le_bytes()[..usize::from(offsets.get())]);
        checksum_field.copy_from_slice(&checksum::jenkins_lookup3(content).to_le_bytes());
    }
}

impl Encode for HeaderEncoding<'_> {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        self.fields.encoded_size()
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let required = self.encoded_len();
        let available = output.len();
        let header = output
            .get_mut(..required)
            .ok_or_else(|| UnexpectedEof::new(required, available))?;
        self.write_header(header);
        Ok(required)
    }
}

struct AllocatedIndexBlock<'a> {
    stats: &'a ExtensibleArrayStats,
    address: StoredAddress,
}

/// Checks the client and supplied element size for a header the writer will emit.
///
/// # Errors
///
/// Returns the client and element size errors described in
/// [`ExtensibleArrayHeaderWithoutIndexBlock::new`].
pub(super) fn check_elements(
    offsets: OffsetWidth,
    client_id: u8,
    element_size: u8,
) -> Result<ElementEncoding, FormatError> {
    let elements =
        ElementEncoding::new(offsets, client_id, element_size).map_err(|error| match error {
            InvalidElementEncoding::Client(client) => {
                FormatError::InvalidExtensibleArrayClient(client)
            }
            InvalidElementEncoding::EntrySize => FormatError::InvalidExtensibleArrayGeometry,
        })?;
    let size_is_valid = match elements.client() {
        ArrayClient::Chunks => elements.stride().get() == element_size,
        ArrayClient::FilteredChunks { chunk_size_len } => {
            chunk_size_len >= MIN_FILTERED_CHUNK_SIZE_LEN
        }
    };
    if !size_is_valid {
        return Err(FormatError::InvalidExtensibleArrayGeometry);
    }
    Ok(elements)
}

// The writer requires at least two bytes for the stored chunk size in a filtered element.
const MIN_FILTERED_CHUNK_SIZE_LEN: u8 = 2;
