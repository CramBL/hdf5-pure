//! The Link Info message (type 0x0002) of a group: the dense storage of its links, and whether it
//! tracks and indexes their creation order.
//!
//! The message is defined in "The Link Info Message" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo

use alloc::vec;
use alloc::vec::Vec;

use crate::address::{DecodedStoredAddress, StoredAddress, StoredAddressExt};
use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::codec::{Decode, Encode};
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// A version 0 Link Info body with addresses checked at one file offset width.
///
/// The maximum creation order and creation-order index fields have independent presence bits.
/// A present index field can store the undefined address. The body layout is defined in
/// "The Link Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkInfoMessage {
    offset_width: OffsetWidth,
    flags: u8,
    max_creation_order: u64,
    fractal_heap_address: StoredAddress,
    btree_name_index_address: StoredAddress,
    creation_order_index: StoredAddress,
}

impl LinkInfoMessage {
    /// Constructs a body with the supplied optional fields and checked stored addresses.
    ///
    /// Pass [`None`] for an undefined heap or name-index address. Use
    /// [`LinkCreationOrderIndex`] to select whether the creation-order index field is present
    /// and whether its address is defined.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if a defined address exceeds `offset_width`, or
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals that width's undefined marker.
    /// Checks the heap address, name-index address, and creation-order index address in that order.
    pub fn new(
        offset_width: OffsetWidth,
        max_creation_order: Option<u64>,
        fractal_heap_address: Option<StoredAddress>,
        btree_name_index_address: Option<StoredAddress>,
        creation_order_index: LinkCreationOrderIndex,
    ) -> Result<Self, FormatError> {
        let stored = |address: Option<StoredAddress>| {
            address
                .map_or(
                    DecodedStoredAddress::Undefined,
                    DecodedStoredAddress::Defined,
                )
                .to_stored(offset_width)
        };
        let heap = stored(fractal_heap_address)?;
        let name_index = stored(btree_name_index_address)?;
        let creation_index = match creation_order_index {
            LinkCreationOrderIndex::Absent => None,
            LinkCreationOrderIndex::Undefined => Some(stored(None)?),
            LinkCreationOrderIndex::Defined(address) => Some(stored(Some(address))?),
        };
        Ok(Self::from_stored(
            offset_width,
            max_creation_order,
            heap,
            name_index,
            creation_index,
        ))
    }

    /// Constructs a body with undefined heap and name-index addresses.
    ///
    /// When `indexed` is `true`, includes an undefined creation-order index address.
    /// `max_creation_order` independently selects whether to include the tracking field.
    pub fn compact(
        offset_width: OffsetWidth,
        max_creation_order: Option<u64>,
        indexed: bool,
    ) -> Self {
        let undefined = StoredAddress::undefined(offset_width);
        Self::from_stored(
            offset_width,
            max_creation_order,
            undefined,
            undefined,
            indexed.then_some(undefined),
        )
    }

    /// Reads the message at the start of `data` after checking the raw address width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] if `offset_size` is outside the supported
    /// widths of 2, 4, and 8 bytes. See [`Self::parse_with_offset_width`] for body errors.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<Self, FormatError> {
        Self::parse_with_offset_width(data, OffsetWidth::try_from(offset_size)?)
    }

    /// Parses the version 0 body at the start of `data` using the file's offset width.
    ///
    /// Ignores bytes after the selected fields, including object header padding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] with a required length of two bytes if the
    /// prefix is incomplete. After reading the prefix, checks the version and flags in order,
    /// returning [`FormatError::InvalidLinkInfoVersion`] for a version other than 0 or
    /// [`FormatError::InvalidLinkInfoFlags`] for reserved flag bits. If a selected field is
    /// incomplete, returns [`FormatError::UnexpectedEof`] with the complete body length and
    /// the original input length.
    pub fn parse_with_offset_width(
        data: &[u8],
        offset_width: OffsetWidth,
    ) -> Result<Self, FormatError> {
        Self::decode(data, offset_width)
    }

    /// Sets the tracking field to `maximum`, or omits it if `maximum` is [`None`].
    pub fn set_max_creation_order(&mut self, maximum: Option<u64>) {
        self.flags =
            (self.flags & !TRACKS_CREATION_ORDER) | maximum.map_or(0, |_| TRACKS_CREATION_ORDER);
        self.max_creation_order = maximum.unwrap_or_default();
    }

    /// Stores a defined address at the retained offset width.
    ///
    /// Setting the creation-order index address includes that field in the body.
    ///
    /// # Errors
    ///
    /// Returns the address errors of [`Self::new`] and leaves the message unchanged on error.
    pub(crate) fn set_defined_address(
        &mut self,
        field: LinkInfoAddressField,
        address: StoredAddress,
    ) -> Result<(), FormatError> {
        let address = DecodedStoredAddress::Defined(address).to_stored(self.offset_width)?;
        match field {
            LinkInfoAddressField::Heap => self.fractal_heap_address = address,
            LinkInfoAddressField::NameIndex => self.btree_name_index_address = address,
            LinkInfoAddressField::CreationIndex => {
                self.flags |= INDEXES_CREATION_ORDER;
                self.creation_order_index = address;
            }
        }
        Ok(())
    }

    /// Returns the offset width retained for address decoding and encoding.
    pub const fn offset_width(&self) -> OffsetWidth {
        self.offset_width
    }

    /// Returns the stored maximum creation order if the tracking field is present.
    pub fn max_creation_order(&self) -> Option<u64> {
        (self.flags & TRACKS_CREATION_ORDER != 0).then_some(self.max_creation_order)
    }

    /// Returns the dense link heap address, or [`None`] if undefined.
    pub fn fractal_heap_address(&self) -> Option<StoredAddress> {
        self.defined_address(self.fractal_heap_address)
    }

    /// Returns the name-index B-tree address, or [`None`] if undefined.
    pub fn btree_name_index_address(&self) -> Option<StoredAddress> {
        self.defined_address(self.btree_name_index_address)
    }

    /// Returns whether the creation-order index field is present, including an undefined address.
    pub const fn indexes_creation_order(&self) -> bool {
        self.flags & INDEXES_CREATION_ORDER != 0
    }

    /// Returns the creation-order index field's presence and address.
    pub fn creation_order_index(&self) -> LinkCreationOrderIndex {
        if !self.indexes_creation_order() {
            LinkCreationOrderIndex::Absent
        } else {
            self.defined_address(self.creation_order_index).map_or(
                LinkCreationOrderIndex::Undefined,
                LinkCreationOrderIndex::Defined,
            )
        }
    }

    /// Returns the exact body length in bytes, excluding object header padding.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the body to the start of `output` and returns the number of bytes written.
    ///
    /// Leaves bytes beyond the body unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] with the required and available lengths if
    /// `output` is shorter than [`Self::encoded_size`]. Leaves `output` unchanged on error.
    ///
    /// # Examples
    ///
    /// Writes a Link Info body with an undefined creation-order index address.
    ///
    /// ```
    /// use hdf5_pure_format::__private::LinkInfoMessage;
    /// use hdf5_pure_format::OffsetWidth;
    ///
    /// let info = LinkInfoMessage::compact(OffsetWidth::Two, None, true);
    /// let mut output = [0xa5; 9];
    /// assert_eq!(info.write_into(&mut output)?, 8);
    /// assert_eq!(output, [0, 2, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xa5]);
    /// assert_eq!(LinkInfoMessage::parse_with_offset_width(&output, OffsetWidth::Two)?, info);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }

    /// Returns the encoded body in an owned buffer, excluding object header padding.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = vec![0; self.encoded_size()];
        self.write_body(&mut body);
        body
    }

    /// Constructs a body from addresses that fit `offset_width`.
    ///
    /// Pass [`None`] to omit the creation-order index field.
    fn from_stored(
        offset_width: OffsetWidth,
        max_creation_order: Option<u64>,
        fractal_heap_address: StoredAddress,
        btree_name_index_address: StoredAddress,
        creation_order_index: Option<StoredAddress>,
    ) -> Self {
        Self {
            offset_width,
            flags: max_creation_order.map_or(0, |_| TRACKS_CREATION_ORDER)
                | creation_order_index.map_or(0, |_| INDEXES_CREATION_ORDER),
            max_creation_order: max_creation_order.unwrap_or_default(),
            fractal_heap_address,
            btree_name_index_address,
            creation_order_index: creation_order_index
                .unwrap_or_else(|| StoredAddress::undefined(offset_width)),
        }
    }

    fn defined_address(&self, address: StoredAddress) -> Option<StoredAddress> {
        (!address.is_undefined(self.offset_width)).then_some(address)
    }

    /// Returns the encoded body length for the message record's length field.
    pub(crate) fn body_size(&self) -> u8 {
        body_size(self.flags, self.offset_width)
    }

    /// Writes the body into a slice of exactly [`Self::encoded_size`] bytes.
    pub(crate) fn write_body(&self, output: &mut [u8]) {
        let (prefix, body) = output.split_at_mut(usize::from(PREFIX_SIZE));
        prefix.copy_from_slice(&[LINK_INFO_VERSION, self.flags]);
        let addresses = if let Some(maximum) = self.max_creation_order() {
            let (field, addresses) = body.split_at_mut(usize::from(MAXIMUM_SIZE));
            field.copy_from_slice(&maximum.to_le_bytes());
            addresses
        } else {
            body
        };
        let width = usize::from(self.offset_width.get());
        for (field, address) in addresses.chunks_exact_mut(width).zip([
            self.fractal_heap_address,
            self.btree_name_index_address,
            self.creation_order_index,
        ]) {
            field.copy_from_slice(&address.get().to_le_bytes()[..width]);
        }
    }
}

impl<'input> Decode<'input> for LinkInfoMessage {
    type Context = OffsetWidth;
    type Error = FormatError;

    fn decode(input: &'input [u8], offset_width: OffsetWidth) -> Result<Self, Self::Error> {
        let mut cursor = ByteCursor::new(input);
        let [version, flags] = cursor.array::<{ PREFIX_SIZE as usize }>()?;
        if version != LINK_INFO_VERSION {
            return Err(FormatError::InvalidLinkInfoVersion(version));
        }
        if flags & !LINK_INFO_FLAGS != 0 {
            return Err(FormatError::InvalidLinkInfoFlags(flags));
        }

        let incomplete =
            |available| UnexpectedEof::new(usize::from(body_size(flags, offset_width)), available);
        let maximum_bytes = (flags & TRACKS_CREATION_ORDER != 0)
            .then(|| cursor.take_array::<{ MAXIMUM_SIZE as usize }>())
            .transpose()
            .map_err(|error| incomplete(error.available))?;
        let address_bytes = cursor.remaining();
        let indexed = flags & INDEXES_CREATION_ORDER != 0;
        let (heap, name_index, creation_index) = match offset_width {
            OffsetWidth::Two => read_addresses(address_bytes, indexed, |bytes| {
                u64::from(u16::from_le_bytes(bytes))
            }),
            OffsetWidth::Four => read_addresses(address_bytes, indexed, |bytes| {
                u64::from(u32::from_le_bytes(bytes))
            }),
            OffsetWidth::Eight => read_addresses(address_bytes, indexed, u64::from_le_bytes),
        }
        .ok_or_else(|| incomplete(input.len()))?;
        Ok(Self::from_stored(
            offset_width,
            maximum_bytes.copied().map(u64::from_le_bytes),
            heap,
            name_index,
            creation_index,
        ))
    }
}

impl Encode for LinkInfoMessage {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        usize::from(self.body_size())
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let len = self.encoded_len();
        let available = output.len();
        let body = output
            .get_mut(..len)
            .ok_or_else(|| UnexpectedEof::new(len, available))?;
        self.write_body(body);
        Ok(len)
    }
}

/// The presence and stored address of the optional creation-order index field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkCreationOrderIndex {
    /// The body omits the creation-order index field.
    Absent,
    /// The body stores a defined index address.
    Defined(StoredAddress),
    /// The body stores the undefined address at its offset width.
    Undefined,
}

impl LinkCreationOrderIndex {
    /// Returns the defined index address when present.
    pub const fn defined_address(self) -> Option<StoredAddress> {
        if let Self::Defined(address) = self {
            Some(address)
        } else {
            None
        }
    }
}

pub(crate) enum LinkInfoAddressField {
    Heap,
    NameIndex,
    CreationIndex,
}

fn body_size(flags: u8, offset_width: OffsetWidth) -> u8 {
    PREFIX_SIZE
        + MAXIMUM_SIZE * u8::from(flags & TRACKS_CREATION_ORDER != 0)
        + offset_width.get() * address_count(flags)
}

fn address_count(flags: u8) -> u8 {
    REQUIRED_ADDRESSES + u8::from(flags & INDEXES_CREATION_ORDER != 0)
}

fn read_addresses<const N: usize>(
    input: &[u8],
    indexed: bool,
    decode: impl Fn([u8; N]) -> u64,
) -> Option<(StoredAddress, StoredAddress, Option<StoredAddress>)> {
    const { assert!(N > 0) };
    let (fields, _) = input.as_chunks::<N>();
    let (heap, name_index, creation_index) = if indexed {
        let [heap, name_index, creation_index] =
            fields.first_chunk::<{ REQUIRED_ADDRESSES as usize + 1 }>()?;
        (heap, name_index, Some(creation_index))
    } else {
        let [heap, name_index] = fields.first_chunk::<{ REQUIRED_ADDRESSES as usize }>()?;
        (heap, name_index, None)
    };
    let read = |bytes: &[u8; N]| StoredAddress::new(decode(*bytes));
    Some((read(heap), read(name_index), creation_index.map(read)))
}

// "The Link Info Message" of the format specification, version 4.0, defines these fields.
const LINK_INFO_VERSION: u8 = 0;
const TRACKS_CREATION_ORDER: u8 = 0x01;
const INDEXES_CREATION_ORDER: u8 = 0x02;
const LINK_INFO_FLAGS: u8 = TRACKS_CREATION_ORDER | INDEXES_CREATION_ORDER;
const PREFIX_SIZE: u8 = 2;
const MAXIMUM_SIZE: u8 = 8;
const REQUIRED_ADDRESSES: u8 = 2;
