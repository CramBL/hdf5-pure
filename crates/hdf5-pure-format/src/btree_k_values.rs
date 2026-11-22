use core::error::Error;
use core::fmt;
use core::num::NonZeroU16;

use crate::bytes::ByteCursor;
use crate::codec::{Decode, Encode};
use crate::error::FormatError;

/// The version 0 superblock extension message for version 1 B-tree K values.
///
/// Each of the three K values is positive. "The B-tree 'K' Values Message" of the
/// [format specification, version 4.0][spec] defines their order and meaning.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_btreek
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BTreeKValues {
    indexed_storage_internal_node_k: NonZeroU16,
    group_internal_node_k: NonZeroU16,
    group_leaf_node_k: NonZeroU16,
}

impl BTreeKValues {
    /// Constructs a message from three positive K values in their stored order.
    pub const fn new(
        indexed_storage_internal_node_k: NonZeroU16,
        group_internal_node_k: NonZeroU16,
        group_leaf_node_k: NonZeroU16,
    ) -> Self {
        Self {
            indexed_storage_internal_node_k,
            group_internal_node_k,
            group_leaf_node_k,
        }
    }

    /// Parses the seven-byte message body at the start of `input`.
    ///
    /// Ignores trailing bytes, including any enclosing object header padding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] with a required length of one byte if
    /// `input` is empty. After reading the version byte, returns
    /// [`FormatError::InvalidBTreeKValuesVersion`] for a version other than 0, or
    /// [`FormatError::UnexpectedEof`] with a required length of seven bytes if the
    /// version 0 body is incomplete. Once the complete body is available, returns
    /// [`FormatError::ZeroBTreeKValue`] for the first zero K value in stored order.
    pub fn parse(input: &[u8]) -> Result<Self, FormatError> {
        Self::decode(input, ())
    }

    /// Returns half the maximum entry count for an indexed storage internal node.
    pub const fn indexed_storage_internal_node_k(&self) -> NonZeroU16 {
        self.indexed_storage_internal_node_k
    }

    /// Returns half the maximum entry count for a group internal node.
    pub const fn group_internal_node_k(&self) -> NonZeroU16 {
        self.group_internal_node_k
    }

    /// Returns half the maximum entry count for a group leaf node.
    pub const fn group_leaf_node_k(&self) -> NonZeroU16 {
        self.group_leaf_node_k
    }

    /// Returns the length of the message body in bytes, always seven.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the message body to the start of `output` and returns the byte count.
    ///
    /// Writes seven bytes and leaves any bytes beyond the body unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`BTreeKValuesOutputTooSmall`] if `output` has fewer than seven bytes.
    /// On error, leaves all of `output` unchanged.
    ///
    /// # Examples
    ///
    /// Reads a message and writes its body while preserving the output suffix.
    ///
    /// ```
    /// use hdf5_pure_format::__private::BTreeKValues;
    ///
    /// let body = [0, 0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a];
    /// let message = BTreeKValues::parse(&body)?;
    /// assert_eq!(message.group_internal_node_k().get(), 0x5678);
    ///
    /// let mut output = [0xff; 8];
    /// assert_eq!(message.write_into(&mut output)?, body.len());
    /// assert_eq!(output, [0, 0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xff]);
    /// # Ok::<(), Box<dyn core::error::Error>>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, BTreeKValuesOutputTooSmall> {
        self.encode_into(output)
    }
}

impl<'input> Decode<'input> for BTreeKValues {
    type Context = ();
    type Error = FormatError;

    fn decode(input: &'input [u8], (): ()) -> Result<Self, Self::Error> {
        let mut cursor = ByteCursor::new(input);
        let version = cursor.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidBTreeKValuesVersion(version));
        }

        let &[
            indexed_storage_low,
            indexed_storage_high,
            group_internal_low,
            group_internal_high,
            group_leaf_low,
            group_leaf_high,
        ] = cursor.take_array::<K_VALUES_SIZE>()?;
        Ok(Self::new(
            read_k(
                [indexed_storage_low, indexed_storage_high],
                INDEXED_STORAGE_INTERNAL_NODE_K_OFFSET,
            )?,
            read_k(
                [group_internal_low, group_internal_high],
                GROUP_INTERNAL_NODE_K_OFFSET,
            )?,
            read_k([group_leaf_low, group_leaf_high], GROUP_LEAF_NODE_K_OFFSET)?,
        ))
    }
}

impl Encode for BTreeKValues {
    type Error = BTreeKValuesOutputTooSmall;

    fn encoded_len(&self) -> usize {
        MESSAGE_SIZE
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let available = output.len();
        let record =
            output
                .first_chunk_mut::<MESSAGE_SIZE>()
                .ok_or(BTreeKValuesOutputTooSmall {
                    required: MESSAGE_SIZE,
                    available,
                })?;
        let Self {
            indexed_storage_internal_node_k,
            group_internal_node_k,
            group_leaf_node_k,
        } = *self;
        let [indexed_storage_low, indexed_storage_high] =
            indexed_storage_internal_node_k.get().to_le_bytes();
        let [group_internal_low, group_internal_high] = group_internal_node_k.get().to_le_bytes();
        let [group_leaf_low, group_leaf_high] = group_leaf_node_k.get().to_le_bytes();
        *record = [
            VERSION,
            indexed_storage_low,
            indexed_storage_high,
            group_internal_low,
            group_internal_high,
            group_leaf_low,
            group_leaf_high,
        ];
        Ok(MESSAGE_SIZE)
    }
}

/// The output is too short to write a complete B-tree K Values message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BTreeKValuesOutputTooSmall {
    /// The required body length in bytes, always seven.
    pub required: usize,
    /// The length of the supplied output in bytes.
    pub available: usize,
}

impl fmt::Display for BTreeKValuesOutputTooSmall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "B-tree K Values encoding needs {} bytes, destination has {}",
            self.required, self.available
        )
    }
}

impl Error for BTreeKValuesOutputTooSmall {}

fn read_k(bytes: [u8; 2], field_offset: usize) -> Result<NonZeroU16, FormatError> {
    let Some(value) = NonZeroU16::new(u16::from_le_bytes(bytes)) else {
        return Err(FormatError::ZeroBTreeKValue { field_offset });
    };
    Ok(value)
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_btreek`, version 4.0, defines version zero.
const VERSION: u8 = 0;
// Section `subsubsec_fmt4_dataobject_hdr_msg_btreek`, version 4.0, defines the version byte
// followed by three two-byte K values.
const INDEXED_STORAGE_INTERNAL_NODE_K_OFFSET: usize = size_of::<u8>();
const GROUP_INTERNAL_NODE_K_OFFSET: usize =
    INDEXED_STORAGE_INTERNAL_NODE_K_OFFSET + size_of::<u16>();
const GROUP_LEAF_NODE_K_OFFSET: usize = GROUP_INTERNAL_NODE_K_OFFSET + size_of::<u16>();
const MESSAGE_SIZE: usize = GROUP_LEAF_NODE_K_OFFSET + size_of::<u16>();
const K_VALUES_SIZE: usize = MESSAGE_SIZE - INDEXED_STORAGE_INTERNAL_NODE_K_OFFSET;
