//! Datatype message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.

pub mod containers;
pub mod float_context;
pub mod leaf;

/// The eight bytes every datatype message begins with: its class and version
/// packed into one byte, the three class bit fields, and the element size.
pub fn header(class: Class, version: u8, bit_fields: [u8; 3], size: u32) -> Vec<u8> {
    let mut datatype = vec![(class.0 & 0x0F) | ((version & 0x0F) << 4)];
    datatype.extend_from_slice(&bit_fields);
    datatype.extend_from_slice(&size.to_le_bytes());
    datatype
}

/// Builds a variable-length message with the supplied parent bytes.
///
/// Accepts unchecked version, flag, size, and parent encodings for malformed fixtures.
pub fn variable_length(version: u8, flags: [u8; 3], size: u32, parent: &[u8]) -> Vec<u8> {
    let mut bytes = header(Class::VARIABLE_LENGTH, version, flags, size);
    bytes.extend_from_slice(parent);
    bytes
}

/// A variable-length sequence body with a descriptor size derived from its offset width.
pub struct VariableLengthSequence<'a> {
    version: leaf::Version,
    descriptor_size: u32,
    parent: &'a [u8],
}

impl<'a> VariableLengthSequence<'a> {
    /// Describes a sequence with the supplied encoded parent body.
    ///
    /// # Panics
    ///
    /// Panics if `offset_width` is other than 2, 4, or 8.
    #[track_caller]
    pub fn new(version: leaf::Version, offset_width: usize, parent: &'a [u8]) -> Self {
        assert!(
            matches!(offset_width, 2 | 4 | 8),
            "a sequence descriptor has a 2-, 4-, or 8-byte address"
        );
        let address_size = u32::try_from(offset_width).expect("the checked width fits u32");
        Self {
            version,
            descriptor_size: SEQUENCE_COUNT_SIZE + address_size + HEAP_INDEX_SIZE,
            parent,
        }
    }

    /// Returns the stored byte size of one sequence descriptor.
    pub fn descriptor_size(&self) -> u32 {
        self.descriptor_size
    }

    /// Writes the sequence's datatype header followed by its parent body.
    pub fn build(self) -> Vec<u8> {
        let Self {
            version,
            descriptor_size,
            parent,
        } = self;
        variable_length(version.get(), SEQUENCE_FLAGS, descriptor_size, parent)
    }
}

/// Builds an array message using the property layout of versions 3 through 5.
///
/// Accepts unchecked header and parent encodings. `rank` and `dimensions` are independent
/// for malformed fixtures. The layout is defined in "The Datatype Message" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub fn array(
    version: u8,
    flags: [u8; 3],
    size: u32,
    rank: u8,
    dimensions: &[u32],
    parent: &[u8],
) -> Vec<u8> {
    let mut bytes = header(Class::ARRAY, version, flags, size);
    bytes.push(rank);
    for dimension in dimensions {
        bytes.extend_from_slice(&dimension.to_le_bytes());
    }
    bytes.extend_from_slice(parent);
    bytes
}

/// Builds a complex message with unchecked header fields and parent bytes.
///
/// Uses the version 5 complex property layout defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec]. Accepts other versions for malformed fixtures.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub fn complex(version: u8, flags: [u8; 3], size: u32, parent: &[u8]) -> Vec<u8> {
    let mut bytes = header(Class::COMPLEX, version, flags, size);
    bytes.extend_from_slice(parent);
    bytes
}

/// A little-endian IEEE 754 double, laid out as the reference library writes
/// one.
pub fn f64_le() -> Vec<u8> {
    let mut datatype = header(
        Class::FLOATING_POINT,
        DATATYPE_VERSION,
        IEEE754_BINARY64_CLASS_FIELDS,
        8,
    );
    // Bit offset(2) and bit precision(2), then the exponent's location and
    // size, the mantissa's, and the exponent bias.
    datatype.extend_from_slice(&0u16.to_le_bytes());
    datatype.extend_from_slice(&64u16.to_le_bytes());
    datatype.extend_from_slice(&[52, 11, 0, 52]);
    datatype.extend_from_slice(&1023u32.to_le_bytes());
    datatype
}

/// A fixed-length string of `size` bytes, null-padded and ASCII.
pub fn fixed_string(size: u32) -> Vec<u8> {
    header(Class::STRING, DATATYPE_VERSION, [0x01, 0, 0], size)
}

/// Encodes a legacy object-reference datatype with the caller-supplied element size.
///
/// Accepts arbitrary sizes for malformed fixtures.
pub fn object_reference(size: u32) -> Vec<u8> {
    header(Class::REFERENCE, DATATYPE_VERSION, OBJECT_REFERENCE, size)
}

/// A legacy object or dataset-region reference datatype fixture.
#[derive(Clone, Copy, Debug)]
pub enum LegacyReference {
    /// An object reference with an eight-byte element size.
    Object,
    /// A dataset-region reference with a twelve-byte element size.
    DatasetRegion,
}

impl LegacyReference {
    /// Writes the reference datatype header with the supplied message version.
    pub fn bytes(self, version: leaf::Version) -> Vec<u8> {
        let flags = match self {
            Self::Object => OBJECT_REFERENCE,
            Self::DatasetRegion => DATASET_REGION_REFERENCE,
        };
        header(Class::REFERENCE, version.get(), flags, self.size())
    }

    /// Returns the element size used by the fixture.
    pub const fn size(self) -> u32 {
        match self {
            Self::Object => LEGACY_OBJECT_REFERENCE_SIZE,
            Self::DatasetRegion => LEGACY_DATASET_REGION_REFERENCE_SIZE,
        }
    }
}

// "The Datatype Message" in format specification version 4.0 defines kind 1 as a
// legacy dataset-region reference for datatype message versions before 4.
const DATASET_REGION_REFERENCE: [u8; 3] = [1, 0, 0];
// `H5T__ref_set_loc` uses the file offset width and adds a heap index for a region reference
// (H5Tref.c, HDF5 2.2.0).
const LEGACY_OBJECT_REFERENCE_SIZE: u32 = 8;
const LEGACY_DATASET_REGION_REFERENCE_SIZE: u32 = LEGACY_OBJECT_REFERENCE_SIZE + HEAP_INDEX_SIZE;

/// A reference to a committed datatype, which stands where an encoding
/// usually is when the message's shared flag is set.
pub fn committed_reference(address: u64) -> Vec<u8> {
    let mut reference = vec![SHARED_MESSAGE_VERSION, COMMITTED];
    reference.extend_from_slice(&address.to_le_bytes());
    reference
}

/// What a datatype describes, which its first byte's low nibble holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Class(pub u8);

impl Class {
    pub const FIXED_POINT: Self = Self(0);
    pub const FLOATING_POINT: Self = Self(1);
    pub const TIME: Self = Self(2);
    pub const STRING: Self = Self(3);
    pub const BIT_FIELD: Self = Self(4);
    pub const OPAQUE: Self = Self(5);
    pub const COMPOUND: Self = Self(6);
    pub const REFERENCE: Self = Self(7);
    pub const ENUMERATED: Self = Self(8);
    pub const VARIABLE_LENGTH: Self = Self(9);
    pub const ARRAY: Self = Self(10);
    /// The native complex class code for datatype message version 5.
    ///
    /// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    pub const COMPLEX: Self = Self(11);
}

/// The version of the shared message reference the reference library writes
/// for a committed datatype.
const SHARED_MESSAGE_VERSION: u8 = 2;

/// The shared message type whose location is an object header.
const COMMITTED: u8 = 2;

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.
const DATATYPE_VERSION: u8 = 1;

// `H5T__vlen_disk_write` writes a four-byte sequence count (H5Tvlen.c, HDF5 2.2.0).
const SEQUENCE_COUNT_SIZE: u32 = 4;

// "Global Heap" in version 4.0 defines the object index in a global heap ID.
const HEAP_INDEX_SIZE: u32 = 4;

// "The Datatype Message" in version 4.0 defines the sequence subtype and its reserved bits.
const SEQUENCE_FLAGS: [u8; 3] = [0; 3];

/// The class bit fields for little-endian IEEE 754 binary64.
///
/// Mantissa normalization and sign location are defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const IEEE754_BINARY64_CLASS_FIELDS: [u8; 3] = [0x20, 0x3F, 0];

/// The version 1 datatype class fields for a legacy object reference.
///
/// The reference type is zero and the remaining bits are reserved, as defined in "The Datatype
/// Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const OBJECT_REFERENCE: [u8; 3] = [0; 3];
