//! Independent fixtures for version 0 Attribute Info message bodies.

use crate::bytes;

/// An address width in bytes.
#[derive(Clone, Copy, Debug)]
pub enum Width {
    Two,
    Four,
    Eight,
}

impl Width {
    /// Returns the number of bytes used by a stored address.
    pub fn get(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    fn undefined_address(self) -> u64 {
        match self {
            Self::Two => u64::from(u16::MAX),
            Self::Four => u64::from(u32::MAX),
            Self::Eight => u64::MAX,
        }
    }
}

/// The fields of an Attribute Info body with addresses that fit the selected width.
#[derive(Clone, Copy, Debug)]
pub struct AttributeInfo {
    offsets: Width,
    max_creation_index: Option<u16>,
    fractal_heap_address: Option<u64>,
    name_index_address: Option<u64>,
    creation_order_index: CreationOrderIndex,
}

impl AttributeInfo {
    /// Creates a fixture with independent tracking and indexing fields.
    ///
    /// Pass `None` for an undefined heap or name-index address and
    /// [`CreationOrderIndex::Undefined`] for a present index with an undefined address.
    ///
    /// # Panics
    ///
    /// Panics if a defined address is at or above the undefined marker for `offsets`.
    ///
    /// # Examples
    ///
    /// Creates a fixture for compact attributes with creation order tracked and indexed.
    ///
    /// ```
    /// use test_util::attribute_info::{AttributeInfo, CreationOrderIndex, Width};
    ///
    /// let fixture = AttributeInfo::new(
    ///     Width::Two,
    ///     Some(1),
    ///     None,
    ///     None,
    ///     CreationOrderIndex::Undefined,
    /// );
    /// assert_eq!(fixture.creation_order_index(), CreationOrderIndex::Undefined);
    /// ```
    #[track_caller]
    pub fn new(
        offsets: Width,
        max_creation_index: Option<u16>,
        fractal_heap_address: Option<u64>,
        name_index_address: Option<u64>,
        creation_order_index: CreationOrderIndex,
    ) -> Self {
        let undefined = offsets.undefined_address();
        let addresses = [
            fractal_heap_address,
            name_index_address,
            match creation_order_index {
                CreationOrderIndex::Defined(address) => Some(address),
                CreationOrderIndex::Absent | CreationOrderIndex::Undefined => None,
            },
        ];
        for address in addresses.into_iter().flatten() {
            assert!(
                address < undefined,
                "defined address {address:#x} exceeds the maximum {:#x}",
                undefined - 1
            );
        }
        Self {
            offsets,
            max_creation_index,
            fractal_heap_address,
            name_index_address,
            creation_order_index,
        }
    }

    /// Returns the width used by every address in the body.
    pub fn offsets(self) -> Width {
        self.offsets
    }

    /// Returns the maximum creation index if tracking is enabled.
    pub fn max_creation_index(self) -> Option<u16> {
        self.max_creation_index
    }

    /// Returns the address of the fractal heap, or `None` if undefined.
    pub fn fractal_heap_address(self) -> Option<u64> {
        self.fractal_heap_address
    }

    /// Returns the address of the name index, or `None` if undefined.
    pub fn name_index_address(self) -> Option<u64> {
        self.name_index_address
    }

    /// Returns the creation-order B-tree address, distinguishing an absent field from a present
    /// field with an undefined address.
    pub fn creation_order_index(self) -> CreationOrderIndex {
        self.creation_order_index
    }

    /// Builds the version 0 body without object header padding.
    pub fn build(self) -> Vec<u8> {
        let Self {
            offsets,
            max_creation_index,
            fractal_heap_address,
            name_index_address,
            creation_order_index,
        } = self;
        let mut flags = 0;
        if max_creation_index.is_some() {
            flags |= TRACKS_CREATION_ORDER;
        }
        if !matches!(creation_order_index, CreationOrderIndex::Absent) {
            flags |= INDEXES_CREATION_ORDER;
        }

        let mut body = vec![VERSION, flags];
        if let Some(maximum) = max_creation_index {
            body.extend_from_slice(&maximum.to_le_bytes());
        }
        let width = usize::from(offsets.get());
        bytes::push_address(&mut body, fractal_heap_address, width);
        bytes::push_address(&mut body, name_index_address, width);
        match creation_order_index {
            CreationOrderIndex::Absent => {}
            CreationOrderIndex::Defined(address) => bytes::push_uint(&mut body, address, width),
            CreationOrderIndex::Undefined => bytes::push_undefined_address(&mut body, width),
        }
        body
    }
}

/// The presence and address of a fixture's creation-order index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CreationOrderIndex {
    /// Omits the creation-order index field.
    Absent,
    /// Stores a defined B-tree address.
    Defined(u64),
    /// Stores the undefined address marker at the fixture's width.
    Undefined,
}

/// The version defined in "The Attribute Info Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const VERSION: u8 = 0;
/// The flag for the Maximum Creation Index field defined in "The Attribute Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const TRACKS_CREATION_ORDER: u8 = 0x01;
/// The flag for the Attribute Creation Order v2 B-tree Address field defined in
/// "The Attribute Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attrinfo
const INDEXES_CREATION_ORDER: u8 = 0x02;
