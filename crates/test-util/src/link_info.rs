use crate::bytes;

/// The checked fields of an independently constructed Link Info body.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    width: Width,
    maximum: Option<u64>,
    heap: Option<u64>,
    name_index: Option<u64>,
    creation_index: CreationIndex,
}

impl Body {
    /// Constructs a body if each defined address is below the width's undefined marker.
    ///
    /// Pass [`None`] for an undefined heap or name-index address. The maximum and
    /// creation-order index fields are independently optional.
    pub fn new(
        width: Width,
        maximum: Option<u64>,
        heap: Option<u64>,
        name_index: Option<u64>,
        creation_index: CreationIndex,
    ) -> Option<Self> {
        let addresses = [heap, name_index, creation_index.address()];
        if addresses
            .into_iter()
            .flatten()
            .any(|address| address >= width.marker())
        {
            return None;
        }
        Some(Self {
            width,
            maximum,
            heap,
            name_index,
            creation_index,
        })
    }

    /// Builds the version 0 body in wire order.
    pub fn build(&self) -> Vec<u8> {
        let mut flags = 0;
        if self.maximum.is_some() {
            flags |= TRACKING;
        }
        if self.creation_index != CreationIndex::Absent {
            flags |= INDEXING;
        }
        let mut body = vec![VERSION, flags];
        if let Some(maximum) = self.maximum {
            body.extend_from_slice(&maximum.to_le_bytes());
        }
        let width = usize::from(self.width.bytes());
        bytes::push_address(&mut body, self.heap, width);
        bytes::push_address(&mut body, self.name_index, width);
        if self.creation_index != CreationIndex::Absent {
            bytes::push_address(&mut body, self.creation_index.address(), width);
        }
        body
    }
}

/// The stored presence and address of the creation-order index field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CreationIndex {
    /// Omits the address field.
    Absent,
    /// Stores a defined index address.
    Defined(u64),
    /// Stores the width's undefined address.
    Undefined,
}

impl CreationIndex {
    fn address(self) -> Option<u64> {
        if let Self::Defined(address) = self {
            Some(address)
        } else {
            None
        }
    }
}

/// The file offset widths supported by the fixture.
#[derive(Clone, Copy, Debug)]
pub enum Width {
    /// Two bytes.
    Two,
    /// Four bytes.
    Four,
    /// Eight bytes.
    Eight,
}

impl Width {
    /// Returns the address width in bytes.
    pub const fn bytes(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    fn marker(self) -> u64 {
        match self {
            Self::Two => u64::from(u16::MAX),
            Self::Four => u64::from(u32::MAX),
            Self::Eight => u64::MAX,
        }
    }
}

// "The Link Info Message" of the format specification, version 4.0, defines version zero
// and separate tracking and indexing presence flags.
const VERSION: u8 = 0;
const TRACKING: u8 = 0x01;
const INDEXING: u8 = 0x02;
