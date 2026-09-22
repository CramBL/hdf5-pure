//! Dataspace message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_simple`, version 4.0.

/// The body of a version 1 dataspace message, which has no type byte and so
/// describes only a simple dataspace.
pub fn v1(rank: u8, flags: Flags, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
    let mut message = Vec::new();
    message.push(V1_VERSION);
    message.push(rank);
    message.push(flags.0);
    message.push(0);
    message.extend_from_slice(&[0; 4]);
    push_dims(&mut message, dims, max_dims);
    message
}

/// The body of a version 2 dataspace message, whose type byte separates a
/// scalar and a null dataspace from a simple one.
pub fn v2(rank: u8, flags: Flags, kind: Kind, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
    let mut message = Vec::new();
    message.push(V2_VERSION);
    message.push(rank);
    message.push(flags.0);
    message.push(kind.0);
    push_dims(&mut message, dims, max_dims);
    message
}

/// A version 2 scalar dataspace, which holds one element and has no dimension.
pub fn scalar() -> Vec<u8> {
    v2(0, Flags::NONE, Kind::SCALAR, &[], None)
}

fn push_dims(message: &mut Vec<u8>, dims: &[u64], max_dims: Option<&[u64]>) {
    message.extend(dims.iter().flat_map(|dim| dim.to_le_bytes()));
    message.extend(
        max_dims
            .unwrap_or_default()
            .iter()
            .flat_map(|dim| dim.to_le_bytes()),
    );
}

/// Which optional dimension lists follow the fixed fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flags(pub u8);

impl Flags {
    pub const NONE: Self = Self(0x00);
    /// A maximum size per dimension follows the sizes.
    pub const MAX_DIMS: Self = Self(0x01);
    /// A permutation index per dimension follows those, which no release of
    /// the reference library has ever written.
    pub const PERMUTATION: Self = Self(0x02);
}

/// What a version 2 dataspace describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Kind(pub u8);

impl Kind {
    /// One element and no dimension.
    pub const SCALAR: Self = Self(0);
    /// A rectangular array of elements.
    pub const SIMPLE: Self = Self(1);
    /// No element at all.
    pub const NULL: Self = Self(2);
}

const V1_VERSION: u8 = 1;

const V2_VERSION: u8 = 2;
