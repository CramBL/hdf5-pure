//! Safe field-wise encoding and decoding for HDF5 compound datatypes.

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};

use crate::convert::{TryToUsize, u32_from};
use crate::datatype::{CompoundMember, Datatype, DatatypeByteOrder};
use crate::error::FormatError;
use crate::type_builders::{
    make_f32_type, make_f64_type, make_i8_type, make_i16_type, make_i32_type, make_i64_type,
    make_u8_type, make_u16_type, make_u32_type, make_u64_type,
};

/// A type that can occupy one field of an HDF5 compound value.
///
/// Implementations encode and decode the field explicitly. They must not copy
/// padding bytes or rely on the Rust memory layout of `Self`.
pub trait CompoundField: Sized {
    /// Canonical datatype used when writing this field.
    fn datatype() -> Result<Datatype, FormatError>;

    /// Append this field's canonical byte representation to `output`.
    fn encode_field(&self, output: &mut Vec<u8>);

    /// Decode this field from bytes described by `datatype`.
    fn decode_field(datatype: &Datatype, bytes: &[u8]) -> Result<Self, FormatError>;
}

/// A safely encoded HDF5 compound element.
///
/// The built-in implementations cover tuples of one through twelve numeric
/// fields. Tuple fields are named `"0"`, `"1"`, and so on in the file. The
/// tuple's Rust memory representation is never inspected.
///
/// User-defined structs can implement this trait by encoding each field
/// explicitly and decoding fields according to the offsets in the supplied
/// [`Datatype`].
///
/// It is the bound of [`with_compound_values`] and [`read_compound`], which the [compound types
/// guide](crate::_guide::compound_types) covers.
///
/// [`with_compound_values`]: crate::DatasetBuilder::with_compound_values
/// [`read_compound`]: crate::Dataset::read_compound
pub trait CompoundType: Sized {
    /// Canonical datatype used when creating a dataset for this type.
    fn datatype() -> Result<Datatype, FormatError>;

    /// Append one canonical compound element to `output`.
    fn encode(&self, output: &mut Vec<u8>);

    /// Decode one element from `bytes` using its exact on-disk datatype.
    fn decode(datatype: &Datatype, bytes: &[u8]) -> Result<Self, FormatError>;
}

fn require_bytes(bytes: &[u8], size: usize) -> Result<&[u8], FormatError> {
    if bytes.len() != size {
        return Err(FormatError::DataSizeMismatch {
            expected: size,
            actual: bytes.len(),
        });
    }
    Ok(bytes)
}

fn integer_order(
    datatype: &Datatype,
    size: u32,
    signed: bool,
    name: &str,
) -> Result<DatatypeByteOrder, FormatError> {
    match datatype {
        Datatype::FixedPoint {
            size: actual_size,
            byte_order,
            signed: actual_signed,
            bit_offset: 0,
            bit_precision,
        } if *actual_size == size
            && *actual_signed == signed
            && u32::from(*bit_precision) == size * 8 =>
        {
            Ok(byte_order.clone())
        }
        _ => Err(FormatError::CompoundFieldTypeMismatch(name.to_string())),
    }
}

fn float_order(
    datatype: &Datatype,
    size: u32,
    name: &str,
) -> Result<DatatypeByteOrder, FormatError> {
    let standard = match (datatype, size) {
        (
            Datatype::FloatingPoint {
                size: 4,
                byte_order,
                bit_offset: 0,
                bit_precision: 32,
                exponent_location: 23,
                exponent_size: 8,
                mantissa_location: 0,
                mantissa_size: 23,
                exponent_bias: 127,
            },
            4,
        ) => Some(byte_order.clone()),
        (
            Datatype::FloatingPoint {
                size: 8,
                byte_order,
                bit_offset: 0,
                bit_precision: 64,
                exponent_location: 52,
                exponent_size: 11,
                mantissa_location: 0,
                mantissa_size: 52,
                exponent_bias: 1023,
            },
            8,
        ) => Some(byte_order.clone()),
        _ => None,
    };
    standard.ok_or_else(|| FormatError::CompoundFieldTypeMismatch(name.to_string()))
}

fn ordered<const N: usize>(bytes: &[u8], order: DatatypeByteOrder) -> Result<[u8; N], FormatError> {
    let bytes = require_bytes(bytes, N)?;
    let mut array = [0u8; N];
    array.copy_from_slice(bytes);
    match order {
        DatatypeByteOrder::LittleEndian => Ok(array),
        DatatypeByteOrder::BigEndian => {
            array.reverse();
            Ok(array)
        }
        DatatypeByteOrder::Vax => Err(FormatError::InvalidByteOrder(2)),
    }
}

macro_rules! impl_integer_field {
    ($ty:ty, $make:ident, $size:expr, $signed:expr) => {
        impl CompoundField for $ty {
            fn datatype() -> Result<Datatype, FormatError> {
                Ok($make())
            }

            fn encode_field(&self, output: &mut Vec<u8>) {
                output.extend_from_slice(&self.to_le_bytes());
            }

            fn decode_field(datatype: &Datatype, bytes: &[u8]) -> Result<Self, FormatError> {
                let order = integer_order(datatype, $size, $signed, "")?;
                Ok(<$ty>::from_le_bytes(ordered(bytes, order)?))
            }
        }
    };
}

impl_integer_field!(i8, make_i8_type, 1, true);
impl_integer_field!(i16, make_i16_type, 2, true);
impl_integer_field!(i32, make_i32_type, 4, true);
impl_integer_field!(i64, make_i64_type, 8, true);
impl_integer_field!(u8, make_u8_type, 1, false);
impl_integer_field!(u16, make_u16_type, 2, false);
impl_integer_field!(u32, make_u32_type, 4, false);
impl_integer_field!(u64, make_u64_type, 8, false);

macro_rules! impl_float_field {
    ($ty:ty, $make:ident, $size:expr) => {
        impl CompoundField for $ty {
            fn datatype() -> Result<Datatype, FormatError> {
                Ok($make())
            }

            fn encode_field(&self, output: &mut Vec<u8>) {
                output.extend_from_slice(&self.to_le_bytes());
            }

            fn decode_field(datatype: &Datatype, bytes: &[u8]) -> Result<Self, FormatError> {
                let order = float_order(datatype, $size, "")?;
                Ok(<$ty>::from_le_bytes(ordered(bytes, order)?))
            }
        }
    };
}

impl_float_field!(f32, make_f32_type, 4);
impl_float_field!(f64, make_f64_type, 8);

fn compound_parts<'a>(
    datatype: &'a Datatype,
    bytes: &[u8],
) -> Result<(&'a [CompoundMember], u32), FormatError> {
    match datatype {
        Datatype::Compound { size, members } => {
            require_bytes(bytes, size.to_usize()?)?;
            Ok((members, *size))
        }
        _ => Err(FormatError::TypeMismatch {
            expected: "Compound",
            actual: "non-Compound",
        }),
    }
}

fn reported_compound_size(bytes: &[u8]) -> u32 {
    u32::try_from(bytes.len()).unwrap_or(u32::MAX)
}

fn decode_named<T: CompoundField>(
    members: &[CompoundMember],
    bytes: &[u8],
    name: &str,
) -> Result<T, FormatError> {
    let member = members
        .iter()
        .find(|member| member.name == name)
        .ok_or_else(|| FormatError::CompoundFieldMissing(name.to_string()))?;
    let start =
        usize::try_from(member.byte_offset).map_err(|_| FormatError::CompoundFieldOutOfBounds {
            name: name.to_string(),
            offset: member.byte_offset,
            field_size: member.datatype.type_size(),
            compound_size: reported_compound_size(bytes),
        })?;
    let end = start
        .checked_add(member.datatype.type_size().to_usize()?)
        .ok_or_else(|| FormatError::CompoundFieldOutOfBounds {
            name: name.to_string(),
            offset: member.byte_offset,
            field_size: member.datatype.type_size(),
            compound_size: reported_compound_size(bytes),
        })?;
    let field_bytes =
        bytes
            .get(start..end)
            .ok_or_else(|| FormatError::CompoundFieldOutOfBounds {
                name: name.to_string(),
                offset: member.byte_offset,
                field_size: member.datatype.type_size(),
                compound_size: reported_compound_size(bytes),
            })?;
    T::decode_field(&member.datatype, field_bytes).map_err(|error| match error {
        FormatError::CompoundFieldTypeMismatch(inner) => {
            let path = if inner.is_empty() {
                name.to_string()
            } else {
                format!("{}.{}", name, inner)
            };
            FormatError::CompoundFieldTypeMismatch(path)
        }
        other => other,
    })
}

macro_rules! impl_compound_tuple {
    ($($type:ident:$index:tt),+) => {
        impl<$($type: CompoundField),+> CompoundType for ($($type,)+) {
            fn datatype() -> Result<Datatype, FormatError> {
                let mut offset = 0u64;
                let mut members = Vec::new();
                $(
                    let datatype = $type::datatype()?;
                    members.push(CompoundMember {
                        name: stringify!($index).to_string(),
                        byte_offset: offset,
                        datatype: datatype.clone(),
                    });
                    offset += u64::from(datatype.type_size());
                )+
                Ok(Datatype::Compound {
                    size: u32_from(offset)?,
                    members,
                })
            }

            fn encode(&self, output: &mut Vec<u8>) {
                $(self.$index.encode_field(output);)+
            }

            fn decode(datatype: &Datatype, bytes: &[u8]) -> Result<Self, FormatError> {
                let (members, _) = compound_parts(datatype, bytes)?;
                Ok(($(decode_named::<$type>(members, bytes, stringify!($index))?,)+))
            }
        }

        impl<$($type: CompoundField),+> CompoundField for ($($type,)+) {
            fn datatype() -> Result<Datatype, FormatError> {
                <Self as CompoundType>::datatype()
            }

            fn encode_field(&self, output: &mut Vec<u8>) {
                <Self as CompoundType>::encode(self, output);
            }

            fn decode_field(
                datatype: &Datatype,
                bytes: &[u8],
            ) -> Result<Self, FormatError> {
                <Self as CompoundType>::decode(datatype, bytes)
            }
        }
    };
}

impl_compound_tuple!(A:0);
impl_compound_tuple!(A:0, B:1);
impl_compound_tuple!(A:0, B:1, C:2);
impl_compound_tuple!(A:0, B:1, C:2, D:3);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4, F:5);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
impl_compound_tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);
