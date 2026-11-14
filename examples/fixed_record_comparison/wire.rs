//! Field encoding and decoding for alignment-one borrowed wire records.

use zerocopy::byteorder::{BE, LE, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::error::FormatError;
use crate::fixed_record::{
    DecodedAddressCodec, FileSpaceStrategyV1Codec, FixedRecord, FixedWire, Length2, Length4,
    Length8, LengthCodec, LengthEncoding, NonzeroBool, Offset2, Offset4, Offset8, OffsetEncoding,
    PageSizeCodec, StoredAddressCodec, U16Be, U16Le, U32Le, U64Be,
};

/// A fixed-record codec with a borrowed wire representation.
///
/// Decoding applies the semantic conversions of [`FixedRecord`]. Encoding accepts the same
/// prepared value, with its width bounds already established.
pub(crate) trait ZField: FixedRecord {
    /// An alignment-one representation whose encoded size equals [`FixedWire::SIZE`].
    type Wire: Copy + FromBytes + Immutable + IntoBytes + KnownLayout + Unaligned;

    /// Applies the shared field policies to a borrowed record.
    ///
    /// # Errors
    ///
    /// Returns a [`FormatError`] if a field does not satisfy the value's semantic constraints.
    fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError>;
    fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value);

    /// Writes one prepared value into the output prefix and returns its encoded size.
    ///
    /// Preserves any output suffix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is shorter than [`FixedWire::SIZE`].
    /// The output is unchanged on error.
    #[inline]
    fn write_view(output: &mut [u8], value: &Self::Value) -> Result<usize, FormatError> {
        let available = output.len();
        let (wire, _) =
            <Self as ZField>::Wire::mut_from_prefix(output).map_err(|_layout_error| {
                FormatError::UnexpectedEof {
                    expected: <Self as FixedRecord>::Wire::SIZE,
                    available,
                }
            })?;
        Self::encode_view(wire, value);
        Ok(<Self as FixedRecord>::Wire::SIZE)
    }
}

macro_rules! primitive {
    ($codec:ident, $wire:ty) => {
        impl ZField for $codec {
            type Wire = $wire;
            #[inline]
            fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError> {
                Ok(wire.get())
            }
            #[inline]
            fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value) {
                wire.set(*value);
            }
        }
    };
}

primitive!(U16Be, U16<BE>);
primitive!(U16Le, U16<LE>);
primitive!(U32Le, U32<LE>);
primitive!(U64Be, U64<BE>);

macro_rules! raw_byte {
    ($codec:ident) => {
        impl ZField for $codec {
            type Wire = u8;
            #[inline]
            fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError> {
                Self::decode(&[*wire])
            }
            #[inline]
            fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value) {
                Self::encode(
                    &mut crate::fixed_record::RecordOutput::Slice {
                        bytes: core::slice::from_mut(wire),
                        pos: 0,
                    },
                    value,
                );
            }
        }
    };
}
raw_byte!(NonzeroBool);
raw_byte!(FileSpaceStrategyV1Codec);

/// An address-width context for little-endian borrowed fields.
pub(crate) trait ZOffset: OffsetEncoding {
    type Endian: Copy + FromBytes + Immutable + IntoBytes + KnownLayout + Unaligned;
    fn bytes(wire: &Self::Endian) -> Self::Wire;
    fn set(wire: &mut Self::Endian, value: u64);
}

/// A length-width context for little-endian borrowed fields.
pub(crate) trait ZLength: LengthEncoding {
    type Endian: Copy + FromBytes + Immutable + IntoBytes + KnownLayout + Unaligned;
    fn bytes(wire: &Self::Endian) -> Self::Wire;
    fn set(wire: &mut Self::Endian, value: u64);
}

macro_rules! widths {
    ($offset:ident, $length:ident, $wire:ty, $integer:ty, $bytes:literal) => {
        impl ZOffset for $offset {
            type Endian = $wire;
            #[inline]
            fn bytes(wire: &Self::Endian) -> Self::Wire {
                wire.get().to_le_bytes()
            }
            #[inline]
            fn set(wire: &mut Self::Endian, value: u64) {
                wire.set(value as $integer);
            }
        }
        impl ZLength for $length {
            type Endian = $wire;
            #[inline]
            fn bytes(wire: &Self::Endian) -> Self::Wire {
                wire.get().to_le_bytes()
            }
            #[inline]
            fn set(wire: &mut Self::Endian, value: u64) {
                wire.set(value as $integer);
            }
        }
    };
}
widths!(Offset2, Length2, U16<LE>, u16, 2);
widths!(Offset4, Length4, U32<LE>, u32, 4);
widths!(Offset8, Length8, U64<LE>, u64, 8);

impl<O: ZOffset> ZField for StoredAddressCodec<O> {
    type Wire = O::Endian;
    #[inline]
    fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError> {
        Self::decode(&O::bytes(wire))
    }
    #[inline]
    fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value) {
        O::set(wire, value.get().get());
    }
}
impl<O: ZOffset> ZField for DecodedAddressCodec<O> {
    type Wire = O::Endian;
    #[inline]
    fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError> {
        Self::decode(&O::bytes(wire))
    }
    #[inline]
    fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value) {
        let value = match value.get() {
            hdf5_pure_core::DecodedStoredAddress::Undefined => O::MAX,
            hdf5_pure_core::DecodedStoredAddress::Defined(address) => address.get(),
        };
        O::set(wire, value);
    }
}
impl<L: ZLength> ZField for LengthCodec<L> {
    type Wire = L::Endian;
    #[inline]
    fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError> {
        Self::decode(&L::bytes(wire))
    }
    #[inline]
    fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value) {
        L::set(wire, value.get());
    }
}
impl<L: ZLength> ZField for PageSizeCodec<L> {
    type Wire = L::Endian;
    #[inline]
    fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError> {
        Self::decode(&L::bytes(wire))
    }
    #[inline]
    fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value) {
        L::set(wire, value.get().get());
    }
}
