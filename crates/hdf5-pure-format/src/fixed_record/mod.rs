//! Fixed records with named field encodings.
//!
//! Decoders read wire fields and create owned semantic values, including addresses and lengths
//! bounded by their selected widths. Callers decode variable bodies and optional tails outside the
//! fixed record.

use core::marker::PhantomData;

use hdf5_pure_core::FileSpaceStrategy;

use crate::address::{DecodedStoredAddress, StoredAddress, StoredAddressExt};
use crate::bytes::Fields;
use crate::error::FormatError;
use crate::width::{LengthWidth, OffsetWidth};

/// A codec for a fixed wire representation and its owned semantic value.
///
/// Construct [`Self::Value`] with its semantic constraints established before encoding. Field
/// conversion during encoding is infallible.
pub(crate) trait FixedRecord {
    /// The fixed fields read before semantic decoding.
    type Wire: FixedWire;
    /// The owned semantic value.
    type Value;

    /// Decodes acquired wire fields into a semantic value.
    ///
    /// # Errors
    ///
    /// Returns a [`FormatError`] if the fields do not satisfy the value's semantic constraints.
    fn decode(wire: &Self::Wire) -> Result<Self::Value, FormatError>;

    /// Writes the value's fields in wire order.
    ///
    /// Implementations write exactly [`Self::Wire::SIZE`](FixedWire::SIZE) bytes.
    fn encode(output: &mut RecordOutput<'_>, value: &Self::Value);

    /// Decodes one record at the cursor and leaves following bytes available.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the complete record is unavailable, leaving the
    /// cursor unchanged. Returns the errors of [`Self::decode`] for invalid semantic fields, after
    /// advancing past the record.
    #[inline]
    fn read(fields: &mut Fields<'_>) -> Result<Self::Value, FormatError> {
        let bytes = fields.take_arrays::<1>(Self::Wire::SIZE)?.as_flattened();
        let wire = Self::Wire::read(&mut Fields::new(bytes, 0))?;
        Self::decode(&wire)
    }

    /// Writes one prepared value into the output prefix and returns its encoded size.
    ///
    /// Preserves the output suffix.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the output is shorter than
    /// [`Self::Wire::SIZE`](FixedWire::SIZE), leaving the entire output unchanged.
    #[inline]
    fn write(output: &mut [u8], value: &Self::Value) -> Result<usize, FormatError> {
        let available = output.len();
        let Some(bytes) = output.get_mut(..Self::Wire::SIZE) else {
            return Err(FormatError::UnexpectedEof {
                expected: Self::Wire::SIZE,
                available,
            });
        };
        Self::encode(&mut RecordOutput { bytes, pos: 0 }, value);
        Ok(Self::Wire::SIZE)
    }
}

/// Fixed fields acquired before their semantic interpretation.
///
/// The encoded extent is independent of the semantic value's native layout.
pub(crate) trait FixedWire: Sized {
    /// The number of encoded bytes.
    const SIZE: usize;

    /// Reads exactly [`Self::SIZE`] bytes from the cursor.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a field is incomplete.
    fn read(fields: &mut Fields<'_>) -> Result<Self, FormatError>;
}

impl<const N: usize> FixedWire for [u8; N] {
    const SIZE: usize = N;

    #[inline]
    fn read(fields: &mut Fields<'_>) -> Result<Self, FormatError> {
        fields.array()
    }
}

/// Checked construction and field access for a record's semantic value.
///
/// Use the generated field type directly, or select a separate value type in the declaration.
/// Implement this trait for that type to establish constraints between fields.
pub(crate) trait RecordValue<F>: Sized {
    /// Creates a value after decoding each field.
    ///
    /// # Errors
    ///
    /// Returns a [`FormatError`] if the fields do not satisfy the value's semantic constraints.
    fn from_fields(fields: F) -> Result<Self, FormatError>;

    /// Borrows the fields encoded for this value.
    fn fields(&self) -> &F;
}

impl<F> RecordValue<F> for F {
    #[inline]
    fn from_fields(fields: F) -> Result<Self, FormatError> {
        Ok(fields)
    }

    #[inline]
    fn fields(&self) -> &F {
        self
    }
}

/// An encoder's acquired output prefix and next field position.
pub(crate) struct RecordOutput<'a> {
    bytes: &'a mut [u8],
    pos: usize,
}

impl RecordOutput<'_> {
    /// Writes one field and advances the output position.
    ///
    /// # Panics
    ///
    /// Panics if the field extends past the prefix.
    #[inline]
    fn array<const N: usize>(&mut self, value: &[u8; N]) {
        // The writer acquires the complete record prefix before encoding its fields.
        self.bytes
            .get_mut(self.pos..)
            .and_then(|bytes| bytes.first_chunk_mut::<N>())
            .expect("fixed-record encoders write their declared extent")
            .copy_from_slice(value);
        self.pos += N;
    }
}

/// Declares fields in wire order and generates their record codec.
///
/// Supply a [`FixedRecord`] codec for each field, including nested records. The optional value type
/// implements [`RecordValue`] for construction from the generated semantic fields.
#[cfg_attr(
    not(test),
    expect(unused_macros, reason = "only unit tests declare fixed records")
)]
macro_rules! fixed_record {
    (
        $(#[$attribute:meta])*
        $visibility:vis $codec:ident $(<$($parameter:ident: $bound:path),+>)?
        => $wire:ident for $fields:ident {
            $($field:ident: $field_codec:ty),+ $(,)?
        }
    ) => {
        $crate::fixed_record::fixed_record! {
            @define [$(#[$attribute])*]
            $visibility $codec $(<$($parameter: $bound),+>)?
            => $wire for $fields => $fields $(<$($parameter),+>)? {
                $($field: $field_codec),+
            }
        }
    };
    (
        $(#[$attribute:meta])*
        $visibility:vis $codec:ident $(<$($parameter:ident: $bound:path),+>)?
        => $wire:ident for $fields:ident => $value:ty {
            $($field:ident: $field_codec:ty),+ $(,)?
        }
    ) => {
        $crate::fixed_record::fixed_record! {
            @define [$(#[$attribute])*]
            $visibility $codec $(<$($parameter: $bound),+>)?
            => $wire for $fields => $value {
                $($field: $field_codec),+
            }
        }
    };
    (
        @define [$(#[$attribute:meta])*]
        $visibility:vis $codec:ident $(<$($parameter:ident: $bound:path),+>)?
        => $wire:ident for $fields:ident => $value:ty {
            $($field:ident: $field_codec:ty),+
        }
    ) => {
        $visibility struct $codec $(<$($parameter: $bound),+>)? (
            core::marker::PhantomData<($(($($parameter,)+),)?)>
        );

        #[repr(C)]
        $visibility struct $wire $(<$($parameter: $bound),+>)? {
            $(pub(crate) $field: <$field_codec as $crate::fixed_record::FixedRecord>::Wire),+
        }

        $(#[$attribute])*
        $visibility struct $fields $(<$($parameter: $bound),+>)? {
            $(pub(crate) $field: <$field_codec as $crate::fixed_record::FixedRecord>::Value),+
        }

        impl $(<$($parameter: $bound),+>)? $crate::fixed_record::FixedWire
            for $wire $(<$($parameter),+>)?
        {
            const SIZE: usize = 0 $(+
                <<$field_codec as $crate::fixed_record::FixedRecord>::Wire
                    as $crate::fixed_record::FixedWire>::SIZE
            )+;

            #[inline]
            fn read(fields: &mut $crate::bytes::Fields<'_>) -> Result<Self, $crate::error::FormatError> {
                Ok(Self {
                    $($field: <<$field_codec as $crate::fixed_record::FixedRecord>::Wire
                        as $crate::fixed_record::FixedWire>::read(fields)?),+
                })
            }
        }

        impl $(<$($parameter: $bound),+>)? $crate::fixed_record::FixedRecord
            for $codec $(<$($parameter),+>)?
        {
            type Wire = $wire $(<$($parameter),+>)?;
            type Value = $value;

            #[inline]
            fn decode(wire: &Self::Wire) -> Result<Self::Value, $crate::error::FormatError> {
                <Self::Value as $crate::fixed_record::RecordValue<$fields $(<$($parameter),+>)?>>
                    ::from_fields($fields {
                        $($field: <$field_codec as $crate::fixed_record::FixedRecord>
                            ::decode(&wire.$field)?),+
                    })
            }

            #[inline]
            fn encode(
                output: &mut $crate::fixed_record::RecordOutput<'_>,
                value: &Self::Value,
            ) {
                let $fields { $($field),+ } =
                    <Self::Value as $crate::fixed_record::RecordValue<$fields $(<$($parameter),+>)?>>
                        ::fields(value);
                $(<$field_codec as $crate::fixed_record::FixedRecord>::encode(output, $field);)+
            }
        }
    };
}

#[cfg_attr(
    not(test),
    expect(unused_imports, reason = "only unit tests declare fixed records")
)]
pub(crate) use fixed_record;

macro_rules! integer_codecs {
    ($($codec:ident: $integer:ty, $bytes:literal, $decode:ident, $encode:ident);+ $(;)?) => {
        $(pub(crate) struct $codec;

        impl FixedRecord for $codec {
            type Wire = [u8; $bytes];
            type Value = $integer;

            #[inline]
            fn decode(wire: &Self::Wire) -> Result<Self::Value, FormatError> {
                Ok(<$integer>::$decode(*wire))
            }

            #[inline]
            fn encode(output: &mut RecordOutput<'_>, value: &Self::Value) {
                output.array(&value.$encode())
            }
        })+
    };
}

integer_codecs! {
    U8: u8, 1, from_le_bytes, to_le_bytes;
    U16Le: u16, 2, from_le_bytes, to_le_bytes;
    U16Be: u16, 2, from_be_bytes, to_be_bytes;
    U32Le: u32, 4, from_le_bytes, to_le_bytes;
    U32Be: u32, 4, from_be_bytes, to_be_bytes;
    U64Le: u64, 8, from_le_bytes, to_le_bytes;
    U64Be: u64, 8, from_be_bytes, to_be_bytes;
}

/// A boolean field decoded from any byte and encoded as zero or one.
pub(crate) struct NonzeroBool;

impl FixedRecord for NonzeroBool {
    type Wire = [u8; 1];
    type Value = bool;

    #[inline]
    fn decode(wire: &Self::Wire) -> Result<Self::Value, FormatError> {
        Ok(wire[0] != 0)
    }

    #[inline]
    fn encode(output: &mut RecordOutput<'_>, value: &Self::Value) {
        output.array(&[u8::from(*value)])
    }
}

/// Declares both directions of a closed one-byte tag mapping.
macro_rules! closed_enum_codec {
    ($visibility:vis $codec:ident for $value:ty {
        $($variant:path => $tag:literal),+ $(,)?
    } error $error:path) => {
        $visibility struct $codec;

        impl $crate::fixed_record::FixedRecord for $codec {
            type Wire = [u8; 1];
            type Value = $value;

            #[inline]
            fn decode(wire: &Self::Wire) -> Result<Self::Value, $crate::error::FormatError> {
                Ok(match wire[0] {
                    $($tag => $variant,)+
                    tag => return Err($error(tag)),
                })
            }

            #[inline]
            fn encode(
                output: &mut $crate::fixed_record::RecordOutput<'_>,
                value: &Self::Value,
            ) {
                <$crate::fixed_record::U8 as $crate::fixed_record::FixedRecord>::encode(
                    output, &match value { $($variant => $tag),+ },
                )
            }
        }
    };
}

// "The File Space Info Message", format specification, version 4.0, defines the version 1
// strategy codes.
closed_enum_codec! {
    pub(crate) FileSpaceStrategyV1Codec for FileSpaceStrategy {
        FileSpaceStrategy::FsmAggr => 0,
        FileSpaceStrategy::Page => 1,
        FileSpaceStrategy::Aggr => 2,
        FileSpaceStrategy::None => 3,
    } error FormatError::InvalidFileSpaceStrategy
}

/// A sealed address-width context for little-endian fields.
pub(crate) trait OffsetEncoding:
    private::OffsetEncoding + Copy + core::fmt::Debug + Eq
{
    /// The address bytes.
    type Wire: FixedWire;
    /// The address width reported in encoding errors.
    const WIDTH: OffsetWidth;
    /// The largest stored address, including the undefined marker.
    const MAX: u64;

    /// Decodes the raw address bits at this width.
    fn decode(wire: &Self::Wire) -> u64;
    /// Writes a stored address whose width bound is established.
    fn encode(output: &mut RecordOutput<'_>, value: &BoundStoredAddress<Self>);
}

/// A sealed length-width context for little-endian fields.
pub(crate) trait LengthEncoding:
    private::LengthEncoding + Copy + core::fmt::Debug + Eq
{
    /// The length bytes.
    type Wire: FixedWire;
    /// The length width reported in encoding errors.
    const WIDTH: LengthWidth;
    /// The largest length represented at this width.
    const MAX: u64;

    /// Decodes the length at this width.
    fn decode(wire: &Self::Wire) -> u64;
    /// Writes a length whose width bound is established.
    fn encode(output: &mut RecordOutput<'_>, value: &BoundLength<Self>);
}

macro_rules! width_encodings {
    ($($offset:ident, $length:ident, $width:ident, $bytes:literal, $integer:ty);+ $(;)?) => {
        $(#[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub(crate) struct $offset;

        impl private::OffsetEncoding for $offset {}

        impl OffsetEncoding for $offset {
            type Wire = [u8; $bytes];
            const WIDTH: OffsetWidth = OffsetWidth::$width;
            const MAX: u64 = <$integer>::MAX as u64;

            #[inline]
            fn decode(wire: &Self::Wire) -> u64 {
                <$integer>::from_le_bytes(*wire) as u64
            }

            #[inline]
            fn encode(output: &mut RecordOutput<'_>, value: &BoundStoredAddress<Self>) {
                output.array(value.get().get().to_le_bytes().first_chunk::<$bytes>()
                    .expect("every supported address width is at most eight bytes"))
            }
        }

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub(crate) struct $length;

        impl private::LengthEncoding for $length {}

        impl LengthEncoding for $length {
            type Wire = [u8; $bytes];
            const WIDTH: LengthWidth = LengthWidth::$width;
            const MAX: u64 = <$integer>::MAX as u64;

            #[inline]
            fn decode(wire: &Self::Wire) -> u64 {
                <$integer>::from_le_bytes(*wire) as u64
            }

            #[inline]
            fn encode(output: &mut RecordOutput<'_>, value: &BoundLength<Self>) {
                output.array(value.get().to_le_bytes().first_chunk::<$bytes>()
                    .expect("every supported length width is at most eight bytes"))
            }
        })+
    };
}

width_encodings! {
    Offset2, Length2, Two, 2, u16;
    Offset4, Length4, Four, 4, u32;
    Offset8, Length8, Eight, 8, u64;
}

/// A stored address known to fit the address encoding `O`.
///
/// The raw bits include zero and the width's undefined marker. A caller resolves the address and
/// checks its target separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BoundStoredAddress<O> {
    address: StoredAddress,
    encoding: PhantomData<O>,
}

impl<O: OffsetEncoding> BoundStoredAddress<O> {
    /// Creates an address bounded by `O`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if the address exceeds [`OffsetEncoding::MAX`].
    pub(crate) fn new(address: StoredAddress) -> Result<Self, FormatError> {
        if address.get() > O::MAX {
            return Err(FormatError::AddressTooLarge {
                address: address.get(),
                offset_size: O::WIDTH.get(),
            });
        }
        Ok(Self {
            address,
            encoding: PhantomData,
        })
    }

    /// Returns the stored address bits.
    pub(crate) const fn get(&self) -> StoredAddress {
        self.address
    }
}

/// A raw stored-address field at address encoding `O`.
pub(crate) struct StoredAddressCodec<O>(PhantomData<O>);

impl<O: OffsetEncoding> FixedRecord for StoredAddressCodec<O> {
    type Wire = O::Wire;
    type Value = BoundStoredAddress<O>;

    #[inline]
    fn decode(wire: &Self::Wire) -> Result<Self::Value, FormatError> {
        Ok(BoundStoredAddress {
            address: StoredAddress::new(O::decode(wire)),
            encoding: PhantomData,
        })
    }

    #[inline]
    fn encode(output: &mut RecordOutput<'_>, value: &Self::Value) {
        O::encode(output, value)
    }
}

/// A decoded address whose defined value fits the address encoding `O`.
///
/// The undefined marker is all ones at this width. A defined address is less than that marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BoundDecodedAddress<O> {
    stored: BoundStoredAddress<O>,
}

impl<O: OffsetEncoding> BoundDecodedAddress<O> {
    /// Creates a decoded address bounded by `O`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if a defined address exceeds the width, or
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals the width's undefined marker.
    pub(crate) fn new(address: DecodedStoredAddress) -> Result<Self, FormatError> {
        let stored = match address {
            DecodedStoredAddress::Undefined => BoundStoredAddress {
                address: StoredAddress::new(O::MAX),
                encoding: PhantomData,
            },
            DecodedStoredAddress::Defined(address) => {
                let stored = BoundStoredAddress::new(address)?;
                if address.get() == O::MAX {
                    return Err(FormatError::DefinedAddressIsUndefined {
                        address: address.get(),
                        offset_size: O::WIDTH.get(),
                    });
                }
                stored
            }
        };
        Ok(Self { stored })
    }

    /// Returns the address with its undefined marker decoded.
    pub(crate) const fn get(&self) -> DecodedStoredAddress {
        if self.stored.address.get() == O::MAX {
            DecodedStoredAddress::Undefined
        } else {
            DecodedStoredAddress::Defined(self.stored.address)
        }
    }
}

/// An address field with decoded undefined-address semantics at encoding `O`.
pub(crate) struct DecodedAddressCodec<O>(PhantomData<O>);

impl<O: OffsetEncoding> FixedRecord for DecodedAddressCodec<O> {
    type Wire = O::Wire;
    type Value = BoundDecodedAddress<O>;

    #[inline]
    fn decode(wire: &Self::Wire) -> Result<Self::Value, FormatError> {
        Ok(BoundDecodedAddress {
            stored: StoredAddressCodec::decode(wire)?,
        })
    }

    #[inline]
    fn encode(output: &mut RecordOutput<'_>, value: &Self::Value) {
        O::encode(output, &value.stored)
    }
}

/// A length known to fit the length encoding `L`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BoundLength<L> {
    value: u64,
    encoding: PhantomData<L>,
}

impl<L: LengthEncoding> BoundLength<L> {
    /// Creates a length bounded by `L`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if the value exceeds [`LengthEncoding::MAX`].
    pub(crate) fn new(value: u64) -> Result<Self, FormatError> {
        if value > L::MAX {
            return Err(FormatError::LengthTooLarge {
                length: value,
                length_size: L::WIDTH.get(),
            });
        }
        Ok(Self {
            value,
            encoding: PhantomData,
        })
    }

    /// Returns the stored length.
    pub(crate) const fn get(&self) -> u64 {
        self.value
    }
}

/// A length field at length encoding `L`.
pub(crate) struct LengthCodec<L>(PhantomData<L>);

impl<L: LengthEncoding> FixedRecord for LengthCodec<L> {
    type Wire = L::Wire;
    type Value = BoundLength<L>;

    #[inline]
    fn decode(wire: &Self::Wire) -> Result<Self::Value, FormatError> {
        Ok(BoundLength {
            value: L::decode(wire),
            encoding: PhantomData,
        })
    }

    #[inline]
    fn encode(output: &mut RecordOutput<'_>, value: &Self::Value) {
        L::encode(output, value)
    }
}

mod private {
    pub(crate) trait OffsetEncoding {}
    pub(crate) trait LengthEncoding {}
}

#[cfg(test)]
mod tests;
