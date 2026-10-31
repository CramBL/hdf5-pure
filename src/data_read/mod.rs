//! Raw data reading and typed conversion for HDF5 datasets.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use core::marker::PhantomData;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::MetadataFrame;
use hdf5_pure_core::PhysicalAddress;

#[cfg(test)]
use crate::VariableLengthStringType;
#[cfg(test)]
use crate::VariableLengthType;
use crate::chunk_cache::ChunkCache;
use crate::chunked_read;
#[cfg(test)]
use crate::data_layout::DataLayout;
pub(crate) use crate::data_read::primitive::NumericReadTarget;
use crate::data_read::primitive::{FromEndianBytes, H5Conversion};
#[cfg(test)]
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
#[cfg(test)]
use crate::datatype::byte_order::DatatypeByteOrder;
use crate::datatype::byte_order::FixedWidthByteOrder;
use crate::datatype::layout::StandardWidth;
use crate::datatype::numeric::{NumericDecodePlan, NumericEncoding};
use crate::error::FormatError;
#[cfg(test)]
use crate::fill_value::FillPattern;
use crate::read_spec::{RawReadSpec, RawReadStorage};
use crate::source;
use crate::source::Source;
use crate::width::FormatWidths;
#[cfg(test)]
use crate::{FixedPointType, FloatingPointType};

mod primitive;

#[cfg(test)]
const TEST_WIDTHS: FormatWidths = FormatWidths {
    offsets: crate::width::OffsetWidth::Eight,
    lengths: crate::width::LengthWidth::Eight,
};

/// Read raw bytes for a dataset given its layout and the file data buffer,
/// using default filter/size parameters.
///
/// A test-only convenience wrapper over [`read_raw_data_full`].
/// The read paths call `read_raw_data_full` directly with the real pipeline and offset sizes.
#[cfg(test)]
pub fn read_raw_data(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
) -> Result<Vec<u8>, FormatError> {
    read_raw_data_full(
        file_data,
        RawReadSpec::parse(layout, dataspace, datatype, None, FillPattern::ZERO)?,
        TEST_WIDTHS,
    )
}

/// Read raw bytes with full parameters including filter pipeline and sizes.
pub fn read_raw_data_full(
    file_data: &[u8],
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (dataspace, fill) = (spec.dataspace(), spec.fill());
    let expected_size = spec.byte_len()?;

    // Zero-element datasets have no data to read.
    if dataspace.num_elements() == 0 {
        return Ok(Vec::new());
    }

    match spec.storage() {
        RawReadStorage::Compact { data } => Ok(data.to_vec()),
        RawReadStorage::Contiguous { address } => Ok(source::bytes_in(
            file_data,
            FileRange::new(PhysicalAddress::new(address.get()), spec.byte_len_u64())?,
        )?
        .to_vec()),
        RawReadStorage::ContiguousUnallocated => fill.buffer(expected_size),
        RawReadStorage::Chunked { .. } => {
            chunked_read::read_chunked_data_cached(file_data, spec, widths, &ChunkCache::new())
        }
        RawReadStorage::Virtual => Err(FormatError::UnsupportedVirtualLayout),
    }
}

/// Reads raw bytes with index and decoded-chunk caching.
///
/// The reader resolves metadata positions from the frame's origin and borrows payload bytes
/// from its slice at local offsets.
/// Compact and contiguous layouts use [`read_raw_data_full`].
///
/// # Errors
///
/// Returns the errors of [`read_raw_data_full`] and
/// [`chunked_read::read_chunked_data_cached`].
pub fn read_raw_data_cached(
    file_data: MetadataFrame<'_>,
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    match spec.storage() {
        RawReadStorage::Chunked { .. } => {
            chunked_read::read_chunked_data_cached_framed(file_data, spec, widths, cache)
        }
        _ => read_raw_data_full(file_data.bytes(), spec, widths),
    }
}

/// Read raw bytes for a dataset from a [`Source`] (streaming counterpart of
/// [`read_raw_data_full`]).
pub fn read_raw_data_full_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
) -> Result<Vec<u8>, FormatError> {
    let (dataspace, fill) = (spec.dataspace(), spec.fill());
    let expected_size = spec.byte_len()?;

    if dataspace.num_elements() == 0 {
        return Ok(Vec::new());
    }

    match spec.storage() {
        RawReadStorage::Compact { data } => Ok(data.to_vec()),
        RawReadStorage::Contiguous { address } => {
            source.read_exact_at(address.get(), expected_size)
        }
        RawReadStorage::ContiguousUnallocated => fill.buffer(expected_size),
        RawReadStorage::Chunked { .. } => {
            chunked_read::read_chunked_data_from_source(source, spec, widths)
        }
        RawReadStorage::Virtual => Err(FormatError::UnsupportedVirtualLayout),
    }
}

/// Streaming counterpart of [`read_raw_data_cached`].
pub fn read_raw_data_cached_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    match spec.storage() {
        RawReadStorage::Chunked { .. } => {
            chunked_read::read_chunked_data_cached_from_source(source, spec, widths, cache)
        }
        _ => read_raw_data_full_from_source(source, spec, widths),
    }
}

/// Returns the datatype that governs numeric decoding of `dt`.
///
/// An HDF5 enumeration is stored as values of its integer base type, so
/// the numeric readers decode enum data through that base type, inheriting its signedness,
/// byte order, precision, and width. The unwrap is recursive for defensiveness (an
/// enum's base is always a leaf integer in practice) and returns any non-enum
/// datatype unchanged, so the value+name round-trip works: the readers surface
/// the codes while [`crate::DType::Enum`] surfaces the member names.
pub(crate) fn effective_numeric(dt: &Datatype) -> &Datatype {
    match dt {
        Datatype::Enumeration { base_type, .. } => effective_numeric(base_type),
        other => other,
    }
}

/// Convert raw bytes to `f64` values.
pub fn read_as_f64(raw: &[u8], datatype: &Datatype) -> Result<Vec<f64>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(raw)
}

/// Decode `src` and **append** the values to `dst`, the appending form of
/// [`read_as_f64`].
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer
#[cfg(test)]
fn read_as_f64_into(
    src: &[u8],
    datatype: &Datatype,
    dst: &mut Vec<f64>,
) -> Result<(), FormatError> {
    read_numeric_into(src, datatype, dst)
}

/// Convert raw bytes to `u64` values.
pub fn read_as_u64(src: &[u8], datatype: &Datatype) -> Result<Vec<u64>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

/// Convert raw bytes to `u32` values (counterpart of [`read_as_u64`] for the
/// narrower element type, used by [`crate::Dataset::read_u32`]).
pub fn read_as_u32(src: &[u8], datatype: &Datatype) -> Result<Vec<u32>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

/// Convert raw bytes to `u16` values (counterpart of [`read_as_u64`] for the
/// narrower element type, used by [`crate::Dataset::read_u16`]).
pub fn read_as_u16(src: &[u8], datatype: &Datatype) -> Result<Vec<u16>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

#[cfg(test)]
fn read_as_u64_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u64>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

#[cfg(test)]
fn read_as_u32_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u32>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

#[cfg(test)]
fn read_as_u16_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u16>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

pub fn read_as_u8(src: &[u8], datatype: &Datatype) -> Result<Vec<u8>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

#[cfg(test)]
fn read_as_u8_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u8>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

/// Convert raw bytes to `f32` values.
pub fn read_as_f32(src: &[u8], datatype: &Datatype) -> Result<Vec<f32>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_f32`].
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer
#[cfg(test)]
fn read_as_f32_into(
    src: &[u8],
    datatype: &Datatype,
    dst: &mut Vec<f32>,
) -> Result<(), FormatError> {
    read_numeric_into(src, datatype, dst)
}

/// Convert raw bytes to `i64` values.
pub fn read_as_i64(raw: &[u8], datatype: &Datatype) -> Result<Vec<i64>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(raw)
}

/// Convert raw bytes to `i32` values.
pub fn read_as_i32(src: &[u8], datatype: &Datatype) -> Result<Vec<i32>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

/// Convert raw bytes to `i16` values (counterpart of [`read_as_i32`] for the
/// narrower element type, used by [`crate::Dataset::read_i16`]).
pub fn read_as_i16(src: &[u8], datatype: &Datatype) -> Result<Vec<i16>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

pub fn read_as_i8(src: &[u8], datatype: &Datatype) -> Result<Vec<i8>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

#[cfg(test)]
fn read_as_i64_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i64>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

/// Decode `src` and **append** the values to `dst`, the appending form of
/// [`read_as_i32`].
///
/// On error `dst` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer
#[cfg(test)]
fn read_as_i32_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i32>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

/// Decode `src` and **append** the values to `dst`, the appending form of
/// [`read_as_i16`].
///
/// On error `dst` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer
#[cfg(test)]
fn read_as_i16_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i16>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

#[cfg(test)]
fn read_as_i8_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i8>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

/// Read fixed-length strings from raw bytes.
pub fn read_as_strings(src: &[u8], datatype: &Datatype) -> Result<Vec<String>, FormatError> {
    match datatype {
        Datatype::String { size, padding, .. } => {
            let Some(elem_size) = NonZeroUsize::new(*size as usize) else {
                return Ok(Vec::new());
            };
            if !src.len().is_multiple_of(elem_size.get()) {
                return Err(FormatError::DataSizeMismatch {
                    expected: 0,
                    actual: src.len(),
                });
            }
            let count = src.len() / elem_size;
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let chunk = &src[i * elem_size.get()..(i + 1) * elem_size.get()];
                let s = match padding {
                    crate::datatype::StringPadding::NullTerminate => {
                        let end = chunk.iter().position(|&b| b == 0).unwrap_or(chunk.len());
                        String::from_utf8_lossy(&chunk[..end]).into_owned()
                    }
                    crate::datatype::StringPadding::NullPad => {
                        let end = chunk.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
                        String::from_utf8_lossy(&chunk[..end]).into_owned()
                    }
                    crate::datatype::StringPadding::SpacePad => {
                        let end = chunk.iter().rposition(|&b| b != b' ').map_or(0, |p| p + 1);
                        String::from_utf8_lossy(&chunk[..end]).into_owned()
                    }
                };
                result.push(s);
            }
            Ok(result)
        }
        _ => Err(FormatError::TypeMismatch {
            expected: "String",
            actual: crate::datatype::datatype_class_name(datatype),
        }),
    }
}

/// Appends one converted value per input element using a supported numeric encoding.
///
/// # Errors
///
/// Returns [`FormatError::TypeMismatch`] if the datatype or its enumeration base is not numeric.
/// Returns [`FormatError::EnumBaseSizeMismatch`] if an enumeration's size differs from its base.
/// Returns [`FormatError::NumericElementTooWide`] if the element size exceeds eight bytes, or
/// [`FormatError::ValueTooLargeForPlatform`] if it exceeds the target's `usize` range.
/// Returns [`FormatError::UnsupportedFloatingPointLayout`] if the floating-point encoding is
/// not IEEE 754 binary32 or binary64 with the matching element size.
/// Returns [`FormatError::UnsupportedVaxByteOrder`] if the byte order is VAX.
/// Returns [`FormatError::DataSizeMismatch`] if the input ends within an element.
#[cfg(test)]
fn read_numeric_into<T: NumericReadTarget>(
    src: &[u8],
    datatype: &Datatype,
    dst: &mut Vec<T>,
) -> Result<(), FormatError> {
    NumericDecoder::parse(datatype)?.append(src, dst)
}

/// A supported numeric source encoding paired with its destination type.
///
/// The caller can decode several buffers with the same parsed source encoding.
pub(crate) struct NumericDecoder<T> {
    plan: NumericDecodePlan,
    target: PhantomData<fn() -> T>,
}

impl<T: NumericReadTarget> NumericDecoder<T> {
    /// Parses the source encoding and element width for conversion to `T`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::TypeMismatch`] if the datatype or its enumeration base is not numeric.
    /// Returns [`FormatError::EnumBaseSizeMismatch`] if an enumeration's size differs from its base.
    /// Returns [`FormatError::NumericElementTooWide`] if the element size exceeds eight bytes, or
    /// [`FormatError::ValueTooLargeForPlatform`] if it exceeds the target's [`usize`] range.
    /// Returns [`FormatError::UnsupportedFloatingPointLayout`] if the floating-point encoding is
    /// not IEEE 754 binary32 or binary64 with the matching element size.
    /// Returns [`FormatError::UnsupportedVaxByteOrder`] if the byte order is VAX.
    pub(crate) fn parse(datatype: &Datatype) -> Result<Self, FormatError> {
        Ok(Self {
            plan: NumericDecodePlan::try_from(datatype)?,
            target: PhantomData,
        })
    }

    /// Borrows exactly one source element for conversion to `T`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if the byte length of `src` differs from
    /// the source element width.
    pub(crate) fn scalar<'a>(&'a self, src: &'a [u8]) -> Result<NumericScalar<'a, T>, FormatError> {
        let expected = self.plan.element_size();
        if src.len() != expected {
            return Err(FormatError::DataSizeMismatch {
                expected,
                actual: src.len(),
            });
        }
        Ok(NumericScalar { decoder: self, src })
    }

    /// Converts whole source elements into a new vector.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if `src` contains an incomplete element.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes.
    pub(crate) fn decode(&self, src: &[u8]) -> Result<Vec<T>, FormatError> {
        let mut dst = Vec::new();
        self.append(src, &mut dst)?;
        Ok(dst)
    }

    /// Appends converted source elements, preserving the existing values in `dst`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if `src` contains an incomplete element.
    /// On error, `dst` is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes.
    pub(crate) fn append(&self, src: &[u8], dst: &mut Vec<T>) -> Result<(), FormatError> {
        let element_size = self.plan.element_size();
        if !src.len().is_multiple_of(element_size) {
            return Err(FormatError::DataSizeMismatch {
                expected: src.len() - src.len() % element_size,
                actual: src.len(),
            });
        }
        self.convert(src, dst)
    }

    /// Selects the source primitive and its conversion for `output`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if `src` contains an incomplete fixed-width
    /// element, or if its byte length differs from the source width for a scalar destination.
    ///
    /// # Panics
    ///
    /// Panics if a vector destination's new capacity exceeds [`isize::MAX`] bytes.
    fn convert<O: NumericOutput<T>>(&self, src: &[u8], output: O) -> Result<O::Value, FormatError> {
        match self.plan.encoding() {
            NumericEncoding::FixedPoint(layout) if layout.signed() => output.fixed_point(
                src,
                self.plan.element_size(),
                |bytes| layout.read_signed(bytes),
                T::I64Conversion::default(),
            ),
            NumericEncoding::FixedPoint(layout) => output.fixed_point(
                src,
                self.plan.element_size(),
                |bytes| layout.read_unsigned(bytes),
                T::U64Conversion::default(),
            ),
            NumericEncoding::Float32(order) => {
                output.fixed_width::<4, f32, T::F32Conversion>(src, order)
            }
            NumericEncoding::Float64(order) => {
                output.fixed_width::<8, f64, T::F64Conversion>(src, order)
            }
            NumericEncoding::StandardFixedPoint { signed, layout } => {
                match (signed, layout.width) {
                    (true, StandardWidth::OneByte) => {
                        output.fixed_width::<1, i8, T::I8Conversion>(src, layout.order)
                    }
                    (true, StandardWidth::TwoBytes) => {
                        output.fixed_width::<2, i16, T::I16Conversion>(src, layout.order)
                    }
                    (true, StandardWidth::FourBytes) => {
                        output.fixed_width::<4, i32, T::I32Conversion>(src, layout.order)
                    }
                    (true, StandardWidth::EightBytes) => {
                        output.fixed_width::<8, i64, T::I64Conversion>(src, layout.order)
                    }
                    (false, StandardWidth::OneByte) => {
                        output.fixed_width::<1, u8, T::U8Conversion>(src, layout.order)
                    }
                    (false, StandardWidth::TwoBytes) => {
                        output.fixed_width::<2, u16, T::U16Conversion>(src, layout.order)
                    }
                    (false, StandardWidth::FourBytes) => {
                        output.fixed_width::<4, u32, T::U32Conversion>(src, layout.order)
                    }
                    (false, StandardWidth::EightBytes) => {
                        output.fixed_width::<8, u64, T::U64Conversion>(src, layout.order)
                    }
                }
            }
        }
    }
}

/// One source element paired with its numeric decoder.
///
/// The source byte length equals the element width of the parsed [`NumericDecoder`].
pub(crate) struct NumericScalar<'a, T> {
    decoder: &'a NumericDecoder<T>,
    src: &'a [u8],
}

impl<T: NumericReadTarget> NumericScalar<'_, T> {
    /// Converts the source element to the decoder's destination type.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if a fixed-width conversion receives a
    /// byte slice of a different width.
    pub(crate) fn decode(self) -> Result<T, FormatError> {
        self.decoder.convert(self.src, ScalarOutput)
    }
}

trait NumericOutput<T> {
    type Value;

    fn fixed_width<const W: usize, S: FromEndianBytes<W>, C: H5Conversion<S, T> + Default>(
        self,
        src: &[u8],
        order: FixedWidthByteOrder,
    ) -> Result<Self::Value, FormatError>;

    fn fixed_point<S>(
        self,
        src: &[u8],
        element_size: usize,
        read: impl Fn(&[u8]) -> S,
        conversion: impl H5Conversion<S, T>,
    ) -> Result<Self::Value, FormatError>;
}

impl<T> NumericOutput<T> for &mut Vec<T> {
    type Value = ();

    fn fixed_width<const W: usize, S: FromEndianBytes<W>, C: H5Conversion<S, T> + Default>(
        self,
        src: &[u8],
        order: FixedWidthByteOrder,
    ) -> Result<(), FormatError> {
        primitive::decode_fixed_width_into::<W, S, T, C>(src, order, C::default(), self)
    }

    fn fixed_point<S>(
        self,
        src: &[u8],
        element_size: usize,
        read: impl Fn(&[u8]) -> S,
        conversion: impl H5Conversion<S, T>,
    ) -> Result<(), FormatError> {
        self.reserve(src.len() / element_size);
        self.extend(
            src.chunks_exact(element_size)
                .map(|bytes| conversion.convert(read(bytes))),
        );
        Ok(())
    }
}

struct ScalarOutput;

impl<T> NumericOutput<T> for ScalarOutput {
    type Value = T;

    fn fixed_width<const W: usize, S: FromEndianBytes<W>, C: H5Conversion<S, T> + Default>(
        self,
        src: &[u8],
        order: FixedWidthByteOrder,
    ) -> Result<T, FormatError> {
        let bytes = src
            .try_into()
            .map_err(|_error| FormatError::DataSizeMismatch {
                expected: W,
                actual: src.len(),
            })?;
        Ok(C::default().convert(match order {
            FixedWidthByteOrder::LittleEndian => S::from_le_slice(bytes),
            FixedWidthByteOrder::BigEndian => S::from_be_slice(bytes),
        }))
    }

    fn fixed_point<S>(
        self,
        src: &[u8],
        _element_size: usize,
        read: impl Fn(&[u8]) -> S,
        conversion: impl H5Conversion<S, T>,
    ) -> Result<T, FormatError> {
        Ok(conversion.convert(read(src)))
    }
}

#[cfg(test)]
mod tests;
