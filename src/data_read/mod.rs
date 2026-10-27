//! Raw data reading and typed conversion for HDF5 datasets.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use core::marker::PhantomData;
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::PhysicalAddress;

#[cfg(test)]
use crate::VariableLengthStringType;
#[cfg(test)]
use crate::VariableLengthType;
use crate::chunk_cache::ChunkCache;
use crate::chunked_read;
#[cfg(test)]
use crate::data_layout::DataLayout;
use crate::data_read::primitive::{FromEndianBytes, H5Conversion, NumericReadTarget};
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

#[cfg(test)]
mod numeric_tests;
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

/// Read raw bytes with chunk cache support.
///
/// For chunked layouts the `cache` is used to avoid repeated B-tree
/// traversals and to cache decompressed chunk data.  For compact and
/// contiguous layouts this behaves identically to [`read_raw_data_full`].
pub fn read_raw_data_cached(
    file_data: &[u8],
    spec: RawReadSpec<'_>,
    widths: FormatWidths,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    match spec.storage() {
        RawReadStorage::Chunked { .. } => {
            chunked_read::read_chunked_data_cached(file_data, spec, widths, cache)
        }
        _ => read_raw_data_full(file_data, spec, widths),
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
pub fn read_as_f64_into(
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

pub fn read_as_u64_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u64>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

pub fn read_as_u32_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u32>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

pub fn read_as_u16_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u16>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

pub fn read_as_u8(src: &[u8], datatype: &Datatype) -> Result<Vec<u8>, FormatError> {
    NumericDecoder::parse(datatype)?.decode(src)
}

pub fn read_as_u8_into(src: &[u8], dt: &Datatype, dst: &mut Vec<u8>) -> Result<(), FormatError> {
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
pub fn read_as_f32_into(
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

pub fn read_as_i64_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i64>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

/// Decode `src` and **append** the values to `dst`, the appending form of
/// [`read_as_i32`].
///
/// On error `dst` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer
pub fn read_as_i32_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i32>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

/// Decode `src` and **append** the values to `dst`, the appending form of
/// [`read_as_i16`].
///
/// On error `dst` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer
pub fn read_as_i16_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i16>) -> Result<(), FormatError> {
    read_numeric_into(src, dt, dst)
}

pub fn read_as_i8_into(src: &[u8], dt: &Datatype, dst: &mut Vec<i8>) -> Result<(), FormatError> {
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
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::*;
    use crate::FillTime;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::convert;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::layout::FloatingPointLayout;
    use crate::datatype::{CharacterSet, StringPadding};
    use crate::fill_value::FillPattern;

    fn make_f64_le_type() -> Datatype {
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64,
            )
            .unwrap(),
        )
    }

    fn make_f32_be_type() -> Datatype {
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                4,
                DatatypeByteOrder::BigEndian,
                FloatingPointLayout::IEEE754_BINARY32,
            )
            .unwrap(),
        )
    }

    fn make_i32_le_type() -> Datatype {
        Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )
    }

    fn make_i16_le_type() -> Datatype {
        Datatype::FixedPoint(
            FixedPointType::new(2, DatatypeByteOrder::LittleEndian, true, 0, 16).unwrap(),
        )
    }

    fn make_u8_type() -> Datatype {
        Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap(),
        )
    }

    fn make_simple_dataspace(dims: &[u64]) -> Dataspace {
        Dataspace {
            space_type: DataspaceType::Simple,
            rank: dims.len() as u8,
            dimensions: dims.to_vec(),
            max_dimensions: None,
        }
    }

    #[test]
    fn read_f64_compact() {
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f64.to_le_bytes());
        data.extend_from_slice(&2.0f64.to_le_bytes());
        data.extend_from_slice(&3.0f64.to_le_bytes());
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        assert_eq!(raw, data);
        let values = read_as_f64(&raw, &dt).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn read_i32_contiguous() {
        let dt = make_i32_le_type();
        let ds = make_simple_dataspace(&[4]);
        let mut file_data = vec![0u8; 1024];
        let offset = 256usize;
        let vals: Vec<i32> = vec![10, -20, 30, -40];
        for (i, v) in vals.iter().enumerate() {
            let bytes = v.to_le_bytes();
            file_data[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&bytes);
        }
        let layout = DataLayout::Contiguous {
            address: Some(StoredAddress::new(offset as u64)),
            size: 16,
        };
        let raw = read_raw_data(&file_data, &layout, &ds, &dt).unwrap();
        let result = read_as_i32(&raw, &dt).unwrap();
        assert_eq!(result, vec![10, -20, 30, -40]);
    }

    #[test]
    fn read_u8_data() {
        let dt = make_u8_type();
        let ds = make_simple_dataspace(&[5]);
        let data = vec![10u8, 20, 30, 40, 50];
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_u64(&raw, &dt).unwrap();
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }

    fn make_enum_type(base: Datatype, members: &[(&str, i64)]) -> Datatype {
        let width = base.type_size() as usize;
        Datatype::Enumeration {
            size: base.type_size(),
            base_type: Box::new(base),
            members: members
                .iter()
                .map(|(name, v)| {
                    crate::datatype::__private::enum_member(
                        (*name).to_string(),
                        v.to_le_bytes()[..width].to_vec(),
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn read_enum_decodes_through_base_type() {
        let dt = make_enum_type(make_i32_le_type(), &[("A", 0), ("B", 1), ("C", 2)]);
        let mut raw = Vec::new();
        for v in [0i32, 2, 1, 0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(read_as_i32(&raw, &dt).unwrap(), vec![0, 2, 1, 0]);
        assert_eq!(read_as_i64(&raw, &dt).unwrap(), vec![0, 2, 1, 0]);

        // A u8-based enum decodes unsigned through its u8 base.
        let dt8 = make_enum_type(make_u8_type(), &[("OFF", 0), ("ON", 1)]);
        let raw8 = vec![0u8, 1, 1, 0];
        assert_eq!(read_as_u64(&raw8, &dt8).unwrap(), vec![0, 1, 1, 0]);
    }

    #[test]
    fn read_f32_be() {
        let dt = make_f32_be_type();
        let ds = make_simple_dataspace(&[2]);
        let mut data = Vec::new();
        // Store as big-endian
        data.extend_from_slice(&1.5f32.to_be_bytes());
        data.extend_from_slice(&2.5f32.to_be_bytes());
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_f32(&raw, &dt).unwrap();
        assert_eq!(result, vec![1.5, 2.5]);
    }

    #[test]
    fn read_i16_le() {
        let dt = make_i16_le_type();
        let ds = make_simple_dataspace(&[3]);
        let mut data = Vec::new();
        data.extend_from_slice(&(-100i16).to_le_bytes());
        data.extend_from_slice(&200i16.to_le_bytes());
        data.extend_from_slice(&(-300i16).to_le_bytes());
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_i64(&raw, &dt).unwrap();
        assert_eq!(result, vec![-100, 200, -300]);
    }

    #[test]
    fn read_strings_compact() {
        let dt = Datatype::String {
            size: 5,
            padding: StringPadding::NullPad,
            charset: CharacterSet::Ascii,
        };
        let ds = make_simple_dataspace(&[2]);
        let mut data = Vec::new();
        data.extend_from_slice(b"hello");
        data.extend_from_slice(b"hi\0\0\0");
        let layout = DataLayout::Compact { data: data.clone() };
        let raw = read_raw_data(&[], &layout, &ds, &dt).unwrap();
        let result = read_as_strings(&raw, &dt).unwrap();
        assert_eq!(result, vec!["hello", "hi"]);
    }

    #[test]
    fn type_mismatch_f64_on_string() {
        let dt = Datatype::String {
            size: 4,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        };
        let raw = vec![0u8; 8];
        let err = read_as_f64(&raw, &dt).unwrap_err();
        assert!(matches!(err, FormatError::TypeMismatch { .. }));
    }

    #[test]
    fn size_mismatch_compact() {
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let data = vec![0u8; 16]; // wrong: should be 24
        let layout = DataLayout::Compact { data };
        let err = read_raw_data(&[], &layout, &ds, &dt).unwrap_err();
        assert_eq!(
            err,
            FormatError::LogicalDataSizeMismatch {
                expected: 24,
                actual: 16
            },
        );
    }

    #[test]
    fn unallocated_contiguous_reads_as_the_fill_value() {
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let layout = DataLayout::Contiguous {
            address: None,
            size: 24,
        };
        assert_eq!(
            read_raw_data(&[], &layout, &ds, &dt).unwrap(),
            vec![0u8; 24]
        );

        let seven = 7.0f64.to_le_bytes();
        let filled = read_raw_data_full(
            &[],
            RawReadSpec::parse(
                &layout,
                &ds,
                &dt,
                None,
                FillPattern::new(FillTime::IfSet, Some(&seven), convert::nz(8)),
            )
            .unwrap(),
            TEST_WIDTHS,
        )
        .unwrap();
        assert_eq!(read_as_f64(&filled, &dt).unwrap(), vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn string_type_mismatch_on_read_as_strings() {
        let dt = make_i32_le_type();
        let raw = vec![0u8; 8];
        let err = read_as_strings(&raw, &dt).unwrap_err();
        assert!(matches!(err, FormatError::TypeMismatch { .. }));
    }

    #[test]
    fn read_f64_from_i32() {
        // read_as_f64 should work on FixedPoint types too
        let dt = make_i32_le_type();
        let mut raw = Vec::new();
        raw.extend_from_slice(&42i32.to_le_bytes());
        raw.extend_from_slice(&(-7i32).to_le_bytes());
        let result = read_as_f64(&raw, &dt).unwrap();
        assert_eq!(result, vec![42.0, -7.0]);
    }

    #[test]
    fn read_strings_space_padded() {
        let dt = Datatype::String {
            size: 8,
            padding: StringPadding::SpacePad,
            charset: CharacterSet::Ascii,
        };
        let raw = b"hello   world   ";
        let result = read_as_strings(raw, &dt).unwrap();
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn read_strings_null_terminated() {
        let dt = Datatype::String {
            size: 6,
            padding: StringPadding::NullTerminate,
            charset: CharacterSet::Ascii,
        };
        let raw = b"abc\0\0\0de\0\0\0\0";
        let result = read_as_strings(raw, &dt).unwrap();
        assert_eq!(result, vec!["abc", "de"]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_contiguous_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let mut file_data = vec![0u8; 1024];
        let offset = 256usize;
        for (i, v) in [1.0f64, 2.0, 3.0].iter().enumerate() {
            file_data[offset + i * 8..offset + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let layout = DataLayout::Contiguous {
            address: Some(StoredAddress::new(offset as u64)),
            size: 24,
        };

        let spec = RawReadSpec::plain(&layout, &ds, &dt);
        let buffered = read_raw_data_full(&file_data, spec, TEST_WIDTHS).unwrap();
        let from_mem =
            read_raw_data_full_from_source(&BytesSource::new(&file_data), spec, TEST_WIDTHS)
                .unwrap();
        let from_seek = read_raw_data_full_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap(),
            spec,
            TEST_WIDTHS,
        )
        .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(read_as_f64(&from_seek, &dt).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_compact_matches_buffered() {
        use crate::source::BytesSource;
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[2]);
        let mut data = Vec::new();
        for v in [7.0f64, 8.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let layout = DataLayout::Compact { data };
        let spec = RawReadSpec::plain(&layout, &ds, &dt);
        let buffered = read_raw_data_full(&[], spec, TEST_WIDTHS).unwrap();
        let streamed =
            read_raw_data_full_from_source(&BytesSource::new(Vec::new()), spec, TEST_WIDTHS)
                .unwrap();
        assert_eq!(buffered, streamed);
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_contiguous_unallocated_parity() {
        use crate::source::BytesSource;
        let dt = make_f64_le_type();
        let ds = make_simple_dataspace(&[3]);
        let layout = DataLayout::Contiguous {
            address: None,
            size: 24,
        };
        let seven = 7.0f64.to_le_bytes();
        for fill in [
            FillPattern::ZERO,
            FillPattern::new(FillTime::IfSet, Some(&seven), convert::nz(8)),
        ] {
            let spec = RawReadSpec::parse(&layout, &ds, &dt, None, fill).unwrap();
            let buffered = read_raw_data_full(&[], spec, TEST_WIDTHS).unwrap();
            let streamed =
                read_raw_data_full_from_source(&BytesSource::new(Vec::new()), spec, TEST_WIDTHS)
                    .unwrap();
            assert_eq!(buffered, streamed);
            assert_eq!(buffered.len(), 24);
        }
    }

    #[test]
    fn unsigned_subbyte_precision_masks_padding() {
        // 12 significant bits stored in 2 bytes: the top 4 bits are padding and
        // must be masked off (previously returned as part of the value).
        let dt = Datatype::FixedPoint(
            FixedPointType::new(2, DatatypeByteOrder::LittleEndian, false, 0, 12).unwrap(),
        );
        // Stored word 0xF123: low 12 bits = 0x123 (= 291), high nibble padding.
        let raw = 0xF123u16.to_le_bytes();
        assert_eq!(read_as_u64(&raw, &dt).unwrap(), vec![0x123]);
    }

    #[test]
    fn unsigned_bit_offset_shifts_value() {
        // 8 significant bits living in bits [4, 12) of a 16-bit word: Both the
        // low 4 (offset) and high 4 bits are padding.
        let dt = Datatype::FixedPoint(
            FixedPointType::new(2, DatatypeByteOrder::LittleEndian, false, 4, 8).unwrap(),
        );
        // 0xCAB7 -> (>>4) = 0xCAB -> &0xFF = 0xAB (= 171).
        let raw = 0xCAB7u16.to_le_bytes();
        assert_eq!(read_as_u64(&raw, &dt).unwrap(), vec![0xAB]);
    }

    #[test]
    fn signed_subbyte_precision_sign_extends() {
        // 4-bit signed value 0b1111 == -1
        // The high nibble is padding that must not leak in, and the sign bit is at bit 3 (not bit 7).
        let dt = Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::LittleEndian, true, 0, 4).unwrap(),
        );
        assert_eq!(read_as_i64(&[0x3F], &dt).unwrap(), vec![-1]);
        // 0b0111 == +7 (still inside 4-bit range, stays positive).
        assert_eq!(read_as_i64(&[0x07], &dt).unwrap(), vec![7]);
    }

    #[test]
    fn standard_width_integers_unchanged() {
        // Regression guard: full-width integers must decode exactly as before.
        let i32t = make_i32_le_type();
        assert_eq!(
            read_as_i64(&(-5i32).to_le_bytes(), &i32t).unwrap(),
            vec![-5]
        );
        let i16t = make_i16_le_type();
        assert_eq!(
            read_as_i64(&(-12345i16).to_le_bytes(), &i16t).unwrap(),
            vec![-12345]
        );
        let u8t = make_u8_type();
        assert_eq!(read_as_u64(&[200u8], &u8t).unwrap(), vec![200]);
    }

    fn make_int(size: u32, signed: bool, be: bool) -> Datatype {
        Datatype::FixedPoint(
            FixedPointType::new(
                size,
                if be {
                    DatatypeByteOrder::BigEndian
                } else {
                    DatatypeByteOrder::LittleEndian
                },
                signed,
                0,
                (size * 8) as u16,
            )
            .unwrap(),
        )
    }

    #[test]
    fn fast_path_big_endian_signed_matches_values() {
        // i32 big-endian: fast path uses `from_be_bytes` verify against known values.
        let dt = make_int(4, true, true);
        let vals = [1i32, -1, 2_000_000_000, -2_000_000_000, 0];
        let mut raw = Vec::new();
        for v in vals {
            raw.extend_from_slice(&v.to_be_bytes());
        }
        assert_eq!(
            read_as_i64(&raw, &dt).unwrap(),
            vals.iter().map(|&v| v as i64).collect::<Vec<_>>()
        );
        assert_eq!(read_as_i32(&raw, &dt).unwrap(), vals.to_vec());
        // Read big-endian i32 as f64 (FixedPoint -> f64 coercion).
        assert_eq!(
            read_as_f64(&raw, &dt).unwrap(),
            vals.iter().map(|&v| v as f64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn fast_path_unsigned_read_as_signed_preserves_value_when_representable() {
        let dt = make_int(4, false, false);
        let raw = 0xFFFF_FFFFu32.to_le_bytes();

        assert_eq!(read_as_i64(&raw, &dt).unwrap(), vec![0xFFFF_FFFF]);
        assert_eq!(read_as_u64(&raw, &dt).unwrap(), vec![0xFFFF_FFFF]);
    }

    #[test]
    fn fast_path_narrowing_readers_clamp_out_of_range_values() {
        for be in [false, true] {
            let i64t = make_int(8, true, be);
            let vals = [1i64, -1, 70_000, -70_000, i64::from(i32::MAX)];

            let mut raw = Vec::new();
            for v in vals {
                if be {
                    raw.extend_from_slice(&v.to_be_bytes());
                } else {
                    raw.extend_from_slice(&v.to_le_bytes());
                }
            }

            assert_eq!(
                read_as_i16(&raw, &i64t).unwrap(),
                vec![1, -1, i16::MAX, i16::MIN, i16::MAX],
                "i16 narrow be={be}"
            );

            let u64t = make_int(8, false, be);

            assert_eq!(
                read_as_u16(&raw, &u64t).unwrap(),
                vec![1, u16::MAX, u16::MAX, u16::MAX, u16::MAX],
                "u16 narrow be={be}"
            );

            assert_eq!(
                read_as_u32(&raw, &u64t).unwrap(),
                vec![1, u32::MAX, 70_000, u32::MAX, 2_147_483_647],
                "u32 narrow be={be}"
            );
        }
    }

    #[test]
    fn fast_path_all_widths_roundtrip_f32() {
        // f32/f64 and every integer width, LE and BE, decode to f32 correctly.
        let f4 = read_as_f32(&1.5f32.to_be_bytes(), &make_f32_be_type()).unwrap();
        assert_eq!(f4, vec![1.5]);
        let f8le = Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64,
            )
            .unwrap(),
        );
        assert_eq!(
            read_as_f32(&2.25f64.to_le_bytes(), &f8le).unwrap(),
            vec![2.25f32]
        );
        // Signed/unsigned 1/2/4/8-byte ints decode to f32.
        for (size, be) in [(1u32, false), (2, true), (4, false), (8, true)] {
            let dt = make_int(size, true, be);
            let v: i64 = -3;
            let bytes = if be { v.to_be_bytes() } else { v.to_le_bytes() };
            let raw = if be {
                &bytes[8 - size as usize..]
            } else {
                &bytes[..size as usize]
            };
            assert_eq!(read_as_f32(raw, &dt).unwrap(), vec![-3.0f32]);
        }
    }
}

/// Rejecting a numeric element wider than the 64-bit word the readers model one as.
#[cfg(test)]
mod wide_element_tests {
    use crate::CharacterSet;
    use crate::MantissaNormalization;
    use crate::StringPadding;

    use super::*;

    /// A fixed-point type `size` bytes wide, every bit of it significant.
    fn wide_int(size: u32, signed: bool, byte_order: DatatypeByteOrder) -> Datatype {
        Datatype::FixedPoint(
            FixedPointType::new(
                size,
                byte_order,
                signed,
                0,
                u16::try_from(size * 8).expect("test widths fit a u16"),
            )
            .unwrap(),
        )
    }

    /// A 16-byte float, the width an 80-bit `long double` is stored at.
    fn wide_float() -> Datatype {
        Datatype::FloatingPoint(
            FloatingPointType::new(
                16,
                DatatypeByteOrder::LittleEndian,
                MantissaNormalization::LeadingOne,
                79,
                0,
                80,
                64,
                15,
                0,
                64,
                16383,
            )
            .unwrap(),
        )
    }

    /// One numeric reader with its output type erased, so every one of them can
    /// be put to the same buffer in a loop.
    type Reader = (
        &'static str,
        fn(&[u8], &Datatype) -> Result<(), FormatError>,
    );

    /// Every `read_as_*` a caller can reach.
    ///
    /// What this pins is that
    /// none of them stops doing so: a reader that reached for
    /// `element_size_usize` directly would decode a wide element again, and it
    /// is the only one of the eight that would.
    const READERS: [Reader; 8] = [
        ("i64", |raw, dt| read_as_i64(raw, dt).map(|_| ())),
        ("u64", |raw, dt| read_as_u64(raw, dt).map(|_| ())),
        ("i32", |raw, dt| read_as_i32(raw, dt).map(|_| ())),
        ("u32", |raw, dt| read_as_u32(raw, dt).map(|_| ())),
        ("i16", |raw, dt| read_as_i16(raw, dt).map(|_| ())),
        ("u16", |raw, dt| read_as_u16(raw, dt).map(|_| ())),
        ("f64", |raw, dt| read_as_f64(raw, dt).map(|_| ())),
        ("f32", |raw, dt| read_as_f32(raw, dt).map(|_| ())),
    ];

    #[test]
    fn every_numeric_reader_refuses_a_wide_element() {
        let raw = [0xFFu8; 16];
        for dt in [
            wide_int(16, false, DatatypeByteOrder::LittleEndian),
            wide_int(16, true, DatatypeByteOrder::BigEndian),
            wide_float(),
        ] {
            for (name, read) in READERS {
                assert!(
                    matches!(
                        read(&raw, &dt),
                        Err(FormatError::NumericElementTooWide { size: 16 })
                    ),
                    "read_as_{name} accepted a {dt}"
                );
            }
        }
    }

    /// Eight bytes is the widest element that decodes, and it decodes at full
    /// range in both signednesses. The ceiling above is only worth having if
    /// this floor holds: A guard that rejected everything would pass every one
    /// of those assertions.
    #[test]
    fn eight_bytes_decodes_at_full_range_and_nine_does_not() {
        let unsigned = wide_int(8, false, DatatypeByteOrder::LittleEndian);
        assert_eq!(
            read_as_u64(&u64::MAX.to_le_bytes(), &unsigned).unwrap(),
            vec![u64::MAX]
        );
        let signed = wide_int(8, true, DatatypeByteOrder::LittleEndian);
        assert_eq!(
            read_as_i64(&i64::MIN.to_le_bytes(), &signed).unwrap(),
            vec![i64::MIN]
        );

        // One byte more is rejected. This is the row that matters most in the
        // table above: nine bytes holding 2^64 decoded to 0, which reads back
        // exactly like nine bytes that hold 0.
        assert!(matches!(
            read_as_u64(
                &[0u8; 9],
                &wide_int(9, false, DatatypeByteOrder::LittleEndian)
            ),
            Err(FormatError::NumericElementTooWide { size: 9 })
        ));
    }

    #[test]
    fn an_enumeration_over_a_wide_base_is_rejected() {
        let dt = Datatype::Enumeration {
            size: 16,
            base_type: Box::new(wide_int(16, false, DatatypeByteOrder::LittleEndian)),
            members: Vec::new(),
        };
        assert!(matches!(
            read_as_u64(&[0xFFu8; 16], &dt),
            Err(FormatError::NumericElementTooWide { size: 16 })
        ));
    }

    /// A variable-length datatype reports a 16-byte element, so it reaches the
    /// width check as a wide one. It must still be rejected for not being a
    /// number: the width check runs after `ensure_numeric`, and swapping the two
    /// would report a wide element for a datatype that is not numeric at all.
    #[test]
    fn a_non_numeric_datatype_is_rejected_for_its_class_not_its_width() {
        let vlen = Datatype::VariableLength(VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                StringPadding::NullTerminate,
                CharacterSet::Ascii,
                wide_int(1, false, DatatypeByteOrder::LittleEndian),
            )
            .unwrap(),
        ));
        assert_eq!(
            vlen.type_size(),
            16,
            "the premise: it looks like a wide one"
        );
        assert!(matches!(
            read_as_u64(&[0u8; 16], &vlen),
            Err(FormatError::TypeMismatch { .. })
        ));
    }

    /// The datatype is judged before the buffer. A wide element is undecodable
    /// however many bytes arrive with it, so reporting a length mismatch first
    /// would name the wrong thing and hide the real one behind a fixed buffer.
    #[test]
    fn a_wide_element_is_rejected_before_its_length_is_judged() {
        let dt = wide_int(16, false, DatatypeByteOrder::LittleEndian);
        // Three bytes is not a whole number of 16-byte elements either.
        assert!(matches!(
            read_as_u64(&[0xFFu8; 3], &dt),
            Err(FormatError::NumericElementTooWide { size: 16 })
        ));
    }

    #[test]
    fn reads_standard_unsigned_u8_as_f32() {
        let datatype = Datatype::FixedPoint(
            FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap(),
        );

        let values = read_as_f32(&[0xff], &datatype).unwrap();

        assert_eq!(values, vec![255.0]);
    }
}
