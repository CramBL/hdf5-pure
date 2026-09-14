//! Raw data reading and typed conversion for HDF5 datasets.

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use core::num::NonZeroUsize;

use crate::chunk_cache::ChunkCache;
use crate::chunked_read::{
    read_chunked_data_cached, read_chunked_data_cached_from_source, read_chunked_data_from_source,
};
use crate::convert::slice_range;
use crate::data_layout::DataLayout;
#[cfg(test)]
use crate::dataspace::Dataspace;
use crate::datatype::{Datatype, DatatypeByteOrder};
use crate::error::FormatError;
use crate::read_spec::RawReadSpec;
use crate::source::Source;

/// Read raw bytes for a dataset given its layout and the file data buffer,
/// using default filter/size parameters.
///
/// A test-only convenience wrapper over [`read_raw_data_full`]; the read paths
/// call `read_raw_data_full` directly with the real pipeline and offset sizes.
#[cfg(test)]
pub fn read_raw_data(
    file_data: &[u8],
    layout: &DataLayout,
    dataspace: &Dataspace,
    datatype: &Datatype,
) -> Result<Vec<u8>, FormatError> {
    read_raw_data_full(
        file_data,
        RawReadSpec::plain(layout, dataspace, datatype),
        8,
        8,
    )
}

/// Read raw bytes with full parameters including filter pipeline and sizes.
pub fn read_raw_data_full(
    file_data: &[u8],
    spec: RawReadSpec<'_>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let RawReadSpec {
        layout,
        dataspace,
        fill,
        ..
    } = spec;
    // The one definition of "these bytes are the dataset the dataspace
    // describes"; the windowed readers make the same check through it.
    let expected_size = spec.stored_byte_len()?;

    // Zero-element datasets have no data to read.
    if dataspace.num_elements() == 0 {
        return Ok(Vec::new());
    }

    match layout {
        DataLayout::Compact { data } => Ok(data.clone()),
        DataLayout::Contiguous { address, size } => {
            // No address means the storage was never allocated — the same lazy
            // allocation a chunked dataset gets, and the reference C library
            // reads it as the fill value rather than refusing it.
            let Some(addr) = *address else {
                return fill.buffer(expected_size);
            };
            let r = slice_range(addr, *size)?;
            if r.end > file_data.len() {
                return Err(FormatError::UnexpectedEof {
                    expected: r.end,
                    available: file_data.len(),
                });
            }
            Ok(file_data[r].to_vec())
        }
        // The live reader routes chunked layouts through `read_raw_data_cached`
        // and its own cache; this arm keeps the function total for the other
        // callers, at the cost of a cache that lives only for the call.
        DataLayout::Chunked { .. } => read_chunked_data_cached(
            file_data,
            spec,
            offset_size,
            length_size,
            &ChunkCache::new(),
        ),
        DataLayout::Virtual => Err(FormatError::UnsupportedVirtualLayout),
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
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    match spec.layout {
        DataLayout::Chunked { .. } => {
            read_chunked_data_cached(file_data, spec, offset_size, length_size, cache)
        }
        _ => read_raw_data_full(file_data, spec, offset_size, length_size),
    }
}

// ---------------------------------------------------------------------------
// Streaming variants (read from a `Source` instead of a whole-file buffer)
// ---------------------------------------------------------------------------
//
// These mirror `read_raw_data_full` / `read_raw_data` / `read_raw_data_cached`
// but fetch bytes on demand via `Source::read_exact_at`, so a 32-bit host
// can read a dataset out of a file larger than its address space. They always
// return an owned `Vec<u8>`: a lazy source cannot hand out a borrow, and the
// public API already returns owned data. The zero-copy `read_raw_data_zerocopy`
// stays the in-memory-only fast path and is intentionally not given a streaming
// variant.

/// Read raw bytes for a dataset from a [`Source`] (streaming counterpart of
/// [`read_raw_data_full`]).
pub fn read_raw_data_full_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<u8>, FormatError> {
    let RawReadSpec {
        layout,
        dataspace,
        fill,
        ..
    } = spec;
    // See the buffered reader: one definition, shared with the windowed ones.
    let expected_size = spec.stored_byte_len()?;

    if dataspace.num_elements() == 0 {
        return Ok(Vec::new());
    }

    match layout {
        DataLayout::Compact { data } => Ok(data.clone()),
        DataLayout::Contiguous { address, .. } => {
            // Unallocated contiguous storage reads as the fill value; see the
            // buffered reader's matching arm.
            let Some(addr) = *address else {
                return fill.buffer(expected_size);
            };
            // The single point of I/O; `read_exact_at` bounds-checks against the
            // source length (in u64) and errors instead of truncating.
            source.read_exact_at(addr, expected_size)
        }
        DataLayout::Chunked { .. } => {
            read_chunked_data_from_source(source, spec, offset_size, length_size)
        }
        DataLayout::Virtual => Err(FormatError::UnsupportedVirtualLayout),
    }
}

/// Streaming counterpart of [`read_raw_data_cached`].
pub fn read_raw_data_cached_from_source<S: Source + ?Sized>(
    source: &S,
    spec: RawReadSpec<'_>,
    offset_size: u8,
    length_size: u8,
    cache: &ChunkCache,
) -> Result<Vec<u8>, FormatError> {
    match spec.layout {
        DataLayout::Chunked { .. } => {
            read_chunked_data_cached_from_source(source, spec, offset_size, length_size, cache)
        }
        _ => read_raw_data_full_from_source(source, spec, offset_size, length_size),
    }
}

fn datatype_name(dt: &Datatype) -> &'static str {
    match dt {
        Datatype::FixedPoint { .. } => "FixedPoint",
        Datatype::FloatingPoint { .. } => "FloatingPoint",
        Datatype::String { .. } => "String",
        Datatype::Time { .. } => "Time",
        Datatype::BitField { .. } => "BitField",
        Datatype::Opaque { .. } => "Opaque",
        Datatype::Compound { .. } => "Compound",
        Datatype::Reference { .. } => "Reference",
        Datatype::Enumeration { .. } => "Enumeration",
        Datatype::VariableLength { .. } => "VariableLength",
        Datatype::Array { .. } => "Array",
    }
}

fn ensure_numeric(dt: &Datatype, expected: &'static str) -> Result<(), FormatError> {
    match dt {
        Datatype::FixedPoint { .. } | Datatype::FloatingPoint { .. } => Ok(()),
        _ => Err(FormatError::TypeMismatch {
            expected,
            actual: datatype_name(dt),
        }),
    }
}

/// Returns the datatype that governs numeric decoding of `dt`.
///
/// An HDF5 enumeration is stored as values of its integer base type, so — like
/// `hdf5-rs`, which reads an enum dataset as its base — the numeric readers
/// decode enum data through that base type, inheriting its signedness, byte
/// order, precision, and width. The unwrap is recursive for defensiveness (an
/// enum's base is always a leaf integer in practice) and returns any non-enum
/// datatype unchanged, so the value+name round-trip works: the readers surface
/// the codes while [`crate::DType::Enum`] surfaces the member names.
pub(crate) fn effective_numeric(dt: &Datatype) -> &Datatype {
    match dt {
        Datatype::Enumeration { base_type, .. } => effective_numeric(base_type),
        other => other,
    }
}

fn get_byte_order(dt: &Datatype) -> DatatypeByteOrder {
    match dt {
        Datatype::FixedPoint { byte_order, .. } => byte_order.clone(),
        Datatype::FloatingPoint { byte_order, .. } => byte_order.clone(),
        _ => DatatypeByteOrder::LittleEndian,
    }
}

/// The 64-bit word every numeric decoder below models one element as.
const DECODED_WORD_BYTES: usize = 8;

/// The element width of a numeric datatype, proven decodable.
///
/// A width past [`DECODED_WORD_BYTES`] is refused here rather than returned,
/// which is what keeps a decoder from being written that skips the check: a
/// width is the one thing every one of them needs, and this is the only way to
/// get it. Running before the raw buffer's length is judged is deliberate too —
/// an element this refuses is undecodable however many bytes arrive with it, so
/// the datatype is what the error should name. Running *after* [`ensure_numeric`]
/// matters as much in the other direction: a variable-length datatype reports a
/// 16-byte element, and it should be refused for not being a number rather than
/// for being a wide one.
///
/// Refusing on the *storage* width rather than on the significant bits is the
/// substantive choice, and it does over-refuse: a wider element whose
/// `bit_precision` fits in 64 decoded correctly before this change, under
/// little-endian order. It did not under big-endian, where [`reorder_bytes`]
/// takes the element's *leading* eight bytes and those are the most significant
/// ones — and a reader whose correctness turns on the file's byte order is worse
/// than one that says no.
///
/// Issue #361.
fn numeric_elem_size(dt: &Datatype) -> Result<NonZeroUsize, FormatError> {
    let size = dt.element_size_usize()?;
    if size.get() > DECODED_WORD_BYTES {
        return Err(FormatError::NumericElementTooWide { size: size.get() });
    }
    Ok(size)
}

/// The width of a numeric element in the standard layout: 1, 2, 4, or 8 bytes.
///
/// An element of this width holds the bytes of one storage scalar, so a `chunks_exact` loop
/// decodes a whole buffer with one `from_le_bytes` or `from_be_bytes` per element. A match over
/// the four variants covers every element that loop takes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StandardWidth {
    OneByte,
    TwoBytes,
    FourBytes,
    EightBytes,
}

impl StandardWidth {
    /// Creates a width if the element is in the standard layout.
    ///
    /// An element is in that layout at a size of 1, 2, 4, or 8 bytes, at bit offset zero, at a
    /// precision equal to the full width in bits, and in plain little- or big-endian order.
    /// Returns `None` for the rest, which [`reorder_bytes`] and [`read_raw_word`] decode one
    /// element at a time.
    fn new(
        elem_size: NonZeroUsize,
        order: &DatatypeByteOrder,
        bit_offset: u16,
        bit_precision: u16,
    ) -> Option<Self> {
        let (width, full_precision) = match elem_size.get() {
            1 => (Self::OneByte, 8),
            2 => (Self::TwoBytes, 16),
            4 => (Self::FourBytes, 32),
            8 => (Self::EightBytes, 64),
            _ => return None,
        };
        (bit_offset == 0
            && bit_precision == full_precision
            && matches!(
                order,
                DatatypeByteOrder::LittleEndian | DatatypeByteOrder::BigEndian
            ))
        .then_some(width)
    }
}

/// Bulk-decode `$raw` — already validated as a whole multiple of the storage
/// width — appending to `$dst`, a `Vec<$out>` the caller has already reserved
/// room in. `$store` is the width-matched storage scalar (e.g. `i32` for a
/// 4-byte signed integer, `f64` for an 8-byte float). Each element is decoded
/// with the correct endianness via `from_le_bytes`/`from_be_bytes` and converted
/// to the requested `$out` element type with `as`, which reproduces the
/// per-element slow path exactly for the full-width case: integer storage is
/// bit-reinterpreted then sign/zero-extended or narrowed; float storage is
/// value-converted. `chunks_exact` yields guaranteed `$store`-sized slices, so
/// `try_into` never fails and the bounds check is hoisted out of the loop.
macro_rules! bulk_decode {
    ($dst:expr, $raw:expr, $order:expr, $store:ty, $out:ty) => {{
        const W: usize = core::mem::size_of::<$store>();
        let dst: &mut Vec<$out> = $dst;
        match $order {
            DatatypeByteOrder::BigEndian => {
                for c in $raw.chunks_exact(W) {
                    let a: [u8; W] = c.try_into().unwrap();
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_possible_wrap,
                        clippy::unnecessary_cast
                    )]
                    dst.push(<$store>::from_be_bytes(a) as $out);
                }
            }
            // LittleEndian here: `StandardWidth::new` returns `None` for Vax order.
            _ => {
                for c in $raw.chunks_exact(W) {
                    let a: [u8; W] = c.try_into().unwrap();
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_possible_wrap,
                        clippy::unnecessary_cast
                    )]
                    dst.push(<$store>::from_le_bytes(a) as $out);
                }
            }
        }
    }};
}

/// Convert raw bytes to `f64` values.
pub fn read_as_f64(raw: &[u8], datatype: &Datatype) -> Result<Vec<f64>, FormatError> {
    let mut out = Vec::new();
    read_as_f64_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_f64`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_f64_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<f64>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FloatingPoint or FixedPoint")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    // Fast path: standard full-width layout, bulk-decoded with `from_*_bytes`.
    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match datatype {
            Datatype::FloatingPoint { size: 4, .. } => {
                bulk_decode!(out, raw, order, f32, f64);
                return Ok(());
            }
            Datatype::FloatingPoint { size: 8, .. } => {
                bulk_decode!(out, raw, order, f64, f64);
                return Ok(());
            }
            Datatype::FixedPoint { signed: true, .. } => {
                match width {
                    StandardWidth::OneByte => bulk_decode!(out, raw, order, i8, f64),
                    StandardWidth::TwoBytes => bulk_decode!(out, raw, order, i16, f64),
                    StandardWidth::FourBytes => bulk_decode!(out, raw, order, i32, f64),
                    StandardWidth::EightBytes => bulk_decode!(out, raw, order, i64, f64),
                }
                return Ok(());
            }
            Datatype::FixedPoint { signed: false, .. } => {
                match width {
                    StandardWidth::OneByte => bulk_decode!(out, raw, order, u8, f64),
                    StandardWidth::TwoBytes => bulk_decode!(out, raw, order, u16, f64),
                    StandardWidth::FourBytes => bulk_decode!(out, raw, order, u32, f64),
                    StandardWidth::EightBytes => bulk_decode!(out, raw, order, u64, f64),
                }
                return Ok(());
            }
            // Other classes (e.g. a 2-byte float, with no `f16`) fall through to
            // the slow path, which errors exactly as before.
            _ => {}
        }
    }

    for i in 0..count {
        let chunk = &raw[i * elem_size.get()..(i + 1) * elem_size.get()];
        let val = convert_to_f64(chunk, datatype, &order)?;
        out.push(val);
    }
    Ok(())
}

fn convert_to_f64(
    bytes: &[u8],
    dt: &Datatype,
    order: &DatatypeByteOrder,
) -> Result<f64, FormatError> {
    match dt {
        Datatype::FloatingPoint { size, .. } => match size {
            4 => {
                let v = read_f32_bytes(bytes, order);
                Ok(v as f64)
            }
            8 => Ok(read_f64_bytes(bytes, order)),
            _ => Err(FormatError::DataSizeMismatch {
                expected: 8,
                actual: *size as usize,
            }),
        },
        Datatype::FixedPoint {
            size,
            signed,
            bit_offset,
            bit_precision,
            ..
        } => {
            if *signed {
                let v = read_signed_int(bytes, *size as usize, order, *bit_offset, *bit_precision);
                Ok(v as f64)
            } else {
                let v =
                    read_unsigned_int(bytes, *size as usize, order, *bit_offset, *bit_precision);
                Ok(v as f64)
            }
        }
        _ => Err(FormatError::TypeMismatch {
            expected: "numeric",
            actual: datatype_name(dt),
        }),
    }
}

/// Convert raw bytes to `i64` values.
pub fn read_as_i64(raw: &[u8], datatype: &Datatype) -> Result<Vec<i64>, FormatError> {
    let mut out = Vec::new();
    read_as_i64_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_i64`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_i64_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<i64>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FixedPoint (signed)")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    // Fast path: standard full-width layout, bulk-decoded then sign-extended.
    // Signed storage types reproduce `read_signed_int`'s sign-extension for the
    // full-width case (and a float read as i64 bit-reinterprets identically).
    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match width {
            StandardWidth::OneByte => bulk_decode!(out, raw, order, i8, i64),
            StandardWidth::TwoBytes => bulk_decode!(out, raw, order, i16, i64),
            StandardWidth::FourBytes => bulk_decode!(out, raw, order, i32, i64),
            StandardWidth::EightBytes => bulk_decode!(out, raw, order, i64, i64),
        }
        return Ok(());
    }

    for i in 0..count {
        let chunk = &raw[i * elem_size.get()..(i + 1) * elem_size.get()];
        let v = read_signed_int(chunk, elem_size.get(), &order, bit_offset, bit_precision);
        out.push(v);
    }
    Ok(())
}

/// Convert raw bytes to `u64` values.
pub fn read_as_u64(raw: &[u8], datatype: &Datatype) -> Result<Vec<u64>, FormatError> {
    let mut out = Vec::new();
    read_as_u64_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_u64`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_u64_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<u64>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FixedPoint (unsigned)")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    // Fast path: standard full-width layout, bulk-decoded with zero-extension
    // (unsigned storage types reproduce `read_unsigned_int`'s magnitude).
    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match width {
            StandardWidth::OneByte => bulk_decode!(out, raw, order, u8, u64),
            StandardWidth::TwoBytes => bulk_decode!(out, raw, order, u16, u64),
            StandardWidth::FourBytes => bulk_decode!(out, raw, order, u32, u64),
            StandardWidth::EightBytes => bulk_decode!(out, raw, order, u64, u64),
        }
        return Ok(());
    }

    for i in 0..count {
        let chunk = &raw[i * elem_size.get()..(i + 1) * elem_size.get()];
        let v = read_unsigned_int(chunk, elem_size.get(), &order, bit_offset, bit_precision);
        out.push(v);
    }
    Ok(())
}

/// Convert raw bytes to `f32` values.
pub fn read_as_f32(raw: &[u8], datatype: &Datatype) -> Result<Vec<f32>, FormatError> {
    let mut out = Vec::new();
    read_as_f32_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_f32`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_f32_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<f32>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FloatingPoint")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    // Fast path: standard full-width layout, bulk-decoded with `from_*_bytes`.
    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match datatype {
            Datatype::FloatingPoint { size: 4, .. } => {
                bulk_decode!(out, raw, order, f32, f32);
                return Ok(());
            }
            Datatype::FloatingPoint { size: 8, .. } => {
                bulk_decode!(out, raw, order, f64, f32);
                return Ok(());
            }
            Datatype::FixedPoint { signed: true, .. } => {
                match width {
                    StandardWidth::OneByte => bulk_decode!(out, raw, order, i8, f32),
                    StandardWidth::TwoBytes => bulk_decode!(out, raw, order, i16, f32),
                    StandardWidth::FourBytes => bulk_decode!(out, raw, order, i32, f32),
                    StandardWidth::EightBytes => bulk_decode!(out, raw, order, i64, f32),
                }
                return Ok(());
            }
            Datatype::FixedPoint { signed: false, .. } => {
                match width {
                    StandardWidth::OneByte => bulk_decode!(out, raw, order, u8, f32),
                    StandardWidth::TwoBytes => bulk_decode!(out, raw, order, u16, f32),
                    StandardWidth::FourBytes => bulk_decode!(out, raw, order, u32, f32),
                    StandardWidth::EightBytes => bulk_decode!(out, raw, order, u64, f32),
                }
                return Ok(());
            }
            _ => {}
        }
    }

    for i in 0..count {
        let chunk = &raw[i * elem_size.get()..(i + 1) * elem_size.get()];
        match datatype {
            Datatype::FloatingPoint { size: 4, .. } => {
                out.push(read_f32_bytes(chunk, &order));
            }
            Datatype::FloatingPoint { size: 8, .. } => {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "read_as_f32 narrows stored f64 values to the requested f32"
                )]
                out.push(read_f64_bytes(chunk, &order) as f32);
            }
            Datatype::FixedPoint {
                signed: true,
                size,
                bit_offset,
                bit_precision,
                ..
            } => {
                out.push(
                    read_signed_int(chunk, *size as usize, &order, *bit_offset, *bit_precision)
                        as f32,
                );
            }
            Datatype::FixedPoint {
                signed: false,
                size,
                bit_offset,
                bit_precision,
                ..
            } => {
                out.push(read_unsigned_int(
                    chunk,
                    *size as usize,
                    &order,
                    *bit_offset,
                    *bit_precision,
                ) as f32);
            }
            _ => {
                return Err(FormatError::TypeMismatch {
                    expected: "numeric",
                    actual: datatype_name(datatype),
                });
            }
        }
    }
    Ok(())
}

/// Convert raw bytes to `i32` values.
pub fn read_as_i32(raw: &[u8], datatype: &Datatype) -> Result<Vec<i32>, FormatError> {
    let mut out = Vec::new();
    read_as_i32_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_i32`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_i32_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<i32>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FixedPoint")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    // Fast path: standard full-width layout, bulk-decoded then narrowed to i32
    // (matches `read_signed_int(..) as i32` for the full-width case).
    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match width {
            StandardWidth::OneByte => bulk_decode!(out, raw, order, i8, i32),
            StandardWidth::TwoBytes => bulk_decode!(out, raw, order, i16, i32),
            StandardWidth::FourBytes => bulk_decode!(out, raw, order, i32, i32),
            StandardWidth::EightBytes => bulk_decode!(out, raw, order, i64, i32),
        }
        return Ok(());
    }

    for i in 0..count {
        let chunk = &raw[i * elem_size.get()..(i + 1) * elem_size.get()];
        let v = read_signed_int(chunk, elem_size.get(), &order, bit_offset, bit_precision);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "read_as_i32 narrows each stored signed value to the requested i32"
        )]
        out.push(v as i32);
    }
    Ok(())
}

/// Convert raw bytes to `i16` values (counterpart of [`read_as_i32`] for the
/// narrower element type, used by [`crate::Dataset::read_i16`]).
pub fn read_as_i16(raw: &[u8], datatype: &Datatype) -> Result<Vec<i16>, FormatError> {
    let mut out = Vec::new();
    read_as_i16_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_i16`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_i16_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<i16>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FixedPoint")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match width {
            StandardWidth::OneByte => bulk_decode!(out, raw, order, i8, i16),
            StandardWidth::TwoBytes => bulk_decode!(out, raw, order, i16, i16),
            StandardWidth::FourBytes => bulk_decode!(out, raw, order, i32, i16),
            StandardWidth::EightBytes => bulk_decode!(out, raw, order, i64, i16),
        }
        return Ok(());
    }

    // Slow path: decode wide, then narrow (matches the prior `read_i16` route
    // through `read_as_i32`).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "read_as_i16 narrows each stored value to the requested i16"
    )]
    out.extend(read_as_i32(raw, datatype)?.into_iter().map(|v| v as i16));
    Ok(())
}

/// Convert raw bytes to `u32` values (counterpart of [`read_as_u64`] for the
/// narrower element type, used by [`crate::Dataset::read_u32`]).
pub fn read_as_u32(raw: &[u8], datatype: &Datatype) -> Result<Vec<u32>, FormatError> {
    let mut out = Vec::new();
    read_as_u32_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_u32`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_u32_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<u32>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FixedPoint (unsigned)")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match width {
            StandardWidth::OneByte => bulk_decode!(out, raw, order, u8, u32),
            StandardWidth::TwoBytes => bulk_decode!(out, raw, order, u16, u32),
            StandardWidth::FourBytes => bulk_decode!(out, raw, order, u32, u32),
            StandardWidth::EightBytes => bulk_decode!(out, raw, order, u64, u32),
        }
        return Ok(());
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "read_as_u32 narrows each stored value to the requested u32"
    )]
    out.extend(read_as_u64(raw, datatype)?.into_iter().map(|v| v as u32));
    Ok(())
}

/// Convert raw bytes to `u16` values (counterpart of [`read_as_u64`] for the
/// narrower element type, used by [`crate::Dataset::read_u16`]).
pub fn read_as_u16(raw: &[u8], datatype: &Datatype) -> Result<Vec<u16>, FormatError> {
    let mut out = Vec::new();
    read_as_u16_into(raw, datatype, &mut out)?;
    Ok(out)
}

/// Decode `raw` and **append** the values to `out`, the appending form of
/// [`read_as_u16`].
///
/// A whole-dataset typed read sweeps its dataset in row windows and decodes
/// each window into one output buffer, so the decoders must add to a buffer
/// rather than each produce one (issue #289).
///
/// On error `out` holds an unspecified prefix of this call's values: every
/// caller abandons the buffer, and nothing else is worth the copy that
/// avoiding it would cost.
pub fn read_as_u16_into(
    raw: &[u8],
    datatype: &Datatype,
    out: &mut Vec<u16>,
) -> Result<(), FormatError> {
    let datatype = effective_numeric(datatype);
    ensure_numeric(datatype, "FixedPoint (unsigned)")?;
    let elem_size = numeric_elem_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size.get()) {
        return Err(FormatError::DataSizeMismatch {
            expected: 0,
            actual: raw.len(),
        });
    }
    let count = raw.len() / elem_size;
    let order = get_byte_order(datatype);
    let (bit_offset, bit_precision) = int_bits(datatype);
    out.reserve(count);

    if let Some(width) = StandardWidth::new(elem_size, &order, bit_offset, bit_precision) {
        match width {
            StandardWidth::OneByte => bulk_decode!(out, raw, order, u8, u16),
            StandardWidth::TwoBytes => bulk_decode!(out, raw, order, u16, u16),
            StandardWidth::FourBytes => bulk_decode!(out, raw, order, u32, u16),
            StandardWidth::EightBytes => bulk_decode!(out, raw, order, u64, u16),
        }
        return Ok(());
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "read_as_u16 narrows each stored value to the requested u16"
    )]
    out.extend(read_as_u64(raw, datatype)?.into_iter().map(|v| v as u16));
    Ok(())
}

/// Read fixed-length strings from raw bytes.
pub fn read_as_strings(raw: &[u8], datatype: &Datatype) -> Result<Vec<String>, FormatError> {
    match datatype {
        Datatype::String { size, padding, .. } => {
            let Some(elem_size) = NonZeroUsize::new(*size as usize) else {
                return Ok(Vec::new());
            };
            if !raw.len().is_multiple_of(elem_size.get()) {
                return Err(FormatError::DataSizeMismatch {
                    expected: 0,
                    actual: raw.len(),
                });
            }
            let count = raw.len() / elem_size;
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let chunk = &raw[i * elem_size.get()..(i + 1) * elem_size.get()];
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
            actual: datatype_name(datatype),
        }),
    }
}

// --- Low-level byte conversion helpers ---

/// An element's bytes as a little-endian 64-bit word, honoring `order`.
///
/// Only the leading eight are taken, and under big-endian order those are an
/// element's *most* significant bytes — so this is faithful only for an element
/// no wider than the word itself. [`numeric_elem_size`] is what holds that true
/// on the fixed-point path; the float callers pass an 8-byte `f64`.
fn reorder_bytes(bytes: &[u8], order: &DatatypeByteOrder) -> [u8; 8] {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    match order {
        DatatypeByteOrder::LittleEndian | DatatypeByteOrder::Vax => {
            buf[..len].copy_from_slice(&bytes[..len]);
        }
        DatatypeByteOrder::BigEndian => {
            // Reverse bytes into LE order
            for i in 0..len {
                buf[i] = bytes[len - 1 - i];
            }
        }
    }
    buf
}

fn read_f64_bytes(bytes: &[u8], order: &DatatypeByteOrder) -> f64 {
    let buf = reorder_bytes(bytes, order);
    f64::from_le_bytes(buf)
}

fn read_f32_bytes(bytes: &[u8], order: &DatatypeByteOrder) -> f32 {
    let mut buf = [0u8; 4];
    let len = bytes.len().min(4);
    match order {
        DatatypeByteOrder::LittleEndian | DatatypeByteOrder::Vax => {
            buf[..len].copy_from_slice(&bytes[..len]);
        }
        DatatypeByteOrder::BigEndian => {
            for i in 0..len {
                buf[i] = bytes[len - 1 - i];
            }
        }
    }
    f32::from_le_bytes(buf)
}

/// Fixed-point `(bit_offset, bit_precision)` for a datatype. Non-fixed-point
/// numeric types (the float-reinterpreted-as-int fallbacks) report a full-width
/// standard layout, so their decoding is byte-for-byte unchanged.
fn int_bits(dt: &Datatype) -> (u16, u16) {
    match dt {
        Datatype::FixedPoint {
            bit_offset,
            bit_precision,
            ..
        } => (*bit_offset, *bit_precision),
        _ => {
            // Full-width standard layout. `try_from` clamps the (always small,
            // <= 64) bit width without a narrowing cast.
            let bits = u16::try_from(u64::from(dt.type_size()) * 8).unwrap_or(u16::MAX);
            (0, bits)
        }
    }
}

/// Read the raw stored integer word (zero-extended to `u64`), honoring byte
/// order.
///
/// `size` is at most [`DECODED_WORD_BYTES`], since every path here takes its
/// width from [`numeric_elem_size`], which refuses a wider element rather than
/// have this keep part of one. The `min` is belt-and-braces against that
/// changing, not a truncation policy.
fn read_raw_word(bytes: &[u8], size: usize, order: &DatatypeByteOrder) -> u64 {
    let buf = reorder_bytes(bytes, order);
    let mut val = 0u64;
    for (i, &byte) in buf.iter().enumerate().take(size.min(8)) {
        val |= (byte as u64) << (i * 8);
    }
    val
}

/// Extract the significant bits of a fixed-point value per the HDF5 layout:
/// drop the `bit_offset` low padding bits and mask to `bit_precision`
/// significant bits, right-justified. Returns the unsigned magnitude. For the
/// standard full-width case (`bit_offset == 0`, `bit_precision == size*8`) this
/// is the identity, so normal integers decode exactly as before.
fn extract_unsigned_bits(raw: u64, bit_offset: u16, bit_precision: u16) -> u64 {
    let off = u32::from(bit_offset);
    let prec = u32::from(bit_precision);
    let shifted = if off >= 64 { 0 } else { raw >> off };
    if prec == 0 {
        0
    } else if prec >= 64 {
        shifted
    } else {
        shifted & ((1u64 << prec) - 1)
    }
}

fn read_unsigned_int(
    bytes: &[u8],
    size: usize,
    order: &DatatypeByteOrder,
    bit_offset: u16,
    bit_precision: u16,
) -> u64 {
    let raw = read_raw_word(bytes, size, order);
    extract_unsigned_bits(raw, bit_offset, bit_precision)
}

#[expect(
    clippy::cast_possible_wrap,
    reason = "reinterprets raw bits as a signed integer; sign reinterpretation and \
              sign-extension are the intended operations"
)]
fn read_signed_int(
    bytes: &[u8],
    size: usize,
    order: &DatatypeByteOrder,
    bit_offset: u16,
    bit_precision: u16,
) -> i64 {
    let magnitude = read_unsigned_int(bytes, size, order, bit_offset, bit_precision);
    let prec = u32::from(bit_precision);
    if prec == 0 || prec >= 64 {
        magnitude as i64
    } else {
        // Sign-extend from the precision boundary: replicate bit (prec - 1).
        let shift = 64 - prec;
        ((magnitude << shift) as i64) >> shift
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use rstest::rstest;

    use super::*;
    use crate::convert::nz;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::{CharacterSet, StringPadding};
    use crate::fill_value::FillPattern;

    fn make_f64_le_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        }
    }

    fn make_f32_be_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 4,
            byte_order: DatatypeByteOrder::BigEndian,
            bit_offset: 0,
            bit_precision: 32,
            exponent_location: 23,
            exponent_size: 8,
            mantissa_location: 0,
            mantissa_size: 23,
            exponent_bias: 127,
        }
    }

    fn make_i32_le_type() -> Datatype {
        Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    fn make_i16_le_type() -> Datatype {
        Datatype::FixedPoint {
            size: 2,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 16,
        }
    }

    fn make_u8_type() -> Datatype {
        Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        }
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
            address: Some(offset as u64),
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
                .map(|(name, v)| crate::datatype::EnumMember {
                    name: (*name).to_string(),
                    value: v.to_le_bytes()[..width].to_vec(),
                })
                .collect(),
        }
    }

    #[test]
    fn read_enum_decodes_through_base_type() {
        // An enum datatype is decoded via its integer base type; before the
        // enum-unwrap this hit `ensure_numeric` and returned a `TypeMismatch`.
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
        assert!(matches!(err, FormatError::DataSizeMismatch { .. }));
    }

    /// Contiguous storage that was never allocated reads as the fill value —
    /// zeros when none is defined, the value itself when one is — rather than
    /// failing. This is the reference C library's behavior for a dataset created
    /// and not written, and the fill pattern is the *only* thing that decides
    /// what comes back, since there are no file bytes to read.
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
            RawReadSpec {
                layout: &layout,
                dataspace: &ds,
                datatype: &dt,
                pipeline: None,
                fill: FillPattern::new(Some(&seven), nz(8)),
            },
            8,
            8,
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
            address: Some(offset as u64),
            size: 24,
        };

        let spec = RawReadSpec::plain(&layout, &ds, &dt);
        let buffered = read_raw_data_full(&file_data, spec, 8, 8).unwrap();
        let from_mem =
            read_raw_data_full_from_source(&BytesSource::new(&file_data), spec, 8, 8).unwrap();
        let from_seek = read_raw_data_full_from_source(
            &ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap(),
            spec,
            8,
            8,
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
        let buffered = read_raw_data_full(&[], spec, 8, 8).unwrap();
        let streamed =
            read_raw_data_full_from_source(&BytesSource::new(Vec::new()), spec, 8, 8).unwrap();
        assert_eq!(buffered, streamed);
    }

    /// The buffered and streaming readers must agree over unallocated
    /// contiguous storage — including on *what* it reads as, not merely that
    /// both succeed. Both fill patterns, since a shared zeroed allocation would
    /// make them agree for the wrong reason.
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
        for fill in [FillPattern::ZERO, FillPattern::new(Some(&seven), nz(8))] {
            let spec = RawReadSpec {
                layout: &layout,
                dataspace: &ds,
                datatype: &dt,
                pipeline: None,
                fill,
            };
            let buffered = read_raw_data_full(&[], spec, 8, 8).unwrap();
            let streamed =
                read_raw_data_full_from_source(&BytesSource::new(Vec::new()), spec, 8, 8).unwrap();
            assert_eq!(buffered, streamed);
            assert_eq!(buffered.len(), 24);
        }
    }

    // --- Sub-byte precision / bit-offset integers (#6) ---

    #[test]
    fn unsigned_subbyte_precision_masks_padding() {
        // 12 significant bits stored in 2 bytes; the top 4 bits are padding and
        // must be masked off (previously returned as part of the value).
        let dt = Datatype::FixedPoint {
            size: 2,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 0,
            bit_precision: 12,
        };
        // Stored word 0xF123: low 12 bits = 0x123 (= 291), high nibble padding.
        let raw = 0xF123u16.to_le_bytes();
        assert_eq!(read_as_u64(&raw, &dt).unwrap(), vec![0x123]);
    }

    #[test]
    fn unsigned_bit_offset_shifts_value() {
        // 8 significant bits living in bits [4, 12) of a 16-bit word; both the
        // low 4 (offset) and high 4 bits are padding.
        let dt = Datatype::FixedPoint {
            size: 2,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: false,
            bit_offset: 4,
            bit_precision: 8,
        };
        // 0xCAB7 -> (>>4) = 0xCAB -> &0xFF = 0xAB (= 171).
        let raw = 0xCAB7u16.to_le_bytes();
        assert_eq!(read_as_u64(&raw, &dt).unwrap(), vec![0xAB]);
    }

    #[test]
    fn signed_subbyte_precision_sign_extends() {
        // 4-bit signed value 0b1111 == -1; the high nibble is padding that must
        // not leak in, and the sign bit is at bit 3, not bit 7.
        let dt = Datatype::FixedPoint {
            size: 1,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 4,
        };
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

    // --- Bulk fast-path equivalence guards ---
    //
    // The native-endian fast paths must produce byte-identical results to the
    // general per-element path for every width, endianness, and cross-type
    // coercion the readers accept.

    fn make_int(size: u32, signed: bool, be: bool) -> Datatype {
        Datatype::FixedPoint {
            size,
            byte_order: if be {
                DatatypeByteOrder::BigEndian
            } else {
                DatatypeByteOrder::LittleEndian
            },
            signed,
            bit_offset: 0,
            bit_precision: (size * 8) as u16,
        }
    }

    #[test]
    fn fast_path_big_endian_signed_matches_values() {
        // i32 big-endian: fast path uses from_be_bytes; verify against known vals.
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
    fn fast_path_unsigned_read_as_signed_reinterprets() {
        // An unsigned u32 read via read_as_i64 must sign-reinterpret the full
        // width, exactly as the per-element read_signed_int did.
        let dt = make_int(4, false, false);
        let raw = 0xFFFF_FFFFu32.to_le_bytes();
        assert_eq!(read_as_i64(&raw, &dt).unwrap(), vec![-1]);
        // And read as u64 zero-extends.
        assert_eq!(read_as_u64(&raw, &dt).unwrap(), vec![0xFFFF_FFFF]);
    }

    #[test]
    fn fast_path_narrowing_readers_match_wide_then_narrow() {
        // read_as_i16/u16/u32 must equal the old "decode wide, then `as`" route
        // for both LE and BE and for narrowing from a wider stored width.
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
            let wide = read_as_i64(&raw, &i64t).unwrap();
            let i16s = read_as_i16(&raw, &i64t).unwrap();
            assert_eq!(
                i16s,
                wide.iter().map(|&v| v as i16).collect::<Vec<_>>(),
                "i16 narrow be={be}"
            );

            let u64t = make_int(8, false, be);
            let uwide = read_as_u64(&raw, &u64t).unwrap();
            assert_eq!(
                read_as_u16(&raw, &u64t).unwrap(),
                uwide.iter().map(|&v| v as u16).collect::<Vec<_>>(),
                "u16 narrow be={be}"
            );
            assert_eq!(
                read_as_u32(&raw, &u64t).unwrap(),
                uwide.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                "u32 narrow be={be}"
            );
        }
    }

    #[rstest]
    #[case(
        nz(1),
        DatatypeByteOrder::LittleEndian,
        0,
        8,
        Some(StandardWidth::OneByte)
    )]
    #[case(
        nz(2),
        DatatypeByteOrder::BigEndian,
        0,
        16,
        Some(StandardWidth::TwoBytes)
    )]
    #[case(
        nz(4),
        DatatypeByteOrder::LittleEndian,
        0,
        32,
        Some(StandardWidth::FourBytes)
    )]
    #[case(
        nz(8),
        DatatypeByteOrder::BigEndian,
        0,
        64,
        Some(StandardWidth::EightBytes)
    )]
    #[case(nz(3), DatatypeByteOrder::LittleEndian, 0, 24, None)]
    #[case(nz(16), DatatypeByteOrder::LittleEndian, 0, 128, None)]
    #[case(nz(4), DatatypeByteOrder::Vax, 0, 32, None)]
    #[case(nz(2), DatatypeByteOrder::LittleEndian, 4, 16, None)]
    #[case(nz(2), DatatypeByteOrder::LittleEndian, 0, 12, None)]
    fn a_width_is_standard_at_full_precision_in_plain_byte_order(
        #[case] elem_size: NonZeroUsize,
        #[case] order: DatatypeByteOrder,
        #[case] bit_offset: u16,
        #[case] bit_precision: u16,
        #[case] expected: Option<StandardWidth>,
    ) {
        assert_eq!(
            StandardWidth::new(elem_size, &order, bit_offset, bit_precision),
            expected
        );
    }

    #[test]
    fn fast_path_all_widths_roundtrip_f32() {
        // f32/f64 and every integer width, LE and BE, decode to f32 correctly.
        let f4 = read_as_f32(&1.5f32.to_be_bytes(), &make_f32_be_type()).unwrap();
        assert_eq!(f4, vec![1.5]);
        let f8le = Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
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

/// Refusing a numeric element wider than the 64-bit word the readers model one
/// as, rather than decoding it from part of itself (issue #361).
///
/// What these widths returned before the refusal, measured with the guard
/// disabled — the first three rows are the issue's own report, the fourth is
/// why the refusal is on storage width rather than on significant bits:
///
/// | stored | decoded as |
/// | --- | --- |
/// | 16-byte unsigned, `[0xFF; 16]` | `18446744073709551615` (2^64 - 1) |
/// | 16-byte signed, `[0xFF; 16]` | `-1` |
/// | 9-byte unsigned little-endian, 2^64 | `0` |
/// | 9-byte unsigned **big-endian**, 2^64 | `72057594037927936` (2^56) |
#[cfg(test)]
mod wide_element_tests {
    use super::*;

    /// A fixed-point type `size` bytes wide, every bit of it significant.
    fn wide_int(size: u32, signed: bool, byte_order: DatatypeByteOrder) -> Datatype {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            bit_offset: 0,
            bit_precision: u16::try_from(size * 8).expect("test widths fit a u16"),
        }
    }

    /// A 16-byte float, the width an 80-bit `long double` is stored at.
    fn wide_float() -> Datatype {
        Datatype::FloatingPoint {
            size: 16,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 80,
            exponent_location: 64,
            exponent_size: 15,
            mantissa_location: 0,
            mantissa_size: 64,
            exponent_bias: 16383,
        }
    }

    /// One numeric reader with its output type erased, so every one of them can
    /// be put to the same buffer in a loop.
    type Reader = (
        &'static str,
        fn(&[u8], &Datatype) -> Result<(), FormatError>,
    );

    /// Every `read_as_*` a caller can reach.
    ///
    /// All eight take their width from [`numeric_elem_size`], so the refusal is
    /// structural rather than eight remembered calls. What this pins is that
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

    /// Every reader, not just the two the issue was reported through. A wide
    /// element is undecodable for all of them, and one that let it past would
    /// hand back bytes the others refuse.
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
    /// this floor holds: a guard that refused everything would pass every one
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

        // One byte more is refused. This is the row that matters most in the
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

    /// An enumeration is stored as values of its base type, and the readers
    /// decode it through that base — so a wide base is a wide element, and the
    /// unwrapping must not step around the guard.
    #[test]
    fn an_enumeration_over_a_wide_base_is_refused() {
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
    /// width check as a wide one. It must still be refused for not being a
    /// number: the width check runs after `ensure_numeric`, and swapping the two
    /// would report a wide element for a datatype that is not numeric at all.
    #[test]
    fn a_non_numeric_datatype_is_refused_for_its_class_not_its_width() {
        let vlen = Datatype::VariableLength {
            is_string: true,
            padding: None,
            charset: None,
            base_type: Box::new(wide_int(1, false, DatatypeByteOrder::LittleEndian)),
        };
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
    fn a_wide_element_is_refused_before_its_length_is_judged() {
        let dt = wide_int(16, false, DatatypeByteOrder::LittleEndian);
        // Three bytes is not a whole number of 16-byte elements either.
        assert!(matches!(
            read_as_u64(&[0xFFu8; 3], &dt),
            Err(FormatError::NumericElementTooWide { size: 16 })
        ));
    }
}
