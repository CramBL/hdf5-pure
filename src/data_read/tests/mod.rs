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
use crate::source;

mod numeric;

/// Rejecting a numeric element wider than the 64-bit word the readers model one as.
mod wide_elements;

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
    assert_eq!(
        err,
        FormatError::TypeMismatch {
            expected: "numeric",
            actual: "String"
        }
    );
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
        &source::zero_base_source(&source::BytesSource::new(&[])),
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
    assert_eq!(
        err,
        FormatError::TypeMismatch {
            expected: "String",
            actual: "FixedPoint"
        }
    );
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
    let buffered = read_raw_data_full(
        &file_data,
        &source::zero_base_source(&source::BytesSource::new(&file_data)),
        spec,
        TEST_WIDTHS,
    )
    .unwrap();
    let from_mem = read_raw_data_full_from_source(
        &BytesSource::new(&file_data),
        &source::zero_base_source(&BytesSource::new(&file_data)),
        spec,
        TEST_WIDTHS,
    )
    .unwrap();
    let from_seek = read_raw_data_full_from_source(
        &ReadSeekSource::new(std::io::Cursor::new(&file_data)).unwrap(),
        &source::zero_base_source(&ReadSeekSource::new(std::io::Cursor::new(&file_data)).unwrap()),
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
    let buffered = read_raw_data_full(
        &[],
        &source::zero_base_source(&source::BytesSource::new(&[])),
        spec,
        TEST_WIDTHS,
    )
    .unwrap();
    let streamed = read_raw_data_full_from_source(
        &BytesSource::new(Vec::new()),
        &source::zero_base_source(&BytesSource::new(Vec::new())),
        spec,
        TEST_WIDTHS,
    )
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
        let buffered = read_raw_data_full(
            &[],
            &source::zero_base_source(&source::BytesSource::new(&[])),
            spec,
            TEST_WIDTHS,
        )
        .unwrap();
        let streamed = read_raw_data_full_from_source(
            &BytesSource::new(Vec::new()),
            &source::zero_base_source(&BytesSource::new(Vec::new())),
            spec,
            TEST_WIDTHS,
        )
        .unwrap();
        assert_eq!(buffered, streamed);
        assert_eq!(buffered.len(), 24);
    }
}

#[test]
fn unsigned_subbyte_precision_masks_padding() {
    // 12 significant bits stored in 2 bytes: the top 4 bits are padding and
    // must be masked off.
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
