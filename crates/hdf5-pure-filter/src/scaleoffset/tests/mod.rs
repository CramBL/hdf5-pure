use rstest::rstest;

use super::*;

#[rstest]
#[case::one_byte(1, ORDER_LE)]
#[case::two_bytes(2, ORDER_BE)]
#[case::four_bytes(4, ORDER_LE)]
#[case::eight_bytes(8, ORDER_BE)]
fn full_width_modes_keep_storage_and_preserve_input_length(
    #[case] size: u32,
    #[case] order: u32,
    #[values(false, true)] owned: bool,
) {
    let raw = [1, 2, 3];
    let mut cd = int_filter(size, false, order, 1);
    cd[PARM_SCALEFACTOR] = size * 8;
    let input = view_input(&raw, owned);
    let pointer = input.as_slice().as_ptr();
    let mut spare = Vec::with_capacity(64);
    let spare_pointer = spare.as_ptr();
    let encoded = super::compress(input, &cd, &mut spare).unwrap();
    assert_eq!(encoded.as_slice(), raw);
    assert_eq!(encoded.as_slice().as_ptr(), pointer);
    let decoded = super::decompress(encoded, &cd, Some(8), &mut spare).unwrap();
    assert_eq!(decoded.as_slice(), raw);
    assert_eq!(decoded.as_slice().as_ptr(), pointer);
    assert_eq!(spare.as_ptr(), spare_pointer);
    assert!(spare.is_empty());
    let output = decoded.into_owned();
    assert_eq!(output, raw);
    if owned {
        assert_eq!(output.as_ptr(), pointer.wrapping_sub(2));
    } else {
        assert_ne!(output.as_ptr(), raw.as_ptr());
    }
}

#[rstest]
#[case::integer_one_byte(int_filter(1, false, ORDER_LE, 8), 8)]
#[case::integer_two_bytes(int_filter(2, true, ORDER_BE, 4), 16)]
#[case::integer_four_bytes(int_filter(4, false, ORDER_LE, 2), 32)]
#[case::integer_eight_bytes(int_filter(8, true, ORDER_BE, 1), 64)]
#[case::float_four_bytes(float_filter(4, 0, ORDER_LE, 2), 32)]
#[case::float_eight_bytes(float_filter(8, 0, ORDER_BE, 1), 64)]
fn raw_payload_views_retain_storage_and_exclude_headers_and_trailing_bytes(
    #[case] cd: Vec<u32>,
    #[case] bits: u8,
    #[values(false, true)] owned: bool,
) {
    let raw = [
        bits, 0, 0, 0, 8, 99, 99, 99, 99, 99, 99, 99, 99, 88, 88, 88, 88, 88, 88, 88, 88, 1, 2,
        3, 4, 5, 6, 7, 8, 77, 66,
    ];
    let input = view_input(&raw, owned);
    let pointer = input.as_slice().as_ptr();
    let mut spare = Vec::with_capacity(64);
    let spare_pointer = spare.as_ptr();
    let output = super::decompress(input, &cd, Some(8), &mut spare).unwrap();
    assert_eq!(output.as_slice(), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(output.as_slice().as_ptr(), pointer.wrapping_add(21));
    assert_eq!(spare.as_ptr(), spare_pointer);
    assert!(spare.is_empty());
    let bytes = output.into_owned();
    assert_eq!(bytes, [1, 2, 3, 4, 5, 6, 7, 8]);
    if owned {
        assert_eq!(bytes.as_ptr(), pointer.wrapping_sub(2));
    }
}

#[rstest]
#[case::short_header(&[8, 0, 0, 0], "scaleoffset: chunk shorter than header")]
#[case::width_before_minval(&[9, 0, 0, 0, 8], "scaleoffset: minbits exceeds datatype size")]
#[case::short_minval(&[8, 0, 0, 0, 8], "scaleoffset: chunk too short for minval")]
#[case::short_raw_header(&[8, 0, 0, 0, 0], "scaleoffset: chunk too short for raw payload")]
#[case::short_raw_payload(&[8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7], "scaleoffset: chunk too short for raw payload")]
fn raw_payload_views_preserve_header_error_order(
    #[case] input: &[u8],
    #[case] expected: &str,
    #[values(false, true)] owned: bool,
) {
    let cd = int_filter(1, false, ORDER_LE, 2);
    let mut spare = Vec::with_capacity(64);
    let pointer = spare.as_ptr();
    let error = super::decompress(view_input(input, owned), &cd, Some(2), &mut spare)
        .err()
        .unwrap();
    assert_eq!(error, Error::ScaleOffset(expected.into()));
    assert_eq!(spare.as_ptr(), pointer);
    assert!(spare.is_empty());
    let valid = [
        8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8,
    ];
    let output =
        super::decompress(view_input(&valid, owned), &cd, Some(2), &mut spare).unwrap();
    assert_eq!(output.as_slice(), [7, 8]);
    assert_eq!(spare.as_ptr(), pointer);
    assert!(spare.is_empty());
}

#[rstest]
#[case::full_width(8)]
#[case::above_full_width(9)]
fn full_width_decode_checks_the_declared_size_before_the_width(#[case] bits: u32) {
    let mut cd = int_filter(1, false, ORDER_LE, 2);
    cd[PARM_SCALEFACTOR] = bits;
    assert_eq!(
        decompress(&[], &cd, Some(1)).unwrap_err(),
        Error::ScaleOffset(
            "scaleoffset: chunk declares 2 decoded bytes, more than the 1 the chunk can hold"
                .into()
        )
    );
}

#[test]
fn raw_payload_size_cannot_overflow_the_header_end() {
    let cd = int_filter(1, false, ORDER_LE, u32::MAX);
    let input = [
        8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(
        decompress(&input, &cd, None).unwrap_err(),
        Error::ScaleOffset("scaleoffset: chunk too short for raw payload".into())
    );
}

fn view_input(input: &[u8], owned: bool) -> FilterBuffer<'_> {
    if owned {
        let mut bytes = Vec::with_capacity(input.len() + 4);
        bytes.extend_from_slice(&[55, 44]);
        bytes.extend_from_slice(input);
        bytes.extend_from_slice(&[33, 22]);
        FilterBuffer::borrowed(&[])
            .replace(&mut Vec::new(), bytes)
            .narrow(2..input.len() + 2)
            .unwrap()
    } else {
        FilterBuffer::borrowed(input)
    }
}

fn compress(input: &[u8], cd: &[u32]) -> Result<Vec<u8>, Error> {
    super::compress(FilterBuffer::borrowed(input), cd, &mut Vec::new())
        .map(FilterBuffer::into_owned)
}

fn decompress(input: &[u8], cd: &[u32], max_output: Option<usize>) -> Result<Vec<u8>, Error> {
    super::decompress(
        FilterBuffer::borrowed(input),
        cd,
        max_output,
        &mut Vec::new(),
    )
    .map(FilterBuffer::into_owned)
}

fn int_filter(size: u32, signed: bool, order: u32, nelmts: u32) -> Vec<u32> {
    let ty = ScaleOffsetType {
        class: CLS_INTEGER,
        sign: if signed { SGN_2 } else { SGN_NONE },
        order,
    };
    build_cd_values(
        ScaleOffset::Integer(0),
        ty,
        size,
        nelmts,
        ScaleOffsetFill::Undefined,
    )
    .unwrap()
}

fn float_filter(size: u32, decimals: i32, order: u32, nelmts: u32) -> Vec<u32> {
    let ty = ScaleOffsetType {
        class: CLS_FLOAT,
        sign: SGN_NONE,
        order,
    };
    build_cd_values(
        ScaleOffset::FloatDScale(decimals),
        ty,
        size,
        nelmts,
        ScaleOffsetFill::Undefined,
    )
    .unwrap()
}

#[test]
fn scale_offset_mode_returns_modes_and_rejects_invalid_parameters() {
    // Lossless integer is recovered (the only mode repack will re-emit).
    let f = int_filter(4, true, ORDER_LE, 16);
    assert_eq!(
        scale_offset_mode(&f),
        Some((ScaleOffset::Integer(0), FillAvailability::Undefined))
    );

    // Lossy float D-scale is recognized as such. Repack rejects it upstream.
    let f = float_filter(8, 3, ORDER_LE, 16);
    assert_eq!(
        scale_offset_mode(&f),
        Some((ScaleOffset::FloatDScale(3), FillAvailability::Undefined))
    );

    // A defined fill value is recovered as one, so repack re-applies the
    // availability recorded by the source.
    let mut fill_defined = int_filter(4, true, ORDER_LE, 16);
    fill_defined[PARM_FILAVAIL] = FILL_DEFINED;
    assert_eq!(
        scale_offset_mode(&fill_defined),
        Some((ScaleOffset::Integer(0), FillAvailability::Defined))
    );

    // Neither availability: a value the reference never writes, so what it
    // would mean is undefined and re-applying it would be a guess.
    let mut bad = int_filter(4, true, ORDER_LE, 16);
    bad[PARM_FILAVAIL] = 2;
    assert_eq!(scale_offset_mode(&bad), None);

    // Too-short parameter arrays -> None, never a panic.
    assert_eq!(scale_offset_mode(&[]), None);
    assert_eq!(scale_offset_mode(&[SO_INT, 0]), None);
}

/// `signed char` is the one type `H5Z__scaleoffset_precompress_i` hand-
/// expands directly without the `H5Z_scaleoffset_check_2` macro
/// (`H5Zscaleoffset.c`, HDF5 2.2.0),
/// and the expansion assigns `*minval` in its fill-**undefined** early
/// return where the macro assigns nothing. So a 1-byte signed chunk that
/// falls back to full precision carries the chunk's minimum in its header,
/// and every other width, signedness, and fill mode carries zero.
///
/// Every fixture's minimum is deliberately non-zero: a chunk whose minimum
/// is zero writes the same header under either rule.
#[test]
fn the_fill_undefined_fallback_carries_the_signed_char_minimum() {
    let minval_of = |raw: &[u8], size: u32, signed: bool| {
        let nelmts = raw.len() as u32 / size;
        let f = int_filter(size, signed, ORDER_LE, nelmts);
        let out = compress(raw, &f).unwrap();
        // The fixtures all spread past the point where packing can pay, so
        // each must have taken the full-precision fallback.
        assert_eq!(
            u32::from_le_bytes(out[..4].try_into().unwrap()),
            size * 8,
            "the fixture must reach the full-precision fallback"
        );
        u64::from_le_bytes(out[5..13].try_into().unwrap())
    };

    let i8_raw: Vec<u8> = [-128i8, 126, -128, 126, -1, -2, -3, -4]
        .iter()
        .map(|&v| v as u8)
        .collect();
    assert_eq!(minval_of(&i8_raw, 1, true), (-128i64) as u64);

    // One byte wide but unsigned: the macro-driven path, so zero. Its own
    // fixture, because the signed one read as unsigned spans only 126..255
    //  -  inside the threshold, so it would reach the *other* full-precision
    // return, the one that does carry the minimum in every mode.
    let u8_raw: Vec<u8> = vec![1, 255, 1, 255, 2, 3, 4, 5];
    assert_eq!(minval_of(&u8_raw, 1, false), 0);

    let i16_raw: Vec<u8> = [-32768i16, 32766, -32768, 32766, -1, -2, -3, -4]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        minval_of(&i16_raw, 2, true),
        0,
        "signed, but two bytes wide"
    );

    // And the fill-defined half of the same 1-byte signed chunk, which the
    // crosscheck pins against the C library: the hand-expanded branch
    // assigns nothing there either.
    let mut defined = int_filter(1, true, ORDER_LE, i8_raw.len() as u32);
    defined[PARM_FILAVAIL] = FILL_DEFINED;
    let out = compress(&i8_raw, &defined).unwrap();
    assert_eq!(u64::from_le_bytes(out[5..13].try_into().unwrap()), 0);
}

/// The fill value's round trip through `cd_values`: written the way
/// `H5Z__scaleoffset_set_parms_fillval` writes it and read back by the same
/// [`read_fill_bits`] the decoder uses, at every width and in both byte
/// orders. A `u64` fill spans two entries, which is the case a single-entry
/// writer would silently truncate.
#[test]
fn a_fill_value_round_trips_through_the_filter_parameters() {
    for (size, bits) in [
        (1u32, 0xA5u64),
        (2, 0xBEEF),
        (4, 0xDEAD_BEEF),
        (8, 0x0123_4567_89AB_CDEF),
    ] {
        for order in [ORDER_LE, ORDER_BE] {
            let ty = ScaleOffsetType {
                class: CLS_INTEGER,
                sign: SGN_NONE,
                order,
            };
            // The fill value as the dataset stores it: `size` bytes in the
            // dataset's own byte order.
            let mut bytes = Vec::new();
            write_value(&mut bytes, bits, size as usize, order);
            let cd = build_cd_values(
                ScaleOffset::Integer(0),
                ty,
                size,
                4,
                ScaleOffsetFill::Defined(Some(&bytes)),
            )
            .unwrap();

            assert_eq!(cd[PARM_FILAVAIL], FILL_DEFINED);
            assert_eq!(
                read_fill_bits(&cd, size as usize).unwrap(),
                bits,
                "size {size}, order {order}"
            );
            // Entries past the value stay zero, as the reference leaves them.
            let used = (size as usize).div_ceil(4);
            assert!(cd[PARM_FILVAL + used..].iter().all(|&e| e == 0));
        }
    }

    // The library default is a defined fill value of zero, not an absent one.
    let ty = ScaleOffsetType {
        class: CLS_INTEGER,
        sign: SGN_NONE,
        order: ORDER_LE,
    };
    let cd = build_cd_values(
        ScaleOffset::Integer(0),
        ty,
        4,
        4,
        ScaleOffsetFill::Defined(None),
    )
    .unwrap();
    assert_eq!(cd[PARM_FILAVAIL], FILL_DEFINED);
    assert_eq!(read_fill_bits(&cd, 4).unwrap(), 0);

    // A datatype wider than the filter's own limit is rejected before
    // running the fill value off the end of the parameter array.
    for width in [0u32, 9, 16] {
        assert_eq!(
            build_cd_values(
                ScaleOffset::Integer(0),
                ty,
                width,
                4,
                ScaleOffsetFill::Defined(Some(&vec![0u8; width as usize]))
            )
            .unwrap_err(),
            Error::ScaleOffset(format!("scaleoffset: unsupported datatype size {width}"))
        );
    }

    // A fill value that is not one element wide is rejected before
    // truncated or padded into a different value.
    for bad in [vec![1u8], vec![1u8; 8]] {
        assert_eq!(
            build_cd_values(
                ScaleOffset::Integer(0),
                ty,
                4,
                4,
                ScaleOffsetFill::Defined(Some(&bad))
            )
            .unwrap_err(),
            Error::ScaleOffset(format!(
                "scaleoffset: fill value is {} bytes for a 4-byte datatype",
                bad.len()
            ))
        );
    }
}

/// The rounding these helpers do is C `llround`/`lroundf`  -  round `x` half
/// away from zero  -  and not "round `x + 0.5`", which is a different
/// function. The two disagree at exactly one input per sign and precision:
/// the largest float below one half, where the sum lands *on* the half and
/// rounds up.
///
/// That input reaches the encoder as `value * 10^D - min_scaled`, so it is
/// data-dependent and changes the decoded value.
#[test]
fn rounding_is_llround_and_not_the_rounded_sum() {
    // The half-way cases themselves round away from zero, in both signs and
    // at every magnitude where a half is representable.
    for (x, want) in [
        (0.5f64, 1i64),
        (-0.5, -1),
        (1.5, 2),
        (-1.5, -2),
        (2.5, 3),
        (-2.5, -3),
        (0.0, 0),
        (0.4, 0),
        (-0.4, 0),
        (7.0, 7),
    ] {
        assert_eq!(round_half_away_f64(x), want, "f64 round({x})");
    }

    // The largest double below one half: the sum `x + 0.5` rounds up to
    // exactly 1.0, where rounding `x` itself gives 0.
    let below_half = 0.499_999_999_999_999_94f64;
    assert!(below_half < 0.5 && below_half + 0.5 == 1.0);
    assert_eq!(round_half_away_f64(below_half), 0);
    assert_eq!(round_half_away_f64(-below_half), 0);

    // The `f32` counterpart, where the sum rounds to exactly 1.0.
    let below_half_f32 = 0.499_999_97f32;
    assert!(below_half_f32 < 0.5 && below_half_f32 + 0.5 == 1.0);
    assert_eq!(round_half_away_f32(below_half_f32), 0);
    assert_eq!(round_half_away_f32(-below_half_f32), 0);
    assert_eq!(round_half_away_f32(0.5), 1);
    assert_eq!(round_half_away_f32(-2.5), -3);

    // Past the point where consecutive floats are more than 1 apart every
    // value is already an integer, and beyond the integer range the cast
    // saturates on overflow (C leaves it undefined).
    assert_eq!(round_half_away_f64(1e300), i64::MAX);
    assert_eq!(round_half_away_f64(-1e300), i64::MIN);
    assert_eq!(round_half_away_f32(1e30), i64::MAX);
    assert_eq!(round_half_away_f64(f64::NAN), 0);
    assert_eq!(round_half_away_f32(f32::NAN), 0);
    // Either side of the constant that separates the two branches.
    let integral = (1u64 << 52) as f64;
    assert_eq!(round_half_away_f64(integral), 1 << 52);
    assert_eq!(round_half_away_f64(integral - 1.0), (1 << 52) - 1);
    assert_eq!(round_half_away_f64(-integral), -(1 << 52));
}

#[test]
fn ceil_log2_matches_reference() {
    assert_eq!(ceil_log2(0), 1);
    assert_eq!(ceil_log2(1), 0);
    assert_eq!(ceil_log2(2), 1);
    assert_eq!(ceil_log2(3), 2);
    assert_eq!(ceil_log2(4), 2);
    assert_eq!(ceil_log2(5), 3);
    assert_eq!(ceil_log2(255), 8);
    assert_eq!(ceil_log2(256), 8);
    assert_eq!(ceil_log2(257), 9);
}

fn roundtrip_u32(vals: &[u32], order: u32) {
    let mut raw = Vec::new();
    for &v in vals {
        if order == ORDER_LE {
            raw.extend_from_slice(&v.to_le_bytes());
        } else {
            raw.extend_from_slice(&v.to_be_bytes());
        }
    }
    let f = int_filter(4, false, order, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    assert_eq!(dec, raw);
}

#[test]
fn integer_unsigned_roundtrip_le_and_be() {
    let vals = [100u32, 105, 101, 110, 100, 128];
    roundtrip_u32(&vals, ORDER_LE);
    roundtrip_u32(&vals, ORDER_BE);
}

#[test]
fn integer_signed_roundtrip_with_negatives() {
    let vals: [i16; 6] = [-100, -50, -100, 0, 27, -99];
    for &order in &[ORDER_LE, ORDER_BE] {
        let mut raw = Vec::new();
        for &v in &vals {
            if order == ORDER_LE {
                raw.extend_from_slice(&v.to_le_bytes());
            } else {
                raw.extend_from_slice(&v.to_be_bytes());
            }
        }
        let f = int_filter(2, true, order, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw, "order {order}");
    }
}

#[test]
fn integer_all_equal_uses_minbits_zero() {
    let vals = [7u32; 5];
    let mut raw = Vec::new();
    for &v in &vals {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    // minbits == 0 -> 21-byte header + 1 trailing byte.
    assert_eq!(comp.len(), HEADER_LEN + 1);
    assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 0);
    let dec = decompress(&comp, &f, None).unwrap();
    assert_eq!(dec, raw);
}

#[test]
fn integer_full_range_uses_raw_path() {
    // 0 and u32::MAX force the full-precision (raw) path.
    let vals = [0u32, u32::MAX, 123];
    let mut raw = Vec::new();
    for &v in &vals {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    assert_eq!(comp.len(), HEADER_LEN + raw.len());
    assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 32);
    let dec = decompress(&comp, &f, None).unwrap();
    assert_eq!(dec, raw);
}

#[test]
fn integer_u8_roundtrip() {
    let raw = vec![10u8, 11, 12, 250, 10, 200];
    let f = int_filter(1, false, ORDER_LE, raw.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    assert_eq!(dec, raw);
}

#[test]
fn integer_i64_roundtrip() {
    let vals: [i64; 4] = [-1_000_000, 5, -999_999, 42];
    let mut raw = Vec::new();
    for &v in &vals {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let f = int_filter(8, true, ORDER_LE, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    assert_eq!(dec, raw);
}

#[test]
fn float_dscale_roundtrip_within_tolerance() {
    let vals = [1.234f64, 1.235, 1.250, 1.111, 1.234, 1.999];
    let decimals = 3;
    let mut raw = Vec::new();
    for &v in &vals {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let f = float_filter(8, decimals, ORDER_LE, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    let got: Vec<f64> = dec
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    let tol = 0.5 * 10f64.powi(-decimals);
    for (g, v) in got.iter().zip(vals.iter()) {
        assert!((g - v).abs() <= tol, "got {g}, want {v}");
    }
}

#[test]
fn float32_dscale_roundtrip_be() {
    let vals = [10.25f32, 10.50, 10.75, 10.00, 10.25];
    let decimals = 2;
    let mut raw = Vec::new();
    for &v in &vals {
        raw.extend_from_slice(&v.to_be_bytes());
    }
    let f = float_filter(4, decimals, ORDER_BE, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    let got: Vec<f32> = dec
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_be_bytes(*c))
        .collect();
    let tol = 0.5 * 10f32.powi(-decimals);
    for (g, v) in got.iter().zip(vals.iter()) {
        assert!((g - v).abs() <= tol, "got {g}, want {v}");
    }
}

#[test]
fn decompress_rejects_chunk_shorter_than_packed_header() {
    // A chunk whose header claims minbits=3 but is shorter than the 21-byte
    // header region must report an error before slicing the payload.
    let f = int_filter(4, false, ORDER_LE, 4);
    let mut bad = Vec::new();
    bad.extend_from_slice(&3u32.to_le_bytes()); // minbits = 3
    bad.push(8); // minval size
    bad.extend_from_slice(&0u64.to_le_bytes()); // minval (bytes 5..13)
    // Only 13 bytes total: shorter than the 21-byte header.
    assert_eq!(
        decompress(&bad, &f, None).unwrap_err(),
        Error::ScaleOffset("scaleoffset: chunk too short for packed payload".to_string())
    );
}

#[test]
fn header_byte_layout() {
    // Four u32 values with min=5 and max=9: span 5, minbits 3.
    let vals = [5u32, 9, 6, 5];
    let mut raw = Vec::new();
    for &v in &vals {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 3);
    assert_eq!(comp[4], 8); // sizeof(minval)
    let minval = u64::from_le_bytes(comp[5..13].try_into().unwrap());
    assert_eq!(minval, 5);
    assert_eq!(&comp[13..21], &[0u8; 8]); // padding
}

// A zero-bit chunk expands from its header alone, so the output bound comes from the pipeline.
#[test]
fn decompress_rejects_declared_size_above_output_cap() {
    // A 21-byte header declaring 2^28 elements of 8 bytes.
    let mut chunk = vec![0u8; HEADER_LEN];
    chunk[4] = 8; // sizeof(minval), minbits stays 0

    let huge = int_filter(8, true, ORDER_LE, 1 << 28);
    assert_eq!(
        decompress(&chunk, &huge, Some(4096)).unwrap_err(),
        Error::ScaleOffset(
            "scaleoffset: chunk declares 2147483648 decoded bytes, more than the \
             4096 the chunk can hold"
                .to_string()
        )
    );

    let fits = int_filter(8, true, ORDER_LE, 512);
    assert_eq!(decompress(&chunk, &fits, Some(4096)).unwrap().len(), 4096);
    assert_eq!(decompress(&chunk, &fits, None).unwrap().len(), 4096);
}

#[test]
fn build_cd_values_rejects_mismatched_mode() {
    let int_ty = ScaleOffsetType {
        class: CLS_INTEGER,
        sign: SGN_2,
        order: ORDER_LE,
    };
    assert_eq!(
        build_cd_values(
            ScaleOffset::FloatDScale(2),
            int_ty,
            4,
            10,
            ScaleOffsetFill::Undefined
        )
        .unwrap_err(),
        Error::ScaleOffset(
            "scaleoffset: float D-scale mode requires a floating-point dataset".to_string()
        )
    );
    let float_ty = ScaleOffsetType {
        class: CLS_FLOAT,
        sign: SGN_NONE,
        order: ORDER_LE,
    };
    assert_eq!(
        build_cd_values(
            ScaleOffset::Integer(0),
            float_ty,
            8,
            10,
            ScaleOffsetFill::Undefined
        )
        .unwrap_err(),
        Error::ScaleOffset("scaleoffset: integer mode requires an integer dataset".to_string())
    );
}

/// Deterministic xorshift64 used to drive the property-style tests below.
/// Keeps the dev-dependency footprint minimal while still covering a wide
/// input space. seeds are fixed so failures are reproducible.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, hi: u64) -> u64 {
        // Modulo bias is fine for our coverage purposes.
        self.next() % hi
    }
}

#[test]
fn pack_unpack_equivalence_random() {
    // For every minbits in 1..=64 and across many random widths/lengths,
    // pack(offsets) followed by unpack must return the original offsets.
    let mut rng = Rng::new(0x00C0_FFEE_F00D_1234);
    for _ in 0..400 {
        let minbits = (rng.range(64) + 1) as u32; // 1..=64
        let nelmts = (rng.range(257) + 1) as usize; // 1..=257
        let mask = if minbits == 64 {
            u64::MAX
        } else {
            (1u64 << minbits) - 1
        };
        let offsets: Vec<u64> = (0..nelmts).map(|_| rng.next() & mask).collect();
        let packed = pack_offsets(&offsets, minbits, nelmts).unwrap();
        // Reference layout reserves at least one trailing byte.
        assert_eq!(packed.len(), nelmts * minbits as usize / 8 + 1);
        let unpacked = unpack_bits(&packed, nelmts, minbits).unwrap();
        assert_eq!(
            unpacked, offsets,
            "minbits={minbits}, nelmts={nelmts}, seed=0xC0FFEEF00D1234"
        );
    }
}

fn roundtrip_random<T: Copy>(
    seed: u64,
    size: u32,
    signed: bool,
    order: u32,
    encode: impl Fn(&mut Rng) -> T,
    to_bytes: impl Fn(T, u32) -> Vec<u8>,
) {
    let mut rng = Rng::new(seed);
    for trial in 0..40 {
        let nelmts = (rng.range(199) + 1) as usize;
        let mut raw = Vec::with_capacity(nelmts * size as usize);
        for _ in 0..nelmts {
            raw.extend_from_slice(&to_bytes(encode(&mut rng), order));
        }
        let f = int_filter(size, signed, order, nelmts as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(
            dec, raw,
            "trial {trial}: size={size}, signed={signed}, order={order}"
        );
    }
}

#[test]
fn integer_roundtrip_random_u8() {
    roundtrip_random::<u8>(0x11, 1, false, ORDER_LE, |r| r.next() as u8, |v, _| vec![v]);
}

#[test]
fn integer_roundtrip_random_u16_le_and_be() {
    for &order in &[ORDER_LE, ORDER_BE] {
        roundtrip_random::<u16>(
            0x22 ^ order as u64,
            2,
            false,
            order,
            |r| r.next() as u16,
            |v, o| {
                if o == ORDER_LE {
                    v.to_le_bytes().to_vec()
                } else {
                    v.to_be_bytes().to_vec()
                }
            },
        );
    }
}

#[test]
fn integer_roundtrip_random_i32_narrow_and_wide() {
    // Mix narrow (small span -> small minbits) and wide (forces raw path).
    let mut rng = Rng::new(0x33);
    for trial in 0..40 {
        let nelmts = (rng.range(199) + 1) as usize;
        let narrow = rng.next() & 1 == 0;
        let mut raw = Vec::with_capacity(nelmts * 4);
        for _ in 0..nelmts {
            let v: i32 = if narrow {
                (rng.range(1024) as i32) - 512 // span ~= 1024
            } else {
                rng.next() as i32 // full range
            };
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(4, true, ORDER_LE, nelmts as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw, "trial {trial}: narrow={narrow}");
    }
}

#[test]
fn integer_roundtrip_random_i64() {
    roundtrip_random::<i64>(
        0x44,
        8,
        true,
        ORDER_LE,
        |r| {
            // Bias toward narrow spans so we exercise the packing path,
            // but include the full-range tail too.
            let span = 1u64 << (r.range(48) as u32 + 1);
            let v = (r.next() % span) as i64;
            let off = (r.next() as i64) >> 1;
            v.wrapping_add(off)
        },
        |v, _| v.to_le_bytes().to_vec(),
    );
}

#[test]
fn float_dscale_roundtrip_random_within_tolerance() {
    let mut rng = Rng::new(0x55);
    for trial in 0..30 {
        let nelmts = (rng.range(99) + 1) as usize;
        let decimals = (rng.range(5) as i32) + 1; // 1..=5
        // Keep magnitudes modest so f64 ULP stays well below the D-scale
        // tolerance. Avoid generating values pre-quantized to the same
        // precision (decimals) we then round to, otherwise the input
        // sits exactly on a rounding boundary.
        let mut raw = Vec::with_capacity(nelmts * 8);
        let mut vals = Vec::with_capacity(nelmts);
        for _ in 0..nelmts {
            let v = (rng.next() as i32 as f64) * 1e-9; // |v| <= ~2.1
            vals.push(v);
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = float_filter(8, decimals, ORDER_LE, nelmts as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        let got: Vec<f64> = dec
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        // 0.5 ULP rounding + a few ULP of float arithmetic noise.
        let tol = 0.501 * 10f64.powi(-decimals);
        for (g, v) in got.iter().zip(vals.iter()) {
            assert!(
                (g - v).abs() <= tol,
                "trial {trial}: got {g}, want {v} (tol {tol})"
            );
        }
    }
}

/// Build a synthetic chunk where `minbits == 0`, the compressed header
/// declares `minval = 7`, but the filter's `cd_values` say
/// `filavail = FILL_DEFINED` with `filval = 99`.
///
/// The zero-bit reserved fill code matches every element, so this chunk
/// decodes to five copies of the fill value.
#[test]
fn minbits_zero_with_fill_defined_reads_as_the_fill_value() {
    let nelmts = 5u32;
    let size = 4u32;
    let mut cd = vec![0u32; TOTAL_NPARMS];
    cd[PARM_SCALETYPE] = SO_INT;
    cd[PARM_SCALEFACTOR] = 0;
    cd[PARM_NELMTS] = nelmts;
    cd[PARM_CLASS] = CLS_INTEGER;
    cd[PARM_SIZE] = size;
    cd[PARM_SIGN] = SGN_NONE;
    cd[PARM_ORDER] = ORDER_LE;
    cd[PARM_FILAVAIL] = FILL_DEFINED;
    cd[PARM_FILVAL] = 99;
    let f = cd;
    // 21-byte header (minbits=0, minval=7) + 1-byte trailing pad.
    let mut chunk = Vec::with_capacity(HEADER_LEN + 1);
    chunk.extend_from_slice(&0u32.to_le_bytes());
    chunk.push(8);
    chunk.extend_from_slice(&7u64.to_le_bytes());
    chunk.extend_from_slice(&[0u8; HEADER_LEN - 13]);
    chunk.push(0);
    let out = decompress(&chunk, &f, None).unwrap();
    let got: Vec<u32> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(
        got,
        vec![99u32; nelmts as usize],
        "minbits==0 with FILL_DEFINED reads as the fill value, not minval"
    );
}

/// A chunk with `minbits > 0` and `FILL_DEFINED`: the offsets at the
/// all-ones reserved fill code must reconstruct to `filval`, while every other
/// offset reconstructs to `minval + offset`. Companion to the
/// `minbits == 0` regression above  -  together they cover both sides of
/// the reserved fill code branch.
#[test]
fn fill_defined_sentinel_emits_filval_not_minval() {
    let nelmts = 4u32;
    let size = 4u32;
    let mut cd = vec![0u32; TOTAL_NPARMS];
    cd[PARM_SCALETYPE] = SO_INT;
    cd[PARM_SCALEFACTOR] = 0;
    cd[PARM_NELMTS] = nelmts;
    cd[PARM_CLASS] = CLS_INTEGER;
    cd[PARM_SIZE] = size;
    cd[PARM_SIGN] = SGN_NONE;
    cd[PARM_ORDER] = ORDER_LE;
    cd[PARM_FILAVAIL] = FILL_DEFINED;
    cd[PARM_FILVAL] = 999;
    let f = cd;
    // minbits = 3 → reserved fill code = 7. Offsets [0, 1, 7, 2] expect
    // [minval+0, minval+1, `filval`, minval+2] = [10, 11, 999, 12].
    let mut chunk = Vec::new();
    chunk.extend_from_slice(&3u32.to_le_bytes());
    chunk.push(8);
    chunk.extend_from_slice(&10u64.to_le_bytes());
    chunk.extend_from_slice(&[0u8; HEADER_LEN - 13]);
    chunk.extend_from_slice(&pack_offsets(&[0, 1, 7, 2], 3, 4).unwrap());
    let out = decompress(&chunk, &f, None).unwrap();
    let got: Vec<u32> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(got, vec![10u32, 11, 999, 12]);
}

/// An integer filter carrying a defined fill value, the shape the
/// reference library writes for any dataset with one.
fn int_filter_with_fill(size: u32, signed: bool, order: u32, nelmts: u32, fill: u64) -> Vec<u32> {
    // Through the writer: a test
    // helper that laid them out itself would agree with a writer that laid
    // them out wrongly.
    let mut bytes = Vec::new();
    write_value(&mut bytes, fill, size as usize, order);
    build_cd_values(
        ScaleOffset::Integer(0),
        ScaleOffsetType {
            class: CLS_INTEGER,
            sign: if signed { SGN_2 } else { SGN_NONE },
            order,
        },
        size,
        nelmts,
        ScaleOffsetFill::Defined(Some(&bytes)),
    )
    .unwrap()
}

/// A big-endian dataset's fill value is carried through `cd_values` as a
/// *value*, not as the dataset's bytes: the writer converts on the way in
/// and the decoder converts on the way out, matching what the reference does
/// on a little-endian host. Convert on only one side and a fill element
/// matches nothing  -  or, worse, something else does.
///
/// `minbits` is what shows it. The non-fill values here span three code
/// points, so recognizing the fill value packs them into two bits. missing
/// it stretches the chunk's range across the whole distance to `0x0BAD`.
#[test]
fn a_big_endian_fill_value_is_recognized() {
    let fill = 0x0BADu64;
    let values: Vec<u16> = vec![0x0BAD, 3, 4, 0x0BAD, 5];
    for order in [ORDER_LE, ORDER_BE] {
        let raw: Vec<u8> = values
            .iter()
            .flat_map(|v| {
                if order == ORDER_LE {
                    v.to_le_bytes()
                } else {
                    v.to_be_bytes()
                }
            })
            .collect();
        let f = int_filter_with_fill(2, false, order, values.len() as u32, fill);
        let packed = compress(&raw, &f).unwrap();

        assert_eq!(
            u32::from_le_bytes(packed[..4].try_into().unwrap()),
            2,
            "order {order}: the fill value must be recognized, not packed"
        );
        assert_eq!(
            decompress(&packed, &f, None).unwrap(),
            raw,
            "order {order}: round trip"
        );
    }
}

/// A defined fill value round-trips in mixed, all-fill, and no-fill chunks
/// in both byte orders.
#[test]
fn compress_round_trips_with_a_defined_fill_value() {
    let fill = 0xDEAD_u32;
    for (label, values) in [
        ("mixed", vec![10u32, fill, 11, 12, fill, 10]),
        ("all fill", vec![fill; 6]),
        ("no fill", vec![10u32, 11, 12, 13, 14, 15]),
        // min = 10, max = 13: a span of 4 needs 2 bits, whose all-ones
        // code is the offset 13 would otherwise take.
        ("sentinel collision", vec![10u32, 11, 12, 13, fill, 13]),
    ] {
        for order in [ORDER_LE, ORDER_BE] {
            let f = int_filter_with_fill(4, false, order, values.len() as u32, u64::from(fill));
            let raw: Vec<u8> = values
                .iter()
                .flat_map(|v| {
                    let le = v.to_le_bytes();
                    if order == ORDER_LE {
                        le.to_vec()
                    } else {
                        le.iter().rev().copied().collect()
                    }
                })
                .collect();
            let packed = compress(&raw, &f).unwrap();
            let out = decompress(&packed, &f, None).unwrap();
            assert_eq!(out, raw, "{label}, order {order}");
        }
    }
}

/// A float filter carrying a defined fill value.
fn float_filter_with_fill(
    size: u32,
    decimals: i32,
    order: u32,
    nelmts: u32,
    fill: u64,
) -> Vec<u32> {
    let mut f = float_filter(size, decimals, order, nelmts);
    f[PARM_FILAVAIL] = FILL_DEFINED;
    for (k, entry) in fill
        .to_le_bytes()
        .chunks(4)
        .take((size as usize).div_ceil(4))
        .enumerate()
    {
        let mut w = [0u8; 4];
        w[..entry.len()].copy_from_slice(entry);
        f[PARM_FILVAL + k] = u32::from_le_bytes(w);
    }
    f
}

/// Defined float fill values round-trip for both element widths and byte orders.
#[test]
fn float_compress_round_trips_with_a_defined_fill_value() {
    let decimals = 3;
    for size in [4u32, 8] {
        for order in [ORDER_LE, ORDER_BE] {
            for (label, values) in [
                ("mixed", vec![1.5f64, -999.0, 2.25, -999.0, 3.125]),
                ("all fill", vec![-999.0f64; 5]),
                ("no fill", vec![1.5f64, 2.25, 3.125, 4.0, 5.5]),
                // Scaled by 10^3 this spans exactly 1024 values, so the
                // largest offset is the all-ones code of an unwidened
                // `minbits` and would decode as the fill value. The spans
                // above cannot distinguish a reserved code point from an
                // unreserved one. only a power-of-two span can.
                (
                    "sentinel collision",
                    vec![0.0f64, 1.023, 0.5, -999.0, 1.023],
                ),
            ] {
                let fill_bits = if size == 4 {
                    u64::from((-999.0f32).to_bits())
                } else {
                    (-999.0f64).to_bits()
                };
                let n = values.len() as u32;
                let f = float_filter_with_fill(size, decimals, order, n, fill_bits);
                let raw: Vec<u8> = values
                    .iter()
                    .flat_map(|v| {
                        let le = if size == 4 {
                            (*v as f32).to_bits().to_le_bytes().to_vec()
                        } else {
                            v.to_bits().to_le_bytes().to_vec()
                        };
                        if order == ORDER_LE {
                            le
                        } else {
                            le.iter().rev().copied().collect()
                        }
                    })
                    .collect();
                let packed = compress(&raw, &f).unwrap();
                let out = decompress(&packed, &f, None).unwrap();

                let got: Vec<f64> = out
                    .chunks_exact(size as usize)
                    .map(|c| {
                        let mut b = c.to_vec();
                        if order == ORDER_BE {
                            b.reverse();
                        }
                        if size == 4 {
                            f64::from(f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
                        } else {
                            f64::from_bits(u64::from_le_bytes(b.try_into().unwrap()))
                        }
                    })
                    .collect();
                let tol = 0.501 * 10f64.powi(-decimals);
                assert_eq!(
                    got.len(),
                    values.len(),
                    "{label}, size {size}, order {order}"
                );
                for (g, v) in got.iter().zip(values.iter()) {
                    assert!(
                        (g - v).abs() <= tol,
                        "{label}, size {size}, order {order}: got {g}, want {v}"
                    );
                }
            }
        }
    }
}

/// Both directions pass a full-width integer chunk through without a header
/// and reject a requested width above the datatype width.
#[test]
fn a_full_width_scale_factor_passes_the_chunk_through_both_ways() {
    let raw: Vec<u8> = (0..40u8).collect();

    for size in [1u32, 2, 4, 8] {
        let bits = size * 8;
        let nelmts = raw.len() as u32 / size;

        let mut f = int_filter(size, false, ORDER_LE, nelmts);
        f[PARM_SCALEFACTOR] = bits;
        assert_eq!(compress(&raw, &f).unwrap(), raw, "size {size}: compress");
        assert_eq!(
            decompress(&raw, &f, None).unwrap(),
            raw,
            "size {size}: decompress"
        );

        // One bit past the datatype is not a width at all.
        f[PARM_SCALEFACTOR] = bits + 1;
        let expected = Error::ScaleOffset(format!(
            "scaleoffset: requested minimum bit width {} exceeds the {bits}-bit datatype",
            bits + 1
        ));
        assert_eq!(
            compress(&raw, &f).unwrap_err(),
            expected,
            "size {size}: compress"
        );
        assert_eq!(
            decompress(&raw, &f, None).unwrap_err(),
            expected,
            "size {size}: decompress"
        );
    }
}

/// The pass-through is an early return, so everything that must not be
/// skipped has to sit above it. A `cd_values` whose class and scale type
/// disagree is rejected for every width  -  the reference
/// validates that first.
#[test]
fn a_pass_through_width_does_not_skip_the_class_check() {
    let raw = vec![0u8; 32];
    // Float class carrying the *integer* scale type, at the pass-through
    // width. Reachable only from a malformed parameter array.
    let mut f = float_filter(4, 0, ORDER_LE, 8);
    f[PARM_SCALETYPE] = SO_INT;
    f[PARM_SCALEFACTOR] = 32;
    assert_eq!(
        compress(&raw, &f).unwrap_err(),
        Error::ScaleOffset("scaleoffset: float class requires D-scale scale type".to_string())
    );

    // Integer class carrying float E-scale, likewise.
    let mut f = int_filter(4, false, ORDER_LE, 8);
    f[PARM_SCALETYPE] = SO_FLOAT_ESCALE;
    f[PARM_SCALEFACTOR] = 32;
    assert_eq!(
        compress(&raw, &f).unwrap_err(),
        Error::ScaleOffset("scaleoffset: integer class requires integer scale type".to_string())
    );
}

/// `precompress_integer` has *two* full-precision returns and they carry
/// different `minval`s. The spread guard fires before the chunk's minimum is
/// known to be usable and leaves zero. the `minbits >= full_bits` return
/// below it is reached with the minimum computed, and the reference's
/// trailing `*minval = min` does run there.
///
/// `the_full_precision_fallback_header_matches_the_c_library` reaches only
/// the first. This reaches the second: a spread wide enough that
/// `ceil_log2` saturates the datatype without exceeding `width_max - 2`.
#[test]
fn the_second_full_precision_return_carries_the_chunk_minimum() {
    // u8, min = 1, max = 128: spread 127, inside the guard (<= 253), and
    // with a fill value `ceil_log2(127 + 2) = 8`  -  the full width.
    let values: Vec<u8> = vec![9, 1, 128, 9, 1, 128, 2, 3];
    let f = int_filter_with_fill(1, false, ORDER_LE, values.len() as u32, 9);
    let packed = compress(&values, &f).unwrap();
    assert_eq!(
        u32::from_le_bytes(packed[..4].try_into().unwrap()),
        8,
        "the fixture must reach the full width"
    );
    assert_eq!(
        u64::from_le_bytes(packed[5..13].try_into().unwrap()),
        1,
        "the second full-precision return carries the chunk minimum, not zero"
    );
    // Stored raw after the header, never packed at the full width.
    assert_eq!(&packed[HEADER_LEN..], &values[..]);
}

/// Reserving a code point for the reserved fill code is what makes the round trip
/// above possible, and it is visible in the encoded header: a chunk with a
/// defined fill value covers `span + 1` values where one without covers
/// `span`.
///
/// Swept over every span. The two
/// formulas agree at most spans  -  they differ only where `span + 1` crosses
/// a power of two  -  so a single fixture is as likely to be blind to the
/// reservation as to catch it, and equally blind to a *double* reservation.
#[test]
fn a_defined_fill_value_reserves_exactly_one_code_point() {
    let fill = 0xDEAD_u32;
    for span in 1..=40u32 {
        // `span` distinct values, 0..span, plus one fill element.
        let mut values: Vec<u32> = (0..span).collect();
        values.push(fill);
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let n = values.len() as u32;

        let minbits_of = |f: &[u32]| {
            let packed = compress(&raw, f).unwrap();
            u32::from_le_bytes(packed[..4].try_into().unwrap())
        };

        // Without a fill value the lone `fill` element is just another
        // value, so the span is the whole 0..=fill range. use a separate
        // buffer for that half.
        let plain: Vec<u8> = (0..span).flat_map(|v| v.to_le_bytes()).collect();
        let packed = compress(&plain, &int_filter(4, false, ORDER_LE, span)).unwrap();
        let plain_minbits = u32::from_le_bytes(packed[..4].try_into().unwrap());
        assert_eq!(plain_minbits, ceil_log2(u64::from(span)), "span {span}");

        assert_eq!(
            minbits_of(&int_filter_with_fill(
                4,
                false,
                ORDER_LE,
                n,
                u64::from(fill)
            )),
            ceil_log2(u64::from(span) + 1),
            "span {span}: a defined fill value must reserve one code point, \
             no more and no less"
        );
    }
}

/// Float E-scale is forbidden by the reference library too. we error
/// before touching the payload.
#[test]
fn decompress_rejects_escale() {
    let mut cd = vec![0u32; TOTAL_NPARMS];
    cd[PARM_SCALETYPE] = SO_FLOAT_ESCALE;
    cd[PARM_NELMTS] = 4;
    cd[PARM_CLASS] = CLS_FLOAT;
    cd[PARM_SIZE] = 4;
    cd[PARM_ORDER] = ORDER_LE;
    let f = cd;
    let chunk = vec![0u8; HEADER_LEN + 4];
    assert_eq!(
        decompress(&chunk, &f, None).unwrap_err(),
        Error::ScaleOffset("scaleoffset: float E-scale method is not supported".to_string())
    );
}

/// A corrupt header that claims `minbits > size * 8` must report an error before reaching the bit reader.
#[test]
fn decompress_rejects_oversized_minbits_header() {
    let f = int_filter(4, false, ORDER_LE, 4);
    // size=4 → full_bits=32. Set minbits=33.
    let mut bad = Vec::new();
    bad.extend_from_slice(&33u32.to_le_bytes());
    bad.push(8);
    bad.extend_from_slice(&0u64.to_le_bytes());
    bad.extend_from_slice(&[0u8; HEADER_LEN - 13]);
    bad.extend_from_slice(&[0u8; 16]); // dummy payload
    assert_eq!(
        decompress(&bad, &f, None).unwrap_err(),
        Error::ScaleOffset("scaleoffset: minbits exceeds datatype size".to_string())
    );
}

/// Signed 1-byte ints: covers the `size == 1` arm of `sign_extend`,
/// which the unsigned `u8` round-trip skips.
#[test]
fn integer_i8_roundtrip_with_negatives() {
    let vals: [i8; 6] = [-100, -50, 0, 27, -99, 100];
    let raw: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
    let f = int_filter(1, true, ORDER_LE, vals.len() as u32);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    assert_eq!(dec, raw);
}

/// `nelmts == 1` exercises the streaming min/max loops in
/// `precompress_{integer,float}` with an empty body (the `1..nelmts`
/// range yields nothing). Span is 0, minbits is 0  -  round-trips both
/// integer and float through the all-equal short-circuit.
#[test]
fn single_element_chunk_roundtrip() {
    let raw = 42u32.to_le_bytes().to_vec();
    let f = int_filter(4, false, ORDER_LE, 1);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    assert_eq!(dec, raw);

    let raw = 3.14f64.to_le_bytes().to_vec();
    let f = float_filter(8, 3, ORDER_LE, 1);
    let comp = compress(&raw, &f).unwrap();
    let dec = decompress(&comp, &f, None).unwrap();
    let got = f64::from_le_bytes(dec.as_slice().try_into().unwrap());
    assert!((got - 3.14).abs() <= 0.5e-3);
}

#[test]
fn an_integer_chunk_round_trips_and_a_zero_element_size_is_rejected() {
    let scalar = ScaleOffsetType::integer(false, ScaleOffsetByteOrder::LittleEndian);
    let cd = build_cd_values(
        ScaleOffset::Integer(0),
        scalar,
        1,
        3,
        ScaleOffsetFill::Undefined,
    )
    .unwrap();
    assert_eq!(
        cd,
        [2, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        scale_offset_mode(&cd),
        Some((ScaleOffset::Integer(0), FillAvailability::Undefined))
    );

    let encoded = compress(&[4, 5, 6], &cd).unwrap();
    let mut expected = vec![2, 0, 0, 0, 8];
    expected.extend_from_slice(&4u64.to_le_bytes());
    expected.extend_from_slice(&[0; 8]);
    expected.push(0x18);
    assert_eq!(encoded, expected);
    assert_eq!(decompress(&encoded, &cd, Some(3)).unwrap(), [4, 5, 6]);

    assert_eq!(
        build_cd_values(
            ScaleOffset::Integer(0),
            scalar,
            0,
            3,
            ScaleOffsetFill::Undefined,
        )
        .unwrap_err(),
        Error::ScaleOffset("scaleoffset: unsupported datatype size 0".into())
    );
}
