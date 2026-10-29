use hdf5_pure_format::__private::V3_FLAGS_DEFAULT;

use super::super::*;
use crate::convert;
use hdf5_pure_format::__private;

#[test]
fn an_unreadable_message_is_unknown_not_zero() {
    let elem = NonZeroUsize::new(4).unwrap();
    // A v3 message with the *defined* bit set and a 4-byte value.
    let mut defined = vec![3u8, V3_FLAGS_DEFAULT | 0x20, 4, 0, 0, 0];
    defined.extend_from_slice(&[7u8, 0, 0, 0]);
    assert!(matches!(
        PaddingFill::from_message(MessageType::FILL_VALUE, &defined),
        PaddingFill::Value(ref b) if b == &[7u8, 0, 0, 0]
    ));

    // The default message uses zero fill.
    let default = vec![3u8, V3_FLAGS_DEFAULT];
    assert!(matches!(
        PaddingFill::from_message(MessageType::FILL_VALUE, &default),
        PaddingFill::Zero
    ));

    // Version 9 is unsupported.
    let mut unknown_version = defined.clone();
    unknown_version[0] = 9;
    assert!(matches!(
        PaddingFill::from_message(MessageType::FILL_VALUE, &unknown_version),
        PaddingFill::Unknown
    ));
    assert!(
        PaddingFill::from_message(MessageType::FILL_VALUE, &unknown_version)
            .pattern(elem)
            .apply(&mut [0u8; 4])
            .is_err()
    );

    // Truncated before the fields it declares: likewise undetermined.
    assert!(matches!(
        PaddingFill::from_message(MessageType::FILL_VALUE, &defined[..3]),
        PaddingFill::Unknown
    ));
}

/// The pattern tiles one element's bytes across a buffer of whole elements.
#[test]
fn a_defined_fill_tiles_across_the_buffer() {
    let seven = 7.0f64.to_le_bytes();
    let p = FillPattern::new(FillTime::IfSet, Some(&seven), convert::nz(8));
    assert_eq!(p.buffer(24).unwrap(), seven.repeat(3));
    // A pattern with interior zero bytes must not be mistaken for the zero
    // pattern: only an *all*-zero value is.
    assert!(seven.iter().filter(|&&b| b == 0).count() >= 6);
}

#[test]
fn a_fill_value_of_the_wrong_width_is_dropped() {
    for bytes in [vec![1u8], vec![1, 2, 3], vec![1; 9], Vec::new()] {
        let p = FillPattern::new(FillTime::IfSet, Some(&bytes), convert::nz(8));
        assert_eq!(
            p.buffer(16).unwrap(),
            vec![0u8; 16],
            "a {}-byte fill over an 8-byte element must not tile",
            bytes.len()
        );
    }
}

/// An all-zero fill value *is* the zero pattern, and says so, so every
/// buffer that would tile zeros stays on the plain allocation path.
#[test]
fn an_all_zero_fill_is_the_zero_pattern() {
    assert_eq!(
        FillPattern::new(FillTime::IfSet, Some(&[0u8; 8]), convert::nz(8))
            .buffer(16)
            .unwrap(),
        vec![0u8; 16]
    );
    assert_eq!(
        FillPattern::new(FillTime::IfSet, None, convert::nz(8))
            .buffer(16)
            .unwrap(),
        vec![0u8; 16]
    );
    assert_eq!(FillPattern::ZERO.buffer(16).unwrap(), vec![0u8; 16]);
}

/// A single-byte element is the degenerate tiling case, and a zero-length
/// buffer must not panic.
#[test]
fn single_byte_elements_and_empty_buffers() {
    let p = FillPattern::new(FillTime::IfSet, Some(&[0xAB]), convert::nz(1));
    assert_eq!(p.buffer(5).unwrap(), vec![0xAB; 5]);
    assert_eq!(p.buffer(0).unwrap(), Vec::<u8>::new());
}

#[test]
fn fill_time_never_reads_as_zeros_despite_defined_fill() {
    let seven = 7.0f64.to_le_bytes();
    let p = FillPattern::new(FillTime::Never, Some(&seven), convert::nz(8));
    assert_eq!(p.buffer(24).unwrap(), vec![0u8; 24]);
    assert_eq!(p.element().unwrap(), None);
}

#[test]
fn fill_time_alloc_tiles_defined_fill() {
    let seven = 7.0f64.to_le_bytes();
    let p = FillPattern::new(FillTime::Alloc, Some(&seven), convert::nz(8));
    assert_eq!(p.buffer(24).unwrap(), seven.repeat(3));
    assert_eq!(p.element().unwrap(), Some(&seven[..]));
}

#[test]
fn an_unknown_fill_refuses_to_materialize() {
    assert!(matches!(
        FillPattern::UNKNOWN.buffer(16),
        Err(FormatError::UnreadableFillValue)
    ));
    // An unknown pattern also rejects an empty materialization request.
    assert!(FillPattern::UNKNOWN.buffer(0).is_err());
    assert!(FillPattern::ZERO.buffer(0).is_ok());
}

#[test]
fn the_write_time_is_read_from_every_message_version() {
    // Version 3: bits 2-3 of the flags byte. 0x26 is Late alloc, Never, and Defined,
    // and 0x2a the same with IfSet. libhdf5 writes Defined for
    // `H5D_FILL_TIME_NEVER` whenever a fill value is set (`H5Ofill.c`, HDF5 2.1.0).
    let v3 = |flags: u8| vec![3u8, flags, 4, 0, 0, 0, 7, 0, 0, 0];
    assert!(!__private::fill_value_is_written(MessageType::FILL_VALUE, &v3(0x26)).unwrap());
    assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &v3(0x2a)).unwrap());
    assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &v3(0x22)).unwrap());

    // Versions 1 and 2: a byte of its own at index 2.
    for version in [1u8, 2] {
        let msg = |write_time: u8| vec![version, 2, write_time, 1, 4, 0, 0, 0, 7, 0, 0, 0];
        assert!(!__private::fill_value_is_written(MessageType::FILL_VALUE, &msg(1)).unwrap());
        assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &msg(0)).unwrap());
        assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &msg(2)).unwrap());
    }

    // The legacy message has no write time to read.
    assert!(
        __private::fill_value_is_written(MessageType::FILL_VALUE_OLD, &[4, 0, 0, 0, 7, 0, 0, 0],)
            .unwrap()
    );

    assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &[]).is_err());
    assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &[3]).is_err());
    assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &[2, 0]).is_err());
    assert!(__private::fill_value_is_written(MessageType::FILL_VALUE, &[9, 0]).is_err());
}
