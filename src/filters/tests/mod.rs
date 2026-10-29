use super::*;
use crate::Datatype;
use crate::datatype::byte_order::DatatypeByteOrder;

#[cfg(feature = "std")]
mod lzf_reference;
mod zfp_reference;

#[test]
fn a_context_cannot_be_built_from_a_zero_width_datatype() {
    let degenerate = Datatype::Array {
        base_type: Box::new(Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
        )),
        dimensions: vec![0],
    };
    assert_eq!(
        ChunkContext::from_datatype(&[4], &degenerate).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: 10 }
    );

    let ordinary = crate::datatype::Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    let ctx = ChunkContext::from_datatype(&[4], &ordinary).unwrap();
    assert_eq!(ctx.element_size.get(), 4);
}

#[rstest::rstest]
#[case::deflate(hdf5_pure_filter::__private::Error::DeflateSizeExceeded { limit: 16, required: 17 }, "deflate: output exceeds expected chunk size of 16 bytes (possible decompression bomb)")]
#[case::lzf(hdf5_pure_filter::__private::Error::LzfSizeExceeded { limit: 16, required: 17 }, "lzf: output exceeds expected chunk size")]
fn output_size_limits_preserve_legacy_filter_errors(
    #[case] error: hdf5_pure_filter::__private::Error,
    #[case] expected: &str,
) {
    assert_eq!(
        FormatError::from(error),
        FormatError::FilterError(expected.into())
    );
}

#[test]
fn a_failed_filter_decode_preserves_the_public_error() {
    let lzf = hdf5_pure_filter::__private::decompress_lzf(&[0x1f], None).unwrap_err();
    let err = FormatError::from(lzf);
    let FormatError::FilterError(reason) = err else {
        panic!("expected FilterError, got {err:?}");
    };
    assert_eq!(reason, "lzf: truncated literal run");
}
