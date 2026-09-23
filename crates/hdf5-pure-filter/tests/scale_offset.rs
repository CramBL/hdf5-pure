use hdf5_pure_filter::Error;
use hdf5_pure_filter::ScaleOffset;
use hdf5_pure_filter::ScaleOffsetByteOrder;
use hdf5_pure_filter::ScaleOffsetFill;
use hdf5_pure_filter::ScaleOffsetType;

#[test]
fn public_scale_offset_round_trip_and_zero_width_error() {
    let scalar = ScaleOffsetType::integer(false, ScaleOffsetByteOrder::LittleEndian);
    let cd = hdf5_pure_filter::build_scale_offset_cd_values(
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
        hdf5_pure_filter::scale_offset_mode(&cd),
        Some((
            ScaleOffset::Integer(0),
            hdf5_pure_filter::FillAvailability::Undefined
        ))
    );

    let encoded = hdf5_pure_filter::compress_scale_offset(&[4, 5, 6], &cd).unwrap();
    let mut expected = vec![2, 0, 0, 0, 8];
    expected.extend_from_slice(&4u64.to_le_bytes());
    expected.extend_from_slice(&[0; 8]);
    expected.push(0x18);
    assert_eq!(encoded, expected);
    assert_eq!(
        hdf5_pure_filter::decompress_scale_offset(&encoded, &cd, Some(3)).unwrap(),
        [4, 5, 6]
    );

    assert_eq!(
        hdf5_pure_filter::build_scale_offset_cd_values(
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
