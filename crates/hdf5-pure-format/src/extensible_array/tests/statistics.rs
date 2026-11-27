use alloc::vec;

use rstest::rstest;

use super::*;

#[rstest::rstest]
#[case::two(LengthWidth::Two, u64::from(u16::MAX))]
#[case::four(LengthWidth::Four, u64::from(u32::MAX))]
#[case::eight(LengthWidth::Eight, u64::MAX)]
fn statistics_encode_exact_width_limits(#[case] lengths: LengthWidth, #[case] maximum: u64) {
    let stats = ExtensibleArrayStats {
        nsuper_blks: maximum,
        super_blk_size: maximum,
        ndata_blks: maximum,
        data_blk_size: maximum,
        max_idx_set: maximum,
        nelmts: maximum,
    };
    let mut output = [0xcc; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN];
    let used = stats.encode_fields(&mut output, lengths).unwrap();
    assert_eq!(used, usize::from(lengths.get()) * STATISTICS as usize);
    assert_eq!(&output[..used], vec![0xff; used]);
    assert_eq!(&output[used..], vec![0xcc; output.len() - used]);
}

#[rstest::rstest]
#[case::secondary_count(0)]
#[case::secondary_bytes(1)]
#[case::data_count(2)]
#[case::data_bytes(3)]
#[case::maximum_index(4)]
#[case::elements(5)]
fn statistics_reject_overflow_before_modifying_any_field(
    #[case] field: usize,
    #[values(LengthWidth::Two, LengthWidth::Four)] lengths: LengthWidth,
) {
    let value = match lengths {
        LengthWidth::Two => 65_536,
        LengthWidth::Four => 4_294_967_296,
        LengthWidth::Eight => panic!("eight-byte lengths have no larger u64 value"),
    };
    let mut fields = [0; STATISTICS as usize];
    fields[field] = value;
    let [
        nsuper_blks,
        super_blk_size,
        ndata_blks,
        data_blk_size,
        max_idx_set,
        nelmts,
    ] = fields;
    let stats = ExtensibleArrayStats {
        nsuper_blks,
        super_blk_size,
        ndata_blks,
        data_blk_size,
        max_idx_set,
        nelmts,
    };
    let mut output = [0xcc; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN];
    assert_eq!(
        stats.encode_fields(&mut output, lengths).unwrap_err(),
        FormatError::LengthTooLarge {
            length: value,
            length_size: lengths.get()
        }
    );
    assert_eq!(output, [0xcc; ExtensibleArrayStats::MAX_ENCODED_FIELDS_LEN]);
}

#[rstest]
#[case::two(LengthWidth::Two)]
#[case::four(LengthWidth::Four)]
#[case::eight(LengthWidth::Eight)]
fn statistics_reject_a_short_output_before_modifying_any_field(#[case] lengths: LengthWidth) {
    let stats = ExtensibleArrayStats {
        nsuper_blks: 1,
        super_blk_size: 2,
        ndata_blks: 3,
        data_blk_size: 4,
        max_idx_set: 5,
        nelmts: 6,
    };
    let size = usize::from(lengths.get()) * STATISTICS as usize;
    let mut output = vec![0xcc; size - 1];
    assert_eq!(
        stats.encode_fields(&mut output, lengths).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: size,
            available: size - 1
        }
    );
    assert_eq!(output, vec![0xcc; size - 1]);
}
