use hdf5_pure_core::LengthWidth;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::allocation;
use test_util::global_heap;
use test_util::widths::Widths;

#[rstest]
#[case::empty(LengthWidth::Two, 0)]
#[case::narrow(LengthWidth::Two, 127)]
#[case::maximum_members(LengthWidth::Eight, usize::from(u16::MAX))]
fn global_heap_convenience_encoding_allocates_only_the_output(
    #[case] lengths: LengthWidth,
    #[case] count: usize,
) {
    let payloads = vec![b"alpha".as_slice(); count];
    let objects: Vec<_> = (1..=u16::MAX)
        .zip(&payloads)
        .map(|(index, payload)| global_heap::Object::new(index, payload))
        .collect();
    let expected = global_heap::collection(
        &objects,
        Widths::new(Widths::EIGHT.offset, usize::from(lengths.get())),
    );
    let (encoded, measured) = allocation::measure("global_heap_collection", || {
        format::encode_global_heap_collection(lengths, &payloads).unwrap()
    });
    assert_eq!(encoded, expected);
    assert!(measured.blocks >= 1, "{measured}");
    assert!(
        measured.bytes >= u64::try_from(encoded.len()).unwrap(),
        "{measured}"
    );
    assert!(measured.blocks <= 1, "{measured}");
    assert_eq!(
        measured.live_bytes,
        u64::try_from(encoded.capacity()).unwrap()
    );
}
