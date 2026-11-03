use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::MessageFilter;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountObserver;
use hdf5_pure_format::__private::ObjectReferenceCountUpdateError;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::v2;

#[rstest]
#[case::matching(false)]
#[case::other_chunk(true)]
fn count_update_capability_uses_checked_physical_regions(#[case] other_chunk: bool) {
    let bytes = v2::Header::new()
        .message(Message::reference_count(2))
        .build();
    let observer = ObjectReferenceCountObserver::new(PhysicalAddress::new(0));
    let mut error = None;
    let header = ObjectHeader::parse_from_source_with_chunk_observer(
        bytes.as_slice(),
        AccessMode::ReadWrite,
        PhysicalAddress::new(0),
        FormatWidths::from_sizes(8, 8).unwrap(),
        BaseAddress::ZERO,
        MessageFilter::All,
        &mut |chunk| {
            if error.is_none() {
                error = observer.observe(chunk).err();
            }
        },
    )
    .unwrap();
    if let Some(error) = error {
        panic!("unexpected count observation error: {error:?}");
    }
    let ObjectReferenceCount::Version2(field) = observer.finish(&header).unwrap() else {
        panic!("expected a version 2 count")
    };
    let provided = FileRange::new(
        PhysicalAddress::new(u64::from(other_chunk)),
        u64::try_from(bytes.len()).unwrap(),
    )
    .unwrap();
    let result = field.update(FileRegion::new(provided, &bytes).unwrap(), 1);
    if other_chunk {
        let error = result.unwrap_err();
        let ObjectReferenceCountUpdateError::Chunk {
            expected,
            provided: actual,
        } = error
        else {
            panic!("expected a mismatched chunk, got {error:?}")
        };
        assert_eq!(expected, field.chunk());
        assert_eq!(actual, provided);
    } else {
        let update = result.unwrap();
        assert_eq!(
            update.writes().next().unwrap(),
            (field.range(), [1, 0, 0, 0].as_slice())
        );
        assert_eq!(
            update.writes().map(|(_, bytes)| bytes.len()).sum::<usize>(),
            8
        );
    }
}
