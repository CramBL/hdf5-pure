use alloc::vec;
use core::cell::Cell;

use hdf5_pure_core::__private::MetadataFrame;

use rstest::rstest;

use super::*;
use crate::address::{BaseAddress, BaseAddressExt, OpenedImageContext, StoredAddressExt};
use crate::source::{BaseOffsetSource, BytesSource};
#[cfg(feature = "std")]
use crate::source::{MetadataCacheConfig, MetadataCachingSource, ReadSeekSource};

#[rstest]
#[case::zero_base(0, 0)]
#[case::userblock(512, 512)]
#[case::forward_relocated(0, 512)]
#[case::backward_relocated(512, 0)]
fn stored_reads_apply_the_effective_base_once(
    #[case] encoded_base: u64,
    #[case] effective_base: u64,
) {
    let mut data = vec![0; 1024];
    data[effective_base as usize + 4..effective_base as usize + 7].copy_from_slice(&[9, 8, 7]);
    let source = BytesSource::new(&data);
    let image = OpenedImageContext::reconcile(BaseAddress::new(encoded_base), 1024, effective_base)
        .unwrap();
    let stored = PhysicalSource::for_test(&source, image.addresses()).stored();
    assert_eq!(stored.len(), 1024 - effective_base);
    assert_eq!(
        StoredRead::metadata_origin(&stored).unwrap(),
        PhysicalAddress::new(effective_base)
    );
    let address = StoredAddress::new(4);
    let mut buffer = [0; 3];
    stored.read_at(address, &mut buffer).unwrap();
    assert_eq!(buffer, [9, 8, 7]);
    assert_eq!(stored.read_exact_at(address, 3).unwrap(), [9, 8, 7]);
    assert_eq!(stored.read_metadata_at(address, 3).unwrap(), [9, 8, 7]);
    let bytes = stored.read_metadata(address, 3).unwrap();
    assert_eq!(bytes.as_ref(), &[9, 8, 7]);
    assert_eq!(bytes.offset(), effective_base + 4);
    assert_eq!(bytes.as_ptr(), data[effective_base as usize + 4..].as_ptr());
    assert!(bytes.shared().is_none());
    let range = AddressRange::new(address, 3).unwrap();
    let bytes = stored.read_range(range).unwrap();
    assert_eq!(bytes.offset(), effective_base + 4);
    assert_eq!(bytes.as_ref(), &[9, 8, 7]);
    assert_eq!(bytes.as_ptr(), data[effective_base as usize + 4..].as_ptr());
}

#[cfg(feature = "std")]
#[rstest]
#[case::zero_base(0)]
#[case::userblock(512)]
fn streaming_metadata_keeps_its_owner_and_cache(#[case] base: u64) {
    let data: Vec<u8> = (0..1024).map(|index| (index % 251) as u8).collect();
    let source = MetadataCachingSource::new(
        ReadSeekSource::new(std::io::Cursor::new(&data)).unwrap(),
        MetadataCacheConfig::new(128),
    );
    let context = OpenedImageContext::reconcile(BaseAddress::new(base), 1024, base).unwrap();
    let stored = PhysicalSource::for_test(&source, context.addresses()).stored();
    let first = stored.read_metadata(StoredAddress::new(4), 3).unwrap();
    let second = stored.read_metadata(StoredAddress::new(4), 3).unwrap();
    assert_eq!(first.offset(), base + 4);
    assert_eq!(first.as_ref(), &data[base as usize + 4..base as usize + 7]);
    assert_eq!(first.as_ptr(), second.as_ptr());
    assert!(first.shared().is_some());
    let stats = stored.metadata_cache_stats().unwrap();
    assert_eq!(
        (stats.hits(), stats.misses(), stats.entries(), stats.bytes()),
        (1, 1, 1, 3)
    );
    assert_eq!(
        stored.read_exact_at(StoredAddress::new(4), 3).unwrap(),
        first.as_ref()
    );
    let mut buffer = [0; 3];
    stored.read_at(StoredAddress::new(4), &mut buffer).unwrap();
    assert_eq!(buffer, first.as_ref());
    assert_eq!(stored.metadata_cache_stats().unwrap(), stats);
    stored.reset_metadata_cache_stats();
    let stats = stored.metadata_cache_stats().unwrap();
    assert_eq!(
        (stats.hits(), stats.misses(), stats.entries(), stats.bytes()),
        (0, 0, 1, 3)
    );
}

struct OriginSource {
    origin: u64,
    length: u64,
    last_read: Cell<Option<(u64, usize)>>,
}

impl Source for OriginSource {
    fn len(&self) -> u64 {
        self.length
    }

    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.last_read.set(Some((offset, bytes.len())));
        bytes.fill(42);
        Ok(())
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        Ok(self.origin)
    }
}

#[rstest]
#[case::nonzero_origin(37, 512, 4)]
#[case::high_address(37, 512, u64::from(u32::MAX) + 4)]
fn metadata_retains_the_original_source_origin(
    #[case] origin: u64,
    #[case] base: u64,
    #[case] offset: u64,
) {
    let source = OriginSource {
        origin,
        length: u64::MAX,
        last_read: Cell::new(None),
    };
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let stored = PhysicalSource::for_test(&source, addresses).stored();
    let metadata = stored.read_metadata(StoredAddress::new(offset), 3).unwrap();
    assert_eq!(metadata.as_ref(), &[42; 3]);
    assert_eq!(metadata.offset(), origin + base + offset);
    assert_eq!(source.last_read.get(), Some((base + offset, 3)));
    assert_eq!(
        StoredRead::metadata_origin(&stored).unwrap(),
        PhysicalAddress::new(origin + base)
    );
    assert_eq!(
        stored.read_exact_at(StoredAddress::new(offset), 3).unwrap(),
        [42; 3]
    );
    assert_eq!(source.last_read.get(), Some((base + offset, 3)));
}

#[rstest]
#[case::empty_at_end(4, 0, None)]
#[case::out_of_bounds(3, 2, Some(FormatError::UnexpectedSourceEof { expected: 5, available: 4 }))]
#[case::address_overflow(u64::MAX, 1, Some(FormatError::OffsetOverflow { offset: u64::MAX, length: 1 }))]
fn exact_reads_check_the_stored_extent(
    #[case] offset: u64,
    #[case] length: usize,
    #[case] error: Option<FormatError>,
) {
    let source = BytesSource::new([0; 8]);
    let addresses = AddressContext::from_effective_base(BaseAddress::new(4));
    let stored = PhysicalSource::for_test(&source, addresses).stored();
    let result = stored.read_exact_at(StoredAddress::new(offset), length);
    if let Some(error) = error {
        assert_eq!(result.unwrap_err(), error);
    } else {
        assert_eq!(result.unwrap(), Vec::<u8>::new());
    }
}

#[rstest]
#[case::resolved_start(u64::MAX, 0, 1, 0, FormatError::OffsetOverflow { offset: u64::MAX, length: 1 })]
#[case::resolved_end(u64::MAX - 1, 0, 1, 1, FormatError::OffsetOverflow { offset: u64::MAX, length: 1 })]
#[case::metadata_origin(0, u64::MAX, 1, 1, FormatError::OffsetOverflow { offset: u64::MAX, length: 1 })]
fn metadata_reads_reject_coordinate_overflow(
    #[case] offset: u64,
    #[case] origin: u64,
    #[case] base: u64,
    #[case] length: u64,
    #[case] error: FormatError,
) {
    let source = OriginSource {
        origin,
        length: u64::MAX,
        last_read: Cell::new(None),
    };
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let stored = PhysicalSource::for_test(&source, addresses).stored();
    let range = AddressRange::new(StoredAddress::new(offset), length).unwrap();
    assert_eq!(stored.read_range(range).unwrap_err(), error);
}

#[test]
fn physical_reads_use_source_positions() {
    let source = BytesSource::new([1, 2, 3]);
    let physical = PhysicalSource::for_test(
        &source,
        AddressContext::from_effective_base(BaseAddress::ZERO),
    );
    let mut buffer = [0; 2];
    physical
        .read_at(PhysicalAddress::new(1), &mut buffer)
        .unwrap();
    assert_eq!(buffer, [2, 3]);
    assert_eq!(
        physical.read_exact_at(PhysicalAddress::new(1), 2).unwrap(),
        [2, 3]
    );
    let bytes = physical
        .read_range(FileRange::new(PhysicalAddress::new(1), 2).unwrap())
        .unwrap();
    assert_eq!(bytes.as_ref(), &[2, 3]);
    assert_eq!(bytes.offset(), 1);
    assert_eq!(bytes.as_ptr(), source.0[1..].as_ptr());
}

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn stored_metadata_borrows_without_copying_or_cache_admission() {
    let data = vec![17; 1_048_576 + 512];
    let source = MetadataCachingSource::new(
        BytesSource::new(&data),
        MetadataCacheConfig::new(data.len()).with_max_entry_bytes(data.len()),
    );
    let addresses = AddressContext::from_effective_base(BaseAddress::new(512));
    let stored = PhysicalSource::for_test(&source, addresses).stored();
    assert_eq!(
        stored.metadata_cache_stats().unwrap(),
        MetadataCacheStats::default()
    );
    let range = AddressRange::new(StoredAddress::new(0), 1_048_576).unwrap();
    let expected = &data[512..];
    let (copy, control) = test_util::allocation::measure("stored_source_metadata_copy", || {
        std::hint::black_box(expected.to_vec())
    });
    assert_eq!(copy, expected);
    assert!(control.blocks > 0, "{control}");
    assert!(control.bytes >= range.length(), "{control}");
    drop(copy);
    let ((first, second), measured) =
        test_util::allocation::measure("stored_source_metadata", || {
            (
                stored.read_range(range).unwrap(),
                stored.read_range(range).unwrap(),
            )
        });
    assert_eq!(first.as_ref(), expected);
    assert_eq!(second.as_ref(), expected);
    assert_eq!(first.as_ptr(), expected.as_ptr());
    assert_eq!(second.as_ptr(), expected.as_ptr());
    assert_eq!(first.offset(), 512);
    assert_eq!(second.offset(), 512);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(measured.peak_bytes, 0, "{measured}");
    assert_eq!(measured.live_bytes, 0, "{measured}");
    let stats = stored.metadata_cache_stats().unwrap();
    assert_eq!(
        (stats.hits(), stats.misses(), stats.entries(), stats.bytes()),
        (0, 2, 0, 0)
    );
}

#[test]
fn typed_views_do_not_implement_the_unframed_source_interface() {
    trait AmbiguousIfSource<A> {
        fn marker() {}
    }
    struct ImplementsSource;
    impl<T: ?Sized> AmbiguousIfSource<()> for T {}
    impl<T: Source + ?Sized> AmbiguousIfSource<ImplementsSource> for T {}

    let _ = <StoredSource<'_, BytesSource<[u8; 0]>> as AmbiguousIfSource<_>>::marker;
    let _ = <PhysicalSource<'_, BytesSource<[u8; 0]>> as AmbiguousIfSource<_>>::marker;
}

#[rstest]
#[case::buffer(false)]
#[case::metadata(true)]
fn stored_reads_report_the_backends_physical_bounds(#[case] metadata: bool) {
    let source = BytesSource::new([0; 8]);
    let addresses = AddressContext::from_effective_base(BaseAddress::new(4));
    let stored = PhysicalSource::for_test(&source, addresses).stored();
    let error = if metadata {
        stored.read_metadata(StoredAddress::new(3), 2).unwrap_err()
    } else {
        stored
            .read_at(StoredAddress::new(3), &mut [0; 2])
            .unwrap_err()
    };
    assert_eq!(
        error,
        FormatError::UnexpectedSourceEof {
            expected: 9,
            available: 8
        }
    );
}

enum MetadataOverflowSource {
    Default,
    Frame,
}

#[rstest]
#[case::start_conversion(
    MetadataOverflowSource::Default,
    u64::MAX,
    1,
    FormatError::OffsetOverflow { offset: u64::MAX, length: 512 },
)]
#[case::physical_endpoint(
    MetadataOverflowSource::Default,
    u64::MAX - 512,
    513,
    FormatError::OffsetOverflow { offset: u64::MAX, length: 513 },
)]
#[case::metadata_origin(
    MetadataOverflowSource::Frame,
    u64::MAX - 512,
    1,
    FormatError::OffsetOverflow { offset: 1, length: u64::MAX },
)]
fn metadata_reads_preserve_backend_overflow_errors(
    #[case] backend: MetadataOverflowSource,
    #[case] offset: u64,
    #[case] length: usize,
    #[case] expected: FormatError,
) {
    let default = OriginSource {
        origin: 0,
        length: u64::MAX,
        last_read: Cell::new(None),
    };
    let frame = MetadataFrame::new(PhysicalAddress::new(1), &[0]).unwrap();
    let source: &dyn Source = match backend {
        MetadataOverflowSource::Default => &default,
        MetadataOverflowSource::Frame => &frame,
    };
    let base = BaseAddress::new(512);
    let stored =
        PhysicalSource::for_test(source, AddressContext::from_effective_base(base)).stored();
    assert_eq!(
        stored
            .read_metadata(StoredAddress::new(offset), length)
            .unwrap_err(),
        expected,
    );
    let legacy = BaseOffsetSource {
        inner: source,
        base,
    };
    assert_eq!(legacy.read_metadata(offset, length).unwrap_err(), expected);
    if let Ok(range) = AddressRange::new(StoredAddress::new(offset), length.to_u64()) {
        assert_eq!(stored.read_range(range).unwrap_err(), expected);
    }
    assert_eq!(default.last_read.get(), None);
}

#[test]
fn unframed_sources_do_not_implement_stored_range_reads() {
    trait AmbiguousIfStoredReader<A> {
        fn marker() {}
    }
    struct ImplementsStoredReader;
    impl<T: ?Sized> AmbiguousIfStoredReader<()> for T {}
    impl<T: RangeReader<AddressRange<StoredAddress>> + ?Sized>
        AmbiguousIfStoredReader<ImplementsStoredReader> for T
    {
    }

    let _ = <BytesSource<[u8; 0]> as AmbiguousIfStoredReader<_>>::marker;
    let _ = <PhysicalSource<'_, BytesSource<[u8; 0]>> as AmbiguousIfStoredReader<_>>::marker;
}

#[rstest]
#[case::zero_base(0, 0)]
#[case::userblock(512, 512)]
#[case::forward_relocated(0, 512)]
#[case::backward_relocated(512, 0)]
fn resident_views_apply_the_effective_base_and_borrow_the_original_bytes(
    #[case] encoded: u64,
    #[case] effective: u64,
) {
    let mut image = vec![0; usize::try_from(effective).unwrap()];
    image.extend(0..32u8);
    let context =
        OpenedImageContext::reconcile(BaseAddress::new(encoded), encoded + 32, effective).unwrap();
    let view = PhysicalBytes::for_test(&image, context.addresses())
        .stored()
        .unwrap();
    let mut buffer = [0; 3];
    view.read_at(StoredAddress::new(4), &mut buffer).unwrap();
    assert_eq!(buffer, [4, 5, 6]);
    assert_eq!(
        view.read_exact_at(StoredAddress::new(4), 3).unwrap(),
        [4, 5, 6]
    );
    assert_eq!(
        view.read_metadata_at(StoredAddress::new(4), 3).unwrap(),
        [4, 5, 6]
    );
    let range = AddressRange::new(StoredAddress::new(4), 3).unwrap();
    let borrowed = view.borrow_range(range).unwrap();
    let expected = &image[usize::try_from(effective + 4).unwrap()..];
    assert_eq!(borrowed, &[4, 5, 6]);
    assert_eq!(borrowed.as_ptr(), expected.as_ptr());
    let metadata = view.read_range(range).unwrap();
    assert_eq!(metadata.as_ref(), &[4, 5, 6]);
    assert_eq!(metadata.offset(), effective + 4);
    assert_eq!(metadata.as_ptr(), expected.as_ptr());
    assert!(metadata.shared().is_none());
    assert_eq!(view.len(), 32);
}

#[derive(Clone, Copy)]
enum ResidentRead {
    Buffer,
    Exact,
    Metadata,
    MetadataVector,
}

#[rstest]
#[case::empty_at_end(32, 0, &[])]
#[case::last_byte(31, 1, &[31])]
#[case::interior(4, 3, &[4, 5, 6])]
fn resident_borrowed_ranges_use_stored_slice_positions(
    #[case] start: u64,
    #[case] length: u64,
    #[case] expected: &[u8],
) {
    let mut image = vec![0; 512];
    image.extend(0..32u8);
    let view = PhysicalBytes::for_test(
        &image,
        AddressContext::from_effective_base(BaseAddress::new(512)),
    )
    .stored()
    .unwrap();
    let borrowed = view
        .borrow_range(AddressRange::new(StoredAddress::new(start), length).unwrap())
        .unwrap();
    assert_eq!(borrowed, expected);
    assert_eq!(
        borrowed.as_ptr(),
        image[512 + usize::try_from(start).unwrap()..].as_ptr()
    );
}

#[rstest]
#[case::crosses_end(31, 2, 33)]
#[case::empty_past_end(33, 0, 33)]
#[case::above_32bit(u64::from(u32::MAX) + 1, 0, u64::from(u32::MAX) + 1)]
#[case::maximum_address(u64::MAX, 0, u64::MAX)]
fn resident_borrowed_ranges_report_stored_bounds(
    #[case] start: u64,
    #[case] length: u64,
    #[case] expected: u64,
) {
    let image = [0; 544];
    let view = PhysicalBytes::for_test(
        &image,
        AddressContext::from_effective_base(BaseAddress::new(512)),
    )
    .stored()
    .unwrap();
    assert_eq!(
        view.borrow_range(AddressRange::new(StoredAddress::new(start), length).unwrap())
            .unwrap_err(),
        FormatError::UnexpectedSourceEof {
            expected,
            available: 32,
        }
    );
}

#[rstest]
#[case::past_end(
    31,
    2,
    FormatError::UnexpectedSourceEof { expected: 545, available: 544 },
    FormatError::UnexpectedSourceEof { expected: 33, available: 32 },
)]
#[case::start_overflow(
    u64::MAX,
    1,
    FormatError::OffsetOverflow { offset: u64::MAX, length: 512 },
    FormatError::OffsetOverflow { offset: u64::MAX, length: 1 },
)]
#[case::endpoint_overflow(
    u64::MAX - 512,
    513,
    FormatError::OffsetOverflow { offset: u64::MAX, length: 513 },
    FormatError::OffsetOverflow { offset: u64::MAX - 512, length: 513 },
)]
fn resident_read_errors_match_reads_of_the_original_source(
    #[case] start: u64,
    #[case] length: usize,
    #[case] expected: FormatError,
    #[case] exact_expected: FormatError,
    #[values(
        ResidentRead::Buffer,
        ResidentRead::Exact,
        ResidentRead::Metadata,
        ResidentRead::MetadataVector
    )]
    operation: ResidentRead,
) {
    let image = [0; 544];
    let context = AddressContext::from_effective_base(BaseAddress::new(512));
    let resident = PhysicalBytes::for_test(&image, context).stored().unwrap();
    let physical = BytesSource::new(image);
    let reference = PhysicalSource::for_test(&physical, context).stored();
    let address = StoredAddress::new(start);
    let read = |source: &dyn StoredRead| match operation {
        ResidentRead::Buffer => source.read_at(address, &mut vec![0; length]),
        ResidentRead::Exact => source.read_exact_at(address, length).map(drop),
        ResidentRead::Metadata => source.read_metadata(address, length).map(drop),
        ResidentRead::MetadataVector => source.read_metadata_at(address, length).map(drop),
    };
    let expected = match operation {
        ResidentRead::Exact => exact_expected,
        ResidentRead::Buffer | ResidentRead::Metadata | ResidentRead::MetadataVector => expected,
    };
    assert_eq!(read(&resident).unwrap_err(), expected);
    assert_eq!(read(&reference).unwrap_err(), expected);
}

#[test]
fn resident_view_rejects_a_base_beyond_its_source() {
    let context = AddressContext::from_effective_base(BaseAddress::new(4));
    assert_eq!(
        PhysicalBytes::for_test(&[0; 3], context)
            .stored()
            .unwrap_err(),
        FormatError::UnexpectedSourceEof {
            expected: 4,
            available: 3,
        }
    );
}

#[test]
fn only_stored_views_implement_typed_payload_reads() {
    trait AmbiguousIfStoredRead<A> {
        fn marker() {}
    }
    struct ImplementsStoredRead;
    impl<T: ?Sized> AmbiguousIfStoredRead<()> for T {}
    impl<T: StoredRead + ?Sized> AmbiguousIfStoredRead<ImplementsStoredRead> for T {}
    let _ = <BytesSource<[u8; 0]> as AmbiguousIfStoredRead<_>>::marker;
    let _ = <PhysicalSource<'_, BytesSource<[u8; 0]>> as AmbiguousIfStoredRead<_>>::marker;
}

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
#[test]
fn resident_range_reads_borrow_without_allocating() {
    let data = vec![17; 1_048_576 + 512];
    let view = PhysicalBytes::for_test(
        &data,
        AddressContext::from_effective_base(BaseAddress::new(512)),
    )
    .stored()
    .unwrap();
    let range = AddressRange::new(StoredAddress::new(0), 1_048_576).unwrap();
    let expected = &data[512..];
    let (copy, control) = test_util::allocation::measure("resident_range_copy", || {
        std::hint::black_box(expected.to_vec())
    });
    assert_eq!(copy, expected);
    assert!(control.blocks > 0, "{control}");
    assert!(control.bytes >= range.length(), "{control}");
    drop(copy);
    let ((borrowed, metadata), measured) = test_util::allocation::measure("resident_range", || {
        (
            view.borrow_range(range).unwrap(),
            view.read_range(range).unwrap(),
        )
    });
    assert_eq!(borrowed, expected);
    assert_eq!(metadata.as_ref(), expected);
    assert_eq!(borrowed.as_ptr(), expected.as_ptr());
    assert_eq!(metadata.as_ptr(), expected.as_ptr());
    assert_eq!(metadata.offset(), 512);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(measured.peak_bytes, 0, "{measured}");
    assert_eq!(measured.live_bytes, 0, "{measured}");
}

#[test]
fn raw_and_framed_readers_do_not_provide_original_backend_coordinates() {
    trait AmbiguousIfOriginalSource<A> {
        fn marker() {}
    }
    struct ImplementsOriginalSource;
    impl<T: ?Sized> AmbiguousIfOriginalSource<()> for T {}
    impl<T: OriginalSource + ?Sized> AmbiguousIfOriginalSource<ImplementsOriginalSource> for T {}
    let _ = <BytesSource<&[u8]> as AmbiguousIfOriginalSource<_>>::marker;
    let _ = <BaseOffsetSource<'_, BytesSource<&[u8]>> as AmbiguousIfOriginalSource<_>>::marker;
    let _ = <MetadataFrame<'_> as AmbiguousIfOriginalSource<_>>::marker;
    let _ = <StoredSource<'_, BytesSource<&[u8]>> as AmbiguousIfOriginalSource<_>>::marker;
    let _ = <StoredBytes<'_> as AmbiguousIfOriginalSource<_>>::marker;
    let _ = <PhysicalBytes<'_> as AmbiguousIfOriginalSource<_>>::marker;
    let _ = <PhysicalSource<'_, BytesSource<&[u8]>> as AmbiguousIfOriginalSource<_>>::marker;
    let _ = <&[u8] as AmbiguousIfOriginalSource<_>>::marker;
}
