use core::cell::Cell;

use hdf5_pure_format::__private::FractalHeapHeaderLocator;
use rstest::rstest;
use test_util::btree_v2;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap;
use crate::fractal_heap::HeapSource;
use crate::group_v2;
use crate::link_info::LinkInfoMessage;
use crate::link_message::LinkTarget;
use crate::object_header::ObjectHeader;
use crate::object_path::LinkName;
use crate::reader::OpenedSource;
use crate::source::BytesSource;
use crate::source::MetadataCacheConfig;
use crate::source::MetadataCachingSource;
use crate::source::PhysicalBytes;
use crate::source::PhysicalSource;
use crate::source::Source;
use crate::width::FormatWidths;

#[rstest]
#[case::indexed("dataset_015", false)]
#[case::lossy_name("\u{fffd}ataset_015", true)]
fn dense_group_lookup_and_listing_use_the_captured_frame(
    #[case] name: &str,
    #[case] lossy_name: bool,
    #[values(0, 512)] base: u64,
    #[values(false, true)] resident: bool,
) {
    let (mut image, header, info, widths) = dense_image();
    if lossy_name {
        let name_at = image
            .windows(b"dataset_015".len())
            .position(|bytes| bytes == b"dataset_015")
            .unwrap();
        image[name_at] = 0xff;
    }
    let heap_address = info.fractal_heap_address.unwrap();
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let mut physical = vec![0; base.to_usize().unwrap()];
    physical.extend_from_slice(&image);
    let source = BytesSource::new(&physical);
    let opened = OpenedSource::for_test(&source, addresses);
    let entries = if resident {
        group_v2::resolve_group_entries(
            PhysicalBytes::for_test(&physical, addresses),
            AccessMode::ReadOnly,
            &header,
            widths,
            None,
        )
    } else {
        group_v2::resolve_group_entries_from_source(
            &opened,
            AccessMode::ReadOnly,
            &header,
            widths,
            None,
        )
    }
    .unwrap();
    let mut names: Vec<_> = entries.iter().map(|entry| entry.name.clone()).collect();
    names.sort_unstable();
    let mut expected: Vec<_> = (0..20)
        .map(|index| {
            if index == 15 && lossy_name {
                String::from(name)
            } else {
                format!("dataset_{index:03}")
            }
        })
        .collect();
    expected.sort_unstable();
    assert_eq!(names, expected);
    let target = if resident {
        group_v2::find_noncompact_link(
            PhysicalBytes::for_test(&physical, addresses),
            AccessMode::ReadOnly,
            &header,
            widths,
            None,
            LinkName::new(name).unwrap(),
        )
    } else {
        group_v2::find_noncompact_link_from_source(
            &opened,
            AccessMode::ReadOnly,
            &header,
            widths,
            None,
            LinkName::new(name).unwrap(),
        )
    }
    .unwrap();
    assert_eq!(
        target,
        Some(LinkTarget::Hard {
            object_header_address: entries
                .iter()
                .find(|entry| entry.name == name)
                .unwrap()
                .object_header_address,
        })
    );
    let stored = PhysicalSource::new(&opened).stored();
    let heap = HeapSource::new(&stored);
    let (_, range) =
        fractal_heap::read_header_and_range_from_source(&heap, heap_address, widths).unwrap();
    assert_eq!(range.start(), addresses.resolve(heap_address).unwrap());
}

#[rstest]
#[case::invalid_heap(true, false, FormatError::InvalidFractalHeapSignature)]
#[case::heap_before_missing_index(true, true, FormatError::InvalidFractalHeapSignature)]
#[case::missing_index(
    false,
    true,
    FormatError::PathNotFound(String::from("no B-tree v2 name index"))
)]
#[case::invalid_index(false, false, FormatError::InvalidBTreeV2Signature)]
fn dense_group_walks_preserve_storage_error_precedence(
    #[case] invalid_heap: bool,
    #[case] missing_index: bool,
    #[case] expected: FormatError,
    #[values(0, 512)] base: u64,
    #[values(false, true)] resident: bool,
) {
    let (mut image, _, mut info, widths) = dense_image();
    let heap_address = info.fractal_heap_address.unwrap();
    image[if invalid_heap {
        heap_address.get().to_usize().unwrap()
    } else {
        info.btree_name_index_address
            .unwrap()
            .get()
            .to_usize()
            .unwrap()
    }] = 0;
    if missing_index {
        info.btree_name_index_address = None;
    }
    let mut physical = vec![0; base.to_usize().unwrap()];
    physical.extend_from_slice(&image);
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let source = BytesSource::new(&physical);
    let opened = OpenedSource::for_test(&source, addresses);
    let stored = PhysicalSource::new(&opened).stored();
    let error = if resident {
        group_v2::resolve_dense_entries(
            PhysicalBytes::for_test(&physical, addresses)
                .stored()
                .unwrap(),
            AccessMode::ReadOnly,
            &info,
            heap_address,
            widths,
        )
    } else {
        group_v2::resolve_dense_entries_from_source(
            &stored,
            AccessMode::ReadOnly,
            &info,
            heap_address,
            widths,
        )
    }
    .unwrap_err();
    assert_eq!(error, expected);
}

#[rstest]
#[case::at_eof(0)]
#[case::past_eof(1)]
fn dense_group_heap_bounds_preserve_reader_errors(
    #[case] after_eof: u64,
    #[values(0, 512)] base: u64,
    #[values(false, true)] resident: bool,
) {
    let (image, _, mut info, widths) = dense_image();
    let image_len = image.len().to_u64();
    let heap_address = StoredAddress::new(image_len + after_eof);
    info.fractal_heap_address = Some(heap_address);
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let mut physical = vec![0; base.to_usize().unwrap()];
    physical.extend_from_slice(&image);
    let source = BytesSource::new(&physical);
    let opened = OpenedSource::for_test(&source, addresses);
    let stored = PhysicalSource::new(&opened).stored();
    let prefix = FractalHeapHeaderLocator::new(
        addresses.resolve(heap_address).unwrap(),
        widths.offsets,
        widths.lengths,
    )
    .parse_range()
    .unwrap();
    let (error, expected) = if resident {
        (
            group_v2::resolve_dense_entries(
                PhysicalBytes::for_test(&physical, addresses)
                    .stored()
                    .unwrap(),
                AccessMode::ReadOnly,
                &info,
                heap_address,
                widths,
            ),
            FormatError::InsufficientRegion {
                required_start: prefix.start().get(),
                required_length: prefix.length(),
                provided_start: base,
                provided_length: image_len,
            },
        )
    } else {
        (
            group_v2::resolve_dense_entries_from_source(
                &stored,
                AccessMode::ReadOnly,
                &info,
                heap_address,
                widths,
            ),
            FormatError::UnexpectedSourceEof {
                expected: prefix.end().get(),
                available: physical.len().to_u64(),
            },
        )
    };
    assert_eq!(error.unwrap_err(), expected);
}

#[rstest]
#[case::borrowed(false)]
#[case::owned(true)]
fn dense_group_reads_preserve_metadata_ownership_and_cache_admission(
    #[case] owned: bool,
    #[values(0, 512)] base: u64,
) {
    let (image, _, info, widths) = dense_image();
    let mut physical = vec![0; base.to_usize().unwrap()];
    physical.extend_from_slice(&image);
    let borrowed = BytesSource::new(&physical);
    let allocating = CountingSource {
        inner: BytesSource::new(physical.clone()),
        reads: Cell::new(0),
        origin: 37,
    };
    let source: &dyn Source = if owned { &allocating } else { &borrowed };
    let cached = MetadataCachingSource::new(source, MetadataCacheConfig::new(8192));
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let opened = OpenedSource::for_test(&cached, addresses);
    let stored = PhysicalSource::new(&opened).stored();
    let first = group_v2::resolve_dense_entries_from_source(
        &stored,
        AccessMode::ReadOnly,
        &info,
        info.fractal_heap_address.unwrap(),
        widths,
    )
    .unwrap();
    let before = cached.metadata_cache_stats().unwrap();
    let reads = allocating.reads.get();
    let second = group_v2::resolve_dense_entries_from_source(
        &stored,
        AccessMode::ReadOnly,
        &info,
        info.fractal_heap_address.unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(
        first
            .iter()
            .map(|entry| (&entry.name, entry.object_header_address))
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|entry| (&entry.name, entry.object_header_address))
            .collect::<Vec<_>>()
    );
    assert_eq!(first.len(), 20);
    let after = cached.metadata_cache_stats().unwrap();
    if owned {
        assert!(before.entries() > 0);
        assert!(after.hits() > before.hits());
        assert_eq!(after.misses(), before.misses());
        assert_eq!(allocating.reads.get(), reads);
    } else {
        assert_eq!((after.hits(), after.entries(), after.bytes()), (0, 0, 0));
        assert_eq!(after.misses(), 2 * before.misses());
    }
    let address = info.btree_name_index_address.unwrap();
    let metadata = stored
        .read_metadata(address, btree_v2::SIGNATURE.len())
        .unwrap();
    assert_eq!(metadata.as_ref(), btree_v2::SIGNATURE);
    assert_eq!(
        metadata.offset(),
        base + address.get() + if owned { 37 } else { 0 }
    );
    assert_eq!(metadata.shared().is_some(), owned);
    if !owned {
        assert_eq!(
            metadata.as_ptr(),
            physical[(base + address.get()).to_usize().unwrap()..].as_ptr()
        );
    }
    let retained = metadata.shared();
    drop(metadata);
    drop(cached);
    if let Some(retained) = retained {
        assert_eq!(retained.as_ref(), btree_v2::SIGNATURE);
        assert_eq!(retained.offset(), base + address.get() + 37);
    }
}

fn dense_image() -> (Vec<u8>, ObjectHeader, LinkInfoMessage, FormatWidths) {
    let image = test_util::modern::v2_many_links();
    let superblock = crate::superblock::read(image.as_slice())
        .unwrap()
        .superblock;
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let header = ObjectHeader::parse_with_widths(
        &image,
        AccessMode::ReadOnly,
        superblock.root_group_address.get().to_usize().unwrap(),
        widths,
    )
    .unwrap();
    let info = group_v2::find_link_info(&header, widths.offsets).unwrap();
    assert!(info.fractal_heap_address.is_some());
    assert!(info.btree_name_index_address.is_some());
    (image, header, info, widths)
}

struct CountingSource {
    inner: BytesSource<Vec<u8>>,
    reads: Cell<u64>,
    origin: u64,
}

impl Source for CountingSource {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn metadata_origin(&self) -> Result<u64, FormatError> {
        Ok(self.origin)
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.reads.set(self.reads.get() + 1);
        self.inner.read_at(offset, buffer)
    }
}
