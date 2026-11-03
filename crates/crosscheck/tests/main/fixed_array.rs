#![cfg(feature = "__hdf5-1.10")]

use std::fs;
use std::ops::Range;
use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use hdf5_pure::PhysicalAddress;
use hdf5_pure::StoredAddress;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::BaseAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FixedArrayHeaderLocator;
use hdf5_pure_format::__private::FixedArrayPlan;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::LengthWidth;
use hdf5_pure_format::__private::OffsetWidth;
use rstest::rstest;

use test_util::bytes;
use test_util::checksum;
use test_util_hdf5::reader::ReadBackend;

use crate::test_util::array_index;
use crate::test_util::array_index::Writes;

#[derive(Clone, Copy, Debug)]
struct Fixture {
    len: usize,
    chunk: usize,
    deflate: bool,
    sparse: Option<&'static [Range<usize>]>,
}

impl Fixture {
    fn libhdf5_create(self, path: &Path, sizes: SizeofInfo, userblock: u64) {
        let Self {
            len,
            chunk,
            deflate,
            sparse: _,
        } = self;
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .with_fcpl(|p| p.sizes(sizes).userblock(userblock))
            .create(path)
            .unwrap();
        let mut builder = file.new_dataset::<i32>().chunk((chunk,)).shape((len,));
        if deflate {
            builder = builder.deflate(4);
        }
        self.writes().write(&builder.create("d").unwrap());
        file.close().unwrap();
    }

    fn writes(self) -> Writes {
        Writes {
            len: self.len,
            sparse: self.sparse,
        }
    }
}

#[rstest]
#[case::unpaged(Fixture { len: 64, chunk: 4, deflate: false, sparse: None })]
#[case::unpaged_filtered(Fixture { len: 64, chunk: 4, deflate: true, sparse: None })]
#[case::unpaged_windows(Fixture { len: 769, chunk: 1, deflate: false, sparse: None })]
#[case::unpaged_filtered_windows(Fixture { len: 769, chunk: 1, deflate: true, sparse: None })]
#[case::unpaged_sparse_windows(Fixture { len: 769, chunk: 1, deflate: true, sparse: Some(&[1..10, 350..360, 760..769]) })]
#[case::paged(Fixture { len: 3_000, chunk: 1, deflate: false, sparse: None })]
#[case::paged_filtered(Fixture { len: 3_000, chunk: 1, deflate: true, sparse: None })]
#[case::sparse_pages(Fixture {
    len: 4_000,
    chunk: 1,
    deflate: false,
    sparse: Some(&[0..10, 2_100..2_110, 3_990..4_000]),
})]
#[case::sparse_pages_filtered(Fixture {
    len: 4_000,
    chunk: 1,
    deflate: true,
    sparse: Some(&[0..10, 2_100..2_110, 3_990..4_000]),
})]
fn pure_reads_the_fixed_array_libhdf5_writes(
    #[case] fixture: Fixture,
    #[values((Sizeof::Bytes2, Sizeof::Bytes2), (Sizeof::Bytes4, Sizeof::Bytes4), (Sizeof::Bytes8, Sizeof::Bytes8))]
    widths: (Sizeof, Sizeof),
    #[values(0, 512)] userblock: u64,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let (sizeof_addr, sizeof_size) = widths;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixed_array.h5");
    fixture.libhdf5_create(
        &path,
        SizeofInfo {
            sizeof_addr,
            sizeof_size,
        },
        userblock,
    );

    let file = backend.open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.chunk_index().unwrap(), Some(ChunkIndex::FixedArray));
    assert_eq!(dataset.read_i32().unwrap(), fixture.writes().expected());
    assert_eq!(
        array_index::pure_chunks(&file),
        array_index::libhdf5_chunks(&path, userblock)
    );
}

#[rstest]
#[case::unpaged(64, 4, None)]
#[case::paged(1025, 1, None)]
#[case::sparse_pages(2049, 1, Some(PLACED_SPARSE_WRITES))]
fn native_reads_independently_placed_complete_fixed_array_blocks(
    #[case] len: usize,
    #[case] chunk: usize,
    #[case] sparse: Option<&'static [Range<usize>]>,
    #[values(false, true)] deflate: bool,
    #[values((Sizeof::Bytes2, OffsetWidth::Two), (Sizeof::Bytes4, OffsetWidth::Four), (Sizeof::Bytes8, OffsetWidth::Eight))]
    widths: (Sizeof, OffsetWidth),
    #[values(0, 512)] userblock: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("placed-fixed-array.h5");
    let fixture = Fixture {
        len,
        chunk,
        deflate,
        sparse,
    };
    fixture.libhdf5_create(
        &path,
        SizeofInfo {
            sizeof_addr: widths.0,
            sizeof_size: widths.0,
        },
        userblock,
    );
    let original_chunks = array_index::libhdf5_chunks(&path, userblock);
    let mut image = fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(array_index::pure_chunks(&file), original_chunks);
    let stored_header = file
        .relative_address(PhysicalAddress::new(
            u64::try_from(bytes::sole_signature(&image, b"FAHD")).unwrap(),
        ))
        .unwrap();
    let mut superblock = file.superblock().clone();
    let widths = FormatWidths {
        offsets: widths.1,
        lengths: LengthWidth::try_from(widths.1.get()).unwrap(),
    };
    let base = BaseAddress::new(file.resolve_address(StoredAddress::ZERO).unwrap().get());
    let source = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(image.len()).unwrap()).unwrap(),
        &image,
    )
    .unwrap();
    let parsed = FixedArrayHeaderLocator::new(widths, base, stored_header)
        .parse(source)
        .unwrap();
    let count = u64::try_from(len.div_ceil(chunk)).unwrap();
    let chunk_bytes = u64::try_from(chunk * size_of::<i32>()).unwrap();
    let plan = FixedArrayPlan::new(count, chunk_bytes, widths, deflate).unwrap();
    assert_eq!(
        plan.block_lengths().collect::<Vec<_>>(),
        [
            parsed.range().length(),
            parsed.data_block().unwrap().range().length()
        ]
    );
    let slots: Vec<_> = original_chunks
        .iter()
        .map(|record| record.offset()[0] / u64::try_from(chunk).unwrap())
        .collect();
    let chunks: Vec<_> = original_chunks
        .iter()
        .map(|record| ChunkRecord {
            address: file
                .relative_address(PhysicalAddress::new(record.address()))
                .unwrap(),
            stored_size: record.size(),
            filter_mask: record.filter_mask(),
        })
        .collect();
    let slots = IndexSlots::new(&chunks, &slots, count).unwrap();
    let data_at = file
        .relative_address(PhysicalAddress::new(file.file_size() + 4096))
        .unwrap();
    let placed = plan.place(stored_header, data_at).unwrap();
    let ranges: Vec<_> = placed
        .blocks()
        .map(|(address, length)| {
            FileRange::new(file.resolve_address(address).unwrap(), length).unwrap()
        })
        .collect();
    assert!(ranges[0].end().get() + 4096 < ranges[1].start().get());
    let blocks = placed.serialize_blocks(&slots).unwrap();
    let end = ranges[1].end().get();
    superblock.eof_address = end;
    file.close().unwrap();
    image.resize(usize::try_from(end).unwrap(), 0);
    for (range, (_, bytes)) in ranges.iter().zip(blocks) {
        let start = usize::try_from(range.start().get()).unwrap();
        assert_eq!(u64::try_from(bytes.len()).unwrap(), range.length());
        image[start..usize::try_from(range.end().get()).unwrap()].copy_from_slice(&bytes);
    }
    let superblock = format::serialize_superblock(&superblock).unwrap();
    let start = usize::try_from(userblock).unwrap();
    image[start..start + superblock.len()].copy_from_slice(&superblock);
    fs::write(&path, &image).unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        fixture.writes().expected()
    );
    native.close().unwrap();
    assert_eq!(
        array_index::libhdf5_chunks(&path, userblock),
        original_chunks
    );
    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.dataset("d").unwrap().read_i32().unwrap(),
        fixture.writes().expected()
    );
    assert_eq!(array_index::pure_chunks(&pure), original_chunks);
    pure.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Corruption {
    HeaderVersion,
    DataBlockOwner,
}

#[rstest]
fn pure_rejects_a_corrupted_fixed_array_libhdf5_writes(
    #[values(Corruption::HeaderVersion, Corruption::DataBlockOwner)] corruption: Corruption,
    #[values(64, 3_000)] len: usize,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixed_array.h5");
    let fixture = Fixture {
        len,
        chunk: 1,
        deflate: false,
        sparse: None,
    };
    fixture.libhdf5_create(
        &path,
        SizeofInfo {
            sizeof_addr: Sizeof::Bytes8,
            sizeof_size: Sizeof::Bytes8,
        },
        0,
    );
    let mut bytes = std::fs::read(&path).unwrap();
    let header = bytes::sole_signature(&bytes, b"FAHD");
    let data_block = bytes::sole_signature(&bytes, b"FADB");
    let entries = u64::try_from(len).unwrap();
    let prefix_len = if entries > 1_024 {
        4 + 2 + 8 + entries.div_ceil(1_024).div_ceil(8) + 4
    } else {
        4 + 2 + 8 + entries * 8 + 4
    };
    let expected = match corruption {
        Corruption::HeaderVersion => {
            bytes[header + 4] = 1;
            checksum::restamp(&mut bytes, header, 4 + 4 + 8 + 8 + 4);
            FormatError::InvalidFixedArrayVersion(1)
        }
        Corruption::DataBlockOwner => {
            let owner = u64::try_from(header).unwrap() + 1;
            bytes[data_block + 6..data_block + 14].copy_from_slice(&owner.to_le_bytes());
            checksum::restamp(&mut bytes, data_block, usize::try_from(prefix_len).unwrap());
            FormatError::InvalidFixedArrayBlockOwner {
                expected: owner - 1,
                actual: owner,
            }
        }
    };
    std::fs::write(&path, &bytes).unwrap();

    let err = backend
        .open(&path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap_err();
    let Error::Format(err) = err else {
        panic!("expected Error::Format, got {err:?}");
    };
    assert_eq!(err, expected);
}

const PLACED_SPARSE_WRITES: &[Range<usize>] = &[0..4, 2048..2049];
