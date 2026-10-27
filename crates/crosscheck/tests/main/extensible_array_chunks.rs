#![cfg(feature = "__hdf5-1.10")]
//! The Extensible Array chunk index of the files libhdf5 writes, read back by hdf5-pure.

use std::ops::Range;
use std::path::Path;

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Error;
use hdf5_pure::FormatError;
use rstest::rstest;
use test_util::checksum;
use test_util_hdf5::reader::ReadBackend;

use crate::test_util::array_index;
use crate::test_util::array_index::Chunk;
use crate::test_util::array_index::Writes;

/// A rank 1 dataset of `len` one-element chunks along an unlimited dimension, deflated or not, with
/// all its elements written or those of the ranges in `sparse`.
#[derive(Clone, Copy, Debug)]
struct Fixture {
    len: usize,
    deflate: bool,
    sparse: Option<&'static [Range<usize>]>,
}

impl Fixture {
    fn libhdf5_create(self, path: &Path, widths: (Sizeof, Sizeof), userblock: u64) {
        let (sizeof_addr, sizeof_size) = widths;
        let Self {
            len,
            deflate,
            sparse: _,
        } = self;
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .with_fcpl(|p| {
                p.sizes(SizeofInfo {
                    sizeof_addr,
                    sizeof_size,
                })
                .userblock(userblock)
            })
            .create(path)
            .unwrap();
        let mut builder = file
            .new_dataset::<i32>()
            .chunk((1,))
            .shape((Extent::resizable(len),));
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

    fn pure_chunks(self, path: &Path, backend: ReadBackend) -> Vec<Chunk> {
        let file = backend.open(path).unwrap();
        let dataset = file.dataset("d").unwrap();
        assert_eq!(
            dataset.chunk_index().unwrap(),
            Some(ChunkIndex::ExtensibleArray)
        );
        assert_eq!(dataset.read_i32().unwrap(), self.writes().expected());
        array_index::pure_chunks(&file)
    }
}

// Every length is 8 bytes wide: `H5O__sdspace_decode` in `H5Osdspace.c`, HDF5 2.2.0, reads an
// unlimited maximum dimension stored in fewer bytes as a finite one, and `H5D__earray_idx_init`
// in `H5Dearray.c` then rejects the dataset.
//
// The index block holds 4 elements and the addresses of 6 data blocks over slots 4..244. Super
// block 13, the first whose data blocks are paged, starts at slot 131_060 and holds 64 data blocks
// of two 1_024-slot pages each.
#[rstest]
#[case::inline_elements(Fixture { len: 4, deflate: false, sparse: None })]
#[case::inline_elements_filtered(Fixture { len: 4, deflate: true, sparse: None })]
#[case::direct_data_blocks(Fixture { len: 244, deflate: false, sparse: None })]
#[case::direct_data_blocks_filtered(Fixture { len: 244, deflate: true, sparse: None })]
#[case::super_blocks(Fixture { len: 2_000, deflate: false, sparse: None })]
#[case::super_blocks_filtered(Fixture { len: 2_000, deflate: true, sparse: None })]
#[case::unallocated_blocks(Fixture {
    len: 3_000,
    deflate: false,
    sparse: Some(&[0..2, 100..110, 2_500..2_510]),
})]
#[case::uninitialized_pages(Fixture {
    len: 140_000,
    deflate: false,
    sparse: Some(&[131_060 + 1_030..131_060 + 1_040, 131_060 + 3 * 2_048..131_060 + 3 * 2_048 + 10]),
})]
#[case::uninitialized_pages_filtered(Fixture {
    len: 140_000,
    deflate: true,
    sparse: Some(&[131_060 + 1_030..131_060 + 1_040, 131_060 + 3 * 2_048..131_060 + 3 * 2_048 + 10]),
})]
fn pure_reads_the_extensible_array_libhdf5_writes(
    #[case] fixture: Fixture,
    #[values((Sizeof::Bytes2, Sizeof::Bytes8), (Sizeof::Bytes4, Sizeof::Bytes8), (Sizeof::Bytes8, Sizeof::Bytes8))]
    widths: (Sizeof, Sizeof),
    #[values(0, 512)] userblock: u64,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extensible_array.h5");
    fixture.libhdf5_create(&path, widths, userblock);
    assert_eq!(
        fixture.pure_chunks(&path, backend),
        array_index::libhdf5_chunks(&path, userblock)
    );
}

// 140_000 chunks do not fit the 64 KiB a 2-byte address reaches. The chunks compared are those at
// the edges of the index block, of its data blocks, and of the pages of super block 13: each
// `chunk_info` call walks the index from the first chunk to the one it reports.
#[rstest]
fn pure_reads_the_paged_extensible_array_libhdf5_writes(
    #[values((Sizeof::Bytes4, Sizeof::Bytes8), (Sizeof::Bytes8, Sizeof::Bytes8))] widths: (
        Sizeof,
        Sizeof,
    ),
    #[values(0, 512)] userblock: u64,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extensible_array.h5");
    let fixture = Fixture {
        len: 140_000,
        deflate: false,
        sparse: None,
    };
    fixture.libhdf5_create(&path, widths, userblock);
    let pure = fixture.pure_chunks(&path, backend);
    assert_eq!(pure.len(), fixture.len);
    let indices = [
        0,
        3,
        4,
        243,
        244,
        131_059,
        131_060,
        131_060 + 1_023,
        131_060 + 1_024,
        131_060 + 2_048,
        139_999,
    ];
    assert_eq!(
        indices.map(|index| pure[index].clone()).to_vec(),
        array_index::libhdf5_chunks_at(&path, userblock, &indices)
    );
}

/// The field a corruption test changes in a file libhdf5 writes.
#[derive(Clone, Copy, Debug)]
enum Corruption {
    HeaderVersion,
    IndexBlockOwner,
    DataBlockOffset,
}

// Twenty chunks fill the index block's 4 elements and the first data block, of 16 elements.
#[rstest]
fn pure_rejects_a_corrupted_extensible_array_libhdf5_writes(
    #[values(
        Corruption::HeaderVersion,
        Corruption::IndexBlockOwner,
        Corruption::DataBlockOffset
    )]
    corruption: Corruption,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extensible_array.h5");
    let fixture = Fixture {
        len: 20,
        deflate: false,
        sparse: None,
    };
    fixture.libhdf5_create(&path, (Sizeof::Bytes8, Sizeof::Bytes8), 0);
    let mut bytes = std::fs::read(&path).unwrap();
    let header = test_util::bytes::sole_signature(&bytes, b"EAHD");
    let index_block = test_util::bytes::sole_signature(&bytes, b"EAIB");
    let data_block = test_util::bytes::sole_signature(&bytes, b"EADB");
    let expected = match corruption {
        Corruption::HeaderVersion => {
            bytes[header + 4] = 1;
            checksum::restamp(&mut bytes, header, 12 + 6 * 8 + 8 + 4);
            FormatError::InvalidExtensibleArrayVersion(1)
        }
        Corruption::IndexBlockOwner => {
            let owner = u64::try_from(header).unwrap() + 1;
            bytes[index_block + 6..index_block + 14].copy_from_slice(&owner.to_le_bytes());
            // The prefix, 4 elements, 6 data block addresses, and 25 super block addresses.
            checksum::restamp(&mut bytes, index_block, 14 + 4 * 8 + (6 + 25) * 8 + 4);
            FormatError::InvalidExtensibleArrayBlockOwner {
                expected: owner - 1,
                actual: owner,
            }
        }
        Corruption::DataBlockOffset => {
            bytes[data_block + 14..data_block + 18].copy_from_slice(&1u32.to_le_bytes());
            checksum::restamp(&mut bytes, data_block, 14 + 4 + 16 * 8 + 4);
            FormatError::InvalidExtensibleArrayBlockOffset {
                expected: 0,
                actual: 1,
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
