#![cfg(feature = "__hdf5-1.10")]

use std::iter;
use std::ops::Range;
use std::path::Path;

use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use rstest::rstest;
use tempfile::tempdir;
use test_util::bytes::sole_signature;
use test_util::checksum::restamp;
use test_util_hdf5::reader::ReadBackend;

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
        let dataset = builder.create("d").unwrap();
        for range in self.written() {
            dataset
                .write_slice(&self.values()[range.clone()], range.clone())
                .unwrap();
        }
        file.close().unwrap();
    }

    fn expected(self) -> Vec<i32> {
        let mut expected = vec![0; self.len];
        for range in self.written() {
            expected[range.clone()].copy_from_slice(&self.values()[range.clone()]);
        }
        expected
    }

    fn written(self) -> Vec<Range<usize>> {
        self.sparse.map_or_else(
            || iter::once(0..self.len).collect(),
            <[Range<usize>]>::to_vec,
        )
    }

    fn values(self) -> Vec<i32> {
        (0..self.len)
            .map(|i| i32::try_from(i).unwrap() % 7 + 1)
            .collect()
    }
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Chunk {
    offset: Vec<u64>,
    address: u64,
    size: u64,
    filter_mask: u32,
}

fn libhdf5_chunks(path: &Path, userblock: u64) -> Vec<Chunk> {
    let file = hdf5::File::open(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let mut chunks: Vec<Chunk> = (0..dataset.num_chunks().unwrap())
        .map(|index| {
            let info = dataset.chunk_info(index).unwrap();
            Chunk {
                offset: info.offset,
                address: absolute(info.addr, userblock),
                size: info.size,
                filter_mask: info.filter_mask,
            }
        })
        .collect();
    chunks.sort();
    chunks
}

// `H5D__get_chunk_info` in `H5Dchunk.c` reports a base-relative chunk address in HDF5 1.10.11 and
// 1.12.3, and adds the base address in 1.14.6 and 2.2.0.
fn absolute(address: u64, userblock: u64) -> u64 {
    if cfg!(feature = "__hdf5-1.14") {
        address
    } else {
        address + userblock
    }
}

fn pure_chunks(file: &File) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = file
        .dataset("d")
        .unwrap()
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| Chunk {
            offset: chunk.offset,
            address: chunk.address,
            size: chunk.storage_size,
            filter_mask: chunk.filter_mask,
        })
        .collect();
    chunks.sort();
    chunks
}

#[rstest]
#[case::unpaged(Fixture { len: 64, chunk: 4, deflate: false, sparse: None })]
#[case::unpaged_filtered(Fixture { len: 64, chunk: 4, deflate: true, sparse: None })]
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
    let dir = tempdir().unwrap();
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
    assert_eq!(dataset.read_i32().unwrap(), fixture.expected());
    assert_eq!(pure_chunks(&file), libhdf5_chunks(&path, userblock));
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
    let dir = tempdir().unwrap();
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
    let header = sole_signature(&bytes, b"FAHD");
    let data_block = sole_signature(&bytes, b"FADB");
    let entries = u64::try_from(len).unwrap();
    let prefix_len = if entries > 1_024 {
        4 + 2 + 8 + entries.div_ceil(1_024).div_ceil(8) + 4
    } else {
        4 + 2 + 8 + entries * 8 + 4
    };
    let expected = match corruption {
        Corruption::HeaderVersion => {
            bytes[header + 4] = 1;
            restamp(&mut bytes, header, 4 + 4 + 8 + 8 + 4);
            FormatError::InvalidFixedArrayVersion(1)
        }
        Corruption::DataBlockOwner => {
            let owner = u64::try_from(header).unwrap() + 1;
            bytes[data_block + 6..data_block + 14].copy_from_slice(&owner.to_le_bytes());
            restamp(&mut bytes, data_block, usize::try_from(prefix_len).unwrap());
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
