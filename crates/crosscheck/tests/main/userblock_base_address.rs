#![cfg(feature = "__hdf5-1.10")]
//! Native files with userblocks and relocated superblocks.
//!
//! Exercises object-header continuations, dense storage, dataset layouts, and address
//! resolution through each reader backend.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5::MajorErrorCode;
use hdf5::MinorErrorCode;
use hdf5::dataset::Layout as CLayout;
use hdf5::file::LibraryVersion;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure::MetadataBytes;
use hdf5_pure::MetadataCacheConfig;
use hdf5_pure::Source;
use hdf5_pure::SuperblockVersion;
use hdf5_pure::UserblockSize;
use rstest::rstest;

use test_util::superblock;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::plain(0, 0)]
#[case::userblock(512, 512)]
#[case::jammed(0, 512)]
#[case::unjammed(512, 0)]
fn relocated_c_image_preserves_encoded_superblock_and_reads_data(
    #[case] stored_base: usize,
    #[case] signature_offset: usize,
    #[values(false, true)] latest_format: bool,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Streaming,
        ReadBackend::Bytes,
        ReadBackend::Source
    )]
    backend: ReadBackend,
) {
    let dir = tempfile::tempdir().unwrap();
    let original_path = dir.path().join("original.h5");
    let relocated_path = dir.path().join("relocated.h5");
    let original = hdf5::File::with_options()
        .with_fcpl(|properties| properties.userblock(stored_base as u64))
        .with_fapl(|properties| {
            if latest_format {
                properties.libver_v110()
            } else {
                properties
            }
        })
        .create(&original_path)
        .unwrap();
    original
        .new_dataset::<i32>()
        .shape([3])
        .create("child")
        .unwrap()
        .write(&[1, 2, 3])
        .unwrap();
    {
        let inner = original
            .create_group("outer")
            .unwrap()
            .create_group("inner")
            .unwrap();
        inner
            .new_dataset::<i32>()
            .shape([4])
            .chunk([2])
            .deflate(1)
            .create("data")
            .unwrap()
            .write(&[4, 5, 6, 7])
            .unwrap();
    }
    original.close().unwrap();
    let original_bytes = std::fs::read(&original_path).unwrap();
    let stored_eof = original_bytes.len() as u64;
    let mut relocated_bytes = vec![0; signature_offset];
    relocated_bytes.extend_from_slice(&original_bytes[stored_base..]);
    std::fs::write(&relocated_path, relocated_bytes).unwrap();
    {
        let oracle = hdf5::File::open(&relocated_path).unwrap();
        assert_eq!(
            oracle.dataset("child").unwrap().read_raw::<i32>().unwrap(),
            [1, 2, 3]
        );
        assert_eq!(
            oracle
                .dataset("outer/inner/data")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [4, 5, 6, 7]
        );
    }
    let file = backend.open(&relocated_path).unwrap();
    assert_eq!(file.superblock().base_address.get(), stored_base as u64);
    assert_eq!(file.superblock().eof_address, stored_eof);
    assert_eq!(file.root().groups().unwrap(), ["outer"]);
    assert_eq!(
        file.root().dataset("child").unwrap().read_i32().unwrap(),
        [1, 2, 3]
    );
    assert_eq!(file.group("outer").unwrap().groups().unwrap(), ["inner"]);
    assert_eq!(
        file.dataset("outer/inner/data")
            .unwrap()
            .read_i32()
            .unwrap(),
        [4, 5, 6, 7]
    );
}

#[rstest]
#[case::zero_base(0, 0)]
#[case::userblock(512, 512)]
#[case::forward_relocated(0, 512)]
#[case::backward_relocated(512, 0)]
fn group_lookup_and_listing_follow_relocated_native_images(
    #[case] encoded_base: usize,
    #[case] effective_base: usize,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Streaming,
        ReadBackend::Bytes,
        ReadBackend::Source
    )]
    backend: ReadBackend,
    #[values(false, true)] symbol_table: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let original_path = directory.path().join("original.h5");
    let relocated_path = directory.path().join("relocated.h5");
    let storage_type = if symbol_table {
        hdf5::GroupStorageType::SymbolTable
    } else {
        hdf5::GroupStorageType::Dense
    };
    let native = hdf5::File::with_options()
        .with_fcpl(|properties| properties.userblock(encoded_base as u64))
        .with_fapl(|properties| {
            if symbol_table {
                properties.libver_earliest()
            } else {
                properties.libver_v110()
            }
        })
        .create(&original_path)
        .unwrap();
    {
        let group = native.create_group("dense").unwrap();
        for index in 0..20 {
            group
                .new_dataset::<i32>()
                .shape([1])
                .create(format!("dataset_{index:03}").as_str())
                .unwrap()
                .write(&[index])
                .unwrap();
        }
        group.link_soft("dataset_015", "alias").unwrap();
        assert_eq!(group.info().unwrap().storage_type, storage_type);
        assert_eq!(
            group.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
            [15]
        );
        assert_eq!(group.member_names().unwrap().len(), 21);
    }
    native.close().unwrap();
    let original = std::fs::read(&original_path).unwrap();
    let mut relocated = vec![0; effective_base];
    relocated.extend_from_slice(&original[encoded_base..]);
    std::fs::write(&relocated_path, relocated).unwrap();
    {
        let oracle = hdf5::File::open(&relocated_path).unwrap();
        let group = oracle.group("dense").unwrap();
        assert_eq!(group.info().unwrap().storage_type, storage_type);
        assert_eq!(
            group
                .dataset("dataset_015")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [15]
        );
        assert_eq!(
            group.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
            [15]
        );
    }
    let metadata_reads = Arc::new(AtomicU64::new(0));
    let file = if matches!(backend, ReadBackend::Source) {
        File::from_source(BorrowedMetadataSource {
            bytes: std::fs::read(&relocated_path).unwrap(),
            metadata_reads: Arc::clone(&metadata_reads),
        })
    } else {
        backend.open(&relocated_path)
    }
    .unwrap();
    assert_eq!(file.superblock().base_address.get(), encoded_base as u64);
    assert_eq!(file.root().groups().unwrap(), ["dense"]);
    {
        let group = file.group("dense").unwrap();
        let mut names = group.datasets().unwrap();
        names.sort_unstable();
        let expected: Vec<_> = (0..20).map(|index| format!("dataset_{index:03}")).collect();
        assert_eq!(names, expected);
        assert_eq!(
            group.dataset("dataset_015").unwrap().read_i32().unwrap(),
            [15]
        );
        assert_eq!(group.dataset("alias").unwrap().read_i32().unwrap(), [15]);
        let error = file.dataset("dense/absent").unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::PathNotFound(String::from("dense/absent"))
        );
    }
    if matches!(backend, ReadBackend::Source) {
        assert!(metadata_reads.load(Ordering::Relaxed) > 0);
    }
    file.close().unwrap();
}

#[rstest]
#[case::zero_base(0, 0)]
#[case::userblock(512, 512)]
#[case::forward_relocated(0, 512)]
#[case::backward_relocated(512, 0)]
fn custom_sources_borrow_native_chunk_indexes_after_relocation(
    #[case] encoded_base: usize,
    #[case] effective_base: usize,
    #[values(false, true)] cache_enabled: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-arrays.h5");
    let fixed_values = [2, 3, 5, 7, 11, 13, 17, 19];
    let extensible_values = [23, 29, 31, 37, 41, 43, 47, 53];
    let btree_values = [59, 61, 67, 71, 73, 79, 83, 89];
    let native = hdf5::File::with_options()
        .with_fcpl(|properties| properties.userblock(encoded_base as u64))
        .with_fapl(|properties| properties.libver_v110())
        .create(&path)
        .unwrap();
    {
        let fixed = native
            .new_dataset::<i32>()
            .shape([fixed_values.len()])
            .chunk([2])
            .deflate(1)
            .create("fixed")
            .unwrap();
        fixed.write(&fixed_values).unwrap();
        assert_eq!(fixed.read_raw::<i32>().unwrap(), fixed_values);
        let extensible = native
            .new_dataset::<i32>()
            .shape((0..,))
            .chunk([2])
            .deflate(1)
            .create("extensible")
            .unwrap();
        extensible.resize([extensible_values.len()]).unwrap();
        extensible.write(&extensible_values).unwrap();
        assert_eq!(extensible.read_raw::<i32>().unwrap(), extensible_values);
        let btree = native
            .new_dataset::<i32>()
            .shape((0.., 0..))
            .chunk([1, 2])
            .deflate(1)
            .create("btree")
            .unwrap();
        btree.resize([2, 4]).unwrap();
        btree.write_raw(&btree_values).unwrap();
        assert_eq!(btree.read_raw::<i32>().unwrap(), btree_values);
    }
    native.close().unwrap();
    let original = std::fs::read(&path).unwrap();
    let mut relocated = vec![0; effective_base];
    relocated.extend_from_slice(&original[encoded_base..]);
    let metadata_reads = Arc::new(AtomicU64::new(0));
    let source = BorrowedMetadataSource {
        bytes: relocated,
        metadata_reads: Arc::clone(&metadata_reads),
    };
    let file = if cache_enabled {
        File::from_source_with_options(
            source,
            FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(1_048_576)),
        )
    } else {
        File::from_source(source)
    }
    .unwrap();
    assert_eq!(file.superblock().base_address.get(), encoded_base as u64);
    let root = file.superblock().root_group_address;
    assert_eq!(
        file.resolve_address(root).unwrap().get(),
        effective_base as u64 + root.get()
    );
    {
        let fixed = file.dataset("fixed").unwrap();
        assert_eq!(fixed.chunk_index().unwrap(), Some(ChunkIndex::FixedArray));
        assert_eq!(fixed.read_i32().unwrap(), fixed_values);
        assert_eq!(
            fixed.read_i32_region(&[1], &[5]).unwrap(),
            fixed_values[1..6]
        );
        let extensible = file.dataset("extensible").unwrap();
        assert_eq!(
            extensible.chunk_index().unwrap(),
            Some(ChunkIndex::ExtensibleArray)
        );
        assert_eq!(extensible.read_i32().unwrap(), extensible_values);
        assert_eq!(
            extensible.read_i32_region(&[1], &[5]).unwrap(),
            extensible_values[1..6]
        );
        let btree = file.dataset("btree").unwrap();
        assert_eq!(btree.chunk_index().unwrap(), Some(ChunkIndex::BTreeV2));
        assert_eq!(btree.read_i32().unwrap(), btree_values);
        assert_eq!(
            btree.read_i32_region(&[0, 1], &[2, 2]).unwrap(),
            [61, 67, 79, 83],
        );
    }
    let reads = metadata_reads.load(Ordering::Relaxed);
    assert!(reads > 0);
    if cache_enabled {
        let stats = file.metadata_cache_stats().unwrap();
        assert_eq!(stats.misses(), reads);
        assert_eq!(
            (
                stats.hits(),
                stats.entries(),
                stats.bytes(),
                stats.evictions(),
                stats.invalidations()
            ),
            (0, 0, 0, 0, 0)
        );
    } else {
        assert_eq!(file.metadata_cache_stats(), None);
    }
    file.close().unwrap();
}

#[rstest]
#[case::plain(0, 0)]
#[case::userblock(512, 512)]
#[case::jammed(0, 512)]
#[case::unjammed(512, 0)]
fn snapshot_open_requires_effective_eof(
    #[case] stored_base: usize,
    #[case] signature_offset: usize,
    #[values(false, true)] latest_format: bool,
    #[values(-1, 0, 512)] length_delta: i32,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let dir = tempfile::tempdir().unwrap();
    let original_path = dir.path().join("complete.h5");
    let path = dir.path().join("snapshot.h5");
    completeness_fixture(
        &original_path,
        stored_base as u64,
        if latest_format {
            LibraryVersion::V110
        } else {
            LibraryVersion::Earliest
        },
        32,
    );
    let original = std::fs::read(&original_path).unwrap();
    let mut image = vec![0; signature_offset];
    image.extend_from_slice(&original[stored_base..]);
    let effective_eof = image.len() as u64;
    if length_delta < 0 {
        image.pop().unwrap();
    } else {
        image.resize(image.len() + length_delta as usize, 0xa5);
    }
    let actual_len = image.len() as u64;
    std::fs::write(&path, image).unwrap();
    if length_delta < 0 {
        let error = hdf5::File::open(&path).unwrap_err();
        assert!(error.contains_major(MajorErrorCode::File), "{error}");
        assert!(error.contains_minor(MinorErrorCode::Truncated), "{error}");
        let error = backend.open(&path).unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::TruncatedFile {
                expected_eof: effective_eof,
                actual_len
            }
        );
        if latest_format {
            let lazy = File::open_swmr(&path).unwrap();
            assert_eq!(lazy.dataset("early").unwrap().read_i32().unwrap(), [42]);
        }
        return;
    }
    let oracle = hdf5::File::open(&path).unwrap();
    assert_eq!(
        oracle.dataset("early").unwrap().read_raw::<i32>().unwrap(),
        [42]
    );
    assert_eq!(
        oracle.dataset("tail").unwrap().read_raw::<i32>().unwrap(),
        vec![7; 4096]
    );
    let file = backend.open(&path).unwrap();
    assert_eq!(file.superblock().base_address.get(), stored_base as u64);
    assert_eq!(file.superblock().eof_address, original.len() as u64);
    assert_eq!(file.file_size(), actual_len);
    assert_eq!(file.dataset("early").unwrap().read_i32().unwrap(), [42]);
    assert_eq!(
        file.dataset("tail").unwrap().read_i32().unwrap(),
        vec![7; 4096]
    );
}

#[rstest]
#[case::v0(LibraryVersion::Earliest, 32, SuperblockVersion::Zero)]
#[case::v1(LibraryVersion::Earliest, 33, SuperblockVersion::One)]
#[case::v2(LibraryVersion::V18, 32, SuperblockVersion::Two)]
fn initial_legacy_swmr_open_requires_effective_eof(
    #[case] low: LibraryVersion,
    #[case] indexed_storage_k: u32,
    #[case] version: SuperblockVersion,
    #[values((0, 0), (512, 512), (0, 512), (512, 0))] placement: (usize, usize),
    #[values(-1, 0, 512)] length_delta: i32,
    #[values(false, true)] explicit_options: bool,
) {
    let (stored_base, signature_offset) = placement;
    let dir = tempfile::tempdir().unwrap();
    let original_path = dir.path().join("legacy-complete.h5");
    let path = dir.path().join("legacy-swmr.h5");
    completeness_fixture(&original_path, stored_base as u64, low, indexed_storage_k);
    let original = std::fs::read(&original_path).unwrap();
    assert_eq!(
        superblock::version_at(&original, stored_base),
        version.get()
    );
    let mut image = vec![0; signature_offset];
    image.extend_from_slice(&original[stored_base..]);
    let effective_eof = image.len() as u64;
    if length_delta < 0 {
        image.pop().unwrap();
    } else {
        image.resize(image.len() + length_delta as usize, 0xa5);
    }
    let actual_len = image.len() as u64;
    std::fs::write(&path, &image).unwrap();
    let ours = if explicit_options {
        File::open_swmr_with_options(&path, FileAccessProperties::new())
    } else {
        File::open_swmr(&path)
    };
    if length_delta < 0 {
        let error = hdf5::File::open_as(&path, hdf5::OpenMode::ReadSWMR).unwrap_err();
        assert!(error.contains_major(MajorErrorCode::File), "{error}");
        assert!(error.contains_minor(MinorErrorCode::Truncated), "{error}");
        let error = ours.unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::TruncatedFile {
                expected_eof: effective_eof,
                actual_len
            }
        );
        return;
    }
    {
        let oracle = hdf5::File::open_as(&path, hdf5::OpenMode::ReadSWMR).unwrap();
        assert_eq!(
            oracle.dataset("early").unwrap().read_raw::<i32>().unwrap(),
            [42]
        );
        assert_eq!(
            oracle.dataset("tail").unwrap().read_raw::<i32>().unwrap(),
            vec![7; 4096]
        );
    }
    let mut file = ours.unwrap();
    assert_eq!(file.superblock().version(), version);
    assert_eq!(file.superblock().base_address.get(), stored_base as u64);
    assert_eq!(file.superblock().eof_address, original.len() as u64);
    assert_eq!(file.dataset("early").unwrap().read_i32().unwrap(), [42]);
    assert_eq!(
        file.dataset("tail").unwrap().read_i32().unwrap(),
        vec![7; 4096]
    );
    image.truncate(usize::try_from(effective_eof - 1).unwrap());
    std::fs::write(&path, image).unwrap();
    file.refresh().unwrap();
    assert_eq!(file.dataset("early").unwrap().read_i32().unwrap(), [42]);
}

#[rstest]
#[case::plain(0, 0)]
#[case::userblock(512, 512)]
#[case::jammed(0, 512)]
#[case::unjammed(512, 0)]
fn swmr_open_and_refresh_accept_partially_extended_image(
    #[case] stored_base: usize,
    #[case] signature_offset: usize,
) {
    let dir = tempfile::tempdir().unwrap();
    let original_path = dir.path().join("complete.h5");
    let path = dir.path().join("live.h5");
    completeness_fixture(&original_path, stored_base as u64, LibraryVersion::V110, 32);
    let writer = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_v110())
        .open_rw(&original_path)
        .unwrap();
    writer.start_swmr().unwrap();
    writer.flush().unwrap();
    let original = std::fs::read(&original_path).unwrap();
    writer.close().unwrap();
    let mut image = vec![0; signature_offset];
    image.extend_from_slice(&original[stored_base..]);
    image.pop().unwrap();
    std::fs::write(&path, &image).unwrap();
    {
        let oracle = hdf5::File::open_as(&path, hdf5::OpenMode::ReadSWMR).unwrap();
        assert_eq!(
            oracle.dataset("early").unwrap().read_raw::<i32>().unwrap(),
            [42]
        );
    }
    let mut file = File::open_swmr(&path).unwrap();
    assert_eq!(file.superblock().version(), SuperblockVersion::Three);
    assert_eq!(file.dataset("early").unwrap().read_i32().unwrap(), [42]);
    image.pop().unwrap();
    std::fs::write(&path, image).unwrap();
    file.refresh().unwrap();
    assert_eq!(file.dataset("early").unwrap().read_i32().unwrap(), [42]);
    assert_eq!(file.superblock().base_address.get(), stored_base as u64);
    assert_eq!(file.superblock().eof_address, original.len() as u64);
}

fn completeness_fixture(
    path: &Path,
    stored_base: u64,
    low: LibraryVersion,
    indexed_storage_k: u32,
) {
    let file = hdf5::File::with_options()
        .with_fcpl(|properties| {
            properties
                .userblock(stored_base)
                .istore_k(indexed_storage_k)
        })
        .with_fapl(|properties| {
            properties.libver_bounds(
                low,
                match low {
                    LibraryVersion::Earliest | LibraryVersion::V18 => LibraryVersion::V18,
                    _ => low,
                },
            )
        })
        .create(path)
        .unwrap();
    file.new_dataset::<i32>()
        .shape([1])
        .layout(CLayout::Compact)
        .create("early")
        .unwrap()
        .write(&[42])
        .unwrap();
    file.new_dataset::<i32>()
        .shape([4096])
        .create("tail")
        .unwrap()
        .write(&vec![7; 4096])
        .unwrap();
    file.close().unwrap();
}

#[rstest]
#[case::jammed(0, 512)]
#[case::unjammed(512, 0)]
fn editor_rejects_relocated_image(#[case] stored_base: usize, #[case] signature_offset: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relocated-edit.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(stored_base as u64).unwrap());
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let original = builder.finish().unwrap();
    let mut relocated = vec![0; signature_offset];
    relocated.extend_from_slice(&original[stored_base..]);
    std::fs::write(&path, &relocated).unwrap();
    let error = File::open_rw(&path).unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a file whose superblock is not located at its base address is not editable in place"
            )
        ),
        "{error:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), relocated);
}

/// The size MATLAB uses, and the one every `.mat` this crate writes carries.
const UB: u64 = 512;

/// Returns sorted root dataset names from buffered, streaming, and byte-image reads of `path`.
fn names_via_every_backend(path: &std::path::Path) -> Vec<(&'static str, Vec<String>)> {
    let sorted = |f: &File| -> Vec<String> {
        let mut v = f.root().datasets().expect("list the root group");
        v.sort();
        v
    };
    vec![
        ("open", sorted(&File::open(path).unwrap())),
        (
            "open_streaming",
            sorted(&File::open_streaming(path).unwrap()),
        ),
        (
            "from_bytes",
            sorted(&File::from_bytes(std::fs::read(path).unwrap()).unwrap()),
        ),
    ]
}

fn assert_every_backend_sees(path: &std::path::Path, expected: &[&str]) {
    let mut want: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
    want.sort();
    for (backend, got) in names_via_every_backend(path) {
        assert_eq!(got, want, "backend `{backend}` read the wrong link set");
    }
}

/// Create a file with the C library, with or without a userblock.
fn c_file(path: &std::path::Path, userblock: u64, build: impl FnOnce(&hdf5::File)) {
    let file = hdf5::File::with_options()
        .with_fcpl(|p| {
            if userblock > 0 {
                p.userblock(userblock);
            }
            p
        })
        .create(path)
        .unwrap();
    build(&file);
    file.close().unwrap();
    assert_eq!(
        hdf5::File::open(path).unwrap().userblock(),
        userblock,
        "the C library did not produce the requested userblock"
    );
}

fn c_add_dataset(file: &hdf5::File, name: &str, value: i32) {
    file.new_dataset::<i32>()
        .shape([1])
        .create(name)
        .unwrap()
        .write(&[value])
        .unwrap();
}

/// The reported case: a file this crate wrote, then modified by the C library.
///
/// Adding one link overflows the root header's chunk, so the C library spills the
/// rest into a continuation block and records its address relative to the base.
/// Run against both userblock sizes: the zero case passed throughout and pins
/// that the base is what makes the difference, not the C edit.
#[test]
fn c_modified_file_with_a_userblock_reads_whole() {
    let dir = tempfile::tempdir().unwrap();
    for ub in [0, UB] {
        let path = dir.path().join(format!("ours_then_c_{ub}.h5"));
        let mut b = FileBuilder::new();
        if ub > 0 {
            b.with_userblock(UserblockSize::try_from(ub).unwrap());
        }
        b.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
        b.write(&path).unwrap();

        {
            let f = hdf5::File::open_rw(&path).unwrap();
            c_add_dataset(&f, "added", 4);
        }

        assert_every_backend_sees(&path, &["values", "added"]);

        // The data itself, not merely the names: a continuation block carries the
        // data-layout message of whichever dataset was pushed out of chunk 0.
        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("values").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0],
            "userblock {ub}"
        );
        assert_eq!(
            f.dataset("added").unwrap().read_i32().unwrap(),
            vec![4],
            "userblock {ub}"
        );
    }
}

/// An attribute added by the C library also grows the header past its chunk, so
/// this reaches the continuation block without adding any link.
#[test]
fn c_added_attribute_in_a_userblock_file_reads_whole() {
    let dir = tempfile::tempdir().unwrap();
    for ub in [0, UB] {
        let path = dir.path().join(format!("attr_{ub}.h5"));
        let mut b = FileBuilder::new();
        if ub > 0 {
            b.with_userblock(UserblockSize::try_from(ub).unwrap());
        }
        b.create_dataset("values").with_f64_data(&[1.5]);
        b.write(&path).unwrap();

        {
            let f = hdf5::File::open_rw(&path).unwrap();
            let d = f.dataset("values").unwrap();
            d.new_attr::<i32>()
                .shape([1])
                .create("tag")
                .unwrap()
                .write(&[7])
                .unwrap();
        }

        assert_every_backend_sees(&path, &["values"]);
        let f = File::open(&path).unwrap();
        assert_eq!(f.dataset("values").unwrap().read_f64().unwrap(), vec![1.5]);
        let attrs = f.dataset("values").unwrap().attrs().unwrap();
        assert_eq!(
            attrs.keys().collect::<Vec<_>>(),
            vec!["tag"],
            "userblock {ub}"
        );
    }
}

/// Dense link storage in a userblock file the C library wrote start to finish.
///
/// Nothing this crate wrote is involved, which is what makes this the widest of
/// the three: the Link Info message names a fractal heap and a version 2 B-tree
/// by base-relative address, and reading either at its stored value lands in the
/// userblock. Twenty-four links is past the eight at which the C library converts
/// a group to dense storage.
#[test]
fn c_written_dense_links_in_a_userblock_file() {
    let dir = tempfile::tempdir().unwrap();
    for ub in [0, UB] {
        let path = dir.path().join(format!("dense_{ub}.h5"));
        c_file(&path, ub, |f| {
            for i in 0..24 {
                c_add_dataset(f, &format!("d{i:02}"), i);
            }
        });

        let names: Vec<String> = (0..24).map(|i| format!("d{i:02}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        assert_every_backend_sees(&path, &refs);

        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d17").unwrap().read_i32().unwrap(),
            vec![17],
            "userblock {ub}"
        );
    }
}

/// A group *nested* inside a userblock file, dense on both levels. The base
/// applies at every hop, not only at the root, so this catches a fix applied
/// only where the root group is resolved.
#[test]
fn c_written_nested_dense_group_in_a_userblock_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    c_file(&path, UB, |f| {
        let g = f.create_group("inner").unwrap();
        for i in 0..24 {
            g.new_dataset::<i32>()
                .shape([1])
                .create(format!("n{i:02}").as_str())
                .unwrap()
                .write(&[i])
                .unwrap();
        }
        for i in 0..24 {
            c_add_dataset(f, &format!("r{i:02}"), i);
        }
    });

    let f = File::open(&path).unwrap();
    let mut inner = f.group("inner").unwrap().datasets().unwrap();
    inner.sort();
    let mut want: Vec<String> = (0..24).map(|i| format!("n{i:02}")).collect();
    want.sort();
    assert_eq!(inner, want);
    assert_eq!(
        f.dataset("inner/n23").unwrap().read_i32().unwrap(),
        vec![23]
    );
}

/// The whole point of the fix, stated as the round trip it protects: a `.mat`
/// file that a C-based tool has rewritten still reads here.
///
/// Twelve variables rather than two, because that is what makes this a test
/// rather than a demonstration. A `.mat` with a couple of variables keeps its
/// root links compact and its headers in one chunk, so it never reaches either
/// half of the defect and passes with the whole fix reverted; past eight, the
/// copy writes dense link storage and the base address starts to matter.
#[cfg(feature = "serde")]
#[test]
fn a_mat_file_repacked_by_the_c_library_still_reads() {
    use hdf5_pure::mat;
    use std::collections::BTreeMap;

    const VARS: usize = 12;

    let doc: BTreeMap<String, Vec<f64>> = (0..VARS)
        .map(|i| (format!("var{i:02}"), vec![i as f64, i as f64 + 0.5]))
        .collect();

    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("in.mat");
    mat::to_file(&doc, &src).unwrap();
    assert_eq!(
        hdf5::File::open(&src).unwrap().userblock(),
        UB,
        "a .mat file carries a userblock, which is what puts it on this path"
    );

    // Stand in for `h5repack`: the C library copies every object into a fresh
    // file that keeps the userblock, which is the shape a repacked `.mat` has.
    let dst = dir.path().join("out.mat");
    {
        let input = hdf5::File::open(&src).unwrap();
        let output = hdf5::File::with_options()
            .with_fcpl(|p| {
                p.userblock(UB);
                p
            })
            .create(&dst)
            .unwrap();
        for name in input.member_names().unwrap() {
            input
                .dataset(&name)
                .unwrap()
                .copy_to(&output, &name)
                .unwrap();
        }
    }

    let f = File::open(&dst).unwrap();
    let mut got = f.root().datasets().unwrap();
    got.sort();
    let mut want: Vec<String> = doc.keys().cloned().collect();
    want.sort();
    assert_eq!(got, want, "every variable survived the repack");
    assert_eq!(
        f.dataset("var07").unwrap().read_f64().unwrap(),
        vec![7.0, 7.5]
    );
}

// ---------------------------------------------------------------------------
// The defect class, rather than the two instances of it
// ---------------------------------------------------------------------------

/// Compares `read` results from buffered, streaming, and byte-image reads of `path`.
fn every_backend(path: &std::path::Path, what: &str, read: impl Fn(&File) -> String) {
    let expected = read(&File::open(path).unwrap());
    for (backend, file) in [
        ("open", File::open(path).unwrap()),
        ("open_streaming", File::open_streaming(path).unwrap()),
        (
            "from_bytes",
            File::from_bytes(std::fs::read(path).unwrap()).unwrap(),
        ),
    ] {
        assert_eq!(
            read(&file),
            expected,
            "{what}: backend `{backend}` disagrees with `open`"
        );
    }
}

// Compares native-written storage layouts with and without a userblock.
#[test]
fn every_c_written_shape_reads_under_a_userblock() {
    let dir = tempfile::tempdir().unwrap();

    for ub in [0, UB] {
        // Chunked with a filter: the chunk addresses live in a chunk index, and
        // the data must be found before it can be inflated.
        let path = dir.path().join(format!("chunked_deflate_{ub}.h5"));
        c_file(&path, ub, |f| {
            f.new_dataset::<f64>()
                .shape([64])
                .chunk([8])
                .deflate(4)
                .create("c")
                .unwrap()
                .write(&(0..64).map(f64::from).collect::<Vec<_>>())
                .unwrap();
        });
        every_backend(&path, "chunked+deflate", |f| {
            format!("{:?}", f.dataset("c").unwrap().read_f64().unwrap())
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("c")
                .unwrap()
                .read_f64()
                .unwrap(),
            (0..64).map(f64::from).collect::<Vec<_>>(),
            "chunked+deflate under userblock {ub}"
        );

        // Variable-length strings: the element data is a reference into the
        // global heap, which is reached by its own stored address.
        let path = dir.path().join(format!("vlen_{ub}.h5"));
        c_file(&path, ub, |f| {
            let words: Vec<hdf5::types::VarLenUnicode> = ["alpha", "beta", "gamma"]
                .iter()
                .map(|s| s.parse().unwrap())
                .collect();
            f.new_dataset::<hdf5::types::VarLenUnicode>()
                .shape([3])
                .create("s")
                .unwrap()
                .write(&words)
                .unwrap();
        });
        every_backend(&path, "vlen strings", |f| {
            format!("{:?}", f.dataset("s").unwrap().read_string().unwrap())
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("s")
                .unwrap()
                .read_string()
                .unwrap(),
            vec!["alpha", "beta", "gamma"],
            "vlen strings under userblock {ub}"
        );

        // Enough attributes to force dense storage: a fractal heap and a version
        // 2 B-tree, each named by a stored address in the Attribute Info message.
        let path = dir.path().join(format!("dense_attrs_{ub}.h5"));
        c_file(&path, ub, |f| {
            let d = f.new_dataset::<i32>().shape([1]).create("d").unwrap();
            d.write(&[1]).unwrap();
            for i in 0..24 {
                d.new_attr::<i32>()
                    .shape([1])
                    .create(format!("a{i:02}").as_str())
                    .unwrap()
                    .write(&[i])
                    .unwrap();
            }
        });
        every_backend(&path, "dense attributes", |f| {
            let mut names: Vec<String> = f
                .dataset("d")
                .unwrap()
                .attrs()
                .unwrap()
                .into_keys()
                .collect();
            names.sort();
            names.join(",")
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("d")
                .unwrap()
                .attrs()
                .unwrap()
                .len(),
            24,
            "dense attributes under userblock {ub}"
        );

        // Six levels of nesting: the base applies at every hop, not only at the
        // root, and a header this deep in a C-written file carries a
        // continuation block.
        let path = dir.path().join(format!("nested_{ub}.h5"));
        c_file(&path, ub, |f| {
            let mut g = f.create_group("l0").unwrap();
            for i in 1..6 {
                g = g.create_group(&format!("l{i}")).unwrap();
            }
            g.new_dataset::<i32>()
                .shape([1])
                .create("leaf")
                .unwrap()
                .write(&[42])
                .unwrap();
        });
        every_backend(&path, "nested groups", |f| {
            format!(
                "{:?}",
                f.dataset("l0/l1/l2/l3/l4/l5/leaf")
                    .unwrap()
                    .read_i32()
                    .unwrap()
            )
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("l0/l1/l2/l3/l4/l5/leaf")
                .unwrap()
                .read_i32()
                .unwrap(),
            vec![42],
            "nested groups under userblock {ub}"
        );

        // Sixty-four chunks: enough that the index is a real structure with
        // internal nodes rather than a single-chunk shortcut.
        let path = dir.path().join(format!("many_chunks_{ub}.h5"));
        c_file(&path, ub, |f| {
            f.new_dataset::<i32>()
                .shape([1024])
                .chunk([16])
                .create("b")
                .unwrap()
                .write(&(0..1024).collect::<Vec<i32>>())
                .unwrap();
        });
        every_backend(&path, "64-chunk index", |f| {
            let v = f.dataset("b").unwrap().read_i32().unwrap();
            format!("{} {} {}", v.len(), v[0], v[v.len() - 1])
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("b")
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..1024).collect::<Vec<i32>>(),
            "64-chunk index under userblock {ub}"
        );

        // An unlimited dimension, which the C library indexes with an
        // extensible array rather than the fixed-array form above.
        let path = dir.path().join(format!("unlimited_{ub}.h5"));
        c_file(&path, ub, |f| {
            let d = f
                .new_dataset::<i32>()
                .shape((0..,))
                .chunk([8])
                .create("u")
                .unwrap();
            d.resize([32]).unwrap();
            d.write(&(0..32).collect::<Vec<i32>>()).unwrap();
        });
        every_backend(&path, "extensible-array index", |f| {
            format!("{:?}", f.dataset("u").unwrap().read_i32().unwrap())
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("u")
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..32).collect::<Vec<i32>>(),
            "extensible-array index under userblock {ub}"
        );
    }
}

struct BorrowedMetadataSource {
    bytes: Vec<u8>,
    metadata_reads: Arc<AtomicU64>,
}

impl BorrowedMetadataSource {
    fn bytes_in(&self, offset: u64, length: usize) -> Result<&[u8], FormatError> {
        let end = offset
            .checked_add(length as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset,
                length: length as u64,
            })?;
        if end > self.len() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end,
                available: self.len(),
            });
        }
        let start =
            usize::try_from(offset).map_err(|_error| FormatError::ValueTooLargeForPlatform {
                value: offset,
                target: "usize",
            })?;
        let end_index =
            usize::try_from(end).map_err(|_error| FormatError::ValueTooLargeForPlatform {
                value: end,
                target: "usize",
            })?;
        self.bytes
            .get(start..end_index)
            .ok_or(FormatError::UnexpectedSourceEof {
                expected: end,
                available: self.bytes.len() as u64,
            })
    }
}

impl Source for BorrowedMetadataSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        buffer.copy_from_slice(self.bytes_in(offset, buffer.len())?);
        Ok(())
    }

    fn read_metadata_at(&self, _offset: u64, _length: usize) -> Result<Vec<u8>, FormatError> {
        Err(FormatError::Source("unexpected owned metadata read".into()))
    }

    fn read_metadata(&self, offset: u64, length: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let metadata = MetadataBytes::from_slice(offset, self.bytes_in(offset, length)?)?;
        self.metadata_reads.fetch_add(1, Ordering::Relaxed);
        Ok(metadata)
    }
}
