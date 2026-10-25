use hdf5_pure::AttrValue;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FilterMask;
use hdf5_pure::FormatError;
use hdf5_pure::LibVer;
use hdf5_pure::MaxExtent;
use hdf5_pure::Object;
use hdf5_pure::ScaleOffset;
use hdf5_pure::UserblockSize;
#[cfg(feature = "provenance")]
use hdf5_pure::VerifyResult;
use rstest::rstest;

#[rstest]
#[case(Index::Single)]
#[case(Index::Fixed)]
#[case(Index::Extensible)]
#[case(Index::Edge)]
#[case(Index::Multidimensional)]
fn chunked_references_resolve_every_target(
    #[case] index: Index,
    #[values(Filters::None, Filters::Shuffle, Filters::Deflate, Filters::Combined)]
    filters: Filters,
    #[values(
        FileSpaceStrategy::None,
        FileSpaceStrategy::Aggr,
        FileSpaceStrategy::Page
    )]
    strategy: FileSpaceStrategy,
    #[values(false, true)] dense: bool,
    #[values(UserblockSize::NONE, UserblockSize::try_from(4096).unwrap())] userblock: UserblockSize,
    #[values(Output::Finish, Output::Sink, Output::Write)] output: Output,
) {
    let mut builder = FileBuilder::new();
    builder
        .with_userblock(userblock)
        .with_file_space_settings(FileSpaceSettings::new(strategy));
    let references = builder.create_dataset("refs").with_path_references(&[
        "later",
        "nested/data",
        "/",
        "nested",
        "refs",
        "peer",
    ]);
    #[cfg(feature = "provenance")]
    references.with_provenance("test-suite", "2026-10-23T00:00:00Z", None);
    index.configure(references);
    filters.configure(references);
    if dense {
        for position in 0..9 {
            references.set_attr(&format!("attribute{position}"), AttrValue::I32(position));
        }
    }
    builder
        .create_dataset("peer")
        .with_path_references(&["refs"])
        .with_chunks(&[1]);
    builder.create_dataset("later").with_i32_data(&[7]);
    let mut group = builder.create_group("nested");
    group.create_dataset("data").with_i32_data(&[21]);
    builder.add_group(group.finish());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("references.h5");
    let bytes = match output {
        Output::Finish => builder.finish().unwrap(),
        Output::Sink => {
            let mut bytes = Vec::new();
            builder.finish_to(&mut bytes).unwrap();
            bytes
        }
        Output::Write => {
            builder.write(&path).unwrap();
            std::fs::read(&path).unwrap()
        }
    };
    assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
    let file = File::from_bytes(bytes.clone()).unwrap();
    let references = file.dataset("refs").unwrap();
    assert_eq!(references.chunk_index().unwrap(), Some(index.kind()));
    assert_eq!(
        references.attrs().unwrap().len(),
        if dense { 9 } else { 0 } + if cfg!(feature = "provenance") { 3 } else { 0 }
    );
    #[cfg(feature = "provenance")]
    assert_eq!(references.verify_provenance().unwrap(), VerifyResult::Ok);
    let raw = references.read_raw().unwrap();
    let objects = references.dereference().unwrap();
    assert_eq!(objects.len(), 6);
    for (position, object) in objects.into_iter().enumerate() {
        match (position, object) {
            (0, Object::Dataset(target)) => assert_eq!(target.read_i32().unwrap(), [7]),
            (1, Object::Dataset(target)) => assert_eq!(target.read_i32().unwrap(), [21]),
            (2, Object::Group(target)) => {
                assert_eq!(target.dataset("later").unwrap().read_i32().unwrap(), [7])
            }
            (3, Object::Group(target)) => {
                assert_eq!(target.dataset("data").unwrap().read_i32().unwrap(), [21])
            }
            (4, Object::Dataset(target)) => assert_eq!(target.read_raw().unwrap(), raw),
            (5, Object::Dataset(target)) => {
                assert_eq!(
                    target.read_raw().unwrap(),
                    file.dataset("peer").unwrap().read_raw().unwrap()
                );
                let targets = target.dereference().unwrap();
                let Object::Dataset(target) = &targets[0] else {
                    panic!("expected the cyclic dataset, got {:?}", targets[0]);
                };
                assert_eq!(target.read_raw().unwrap(), raw);
            }
            (position, object) => panic!("unexpected target {position}: {object:?}"),
        }
    }
    let chunks = references.chunks().unwrap();
    assert_eq!(chunks.len(), index.chunks());
    for chunk in chunks {
        assert_eq!(chunk.filter_mask(), FilterMask::NONE);
        let start = usize::try_from(chunk.address).unwrap();
        let length = usize::try_from(chunk.storage_size).unwrap();
        let stored = &bytes[start..start + length];
        if matches!(filters, Filters::None) {
            let element = if chunk.offset.len() == 1 {
                chunk.offset[0]
            } else {
                chunk.offset[0] * 3 + chunk.offset[1]
            };
            let element = usize::try_from(element).unwrap() * 8;
            assert_eq!(&stored[..8], &raw[element..element + 8]);
        }
        if matches!(filters, Filters::None | Filters::Shuffle) {
            assert_eq!(chunk.storage_size, index.chunk_bytes());
        }
    }
}

#[rstest]
#[case(false, 0)]
#[case(false, u64::MAX)]
#[case(true, u64::MAX)]
fn chunked_references_preserve_null_and_undefined_values(
    #[case] path: bool,
    #[case] expected: u64,
) {
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("refs");
    if path {
        dataset.with_path_references(&["missing"]);
    } else {
        dataset.with_reference_data(&[expected]);
    }
    dataset
        .with_chunks(&[1])
        .with_shuffle()
        .with_deflate(DeflateLevel::Six);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("refs").unwrap();
    assert_eq!(dataset.read_raw().unwrap(), expected.to_le_bytes());
    let error = dataset.dereference().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, FormatError::InvalidObjectReference(expected));
}

#[rstest]
#[case(false)]
#[case(true)]
fn chunked_reference_errors_preserve_the_destination(#[case] old_bounds: bool) {
    let mut builder = FileBuilder::new();
    if old_bounds {
        builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    }
    let dataset = builder
        .create_dataset("refs")
        .with_path_references(&["target"])
        .with_chunks(&[1]);
    if !old_bounds {
        dataset.with_scale_offset(ScaleOffset::Integer(0));
    }
    builder.create_dataset("target").with_i32_data(&[7]);
    let mut bytes = b"previous bytes".to_vec();
    let error = builder.finish_to(&mut bytes).unwrap_err();
    assert_eq!(bytes, b"previous bytes");
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(
        error,
        if old_bounds {
            FormatError::LibverTooOldForContent {
                content: "a chunked, filtered, or resizable dataset",
                needs: LibVer::V110.name(),
                writing: LibVer::V18.name(),
            }
        } else {
            FormatError::FilterError("scale-offset requires an integer or floating-point scalar datatype with a definite (little/big endian) byte order".into())
        }
    );
}

#[derive(Clone, Copy)]
enum Index {
    Edge,
    Extensible,
    Fixed,
    Multidimensional,
    Single,
}
impl Index {
    fn configure(self, dataset: &mut DatasetBuilder) {
        match self {
            Self::Single => {
                dataset.with_chunks(&[6]);
            }
            Self::Fixed => {
                dataset.with_chunks(&[2]);
            }
            Self::Extensible => {
                dataset
                    .with_chunks(&[2])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            }
            Self::Edge => {
                dataset.with_chunks(&[4]);
            }
            Self::Multidimensional => {
                dataset.with_shape(&[2, 3]).with_chunks(&[2, 2]);
            }
        }
    }
    fn kind(self) -> ChunkIndex {
        match self {
            Self::Single => ChunkIndex::SingleChunk,
            Self::Extensible => ChunkIndex::ExtensibleArray,
            Self::Edge | Self::Fixed | Self::Multidimensional => ChunkIndex::FixedArray,
        }
    }
    fn chunks(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Fixed | Self::Extensible => 3,
            Self::Edge | Self::Multidimensional => 2,
        }
    }
    fn chunk_bytes(self) -> u64 {
        match self {
            Self::Single => 48,
            Self::Fixed | Self::Extensible => 16,
            Self::Edge | Self::Multidimensional => 32,
        }
    }
}

#[derive(Clone, Copy)]
enum Filters {
    Combined,
    Deflate,
    None,
    Shuffle,
}
impl Filters {
    fn configure(self, dataset: &mut DatasetBuilder) {
        match self {
            Self::Combined => {
                dataset.with_shuffle().with_deflate(DeflateLevel::Six);
            }
            Self::Deflate => {
                dataset.with_deflate(DeflateLevel::Six);
            }
            Self::None => {}
            Self::Shuffle => {
                dataset.with_shuffle();
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Output {
    Finish,
    Sink,
    Write,
}
