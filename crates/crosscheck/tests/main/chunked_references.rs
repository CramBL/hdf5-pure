#![cfg(feature = "__hdf5-1.10")]

use hdf5::ObjectReference;
use hdf5::ObjectReference1;
use hdf5::ReferencedObject;
use hdf5::types::IntSize;
use hdf5::types::TypeDescriptor;
use hdf5_pure::AttrValue;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FilterMask;
use hdf5_pure::MaxExtent;
use hdf5_pure::Object;
use hdf5_pure::UserblockSize;
#[cfg(feature = "provenance")]
use hdf5_pure::VerifyResult;
use rstest::rstest;

#[rstest]
#[case::single_chunk(&[6], None)]
#[case::fixed_array(&[2], None)]
#[case::extensible_array(&[2], Some(&[MaxExtent::Unlimited][..]))]
#[case::partial_edge(&[4], None)]
fn c_resolves_chunked_path_references_and_reads_chunk_records(
    #[case] chunk: &[u64],
    #[case] maxshape: Option<&[MaxExtent]>,
    #[values(false, true)] shuffle: bool,
    #[values(false, true)] deflate: bool,
    #[values(
        FileSpaceStrategy::None,
        FileSpaceStrategy::Aggr,
        FileSpaceStrategy::Page
    )]
    strategy: FileSpaceStrategy,
    #[values(UserblockSize::NONE, UserblockSize::try_from(4096).unwrap())] userblock: UserblockSize,
    #[values(false, true)] dense: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("references.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true))
        .with_userblock(userblock);
    let dataset = builder
        .create_dataset("refs")
        .with_path_references(&["later", "group/data", "/", "group", "refs", "type"])
        .with_chunks(chunk);
    if let Some(maxshape) = maxshape {
        dataset.with_maxshape(maxshape);
    }
    if shuffle {
        dataset.with_shuffle();
    }
    if deflate {
        dataset.with_deflate(DeflateLevel::Six);
    }
    #[cfg(feature = "provenance")]
    dataset.with_provenance("test-suite", "2026-10-23T00:00:00Z", None);
    if dense {
        for position in 0..9 {
            dataset.set_attr(&format!("attribute{position}"), AttrValue::I32(position));
        }
    }
    builder.create_dataset("later").with_i32_data(&[7]);
    let mut group = builder.create_group("group");
    group.create_dataset("data").with_i32_data(&[21]);
    builder.add_group(group.finish());
    builder.commit_datatype("type", hdf5_pure::make_i32_type());
    builder.write(&path).unwrap();
    assert_eq!(
        test_util::fractal_heap::has_fractal_heap(&std::fs::read(&path).unwrap()),
        dense
    );
    let rust = File::open(&path).unwrap();
    let references = rust.dataset("refs").unwrap();
    #[cfg(feature = "provenance")]
    assert_eq!(references.verify_provenance().unwrap(), VerifyResult::Ok);
    let chunks = references.chunks().unwrap();
    let c = hdf5::File::open(&path).unwrap();
    let dataset = c.dataset("refs").unwrap();
    let values = dataset.read_raw::<ObjectReference1>().unwrap();
    let mut targets = Vec::new();
    for reference in values {
        let target = reference.dereference(&c).unwrap();
        targets.push(match target {
            ReferencedObject::Dataset(dataset) => {
                let name = dataset.name();
                if name == "/later" {
                    assert_eq!(dataset.read_raw::<i32>().unwrap(), [7]);
                }
                if name == "/group/data" {
                    assert_eq!(dataset.read_raw::<i32>().unwrap(), [21]);
                }
                name
            }
            ReferencedObject::Group(group) => group.name(),
            ReferencedObject::Datatype(datatype) => {
                assert_eq!(
                    datatype.to_descriptor().unwrap(),
                    TypeDescriptor::Integer(IntSize::U4)
                );
                "/type".into()
            }
        });
    }
    assert_eq!(
        targets,
        ["/later", "/group/data", "/", "/group", "/refs", "/type"]
    );
    assert_eq!(dataset.num_chunks().unwrap(), chunks.len());
    for (index, chunk) in chunks.into_iter().enumerate() {
        let c_chunk = dataset.chunk_info(index).unwrap();
        assert_eq!(c_chunk.offset, chunk.offset);
        // `H5D__get_chunk_info` adds the userblock base to chunk addresses in HDF5 1.14.6
        // and 2.2.0. HDF5 1.10.11 and 1.12.3 return the stored address (`H5Dchunk.c`).
        let address = if cfg!(feature = "__hdf5-1.14") {
            c_chunk.addr
        } else {
            c_chunk.addr + userblock.get()
        };
        assert_eq!(address, chunk.address);
        assert_eq!(c_chunk.size, chunk.storage_size);
        assert_eq!(c_chunk.filter_mask, chunk.filter_mask().raw_bits());
        assert_eq!(chunk.filter_mask(), FilterMask::NONE);
    }
    c.close().unwrap();
}

#[rstest]
#[case::unfiltered(false)]
#[case::filtered(true)]
fn rust_resolves_c_created_chunked_reference_values(#[case] filtered: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c-references.h5");
    let c = hdf5::File::create(&path).unwrap();
    c.new_dataset::<i32>()
        .shape([1])
        .create("target")
        .unwrap()
        .write(&[42])
        .unwrap();
    let dataset = c.new_dataset::<ObjectReference1>().shape([3]).chunk([2]);
    let dataset = if filtered {
        dataset.shuffle().deflate(6)
    } else {
        dataset
    };
    dataset
        .create("refs")
        .unwrap()
        .write(&[ObjectReference1::create(&c, "target").unwrap(); 3])
        .unwrap();
    drop(dataset);
    c.close().unwrap();
    let rust = File::open(&path).unwrap();
    let values: Vec<_> = rust
        .dataset("refs")
        .unwrap()
        .dereference()
        .unwrap()
        .into_iter()
        .map(|object| {
            let Object::Dataset(dataset) = object else {
                panic!("expected a dataset, got {object:?}");
            };
            dataset.read_i32().unwrap()
        })
        .collect();
    assert_eq!(values, [vec![42], vec![42], vec![42]]);
}
