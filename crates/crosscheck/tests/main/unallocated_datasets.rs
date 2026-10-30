#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::Extent;
use hdf5::datatype::ByteOrder as CByteOrder;
use hdf5::filters::Filter;
use hdf5::plist::dataset_create::Layout;
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::MaxExtent;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::ObjectHeader;
use rstest::rstest;
use test_util::extensible_array;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

#[derive(Clone, Copy, Debug)]
enum Creation {
    Buffered,
    Edit,
    Streaming,
}

impl Creation {
    fn write(self, path: &Path, configure: impl FnOnce(&mut DatasetBuilder)) {
        match self {
            Creation::Buffered | Creation::Streaming => {
                let mut builder = FileBuilder::new();
                configure(builder.create_dataset("schema"));
                if matches!(self, Creation::Streaming) {
                    builder.write(path).unwrap();
                } else {
                    std::fs::write(path, builder.finish().unwrap()).unwrap();
                }
            }
            Creation::Edit => {
                FileBuilder::new().write(path).unwrap();
                let file = File::open_rw(path).unwrap();
                file.root().create_dataset("schema", configure).unwrap();
                file.commit().unwrap();
                file.close().unwrap();
            }
        }
    }
}

#[rstest]
#[case::contiguous(false, false, false, &[3, 4])]
#[case::chunked(true, false, false, &[3, 4])]
#[case::filtered(true, true, false, &[3, 4])]
#[case::extensible(true, false, true, &[3, 4])]
#[case::empty_extensible(true, false, true, &[0, 4])]
#[case::empty_filtered_extensible(true, true, true, &[0, 4])]
fn schemas_match_reference_unallocated_storage(
    #[case] chunked: bool,
    #[case] filtered: bool,
    #[case] extensible: bool,
    #[case] shape: &[usize],
    #[values(None, Some(-7_i32))] fill: Option<i32>,
    #[values(Creation::Buffered, Creation::Streaming, Creation::Edit)] creation: Creation,
    #[values(ByteOrder::BigEndian, ByteOrder::LittleEndian)] reference_order: ByteOrder,
) {
    let directory = tempfile::tempdir().unwrap();
    let reference_path = directory.path().join("reference.h5");
    let pure_path = directory.path().join("pure.h5");
    {
        let reference = file::libhdf5_create_v110(&reference_path);
        let datatype = reference_order.i32_datatype();
        let mut schema = reference.new_dataset_builder().empty_as(&datatype);
        if let Some(fill) = fill {
            schema = schema.fill_value(fill);
        }
        if chunked {
            schema = schema.chunk((2, 2));
        }
        if filtered {
            schema = schema.shuffle().deflate(6);
        }
        let dataset = if extensible {
            schema
                .shape((Extent::resizable(shape[0]), Extent::fixed(shape[1])))
                .create("schema")
        } else {
            schema.shape((shape[0], shape[1])).create("schema")
        }
        .unwrap();
        assert_eq!(dataset.storage_size(), 0);
        drop(dataset);
        reference.close().unwrap();
    }
    let configure = |schema: &mut DatasetBuilder| {
        schema.with_unallocated_storage(
            hdf5_pure::make_i32_type(),
            &shape
                .iter()
                .map(|&dimension| u64::try_from(dimension).unwrap())
                .collect::<Vec<_>>(),
        );
        if let Some(fill) = fill {
            schema.with_fill_value(fill);
        }
        if chunked {
            schema.with_chunks(&[2, 2]);
        }
        if filtered {
            schema.with_shuffle().with_deflate(DeflateLevel::Six);
        }
        if extensible {
            schema.with_maxshape(&[MaxExtent::Unlimited, MaxExtent::Fixed(4)]);
        }
    };
    creation.write(&pure_path, configure);
    if shape.contains(&0) {
        assert_deferred_layout(&reference_path);
        assert_deferred_layout(&pure_path);
    }
    let reference = hdf5::File::open(&reference_path)
        .unwrap()
        .dataset("schema")
        .unwrap();
    let dataset = hdf5::File::open(&pure_path)
        .unwrap()
        .dataset("schema")
        .unwrap();
    assert_eq!(reference.shape(), shape);
    assert_eq!(dataset.shape(), shape);
    let reference_datatype = reference.dtype().unwrap();
    let datatype = dataset.dtype().unwrap();
    assert_eq!(
        reference_datatype.byte_order(),
        match reference_order {
            ByteOrder::BigEndian => CByteOrder::BigEndian,
            ByteOrder::LittleEndian => CByteOrder::LittleEndian,
        }
    );
    assert_eq!(datatype.byte_order(), CByteOrder::LittleEndian);
    assert_eq!(reference_datatype, reference_order.i32_datatype());
    assert_eq!(datatype, ByteOrder::LittleEndian.i32_datatype());
    assert_eq!(
        reference.read_raw::<i32>().unwrap(),
        vec![fill.unwrap_or(0); shape.iter().product()]
    );
    assert_eq!(
        dataset.read_raw::<i32>().unwrap(),
        vec![fill.unwrap_or(0); shape.iter().product()]
    );
    assert_eq!(reference.storage_size(), 0);
    assert_eq!(dataset.storage_size(), 0);
    assert_eq!(dataset.offset(), None);
    assert_eq!(
        dataset.layout(),
        if chunked {
            Layout::Chunked
        } else {
            Layout::Contiguous
        }
    );
    assert_eq!(dataset.chunk(), chunked.then(|| vec![2, 2]));
    assert_eq!(
        dataset.space().unwrap().maxdims(),
        if extensible {
            vec![None, Some(4)]
        } else {
            vec![Some(3), Some(4)]
        }
    );
    assert_eq!(
        dataset.dcpl().unwrap().get_fill_value_as::<i32>().unwrap(),
        Some(fill.unwrap_or(0))
    );
    assert_eq!(
        dataset.filters(),
        if filtered {
            vec![Filter::Shuffle, Filter::Deflate(6)]
        } else {
            vec![]
        }
    );
    assert_eq!(
        File::open(&reference_path)
            .unwrap()
            .dataset("schema")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![fill.unwrap_or(0); shape.iter().product()]
    );
    assert_eq!(
        File::open(&pure_path)
            .unwrap()
            .dataset("schema")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![fill.unwrap_or(0); shape.iter().product()]
    );
}

fn assert_deferred_layout(path: &Path) {
    let native = hdf5::File::open(path).unwrap();
    let address = usize::try_from(c_object::header_address(&native, "schema")).unwrap();
    native.close().unwrap();
    let bytes = std::fs::read(path).unwrap();
    let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, address, 8, 8).unwrap();
    let layout = header
        .messages
        .into_iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap()
        .data;
    assert_eq!(&layout[layout.len() - 8..], &[u8::MAX; 8]);
    assert!(matches!(
        DataLayout::parse(&layout, 8, 8).unwrap(),
        DataLayout::Chunked {
            index: ChunkIndexLayout::ExtensibleArray { address: None, .. },
            ..
        }
    ));
    assert!(test_util::bytes::signature_offsets(&bytes, extensible_array::SIGNATURE).is_empty());
    assert!(
        test_util::bytes::signature_offsets(&bytes, extensible_array::INDEX_SIGNATURE).is_empty()
    );
}

#[rstest]
fn schemas_preserve_the_largest_version_four_chunk(
    #[values(Creation::Buffered, Creation::Streaming, Creation::Edit)] creation: Creation,
) {
    let directory = tempfile::tempdir().unwrap();
    let reference_path = directory.path().join("reference.h5");
    let pure_path = directory.path().join("pure.h5");
    let chunk_len = usize::try_from(u32::MAX).unwrap();
    {
        let reference = file::libhdf5_create_v110(&reference_path);
        reference
            .new_dataset::<u8>()
            .shape((Extent::resizable(1),))
            .chunk((chunk_len,))
            .fill_value(42_u8)
            .create("schema")
            .unwrap();
        reference.close().unwrap();
    }
    creation.write(&pure_path, |schema| {
        schema
            .with_unallocated_storage(hdf5_pure::make_u8_type(), &[1])
            .with_chunks(&[u64::from(u32::MAX)])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_fill_value(42_u8);
    });
    let reference = hdf5::File::open(&reference_path)
        .unwrap()
        .dataset("schema")
        .unwrap();
    let dataset = hdf5::File::open(&pure_path)
        .unwrap()
        .dataset("schema")
        .unwrap();
    assert_eq!(reference.chunk(), Some(vec![chunk_len]));
    assert_eq!(dataset.chunk(), reference.chunk());
    assert_eq!(reference.storage_size(), 0);
    assert_eq!(dataset.storage_size(), 0);
    assert_eq!(dataset.shape(), reference.shape());
    assert_eq!(
        File::open(&reference_path)
            .unwrap()
            .dataset("schema")
            .unwrap()
            .chunk_shape()
            .unwrap(),
        Some(vec![u64::from(u32::MAX)])
    );
    assert_eq!(
        File::open(&pure_path)
            .unwrap()
            .dataset("schema")
            .unwrap()
            .chunk_shape()
            .unwrap(),
        Some(vec![u64::from(u32::MAX)])
    );
}
