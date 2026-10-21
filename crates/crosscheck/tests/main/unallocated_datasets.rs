#![cfg(feature = "__hdf5-1.10")]

use hdf5::Extent;
use hdf5::datatype::ByteOrder as CByteOrder;
use hdf5::filters::Filter;
use hdf5::plist::dataset_create::Layout;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::MaxExtent;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

#[rstest]
#[case::contiguous(false, false, false)]
#[case::chunked(true, false, false)]
#[case::filtered(true, true, false)]
#[case::extensible(true, false, true)]
fn schemas_match_reference_unallocated_storage(
    #[case] chunked: bool,
    #[case] filtered: bool,
    #[case] extensible: bool,
    #[values(None, Some(-7_i32))] fill: Option<i32>,
    #[values(false, true)] streaming: bool,
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
                .shape((Extent::resizable(3), Extent::fixed(4)))
                .create("schema")
        } else {
            schema.shape((3, 4)).create("schema")
        }
        .unwrap();
        assert_eq!(dataset.storage_size(), 0);
        drop(dataset);
        reference.close().unwrap();
    }
    let mut builder = FileBuilder::new();
    let schema = builder
        .create_dataset("schema")
        .with_unallocated_storage(hdf5_pure::make_i32_type(), &[3, 4]);
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
    if streaming {
        builder.write(&pure_path).unwrap();
    } else {
        std::fs::write(&pure_path, builder.finish().unwrap()).unwrap();
    }
    let reference = hdf5::File::open(&reference_path)
        .unwrap()
        .dataset("schema")
        .unwrap();
    let dataset = hdf5::File::open(&pure_path)
        .unwrap()
        .dataset("schema")
        .unwrap();
    assert_eq!(reference.shape(), [3, 4]);
    assert_eq!(dataset.shape(), [3, 4]);
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
        vec![fill.unwrap_or(0); 12]
    );
    assert_eq!(
        dataset.read_raw::<i32>().unwrap(),
        vec![fill.unwrap_or(0); 12]
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
        vec![fill.unwrap_or(0); 12]
    );
    assert_eq!(
        File::open(&pure_path)
            .unwrap()
            .dataset("schema")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![fill.unwrap_or(0); 12]
    );
}
