#![cfg(feature = "hdf5")]

use hdf5::filters::Filter;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileBuilder;
use rstest::rstest;

#[rstest]
#[case(DeflateLevel::Zero)]
#[case(DeflateLevel::Six)]
#[case(DeflateLevel::Nine)]
fn reads_c_deflate_values_and_level_metadata(
    #[case] level: DeflateLevel,
    #[values(false, true)] shuffle: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.h5");
    let mut filters = Vec::new();
    if shuffle {
        filters.push(Filter::Shuffle);
    }
    filters.push(Filter::Deflate(level.get()));
    {
        let file = hdf5::File::create(&path).unwrap();
        let dataset = file
            .new_dataset::<i32>()
            .shape([4])
            .chunk([2])
            .set_filters(&filters)
            .create("data")
            .unwrap();
        dataset.write_raw(&[1, 2, 3, 4]).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), [1, 2, 3, 4]);
        file.close().unwrap();
    }
    let dataset = File::open(&path).unwrap().dataset("data").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [1, 2, 3, 4]);
    assert_eq!(
        dataset.filter_pipeline().last().unwrap().client_data,
        [u32::from(level.get())]
    );
}

#[rstest]
#[cfg(feature = "__hdf5-1.10")]
#[case(DeflateLevel::Zero)]
#[case(DeflateLevel::Six)]
#[case(DeflateLevel::Nine)]
fn c_reads_requested_deflate_levels(
    #[case] level: DeflateLevel,
    #[values(false, true)] shuffle: bool,
    #[values(false, true)] staged: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pure.h5");
    let configure = |builder: &mut DatasetBuilder| {
        builder
            .with_i32_data(&[1, 2, 3, 4])
            .with_chunks(&[2])
            .with_deflate(level);
        if shuffle {
            builder.with_shuffle();
        }
    };
    let mut builder = FileBuilder::new();
    if !staged {
        configure(builder.create_dataset("data"));
    }
    builder.write(&path).unwrap();
    if staged {
        let file = File::open_rw(&path).unwrap();
        file.root().create_dataset("data", configure).unwrap();
        file.commit().unwrap();
    }
    let dataset = hdf5::File::open(&path).unwrap().dataset("data").unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [1, 2, 3, 4]);
    let mut expected = Vec::new();
    if shuffle {
        expected.push(Filter::Shuffle);
    }
    expected.push(Filter::Deflate(level.get()));
    assert_eq!(dataset.filters(), expected);
}
