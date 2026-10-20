#[cfg(feature = "deflate")]
use hdf5_pure::DatasetBuilder;
use hdf5_pure::DeflateLevel;
#[cfg(feature = "deflate")]
use hdf5_pure::File;
#[cfg(feature = "deflate")]
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use rstest::rstest;

#[rstest]
#[case(10)]
#[case(u8::MAX)]
fn deflate_level_rejects_invalid_byte(#[case] level: u8) {
    assert_eq!(
        DeflateLevel::try_from(level).unwrap_err(),
        FormatError::InvalidDeflateLevel {
            level: u32::from(level)
        }
    );
}

#[rstest]
#[cfg(feature = "deflate")]
#[case(DeflateLevel::Zero)]
#[case(DeflateLevel::Six)]
#[case(DeflateLevel::Nine)]
fn deflate_preserves_typed_level(
    #[case] level: DeflateLevel,
    #[values(false, true)] shuffle: bool,
    #[values(false, true)] staged: bool,
) {
    let configure = |builder: &mut DatasetBuilder| {
        builder.with_i32_data(&[1, 2, 3, 4]).with_deflate(level);
        if shuffle {
            builder.with_shuffle();
        }
    };
    let file = if staged {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("staged.h5");
        FileBuilder::new().write(&path).unwrap();
        {
            let file = File::open_rw(&path).unwrap();
            file.root().create_dataset("data", configure).unwrap();
            file.commit().unwrap();
        }
        File::from_bytes(std::fs::read(path).unwrap()).unwrap()
    } else {
        let mut builder = FileBuilder::new();
        configure(builder.create_dataset("data"));
        File::from_bytes(builder.finish().unwrap()).unwrap()
    };
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [1, 2, 3, 4]);
    assert_eq!(
        dataset.filter_pipeline().last().unwrap().client_data,
        [u32::from(level.get())]
    );
}
