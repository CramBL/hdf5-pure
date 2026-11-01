use hdf5_pure::Datatype;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use rstest::rstest;

#[rstest]
#[case::inline_dataset(false, false)]
#[case::committed_type(true, false)]
#[case::committed_use(true, true)]
fn oversized_opaque_tags_return_encoding_errors(
    #[case] committed: bool,
    #[case] committed_use: bool,
) {
    let datatype = Datatype::Opaque {
        size: 4,
        tag: vec![b'x'; 248],
    };
    let mut builder = FileBuilder::new();
    if committed {
        builder.commit_datatype("opaque", datatype.clone());
    }
    if !committed || committed_use {
        let dataset = builder.create_dataset("data");
        dataset.with_raw_data(datatype, vec![1, 2, 3, 4], 1);
        if committed_use {
            dataset.with_committed_datatype("opaque");
        }
    }
    let error = builder.finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, FormatError::OpaqueTagTooLong { length: 248 });
}

#[test]
fn staged_opaque_dataset_commit_rejects_an_oversized_tag() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("edited.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("seed").with_u8_data(&[1, 2]);
    builder.write(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.root()
            .create_dataset("invalid", |dataset| {
                dataset.with_raw_data(
                    Datatype::Opaque {
                        size: 4,
                        tag: vec![b'x'; 248],
                    },
                    vec![0; 4],
                    1,
                );
            })
            .unwrap();
        let error = file.commit().unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected an opaque encoding error, got {error:?}");
        };
        assert_eq!(error, FormatError::OpaqueTagTooLong { length: 248 });
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["seed"]);
    assert_eq!(file.dataset("seed").unwrap().read_u8().unwrap(), [1, 2]);
    file.close().unwrap();
}
