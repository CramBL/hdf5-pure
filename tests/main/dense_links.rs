use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure_format::__private::GROUP_HEAP_MAX_MANAGED_OBJECT;
use rstest::rstest;

#[rstest]
#[case::maximum(GROUP_HEAP_MAX_MANAGED_OBJECT, false)]
#[case::above_maximum(GROUP_HEAP_MAX_MANAGED_OBJECT + 1, true)]
fn dense_links_enforce_managed_object_limit(#[case] encoded_len: u64, #[case] rejected: bool) {
    let mut builder = compact_builder();
    let link_name = "a".repeat(usize::try_from(encoded_len).unwrap() - HARD_LINK_OVERHEAD);
    builder.create_dataset(&link_name).with_i32_data(&[7]);
    if rejected {
        let error = builder.finish().unwrap_err();
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(
            error,
            FormatError::DenseLinkObjectTooLarge {
                size: encoded_len,
                limit: GROUP_HEAP_MAX_MANAGED_OBJECT
            }
        );
    } else {
        let file = File::from_bytes(builder.finish().unwrap()).unwrap();
        assert_eq!(file.dataset(&link_name).unwrap().read_i32().unwrap(), [7]);
    }
}

#[test]
fn dense_links_reject_duplicate_names() {
    let mut builder = compact_builder();
    builder.create_dataset("data_0000_λ").with_i32_data(&[7]);
    let error = builder.finish().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, FormatError::DuplicateLinkName("data_0000_λ".into()));
}

fn compact_builder() -> FileBuilder {
    let mut builder = FileBuilder::new();
    for index in 0..MAX_COMPACT_LINKS {
        builder
            .create_dataset(&format!("data_{index:04}_λ"))
            .with_i32_data(&[i32::try_from(index).unwrap()]);
    }
    builder
}

// `H5G_CRT_GINFO_MAX_COMPACT` in `H5Gprivate.h`, HDF5 2.2.0, is the default compact link limit.
const MAX_COMPACT_LINKS: usize = 8;
// Section `subsubsec_fmt4_dataobject_hdr_msg_link`, version 4.0, defines the version, flags,
// two-byte name length, and eight-byte target address of these untracked ASCII hard links.
const HARD_LINK_OVERHEAD: usize = 2 + 2 + 8;
