use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure_format::__private::GROUP_HEAP_MAX_MANAGED_OBJECT;
use rstest::rstest;
use test_util::btree::v2 as btree_v2;
use test_util::bytes;
use test_util::checksum;
use test_util::modern;
use test_util_hdf5::reader::ReadBackend;

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

#[rstest]
fn group_datasets_rejects_an_unsupported_heap_id_version_in_a_dense_link(
    #[values(ReadBackend::Bytes, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let mut image = modern::v2_many_links();
    let leaf = bytes::sole_signature(&image, btree_v2::LEAF_SIGNATURE);
    let heap_id_at = leaf + btree_v2::NODE_PREFIX + size_of::<u32>();
    image[heap_id_at] |= HEAP_ID_VERSION_ONE;
    let leaf_length = btree_v2::NODE_PREFIX
        + 20 * usize::from(btree_v2::LINK_NAME_RECORD_SIZE)
        + checksum::CHECKSUM;
    checksum::restamp(&mut image, leaf, leaf_length);

    let file = backend.open_image(image).unwrap();
    let error = file.root().datasets().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, FormatError::UnsupportedFractalHeapIdVersion(1));
    file.close().unwrap();
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
// "Fractal Heap" in the format specification, version 4.0, defines bits 6 and 7 as the ID version.
const HEAP_ID_VERSION_ONE: u8 = 0x40;
