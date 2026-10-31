//! Complete observations of independent dense attributes across reader backends.

use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FormatError;
use rstest::rstest;
use test_util::btree_v2;
use test_util::checksum;
use test_util::dense_attributes::ChecksumUpdate;
use test_util::dense_attributes::Fixture;
use test_util::dense_attributes::NodeCorruption;
use test_util::temp;
use test_util_hdf5::dense_attributes;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::dataset(Fixture::Dataset)]
#[case::root(Fixture::Root)]
fn independent_dense_attribute_files_have_complete_contents(
    #[case] fixture: Fixture,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let path = temp::temp_path("dense_attributes.h5");
    let bytes = match fixture {
        Fixture::Dataset => test_util::dense_attributes::dense_attrs(),
        Fixture::Root => test_util::dense_attributes::dense_attrs_root(),
    };
    std::fs::write(&path, &bytes).unwrap();
    let file = backend.open(&path).unwrap();
    dense_attributes::assert_contents(&file, fixture, DatatypeByteOrder::LittleEndian);
    dense_attributes::assert_storage(&file, &bytes, fixture, DatatypeByteOrder::LittleEndian);
    drop(file);
}

#[rstest]
#[case::owner_type_with_restamped_checksum(
    NodeCorruption::ClientType(ChecksumUpdate::Recompute),
    FormatError::InvalidBTreeNodeType(11)
)]
#[case::owner_type_with_original_checksum(
    NodeCorruption::ClientType(ChecksumUpdate::Keep),
    FormatError::InvalidBTreeNodeType(11)
)]
#[case::version_with_restamped_checksum(
    NodeCorruption::Version,
    FormatError::InvalidBTreeV2Version(1)
)]
#[case::signature_with_restamped_checksum(
    NodeCorruption::Signature,
    FormatError::InvalidBTreeV2Signature
)]
fn dense_attribute_nodes_reject_structural_corruption_at_attribute_loading(
    #[case] corruption: NodeCorruption,
    #[case] expected: FormatError,
) {
    let image = corruption.build();
    let (stored, computed) = node_checksums(&image, corruption);
    match corruption {
        NodeCorruption::ClientType(ChecksumUpdate::Keep) => assert_ne!(stored, computed),
        _ => assert_eq!(stored, computed),
    }
    assert_node_error(&image, expected);
}

#[cfg(feature = "checksum")]
#[test]
fn dense_attribute_nodes_report_the_complete_checksum_mismatch() {
    let image = NodeCorruption::Checksum.build();
    let (expected, computed) = node_checksums(&image, NodeCorruption::Checksum);
    assert_eq!(expected, computed ^ 1);
    assert_node_error(&image, FormatError::ChecksumMismatch { expected, computed });
}

fn node_checksums(image: &[u8], corruption: NodeCorruption) -> (u32, u32) {
    let signature = match corruption {
        NodeCorruption::Signature => btree_v2::INTERNAL_SIGNATURE,
        _ => btree_v2::LEAF_SIGNATURE,
    };
    let leaf = test_util::bytes::sole_signature(image, signature);
    let end = leaf + DATASET_NODE_ENCODED_LEN;
    let checksum_at = end - core::mem::size_of::<u32>();
    let stored = u32::from_le_bytes(image[checksum_at..end].try_into().unwrap());
    (stored, checksum::lookup3(&image[leaf..checksum_at]))
}

fn assert_node_error(image: &[u8], expected: FormatError) {
    let path = temp::temp_path("malformed_dense_attributes.h5");
    std::fs::write(&path, image).unwrap();
    let buffered = File::open(&path).unwrap();
    let streaming = File::open_streaming(&path).unwrap();
    let buffered_dataset = buffered.dataset("data").unwrap();
    let streaming_dataset = streaming.dataset("data").unwrap();
    let buffered_error = buffered_dataset.attrs().unwrap_err();
    let Error::Format(actual) = &buffered_error else {
        panic!("expected buffered format rejection, got {buffered_error:?}");
    };
    assert_eq!(actual, &expected);
    let streaming_error = streaming_dataset.attrs().unwrap_err();
    let Error::Format(actual) = &streaming_error else {
        panic!("expected streaming format rejection, got {streaming_error:?}");
    };
    assert_eq!(actual, &expected);
}

/// The active leaf length: `6 + 50 * 17 + 4` bytes, excluding allocation padding.
const DATASET_NODE_ENCODED_LEN: usize = 860;
