//! Paged-index reclamation in native files.

use test_util::temp;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::paged_index;

use crate::DatatypeByteOrder;
use crate::edit;

#[test]
fn a_native_chunk_index_is_not_reclaimed_as_raw() {
    let path = temp::temp_path("native_paged_index.h5");
    Fixture::PagedIndex.write(&path);
    let (native, pure) = native_order();
    Fixture::PagedIndex.assert_native(&path, native);
    edit::test_util::assert_index_reclamation(&path, pure);
    paged_index::assert_native_after_edit(&path, native);
    file_space_oracle::assert_persisted_sections(&path);
}

fn native_order() -> (ByteOrder, DatatypeByteOrder) {
    if cfg!(target_endian = "big") {
        (ByteOrder::BigEndian, DatatypeByteOrder::BigEndian)
    } else {
        (ByteOrder::LittleEndian, DatatypeByteOrder::LittleEndian)
    }
}
