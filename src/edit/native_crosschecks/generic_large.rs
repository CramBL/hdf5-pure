//! Generic-large fragments and continuation tails in native files.

use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use rstest::rstest;
use test_util::temp;
use test_util_hdf5::native_fixtures::Fixture;
use test_util_hdf5::numeric::ByteOrder;

use crate::FileLocking;
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::edit;
use crate::edit::WriteEngine;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;

#[rstest]
fn native_generic_large_fragments_are_unavailable(
    #[values(PersistedFreeSpacePolicy::Tolerant, PersistedFreeSpacePolicy::Strict)]
    policy: PersistedFreeSpacePolicy,
) {
    let path = temp::temp_path("native_generic_large.h5");
    Fixture::GenericLarge.write(&path);
    Fixture::GenericLarge.assert_native(&path, native_order());
    edit::test_util::assert_generic_large_sections(&path, policy);
}

#[test]
fn native_generic_large_reclamation_keeps_partial_tails_unavailable() {
    let path = temp::temp_path("native_generic_large_tail.h5");
    Fixture::GenericLarge.write(&path);
    Fixture::GenericLarge.assert_native(&path, native_order());
    let file = File::open(&path).unwrap();
    let address = file.dataset("d").unwrap().header_address().unwrap();
    drop(file);
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let page = engine.space.page_size().unwrap().get();
    let spans = engine.oh_chunk_spans(address).unwrap();
    let large = spans
        .iter()
        .filter(|(_, length)| *length > page)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(large.len(), 1);
    let (start, length) = large[0];
    assert_eq!(start % page, 0);
    let tail_start = start + length / page * page;
    let tail_end = start + length;
    assert!(tail_start < tail_end);
    let tail = Extent::new(
        engine
            .addresses()
            .relative(PhysicalAddress::new(tail_start))
            .unwrap(),
        tail_end - tail_start,
    )
    .unwrap();
    let padding = Extent::new(
        engine
            .addresses()
            .relative(PhysicalAddress::new(tail_end))
            .unwrap(),
        page - tail_end % page,
    )
    .unwrap();
    assert_eq!(
        engine.object_header_free_spans(&large).unwrap(),
        [(start, length, FreeClass::Dead)]
    );
    engine.delete(&ObjectPathBuf::parse("/d")).unwrap();
    engine.commit().unwrap();
    engine.assert_generic_large_tails(tail, padding);
    drop(engine);
    let engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.assert_generic_large_tails(tail, padding);
    drop(engine);
    let file = test_util_hdf5::file::libhdf5_open(&path);
    assert!(file.member_names().unwrap().is_empty());
    assert!(file.attr_names().unwrap().is_empty());
    file.close().unwrap();
}

impl WriteEngine {
    fn assert_generic_large_tails(&self, tail: Extent, padding: Extent) {
        let owned = self.space_ownership().unwrap();
        assert!(owned.analysis_complete);
        for extent in [tail, padding] {
            assert_eq!(
                test_util::range::covered_len(
                    &owned.unclassified_space,
                    &(self.addresses().resolve(extent.start()).unwrap().get()
                        ..self.addresses().resolve(extent.end()).unwrap().get())
                ),
                extent.len()
            );
            assert!(
                self.space
                    .reusable_sections()
                    .all(|free| !free.range().overlaps(extent.range()))
            );
        }
        assert!(
            self.space
                .paged_sections()
                .unwrap()
                .unclassified()
                .contains(&padding)
        );
    }
}

fn native_order() -> ByteOrder {
    if cfg!(target_endian = "big") {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    }
}
