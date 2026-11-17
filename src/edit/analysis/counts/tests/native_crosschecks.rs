use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCount;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::checksum;
use test_util_hdf5::file;

use super::super::CountObservation;
use crate::File;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;

#[rstest]
#[case::version_1(true)]
#[case::version_2(false)]
fn native_count_fields_prepare_only_scalar_and_checksum_bytes(#[case] earliest: bool) {
    let path = test_util::temp::temp_path("native-count-update.h5");
    let native = if earliest {
        file::libhdf5_create_earliest(&path)
    } else {
        file::libhdf5_create_v110(&path)
    };
    let group = native.create_group("g").unwrap();
    group
        .new_attr::<i64>()
        .create("value")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    drop(group);
    native.link_hard("g", "alias").unwrap();
    assert_eq!(native.loc_info_by_name("g").unwrap().num_links, 2);
    native.close().unwrap();
    let file = File::open(&path).unwrap();
    let object = ObjectId::new(PhysicalAddress::new(
        file.group("g").unwrap().header_address().unwrap(),
    ));
    file.close().unwrap();
    let engine = edit_test_util::open_engine(&path);
    let field = match CountObservation::read(&engine, object).unwrap().count {
        ObjectReferenceCount::Version1(field) if earliest => field,
        ObjectReferenceCount::Version2(field) if !earliest => field,
        other => panic!("unexpected native count form: {other:?}"),
    };
    assert_eq!(field.value(), 2);
    let current = engine
        .image()
        .read_exact_at(
            field.chunk().start().get(),
            usize::try_from(field.chunk().length()).unwrap(),
        )
        .unwrap();
    let region = FileRegion::new(field.chunk(), &current).unwrap();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let update = {
        let (copy, control) = allocation::measure(&format!("native_count_copy_{earliest}"), || {
            std::hint::black_box(current.to_vec())
        });
        assert_eq!(copy, current);
        assert!(
            control.blocks > 0 && control.bytes >= u64::try_from(current.len()).unwrap(),
            "{control}"
        );
        let (update, measured) =
            allocation::measure(&format!("native_count_update_{earliest}"), || {
                field.update(region, 1)
            });
        let update = update.unwrap();
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(measured.bytes, 0, "{measured}");
        update
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let update = field.update(region, 1).unwrap();
    let mut output = current.clone();
    for (range, bytes) in update.writes() {
        assert!(field.chunk().contains(range));
        let start = usize::try_from(range.start().get() - field.chunk().start().get()).unwrap();
        output[start..start + bytes.len()].copy_from_slice(bytes);
    }
    let mut expected = current.clone();
    let at = usize::try_from(field.range().start().get() - field.chunk().start().get()).unwrap();
    assert_eq!(&current[at..at + 4], 2_u32.to_le_bytes());
    expected[at..at + 4].copy_from_slice(&1_u32.to_le_bytes());
    if !earliest {
        let at = expected.len() - checksum::CHECKSUM;
        let value = checksum::lookup3(&expected[..at]);
        expected[at..].copy_from_slice(&value.to_le_bytes());
    }
    assert_eq!(output, expected);
    assert_eq!(
        update.writes().map(|(_, bytes)| bytes.len()).sum::<usize>(),
        if earliest { 4 } else { 8 }
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(field.chunk().start().get(), current.len())
            .unwrap(),
        current
    );
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.loc_info_by_name("g").unwrap().num_links, 2);
    assert_eq!(
        native
            .group("g")
            .unwrap()
            .attr("value")
            .unwrap()
            .read_scalar::<i64>()
            .unwrap(),
        7
    );
    native.close().unwrap();
}
