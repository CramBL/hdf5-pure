use rstest::rstest;
use test_util::userblock::Userblock;
use test_util_hdf5::paged;

use super::file_space;
use crate::AttrValue;
use crate::File;
use crate::FreeSectionCoordinates;
use crate::reader::test_util as reader_test_util;

#[rstest]
#[case::before_exact_fill(512, 0, 1, 17, 512)]
#[case::exact_fill(512, 0, 1, 18, 512)]
#[case::after_exact_fill(512, 0, 1, 19, 512)]
#[case::metadata_only(512, 0, 1, 18, 0)]
#[case::raw_manager_fills_metadata(512, 0, 1, 18, 511)]
#[case::large_manager_fills_metadata(512, 0, 1, 18, 513)]
#[case::raw_and_metadata_managers_fill_page(512, 0, 1, 412, 511)]
#[case::large_and_metadata_managers_fill_page(512, 0, 1, 412, 513)]
#[case::userblock(512, 4_096, 1, 18, 512)]
#[case::threshold(512, 0, 4_097, 18, 512)]
#[case::default_page(4_096, 4_096, 1, 3_601, 4_096)]
#[case::larger_page(8_192, 8_192, 1, 7_697, 8_192)]
fn created_page_metadata_tail_is_owned(
    #[case] page: u64,
    #[case] base: u64,
    #[case] threshold: u64,
    #[case] attribute_len: usize,
    #[case] raw_len: usize,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exact_fill.h5");
    paged::write_metadata_tail(&path, page, base, threshold, attribute_len, raw_len);
    let userblock = Userblock::read(&path, usize::try_from(base).unwrap());
    paged::assert_consistent(&path, page);
    if raw_len != 0 {
        paged::assert_pages_homogeneous(&path, page, &["d"]);
    }
    let exact_fill = [(512, 18), (4_096, 3_601), (8_192, 7_697)].contains(&(page, attribute_len))
        && u64::try_from(raw_len).unwrap() == page;
    if exact_fill || (page == 512 && attribute_len == 412 && (raw_len == 511 || raw_len == 513)) {
        assert_eq!(file_space::serialized_manager_padding(&path), 1);
    }
    if attribute_len == 18 && (raw_len == 511 || raw_len == 513) {
        assert_eq!(
            File::open(&path)
                .unwrap()
                .file_space_info()
                .unwrap()
                .unwrap()
                .manager_addrs
                .first(),
            Some(&crate::DecodedStoredAddress::Undefined),
        );
    }
    let file = File::open_rw(&path).unwrap();
    let initial = file.reusable_free_sections().unwrap();
    if exact_fill {
        test_util_hdf5::assert_free_sections!(
            &initial,
            &[(base + page + 1, page - 1)],
            FreeSectionCoordinates::Absolute
        );
    }
    let unclassified = if raw_len == 513 {
        vec![(file_space::raw_extents(&file, "d")[0].end, 511)]
    } else {
        Vec::new()
    };
    assert_eq!(
        file.dataset("d").unwrap().read_u8().unwrap(),
        vec![7; raw_len]
    );
    assert_eq!(
        file.root().attrs().unwrap()["fill"],
        AttrValue::U8Array(vec![0; attribute_len])
    );
    file_space::assert_space_ownership(&file, &unclassified);
    file.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.reusable_free_sections().unwrap(), initial);
    file.root()
        .create_dataset("added", |builder| {
            builder.with_u8_data(&[9; 32]);
        })
        .unwrap();
    file.commit().unwrap();
    if exact_fill && threshold == 1 {
        assert!(
            reader_test_util::ownership(&file)
                .unwrap()
                .live_metadata_space
                .iter()
                .any(|&(at, len)| {
                    initial
                        .sections()
                        .iter()
                        .any(|section| section.address() <= at && at + len <= section.end_address())
                })
        );
    }
    let unclassified = if threshold > page {
        vec![(48, 136), (255, 258)]
    } else {
        unclassified
    };
    file_space::assert_space_ownership(&file, &unclassified);
    let reusable = file.reusable_free_sections().unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_u8().unwrap(),
        vec![7; raw_len]
    );
    assert_eq!(
        file.dataset("added").unwrap().read_u8().unwrap(),
        vec![9; 32]
    );
    file.close().unwrap();
    let reopened = File::open_rw(&path).unwrap();
    assert_eq!(reopened.reusable_free_sections().unwrap(), reusable);
    assert_eq!(
        reopened.dataset("d").unwrap().read_u8().unwrap(),
        vec![7; raw_len]
    );
    assert_eq!(
        reopened.dataset("added").unwrap().read_u8().unwrap(),
        vec![9; 32]
    );
    file_space::assert_space_ownership(&reopened, &unclassified);
    reopened.close().unwrap();
    paged::assert_consistent(&path, page);
    if raw_len != 0 {
        paged::assert_pages_homogeneous(&path, page, &["d", "added"]);
    }
    userblock.assert_unchanged(&path);
}
