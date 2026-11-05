use std::fs;

use rstest::rstest;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;

use crate::LengthWidth;
use crate::OffsetWidth;
use crate::PhysicalAddress;
use crate::SuperblockVersion;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::source::ReadSeekSource;
use crate::superblock;
use crate::width::FormatWidths;

#[rstest]
#[case::long_lengths(OffsetWidth::Two, LengthWidth::Eight, 66)]
#[case::long_offsets(OffsetWidth::Eight, LengthWidth::Two, 90)]
fn native_legacy_superblock_framing_preserves_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] framing_length: u64,
    #[values(0, 512)] base: u64,
) {
    let path = test_util::temp::temp_path("native-legacy-superblock-framing.h5");
    let native = file::libhdf5_create_earliest_with_widths(&path, offsets, lengths, base);
    native
        .new_dataset::<i32>()
        .shape([3])
        .create("data")
        .unwrap()
        .write(&[7, 11, 13])
        .unwrap();
    let root = c_object::header_address(&native, "/");
    native.close().unwrap();
    let bytes = fs::read(&path).unwrap();
    let source = ReadSeekSource::new(fs::File::open(&path).unwrap()).unwrap();
    let (located, range) = superblock::read_with_range(&source).unwrap();
    assert_eq!(located.address, PhysicalAddress::new(base));
    assert_eq!(located.superblock.version(), SuperblockVersion::Zero);
    assert_eq!(located.superblock.base_address, BaseAddress::new(base));
    assert_eq!(located.widths, FormatWidths { offsets, lengths });
    assert_eq!(located.superblock.root_group_address.get(), root);
    assert_eq!(range.start(), PhysicalAddress::new(base));
    assert_eq!(range.length(), framing_length);
    assert_eq!(
        superblock::read_with_range(bytes.as_slice()).unwrap(),
        (located, range)
    );
    drop(source);
    assert_eq!(fs::read(&path).unwrap(), bytes);
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13]
    );
    native.close().unwrap();
}
