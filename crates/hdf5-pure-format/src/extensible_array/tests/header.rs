use super::ExtensibleArrayHeader;

#[test]
fn header_serialized_size() {
    // 12 fixed + 6*8 stats + 8 address + 4 checksum = 72
    assert_eq!(ExtensibleArrayHeader::serialized_size(8, 8), 72);
    // 12 fixed + 6*4 stats + 4 address + 4 checksum = 44
    assert_eq!(ExtensibleArrayHeader::serialized_size(4, 4), 44);
}
