use hdf5_pure_core::__private;
use hdf5_pure_core::Datatype;

#[test]
fn provider_construction_preserves_member_fields() {
    let datatype = Datatype::String {
        size: 3,
        padding: hdf5_pure_core::StringPadding::NullPad,
        charset: hdf5_pure_core::CharacterSet::Ascii,
    };
    let member = __private::compound_member("field".into(), 7, datatype.clone());
    assert_eq!(member.name, "field");
    assert_eq!(member.byte_offset, 7);
    assert_eq!(member.datatype, datatype);

    let enumeration = __private::enum_member("value".into(), vec![1, 2, 3]);
    assert_eq!(enumeration.name, "value");
    assert_eq!(enumeration.value, vec![1, 2, 3]);
}
