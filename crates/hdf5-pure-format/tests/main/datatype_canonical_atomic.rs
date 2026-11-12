use hdf5_pure_core::{CharacterSet, Datatype, FormatError, StringPadding};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{DatatypeMessage, OffsetWidth};
use rstest::rstest;
use test_util::datatype::Class;
use test_util::datatype::leaf::{
    self, Charset, Leaf, OpaqueTag, Order, Padding, VaxVersion, Version,
};

#[rstest]
#[case::integer(Leaf::Integer)]
#[case::unsigned_integer(Leaf::LittleInteger)]
#[case::little_float(Leaf::Float(Order::Little))]
#[case::big_float(Leaf::Float(Order::Big))]
#[case::unnormalized(Leaf::UnnormalizedFloat(Order::Little))]
#[case::implied(Leaf::ImpliedFloat(Order::Big))]
#[case::low_padding(Leaf::LowPaddedFloat(Order::Little))]
#[case::high_padding(Leaf::HighPaddedFloat(Order::Little))]
#[case::internal_padding(Leaf::InternallyPaddedFloat(Order::Little))]
#[case::time(Leaf::Time)]
#[case::little_time(Leaf::LittleTime)]
#[case::ascii_terminated(Leaf::String { padding: Padding::NullTerminate, charset: Charset::Ascii })]
#[case::ascii_null_padded(Leaf::String { padding: Padding::NullPad, charset: Charset::Ascii })]
#[case::ascii_space_padded(Leaf::String { padding: Padding::SpacePad, charset: Charset::Ascii })]
#[case::utf8_terminated(Leaf::String { padding: Padding::NullTerminate, charset: Charset::Utf8 })]
#[case::utf8_null_padded(Leaf::String { padding: Padding::NullPad, charset: Charset::Utf8 })]
#[case::utf8_space_padded(Leaf::string())]
#[case::bitfield(Leaf::Bitfield)]
#[case::little_bitfield(Leaf::LittleBitfield)]
#[case::opaque_empty(Leaf::Opaque(OpaqueTag::Empty))]
#[case::opaque_bytes(Leaf::Opaque(OpaqueTag::NonAscii))]
#[case::opaque_aligned(Leaf::Opaque(OpaqueTag::Aligned))]
#[case::opaque_maximum(Leaf::Opaque(OpaqueTag::Maximum))]
fn datatype_canonical_atomic_uses_version_one_with_complete_properties(#[case] leaf: Leaf) {
    let source = leaf.bytes(Version::Five);
    let datatype = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    assert_eq!(
        format::serialize_datatype(&datatype).unwrap(),
        leaf.bytes(Version::One)
    );
}

#[rstest]
fn datatype_canonical_atomic_uses_version_three_for_vax(
    #[values(VaxVersion::Three, VaxVersion::Four, VaxVersion::Five)] version: VaxVersion,
) {
    let source = leaf::vax_bytes(version);
    let datatype = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    assert_eq!(
        format::serialize_datatype(&datatype).unwrap(),
        leaf::vax_bytes(VaxVersion::Three)
    );
}

#[rstest]
#[case::integer(Leaf::Integer)]
#[case::float(Leaf::Float(Order::Big))]
#[case::time(Leaf::Time)]
#[case::string(Leaf::string())]
#[case::bitfield(Leaf::Bitfield)]
#[case::opaque(Leaf::Opaque(OpaqueTag::Aligned))]
fn datatype_canonical_atomic_preserves_offset_width_independence(
    #[case] leaf: Leaf,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let source = leaf.bytes(Version::Five);
    let datatype = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    assert_eq!(
        format::serialize_datatype_with_offset_width(&datatype, width).unwrap(),
        leaf.bytes(Version::One)
    );
}

#[rstest]
#[case::string(Datatype::String { size: 0, padding: StringPadding::NullPad, charset: CharacterSet::Ascii }, Class::STRING)]
#[case::opaque(Datatype::Opaque { size: 0, tag: vec![] }, Class::OPAQUE)]
fn datatype_canonical_atomic_rejects_zero_element_sizes(
    #[case] datatype: Datatype,
    #[case] class: Class,
) {
    assert_eq!(
        format::serialize_datatype(&datatype).unwrap_err(),
        FormatError::ZeroSizedDatatype { class: class.0 }
    );
}
