use core::mem;

use rstest::rstest;

use super::*;

fixed_record! {
    #[derive(Debug, Eq, PartialEq)]
    FlagCodec => FlagWire for FlagFields {
        persist: NonzeroBool,
    }
}

fixed_record! {
    #[derive(Debug, Eq, PartialEq)]
    PairCodec => PairWire for PairFields {
        little: U16Le,
        big: U32Be,
    }
}

fixed_record! {
    #[derive(Debug, Eq, PartialEq)]
    EnvelopeCodec => EnvelopeWire for EnvelopeFields {
        kind: U8,
        pair: PairCodec,
        tail: U16Le,
    }
}

fixed_record! {
    #[derive(Debug, Eq, PartialEq)]
    AddressLengthCodec<O: OffsetEncoding, L: LengthEncoding>
    => AddressLengthWire for AddressLengthFields {
        address: StoredAddressCodec<O>,
        length: LengthCodec<L>,
    }
}

fixed_record! {
    #[derive(Debug)]
    PhaseChangeCodec => PhaseChangeWire for PhaseChangeFields => PhaseChange {
        max_compact: U16Le,
        min_dense: U16Le,
    }
}

#[derive(Debug)]
struct PhaseChange {
    fields: PhaseChangeFields,
}

impl RecordValue<PhaseChangeFields> for PhaseChange {
    fn from_fields(fields: PhaseChangeFields) -> Result<Self, FormatError> {
        let PhaseChangeFields {
            max_compact,
            min_dense,
        } = fields;
        if min_dense > max_compact {
            return Err(FormatError::InvalidGroupInfoThresholds {
                max_compact,
                min_dense,
            });
        }
        Ok(Self { fields })
    }

    fn fields(&self) -> &PhaseChangeFields {
        &self.fields
    }
}

#[repr(align(8))]
struct Aligned([u8; 32]);

#[rstest]
#[case::little_u16(Primitive::U16Le, &[0x12, 0x34], 0x3412)]
#[case::big_u16(Primitive::U16Be, &[0x12, 0x34], 0x1234)]
#[case::little_u32(Primitive::U32Le, &[0x12, 0x34, 0x56, 0x78], 0x7856_3412)]
#[case::big_u32(Primitive::U32Be, &[0x12, 0x34, 0x56, 0x78], 0x1234_5678)]
#[case::little_u64(Primitive::U64Le, &[1, 2, 3, 4, 5, 6, 7, 8], 0x0807_0605_0403_0201)]
#[case::big_u64(Primitive::U64Be, &[1, 2, 3, 4, 5, 6, 7, 8], 0x0102_0304_0506_0708)]
#[case::byte(Primitive::U8, &[0xfe], 0xfe)]
fn primitive_codecs_read_and_write_the_declared_byte_order(
    #[case] primitive: Primitive,
    #[case] bytes: &[u8],
    #[case] expected: u64,
) {
    match primitive {
        Primitive::U8 => assert_primitive::<U8>(bytes, u8::try_from(expected).unwrap()),
        Primitive::U16Le => assert_primitive::<U16Le>(bytes, u16::try_from(expected).unwrap()),
        Primitive::U16Be => assert_primitive::<U16Be>(bytes, u16::try_from(expected).unwrap()),
        Primitive::U32Le => assert_primitive::<U32Le>(bytes, u32::try_from(expected).unwrap()),
        Primitive::U32Be => assert_primitive::<U32Be>(bytes, u32::try_from(expected).unwrap()),
        Primitive::U64Le => assert_primitive::<U64Le>(bytes, expected),
        Primitive::U64Be => assert_primitive::<U64Be>(bytes, expected),
    }
}

#[derive(Debug)]
enum Primitive {
    U16Be,
    U16Le,
    U32Be,
    U32Le,
    U64Be,
    U64Le,
    U8,
}

fn assert_primitive<C: FixedRecord>(bytes: &[u8], expected: C::Value)
where
    C::Value: core::fmt::Debug + PartialEq,
{
    let mut backing = Aligned([0xcc; 32]);
    backing.0[1..1 + bytes.len()].copy_from_slice(bytes);
    let mut fields = Fields::new(&backing.0, 1);
    assert_eq!(backing.0[1..].as_ptr().addr() % 8, 1);
    assert_eq!(C::read(&mut fields).unwrap(), expected);
    assert_eq!(fields.pos(), 1 + bytes.len());
    assert_eq!(fields.u8().unwrap(), 0xcc);
    let mut output = [0xcc; 32];
    assert_eq!(C::write(&mut output[1..], &expected).unwrap(), bytes.len());
    assert_eq!(&output[1..1 + bytes.len()], bytes);
    assert_eq!(output[0], 0xcc);
    assert_eq!(&output[1 + bytes.len()..], &vec![0xcc; 31 - bytes.len()]);
}

#[test]
fn named_nested_records_share_field_order_and_size() {
    let bytes = [
        0xaa, 7, 0x34, 0x12, 0x56, 0x78, 0x9a, 0xbc, 0xef, 0xcd, 0xbb,
    ];
    let expected = EnvelopeFields {
        kind: 7,
        pair: PairFields {
            little: 0x1234,
            big: 0x5678_9abc,
        },
        tail: 0xcdef,
    };
    let mut fields = Fields::new(&bytes, 1);
    assert_eq!(read_envelope(&mut fields).unwrap(), expected);
    assert_eq!(fields.pos(), 10);
    assert_eq!(fields.u8().unwrap(), 0xbb);
    assert_eq!(EnvelopeWire::SIZE, 9);
    assert_eq!(mem::size_of::<EnvelopeWire>(), 9);
    assert_eq!(mem::align_of::<EnvelopeWire>(), 1);
    assert!(mem::size_of::<EnvelopeFields>() > EnvelopeWire::SIZE);
    let mut output = [0xcc; 11];
    assert_eq!(write_envelope(&mut output[1..], &expected).unwrap(), 9);
    assert_eq!(&output[1..10], &bytes[1..10]);
    assert_eq!([output[0], output[10]], [0xcc; 2]);
}

#[inline(never)]
fn read_envelope(fields: &mut Fields<'_>) -> Result<EnvelopeFields, FormatError> {
    EnvelopeCodec::read(fields)
}

#[inline(never)]
fn write_envelope(output: &mut [u8], value: &EnvelopeFields) -> Result<usize, FormatError> {
    EnvelopeCodec::write(output, value)
}

#[rstest]
fn incomplete_records_leave_the_cursor_and_output_unchanged(
    #[values(0, 1, 2, 3, 4, 5, 6, 7, 8)] len: usize,
) {
    let bytes = [0; 10];
    let mut fields = Fields::new(&bytes[..1 + len], 1);
    assert_eq!(
        read_envelope(&mut fields).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 10,
            available: 1 + len
        }
    );
    assert_eq!(fields.pos(), 1);
    let value = EnvelopeFields {
        kind: 0,
        pair: PairFields { little: 0, big: 0 },
        tail: 0,
    };
    let mut output = [0xcc; 10];
    assert_eq!(
        write_envelope(&mut output[1..1 + len], &value).unwrap_err(),
        FormatError::UnexpectedEof {
            expected: 9,
            available: len
        }
    );
    assert_eq!(output, [0xcc; 10]);
}

#[rstest]
#[case::ordered(&[8, 0, 6, 0], 8, 6)]
#[case::equal(&[8, 0, 8, 0], 8, 8)]
#[case::disabled(&[0; 4], 0, 0)]
#[case::no_reversion(&[8, 0, 0, 0], 8, 0)]
#[case::maximum(&[0xff; 4], u16::MAX, u16::MAX)]
fn a_record_constructor_establishes_cross_field_invariants(
    #[case] bytes: &[u8],
    #[case] max_compact: u16,
    #[case] min_dense: u16,
) {
    let mut fields = Fields::new(bytes, 0);
    let value = PhaseChangeCodec::read(&mut fields).unwrap();
    assert_eq!(value.fields.max_compact, max_compact);
    assert_eq!(value.fields.min_dense, min_dense);
    assert_eq!(fields.pos(), 4);
    let mut output = [0; 4];
    assert_eq!(PhaseChangeCodec::write(&mut output, &value).unwrap(), 4);
    assert_eq!(output, bytes);
}

#[rstest]
#[case::ordered_incorrectly(&[6, 0, 8, 0, 0xaa], 6, 8)]
#[case::zero_maximum(&[0, 0, 1, 0, 0xaa], 0, 1)]
fn a_record_constructor_rejects_invalid_cross_field_values(
    #[case] bytes: &[u8],
    #[case] max_compact: u16,
    #[case] min_dense: u16,
) {
    let mut fields = Fields::new(bytes, 0);
    let error = PhaseChangeCodec::read(&mut fields).unwrap_err();
    assert_eq!(
        error,
        FormatError::InvalidGroupInfoThresholds {
            max_compact,
            min_dense
        }
    );
    assert_eq!(fields.pos(), 4);
    assert_eq!(fields.u8().unwrap(), 0xaa);
}

#[rstest]
#[case::zero(0, false, 0)]
#[case::one(1, true, 1)]
#[case::two(2, true, 1)]
#[case::all_ones(0xff, true, 1)]
fn nonzero_boolean_encoding_normalizes_the_stored_byte(
    #[case] input: u8,
    #[case] expected: bool,
    #[case] encoded: u8,
) {
    let bytes = [input];
    let mut fields = Fields::new(&bytes, 0);
    assert_eq!(
        FlagCodec::read(&mut fields).unwrap(),
        FlagFields { persist: expected }
    );
    let mut output = [0xff];
    FlagCodec::write(&mut output, &FlagFields { persist: expected }).unwrap();
    assert_eq!(output, [encoded]);
}

#[rstest]
#[case::fsm_aggr(0, FileSpaceStrategy::FsmAggr)]
#[case::page(1, FileSpaceStrategy::Page)]
#[case::aggr(2, FileSpaceStrategy::Aggr)]
#[case::none(3, FileSpaceStrategy::None)]
fn a_closed_enum_mapping_supplies_both_directions(
    #[case] code: u8,
    #[case] expected: FileSpaceStrategy,
) {
    let bytes = [code];
    let mut fields = Fields::new(&bytes, 0);
    assert_eq!(
        FileSpaceStrategyV1Codec::read(&mut fields).unwrap(),
        expected
    );
    let mut output = [0xff];
    FileSpaceStrategyV1Codec::write(&mut output, &expected).unwrap();
    assert_eq!(output, [code]);
}

#[rstest]
#[case::above_last(4)]
#[case::all_ones(0xff)]
fn a_closed_enum_mapping_rejects_unknown_tags(#[case] code: u8) {
    let bytes = [code];
    let mut fields = Fields::new(&bytes, 0);
    assert_eq!(
        FileSpaceStrategyV1Codec::read(&mut fields).unwrap_err(),
        FormatError::InvalidFileSpaceStrategy(code)
    );
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn address_and_length_encodings_are_independent(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    match offsets {
        OffsetWidth::Two => assert_length_width::<Offset2>(lengths),
        OffsetWidth::Four => assert_length_width::<Offset4>(lengths),
        OffsetWidth::Eight => assert_length_width::<Offset8>(lengths),
    }
}

fn assert_length_width<O: OffsetEncoding>(lengths: LengthWidth) {
    match lengths {
        LengthWidth::Two => assert_width_pair::<O, Length2>(),
        LengthWidth::Four => assert_width_pair::<O, Length4>(),
        LengthWidth::Eight => assert_width_pair::<O, Length8>(),
    }
}

fn assert_width_pair<O: OffsetEncoding, L: LengthEncoding>() {
    let mut bytes = vec![0xff; usize::from(O::WIDTH.get())];
    bytes.extend_from_slice(&0x1234u64.to_le_bytes()[..usize::from(L::WIDTH.get())]);
    bytes.push(0xaa);
    let mut fields = Fields::new(&bytes, 0);
    let value = AddressLengthCodec::<O, L>::read(&mut fields).unwrap();
    assert_eq!(value.address.get().get(), O::MAX);
    assert_eq!(value.length.get(), 0x1234);
    assert_eq!(fields.pos(), bytes.len() - 1);
    assert_eq!(fields.u8().unwrap(), 0xaa);
    let mut output = vec![0xcc; bytes.len()];
    assert_eq!(
        AddressLengthCodec::<O, L>::write(&mut output, &value).unwrap(),
        bytes.len() - 1
    );
    assert_eq!(&output[..bytes.len() - 1], &bytes[..bytes.len() - 1]);
    assert_eq!(output[bytes.len() - 1], 0xcc);
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn address_codecs_distinguish_raw_bits_from_undefined_addresses(#[case] width: OffsetWidth) {
    match width {
        OffsetWidth::Two => assert_address_bound::<Offset2>(),
        OffsetWidth::Four => assert_address_bound::<Offset4>(),
        OffsetWidth::Eight => assert_address_bound::<Offset8>(),
    }
}

fn assert_address_bound<O: OffsetEncoding>() {
    let bytes = vec![0xff; usize::from(O::WIDTH.get())];
    let mut fields = Fields::new(&bytes, 0);
    assert_eq!(
        StoredAddressCodec::<O>::read(&mut fields).unwrap().get(),
        StoredAddress::new(O::MAX)
    );
    let mut fields = Fields::new(&bytes, 0);
    assert_eq!(
        DecodedAddressCodec::<O>::read(&mut fields).unwrap().get(),
        DecodedStoredAddress::Undefined
    );
    let raw = BoundStoredAddress::<O>::new(StoredAddress::new(O::MAX)).unwrap();
    let mut output = vec![0; bytes.len()];
    StoredAddressCodec::<O>::write(&mut output, &raw).unwrap();
    assert_eq!(output, bytes);
    let undefined = BoundDecodedAddress::<O>::new(DecodedStoredAddress::Undefined).unwrap();
    DecodedAddressCodec::<O>::write(&mut output, &undefined).unwrap();
    assert_eq!(output, bytes);
    assert_eq!(
        BoundDecodedAddress::<O>::new(DecodedStoredAddress::Defined(StoredAddress::new(O::MAX)))
            .unwrap_err(),
        FormatError::DefinedAddressIsUndefined {
            address: O::MAX,
            offset_size: O::WIDTH.get()
        }
    );
    let zero = BoundDecodedAddress::<O>::new(DecodedStoredAddress::Defined(StoredAddress::new(0)))
        .unwrap();
    DecodedAddressCodec::<O>::write(&mut output, &zero).unwrap();
    assert_eq!(output, vec![0; bytes.len()]);
    assert_eq!(
        zero.get(),
        DecodedStoredAddress::Defined(StoredAddress::new(0))
    );
}

#[rstest]
#[case::two(OffsetWidth::Two, 0x1_0000)]
#[case::four(OffsetWidth::Four, 0x1_0000_0000)]
#[case::two_wide_undefined(OffsetWidth::Two, u64::MAX)]
#[case::four_wide_undefined(OffsetWidth::Four, u64::MAX)]
fn address_construction_rejects_values_wider_than_the_encoding(
    #[case] width: OffsetWidth,
    #[case] address: u64,
    #[values(AddressForm::Stored, AddressForm::Decoded)] form: AddressForm,
) {
    let error = match width {
        OffsetWidth::Two => form.oversized::<Offset2>(address),
        OffsetWidth::Four => form.oversized::<Offset4>(address),
        OffsetWidth::Eight => panic!("eight-byte addresses contain every u64 value"),
    };
    assert_eq!(
        error,
        FormatError::AddressTooLarge {
            address,
            offset_size: width.get()
        }
    );
}

#[derive(Clone, Copy, Debug)]
enum AddressForm {
    Decoded,
    Stored,
}

impl AddressForm {
    fn oversized<O: OffsetEncoding>(self, address: u64) -> FormatError {
        match self {
            Self::Stored => BoundStoredAddress::<O>::new(StoredAddress::new(address)).unwrap_err(),
            Self::Decoded => BoundDecodedAddress::<O>::new(DecodedStoredAddress::Defined(
                StoredAddress::new(address),
            ))
            .unwrap_err(),
        }
    }
}

#[rstest]
#[case::two(LengthWidth::Two, 0x1_0000)]
#[case::four(LengthWidth::Four, 0x1_0000_0000)]
fn length_construction_rejects_values_wider_than_the_encoding(
    #[case] width: LengthWidth,
    #[case] length: u64,
) {
    let error = match width {
        LengthWidth::Two => BoundLength::<Length2>::new(length).unwrap_err(),
        LengthWidth::Four => BoundLength::<Length4>::new(length).unwrap_err(),
        LengthWidth::Eight => panic!("eight-byte lengths contain every u64 value"),
    };
    assert_eq!(
        error,
        FormatError::LengthTooLarge {
            length,
            length_size: width.get()
        }
    );
}

#[rstest]
#[case::two_zero(LengthWidth::Two, 0, &[0; 2])]
#[case::two_maximum(LengthWidth::Two, 0xffff, &[0xff; 2])]
#[case::four_zero(LengthWidth::Four, 0, &[0; 4])]
#[case::four_maximum(LengthWidth::Four, 0xffff_ffff, &[0xff; 4])]
#[case::eight_zero(LengthWidth::Eight, 0, &[0; 8])]
#[case::eight_maximum(LengthWidth::Eight, u64::MAX, &[0xff; 8])]
fn length_construction_preserves_the_complete_encoding_range(
    #[case] width: LengthWidth,
    #[case] value: u64,
    #[case] expected: &[u8],
) {
    match width {
        LengthWidth::Two => assert_length_bound::<Length2>(value, expected),
        LengthWidth::Four => assert_length_bound::<Length4>(value, expected),
        LengthWidth::Eight => assert_length_bound::<Length8>(value, expected),
    }
}

fn assert_length_bound<L: LengthEncoding>(value: u64, expected: &[u8]) {
    let bound = BoundLength::<L>::new(value).unwrap();
    assert_eq!(bound.get(), value);
    let mut output = vec![0xaa; expected.len()];
    assert_eq!(
        LengthCodec::<L>::write(&mut output, &bound).unwrap(),
        expected.len()
    );
    assert_eq!(output, expected);
    assert_eq!(
        LengthCodec::<L>::read(&mut Fields::new(expected, 0))
            .unwrap()
            .get(),
        value
    );
}
