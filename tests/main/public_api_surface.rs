//! The types a caller reaches by value through the public API can also be
//! named.
//!
//! Every module in this crate is `pub(crate)`, so a type is public API only if
//! `lib.rs` re-exports it — but reading a `pub` field and calling an inherent
//! method need no name, so a type can be reachable and unwritable at once.
//! `File::superblock` hands back a [`Superblock`] and with it a
//! [`BaseAddress`], and `Error::MissingMessage` carries a [`MessageType`].
//!
//! This file is a separate crate, exactly like a consumer, which is what makes
//! it the gate for two things. A re-export dropped from `lib.rs` stops it
//! compiling. So does a *retyped* field of `Superblock`, which nothing else
//! catches: `cargo-semver-checks` ships no lint for one (`struct_pub_field_missing`
//! covers removal and rename), and it could not see these types anyway, since
//! its lints match items by importable path. That combination is how
//! `Superblock::base_address` changed from `u64` to `BaseAddress` in 0.40.0
//! with no finding and no changelog line, leaving callers a value they could
//! compare and print but not use as a number.

use hdf5_pure::CharacterSet;
use hdf5_pure::Datatype;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FilterDisposition;
use hdf5_pure::FilterId;
use hdf5_pure::FilterParameters;
use hdf5_pure::FormatError;
use hdf5_pure::LengthWidth;
use hdf5_pure::OffsetWidth;
use hdf5_pure::StoredAddress;
use hdf5_pure::StringPadding;
use hdf5_pure::SuperblockStatus;
use hdf5_pure::SuperblockStatusFlag;
use hdf5_pure::SuperblockStatusFlags;
use hdf5_pure::SuperblockVersion;
use hdf5_pure::UserblockSize;
use hdf5_pure::VariableLengthStringType;
use hdf5_pure::VariableLengthType;
use hdf5_pure::{BaseAddress, Error, File, FileBuilder, LibVer, MessageType, Superblock};
use rstest::rstest;

use test_util::temp;

#[test]
fn file_space_settings_have_one_type_across_facade_and_core() {
    let settings: FileSpaceSettings = hdf5_pure_core::FileSpaceSettings::default();
    let copied = settings;
    assert_eq!(settings, copied);
    assert_eq!(settings, Clone::clone(&settings));
}

/// A consumer's own helper over the superblock: both types in one signature,
/// which is the thing that could not be written before.
fn image_start(sb: &Superblock) -> BaseAddress {
    sb.base_address
}

/// What a consumer would use the superblock for — deciding which format a file
/// is in, since HDF5 1.8 cannot open a version 3 superblock at all — written as
/// a function over the type rather than inline at the call site.
fn format_of(sb: &Superblock) -> LibVer {
    LibVer::from_superblock_version(sb.version())
}

/// Checks public field, accessor, and setter types through consumer bindings.
fn field_types(sb: &Superblock) {
    let _: fn(StoredAddress) -> u64 = StoredAddress::get;
    let _: fn(hdf5_pure_core::StoredAddress) -> u64 = StoredAddress::get;
    let _: fn(OffsetWidth) -> StoredAddress = StoredAddress::undefined;
    let _: fn(StoredAddress, OffsetWidth) -> bool = StoredAddress::is_undefined;
    let _: fn(hdf5_pure_core::OffsetWidth) -> StoredAddress = StoredAddress::undefined;
    let _: fn(StoredAddress, hdf5_pure_core::OffsetWidth) -> bool = StoredAddress::is_undefined;
    let _: SuperblockStatus = sb.status();
    let _: hdf5_pure_core::SuperblockStatus = sb.status();
    let _: fn(&Superblock) -> SuperblockStatus = Superblock::status;
    let _: fn(&mut Superblock, SuperblockVersion) -> Result<(), FormatError> =
        Superblock::set_version;
    let _: fn(&mut Superblock, SuperblockStatusFlags) -> Result<(), FormatError> =
        Superblock::set_consistency_flags;
    let _: SuperblockVersion = sb.version();
    let _: hdf5_pure_core::SuperblockVersion = sb.version();
    let _: OffsetWidth = sb.offset_size;
    let _: LengthWidth = sb.length_size;
    let _: BaseAddress = sb.base_address;
    let _: u64 = sb.eof_address;
    let _: StoredAddress = sb.root_group_address;
    let _: Option<u16> = sb.group_leaf_node_k;
    let _: Option<u16> = sb.group_internal_node_k;
    let _: Option<u16> = sb.indexed_storage_internal_node_k;
    let _: Option<StoredAddress> = sb.free_space_address;
    let _: Option<StoredAddress> = sb.driver_info_address;
    let _: SuperblockStatusFlags = sb.consistency_flags();
    let _: hdf5_pure_core::SuperblockStatusFlags = sb.consistency_flags();
    let _: fn(SuperblockStatusFlags, SuperblockStatusFlag) -> bool =
        SuperblockStatusFlags::contains;
    let _: Option<StoredAddress> = sb.superblock_extension_address;
    let _: Option<u32> = sb.checksum;
}

#[test]
fn the_superblock_and_its_base_address_can_both_be_named() {
    let path = temp::temp_path("hdf5_pure_api_surface_superblock.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::MIN);
    b.create_dataset("alpha").with_f64_data(&[1.0]);
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    let file = File::open(&path).unwrap();
    let sb: &Superblock = file.superblock();
    field_types(sb);
    let base: BaseAddress = image_start(sb);
    assert_eq!(base.get(), 512, "the base is the userblock size");
    assert_eq!(
        sb.version(),
        SuperblockVersion::Three,
        "the default writer emits a version 3 superblock"
    );
    assert_eq!(format_of(sb), LibVer::V110);

    // Storable, not merely readable: a consumer can keep one past the file it
    // came from. Only the compiler can assert this — the binding names the type
    // and the read happens after the borrow it came from is gone.
    let kept: Superblock = sb.clone();
    drop(file);
    field_types(&kept);

    // The other half of the same question: bounds that select the 1.8 format
    // are visible through the same accessor.
    let old = temp::temp_path("hdf5_pure_api_surface_libver18.h5");
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("alpha").with_f64_data(&[1.0]);
    std::fs::write(&old, b.finish().unwrap()).unwrap();
    let file = File::open(&old).unwrap();
    assert_eq!(format_of(file.superblock()), LibVer::V18);
    assert_eq!(image_start(file.superblock()).get(), 0, "no userblock");
}

/// What a consumer does with the error: bind the message type out of the
/// variant and answer a question about it. Naming the type is what lets this be
/// a function at all rather than an arm that can only format its binding.
fn missing_message_id(e: &Error) -> Option<u16> {
    match e {
        Error::MissingMessage(t) => Some(t.to_u16()),
        _ => None,
    }
}

#[test]
fn a_header_message_type_can_be_named_where_an_error_hands_one_over() {
    let named = MessageType::from_u16(0x0008);
    assert_eq!(named, MessageType::DATA_LAYOUT);
    assert_eq!(
        missing_message_id(&Error::MissingMessage(named)),
        Some(0x0008)
    );

    const UNKNOWN_TYPE: u16 = 0x7FFF;

    let unknown = MessageType::from_u16(UNKNOWN_TYPE);
    assert_eq!(unknown.unknown_id(), Some(UNKNOWN_TYPE));
    assert_eq!(
        missing_message_id(&Error::MissingMessage(unknown)),
        Some(UNKNOWN_TYPE)
    );

    assert_eq!(missing_message_id(&Error::SwmrUnsupported), None);
}

#[rstest::rstest]
#[case(SuperblockVersion::Zero, 0, LibVer::Earliest)]
#[case(SuperblockVersion::One, 1, LibVer::Earliest)]
#[case(SuperblockVersion::Two, 2, LibVer::V18)]
#[case(SuperblockVersion::Three, 3, LibVer::V110)]
fn a_public_superblock_version_has_a_wire_value_and_a_library_bound(
    #[case] version: SuperblockVersion,
    #[case] byte: u8,
    #[case] bound: LibVer,
) {
    assert_eq!(SuperblockVersion::try_from(byte).unwrap(), version);
    assert_eq!(version.get(), byte);
    assert_eq!(LibVer::from_superblock_version(version), bound);
    assert_eq!(
        match version {
            SuperblockVersion::Zero => 0,
            SuperblockVersion::One => 1,
            SuperblockVersion::Two => 2,
            SuperblockVersion::Three => 3,
        },
        byte
    );
    let version = hdf5_pure_core::SuperblockVersion::try_from(byte).unwrap();
    assert_eq!(
        match version {
            hdf5_pure_core::SuperblockVersion::Zero => 0,
            hdf5_pure_core::SuperblockVersion::One => 1,
            hdf5_pure_core::SuperblockVersion::Two => 2,
            hdf5_pure_core::SuperblockVersion::Three => 3,
        },
        byte
    );
}

#[rstest::rstest]
#[case(4)]
#[case(u8::MAX)]
fn public_superblock_versions_reject_unsupported_bytes(#[case] byte: u8) {
    assert_eq!(
        SuperblockVersion::try_from(byte).unwrap_err(),
        FormatError::UnsupportedVersion(byte)
    );
    assert_eq!(
        hdf5_pure_core::SuperblockVersion::try_from(byte).unwrap_err(),
        hdf5_pure_core::FormatError::UnsupportedVersion(byte)
    );
}

#[rstest::rstest]
#[case(SuperblockStatusFlags::new(), 0x00, false, false)]
#[case(SuperblockStatusFlags::write_marked(), 0x01, true, false)]
#[case(SuperblockStatusFlags::swmr_write_marked(), 0x05, true, true)]
fn status_flags_have_named_constructors_and_typed_queries(
    #[case] flags: SuperblockStatusFlags,
    #[case] bits: u8,
    #[case] write: bool,
    #[case] swmr: bool,
) {
    assert_eq!(flags.bits(), u32::from(bits));
    assert_eq!(u8::try_from(flags).unwrap(), bits);
    assert_eq!(flags.contains(SuperblockStatusFlag::WriteAccess), write);
    assert_eq!(flags.contains(SuperblockStatusFlag::SwmrWriteAccess), swmr);
    assert!(!flags.is_inconsistent());
    assert_eq!(flags.unknown_bits(), 0);
}

#[test]
fn filter_metadata_types_are_public() {
    let _: fn(&hdf5_pure::Dataset) -> Result<Vec<hdf5_pure::FilterId>, Error> =
        hdf5_pure::Dataset::filters;
    let _: fn(&hdf5_pure::Dataset) -> Result<Vec<hdf5_pure::Filter>, Error> =
        hdf5_pure::Dataset::filter_pipeline;
    let _: fn(&hdf5_pure::ScaleOffsetParameters) -> hdf5_pure::ScaleOffset =
        hdf5_pure::ScaleOffsetParameters::mode;
    let id: hdf5_pure::CustomFilterId = 40000_u16.try_into().unwrap();
    let id: hdf5_pure::FilterId = hdf5_pure::FilterId::Custom(id);
    assert_eq!(id.get(), 40000);
    let _: hdf5_pure::FilterParameters =
        hdf5_pure::FilterParameters::Deflate(hdf5_pure::DeflateLevel::Six);
}

#[test]
fn numeric_datatype_payloads_and_accessors_are_public() {
    use core::num::NonZeroU32;

    use hdf5_pure::{
        BitFieldType, Datatype, DatatypeByteOrder, FixedPointLayout, FixedPointType,
        FloatingPointLayout, FloatingPointType, TimeType,
    };

    let integer = FixedPointType::new(2, DatatypeByteOrder::BigEndian, true, 3, 9).unwrap();
    let _: NonZeroU32 = integer.size();
    let layout: FixedPointLayout = integer.layout();
    assert_eq!(
        (layout.signed(), layout.bit_offset(), layout.bit_precision()),
        (true, 3, 9)
    );
    let Datatype::FixedPoint(integer) = Datatype::FixedPoint(integer) else {
        panic!("expected FixedPoint");
    };
    assert_eq!(integer.byte_order(), DatatypeByteOrder::BigEndian);
    let float = FloatingPointType::ieee754_binary64(DatatypeByteOrder::LittleEndian);
    let layout: FloatingPointLayout = float.layout();
    assert_eq!(
        (
            float.size().get(),
            layout.exponent_size(),
            layout.mantissa_size()
        ),
        (8, 11, 52)
    );
    let Datatype::FloatingPoint(float) = Datatype::FloatingPoint(float) else {
        panic!("expected FloatingPoint");
    };
    assert_eq!(float.layout(), FloatingPointLayout::IEEE754_BINARY64);
    assert_eq!(
        Datatype::BitField(BitFieldType::new(1, DatatypeByteOrder::LittleEndian, 4, 4).unwrap())
            .type_size(),
        1
    );
    assert_eq!(
        Datatype::Time(TimeType::new(8, DatatypeByteOrder::LittleEndian, 64).unwrap()).type_size(),
        8
    );
}

#[test]
fn userblock_properties_use_the_public_domain_type() {
    use hdf5_pure::FileCreateProperties;

    let _: fn(UserblockSize) -> u64 = UserblockSize::get;
    let _: fn(hdf5_pure_core::UserblockSize) -> u64 = UserblockSize::get;
    let _: fn(&mut FileBuilder, UserblockSize) -> &mut FileBuilder = FileBuilder::with_userblock;
    let _: fn(FileCreateProperties, UserblockSize) -> FileCreateProperties =
        FileCreateProperties::with_userblock;
    let properties = FileCreateProperties::new().with_userblock(UserblockSize::MIN);
    let size: UserblockSize = properties.userblock();
    assert_eq!(size, UserblockSize::MIN);
}

#[test]
fn creation_properties_use_public_file_space_settings() {
    use hdf5_pure::{FileCreateProperties, FileSpacePageSize};

    let _: fn(&mut FileBuilder, FileSpaceSettings) -> &mut FileBuilder =
        FileBuilder::with_file_space_settings;
    let _: fn(FileCreateProperties, FileSpaceSettings) -> FileCreateProperties =
        FileCreateProperties::with_file_space_settings;
    let _: fn(&FileCreateProperties) -> Option<FileSpaceSettings> =
        FileCreateProperties::file_space_settings;
    let settings = FileSpaceSettings::default().with_page_size(FileSpacePageSize::DEFAULT);
    let properties = FileCreateProperties::new().with_file_space_settings(settings);
    assert_eq!(properties.file_space_settings(), Some(settings));
}

#[cfg(feature = "zfp")]
#[test]
fn zfp_rates_are_public_checked_builder_inputs() {
    use hdf5_pure::{DatasetBuilder, ZfpRate};

    let _: fn(hdf5_pure_core::ZfpRate) -> f64 = ZfpRate::get;
    let _: fn(&mut DatasetBuilder, ZfpRate) -> &mut DatasetBuilder = DatasetBuilder::with_zfp;
    assert_eq!(ZfpRate::try_from(16.0).unwrap().get(), 16.0);
}

#[test]
fn chunk_filter_metadata_types_are_public() {
    let _: fn(&hdf5_pure::Chunk) -> hdf5_pure::FilterMask = hdf5_pure::Chunk::filter_mask;
    let _: fn(&hdf5_pure::Chunk) -> hdf5_pure::ChunkFilterView<'_> = hdf5_pure::Chunk::filters;
    let _: hdf5_pure::FilterDisposition = hdf5_pure::FilterDisposition::BypassedRaw;
    assert_eq!(hdf5_pure::FilterMask::NONE.raw_bits(), 0);
}

#[test]
#[deny(unreachable_patterns)]
fn filter_identifier_matches_require_a_consumer_fallback() {
    let id = FilterId::try_from(40_000).unwrap();
    match id {
        FilterId::Deflate
        | FilterId::Shuffle
        | FilterId::Fletcher32
        | FilterId::Szip
        | FilterId::Nbit
        | FilterId::ScaleOffset
        | FilterId::Lzf
        | FilterId::Zfp
        | FilterId::Custom(_) => assert_eq!(id.get(), 40_000),
        _ => panic!("unexpected filter identifier: {id:?}"),
    }
}

#[test]
#[deny(unreachable_patterns)]
fn filter_parameter_matches_require_a_consumer_fallback() {
    let parameters = FilterParameters::Opaque;
    match parameters {
        FilterParameters::Deflate(_)
        | FilterParameters::Shuffle(_)
        | FilterParameters::Fletcher32
        | FilterParameters::ScaleOffset(_)
        | FilterParameters::Opaque
        | FilterParameters::Pending => assert_eq!(parameters, FilterParameters::Opaque),
        _ => panic!("unexpected filter parameters: {parameters:?}"),
    }
}

#[rstest::rstest]
#[case(FilterDisposition::Applied, (true, true))]
#[case(FilterDisposition::BypassedRaw, (false, false))]
#[case(FilterDisposition::Skipped, (true, false))]
fn filter_dispositions_can_be_matched_exhaustively(
    #[case] disposition: FilterDisposition,
    #[case] expected: (bool, bool),
) {
    assert_eq!(
        match disposition {
            FilterDisposition::Applied => (true, true),
            FilterDisposition::BypassedRaw => (false, false),
            FilterDisposition::Skipped => (true, false),
        },
        expected
    );
}

#[test]
fn variable_length_kinds_and_string_metadata_are_public() {
    let sequence: hdf5_pure_core::VariableLengthType =
        hdf5_pure_format::VariableLengthType::sequence(hdf5_pure::make_u32_type());
    let Datatype::VariableLength(VariableLengthType::Sequence { element_type }) =
        Datatype::VariableLength(sequence)
    else {
        panic!("expected a variable-length sequence");
    };
    assert_eq!(*element_type, hdf5_pure::make_u32_type());

    let string: hdf5_pure_core::VariableLengthStringType =
        hdf5_pure_format::VariableLengthStringType::new(
            StringPadding::SpacePad,
            CharacterSet::Utf8,
        );
    let _: &VariableLengthStringType = &string;
    let datatype = Datatype::VariableLength(VariableLengthType::String(string));
    let hdf5_pure_format::Datatype::VariableLength(hdf5_pure_format::VariableLengthType::String(
        string,
    )) = datatype
    else {
        panic!("expected a variable-length string");
    };
    assert_eq!(string.padding(), &StringPadding::SpacePad);
    assert_eq!(string.charset(), &CharacterSet::Utf8);
    assert_eq!(string.base_type(), &hdf5_pure::make_u8_type());
}

#[rstest]
#[case(SuperblockStatus::Zero(u32::MAX), SuperblockVersion::Zero, u32::MAX)]
#[case(
    SuperblockStatus::One(0x8000_0004),
    SuperblockVersion::One,
    0x8000_0004
)]
#[case(SuperblockStatus::Two(0x84), SuperblockVersion::Two, 0x84)]
#[case(SuperblockStatus::Three(0x04), SuperblockVersion::Three, 0x04)]
fn superblock_status_preserves_version_and_observed_bits(
    #[case] status: SuperblockStatus,
    #[case] version: SuperblockVersion,
    #[case] bits: u32,
) {
    let _: hdf5_pure_core::SuperblockStatus = status;
    assert_eq!(status.version(), version);
    assert_eq!(status.flags().bits(), bits);
    assert_eq!(
        SuperblockStatus::new(version, status.flags()).unwrap(),
        status
    );
}

#[rstest]
#[case(0x100)]
#[case(0x8000_0004)]
#[case(u32::MAX)]
fn modern_superblock_status_rejects_a_wider_legacy_field(
    #[case] bits: u32,
    #[values(SuperblockVersion::Two, SuperblockVersion::Three)] version: SuperblockVersion,
) {
    let flags = SuperblockStatus::Zero(bits).flags();
    assert_eq!(
        SuperblockStatus::new(version, flags).unwrap_err(),
        FormatError::SuperblockStatusFlagsTooWide { bits }
    );
}

#[rstest]
#[case(0x100)]
#[case(0x8000_0004)]
#[case(u32::MAX)]
fn public_superblock_updates_reject_narrowing_without_changing_state(
    #[case] bits: u32,
    #[values(SuperblockVersion::Two, SuperblockVersion::Three)] version: SuperblockVersion,
) {
    let file = File::from_bytes(include_bytes!("../data/c/1.8/v1_superblock.h5").to_vec()).unwrap();
    let mut superblock = file.superblock().clone();
    superblock
        .set_consistency_flags(SuperblockStatus::One(bits).flags())
        .unwrap();
    let before = superblock.clone();
    assert_eq!(
        superblock.set_version(version).unwrap_err(),
        FormatError::SuperblockStatusFlagsTooWide { bits }
    );
    assert_eq!(superblock, before);
    superblock
        .set_consistency_flags(SuperblockStatusFlags::new())
        .unwrap();
    superblock.set_version(version).unwrap();
    let before = superblock.clone();
    assert_eq!(
        superblock
            .set_consistency_flags(SuperblockStatus::One(bits).flags())
            .unwrap_err(),
        FormatError::SuperblockStatusFlagsTooWide { bits }
    );
    assert_eq!(superblock, before);
    assert_eq!(superblock.version(), version);
    assert_eq!(superblock.consistency_flags().bits(), 0);
}
