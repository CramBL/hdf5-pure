use hdf5_pure_core::__private::{
    BaseAddressExt, FileRange, FileRegion, StoredAddressExt, SuperblockFields,
};
use hdf5_pure_core::{BaseAddress, PhysicalAddress, StoredAddress, Superblock, SuperblockStatus};
use hdf5_pure_format::__private::{
    FormatWidths, ModernSuperblockLayout, PreparedSuperblock, SuperblockCandidate,
    SuperblockCandidateRead, SuperblockLocator,
};
use hdf5_pure_format::FormatError;
use rstest::rstest;
use test_util::superblock::{VERSION_0, VERSION_1, VERSION_2, VERSION_3, v2};
use test_util::widths::Widths;

#[rstest]
#[case::version_two(SuperblockStatus::Two(FLAGS), VERSION_2)]
#[case::version_three(SuperblockStatus::Three(FLAGS), VERSION_3)]
fn writer_encodes_independent_address_and_length_widths(
    #[case] status: SuperblockStatus,
    #[case] version: u8,
    #[values(2, 4, 8)] offsets: u8,
    #[values(2, 4, 8)] lengths: u8,
) {
    let expected = ADDRESSES.fixture(offsets, lengths, version, FLAGS).build();
    let widths = FormatWidths::from_sizes(offsets, lengths).unwrap();
    let layout = ModernSuperblockLayout::new(widths);
    let prepared = layout
        .prepare(
            status,
            BaseAddress::new(ADDRESSES.base),
            ADDRESSES.extension.map(StoredAddress::new),
            PhysicalAddress::new(ADDRESSES.eof),
            StoredAddress::new(ADDRESSES.root),
        )
        .unwrap();

    assert_eq!(layout.encoded_size(), expected.len());
    assert_eq!(prepared.encoded_size(), expected.len());
    let mut output = [0xa5; PreparedSuperblock::MAX_ENCODED_SIZE];
    let written = prepared.write_into(&mut output).unwrap();
    assert_eq!(written, expected.len());
    assert_eq!(&output[..written], expected);
    assert_eq!(&output[written..], vec![0xa5; output.len() - written]);

    let mut decoded = ADDRESSES.superblock(widths, status);
    decoded.checksum = Some(stored_checksum(&expected));
    assert_eq!(parse(&output[..written]).unwrap(), decoded);
}

#[rstest]
#[case::absent(|_marker| None)]
#[case::selected_width_marker(Some)]
#[case::all_bits_set(|_marker| Some(u64::MAX))]
fn writer_encodes_each_undefined_extension_form(
    #[case] extension: fn(u64) -> Option<u64>,
    #[values(2, 4, 8)] offsets: u8,
) {
    let addresses = Addresses {
        extension: None,
        ..ADDRESSES
    };
    let expected = addresses.fixture(offsets, 4, VERSION_2, FLAGS).build();
    let widths = FormatWidths::from_sizes(offsets, 4).unwrap();
    let mut superblock = addresses.superblock(widths, SuperblockStatus::Two(FLAGS));
    let marker = undefined_marker(offsets);
    superblock.superblock_extension_address = extension(marker).map(StoredAddress::new);

    let mut output = [0; PreparedSuperblock::MAX_ENCODED_SIZE];
    let prepared = PreparedSuperblock::new(&superblock).unwrap();
    let written = prepared.write_into(&mut output).unwrap();
    assert_eq!(written, expected.len());
    assert_eq!(&output[..written], expected);
    superblock.superblock_extension_address = Some(StoredAddress::new(marker));
    superblock.checksum = Some(stored_checksum(&expected));
    assert_eq!(parse(&output[..written]).unwrap(), superblock);
}

#[rstest]
#[case::zero(|_marker| 0)]
#[case::largest_defined(|marker| marker - 1)]
fn writer_encodes_defined_address_boundaries(
    #[case] address: fn(u64) -> u64,
    #[values(2, 4, 8)] offsets: u8,
) {
    let value = address(undefined_marker(offsets));
    let addresses = Addresses {
        base: value,
        extension: Some(value),
        eof: value,
        root: value,
    };
    let expected = addresses.fixture(offsets, 8, VERSION_3, FLAGS).build();
    let widths = FormatWidths::from_sizes(offsets, 8).unwrap();
    let mut superblock = addresses.superblock(widths, SuperblockStatus::Three(FLAGS));
    let mut output = [0; PreparedSuperblock::MAX_ENCODED_SIZE];

    let written = PreparedSuperblock::new(&superblock)
        .unwrap()
        .write_into(&mut output)
        .unwrap();
    assert_eq!(written, expected.len());
    assert_eq!(&output[..written], expected);
    superblock.checksum = Some(stored_checksum(&expected));
    assert_eq!(parse(&output[..written]).unwrap(), superblock);
}

#[rstest]
#[case::legacy_zero(SuperblockStatus::Zero(u32::MAX), VERSION_0)]
#[case::legacy_one(SuperblockStatus::One(u32::MAX), VERSION_1)]
fn preparation_checks_the_version_before_addresses(
    #[case] status: SuperblockStatus,
    #[case] version: u8,
) {
    let mut superblock = ADDRESSES.superblock(FormatWidths::from_sizes(2, 2).unwrap(), status);
    superblock.base_address = BaseAddress::new(u64::MAX);
    assert_eq!(
        PreparedSuperblock::new(&superblock).unwrap_err(),
        FormatError::UnsupportedVersion(version)
    );
}

#[rstest]
#[case::base(|sb: &mut Superblock, address| sb.base_address = BaseAddress::new(address))]
#[case::end_of_file(|sb: &mut Superblock, address| sb.eof_address = address)]
#[case::root(|sb: &mut Superblock, address| sb.root_group_address = StoredAddress::new(address))]
fn preparation_rejects_undefined_required_addresses(
    #[case] set_address: fn(&mut Superblock, u64),
    #[values(2, 4, 8)] offsets: u8,
) {
    let mut superblock = ADDRESSES.superblock(
        FormatWidths::from_sizes(offsets, 2).unwrap(),
        SuperblockStatus::Two(FLAGS),
    );
    let marker = undefined_marker(offsets);
    set_address(&mut superblock, marker);
    assert_eq!(
        PreparedSuperblock::new(&superblock).unwrap_err(),
        FormatError::DefinedAddressIsUndefined {
            address: marker,
            offset_size: offsets
        }
    );
}

#[rstest]
#[case::base(|sb: &mut Superblock, address| sb.base_address = BaseAddress::new(address))]
#[case::extension(|sb: &mut Superblock, address| sb.superblock_extension_address = Some(StoredAddress::new(address)))]
#[case::end_of_file(|sb: &mut Superblock, address| sb.eof_address = address)]
#[case::root(|sb: &mut Superblock, address| sb.root_group_address = StoredAddress::new(address))]
fn preparation_rejects_addresses_too_large_for_their_width(
    #[case] set_address: fn(&mut Superblock, u64),
    #[values(2, 4)] offsets: u8,
) {
    let mut superblock = ADDRESSES.superblock(
        FormatWidths::from_sizes(offsets, 8).unwrap(),
        SuperblockStatus::Three(FLAGS),
    );
    let address = undefined_marker(offsets) + 1;
    set_address(&mut superblock, address);
    assert_eq!(
        PreparedSuperblock::new(&superblock).unwrap_err(),
        FormatError::AddressTooLarge {
            address,
            offset_size: offsets
        }
    );
}

#[rstest]
#[case::base(
    Some(u64::MAX),
    Some(0x1_0100),
    Some(0x1_0200),
    Some(0x1_0300),
    u64::MAX
)]
#[case::extension(None, Some(0x1_0100), Some(0x1_0200), Some(0x1_0300), 0x1_0100)]
#[case::end_of_file(None, None, Some(0x1_0200), Some(0x1_0300), 0x1_0200)]
fn preparation_reports_the_first_invalid_address_in_wire_order(
    #[case] base: Option<u64>,
    #[case] extension: Option<u64>,
    #[case] eof: Option<u64>,
    #[case] root: Option<u64>,
    #[case] address: u64,
) {
    let widths = FormatWidths::from_sizes(2, 8).unwrap();
    let superblock = Addresses {
        base: base.unwrap_or(ADDRESSES.base),
        extension: extension.or(ADDRESSES.extension),
        eof: eof.unwrap_or(ADDRESSES.eof),
        root: root.unwrap_or(ADDRESSES.root),
    }
    .superblock(widths, SuperblockStatus::Two(FLAGS));
    assert_eq!(
        PreparedSuperblock::new(&superblock).unwrap_err(),
        FormatError::AddressTooLarge {
            address,
            offset_size: 2
        }
    );
}

#[rstest]
#[case::two_byte_offsets(2)]
#[case::four_byte_offsets(4)]
#[case::eight_byte_offsets(8)]
fn writer_preserves_every_short_destination_and_guards_on_retry(#[case] offsets: u8) {
    let expected = ADDRESSES.fixture(offsets, 4, VERSION_3, FLAGS).build();
    let widths = FormatWidths::from_sizes(offsets, 4).unwrap();
    let superblock = ADDRESSES.superblock(widths, SuperblockStatus::Three(FLAGS));
    let prepared = PreparedSuperblock::new(&superblock).unwrap();
    let required = expected.len();
    let original = [0xa5; PreparedSuperblock::MAX_ENCODED_SIZE + 2];
    let mut output = original;
    for available in 0..required {
        assert_eq!(
            prepared.write_into(&mut output[1..1 + available]),
            Err(FormatError::UnexpectedEof {
                expected: required,
                available
            }),
            "destination length {available}"
        );
        assert_eq!(output, original);
    }
    assert_eq!(prepared.write_into(&mut output[1..]).unwrap(), required);
    assert_eq!(output[0], 0xa5);
    assert_eq!(&output[1..1 + required], expected);
    assert_eq!(output[1 + required..], original[1 + required..]);
}

#[rstest]
#[case::version_two(VERSION_2, SuperblockStatus::Two(FLAGS))]
#[case::version_three(VERSION_3, SuperblockStatus::Three(FLAGS))]
fn parser_applies_the_configured_checksum_verification(
    #[case] version: u8,
    #[case] status: SuperblockStatus,
) {
    let mut fixture = ADDRESSES.fixture(4, 2, version, FLAGS).build();
    let computed = stored_checksum(&fixture);
    *fixture.last_mut().unwrap() ^= 0xff;
    let stored = computed ^ 0xff00_0000;

    #[cfg(feature = "checksum")]
    {
        let _ = status;
        assert_eq!(
            parse(&fixture),
            Err(FormatError::ChecksumMismatch {
                expected: stored,
                computed
            })
        );
    }
    #[cfg(not(feature = "checksum"))]
    {
        let mut expected = ADDRESSES.superblock(FormatWidths::from_sizes(4, 2).unwrap(), status);
        expected.checksum = Some(stored);
        assert_eq!(parse(&fixture).unwrap(), expected);
    }
}

fn parse(bytes: &[u8]) -> Result<Superblock, FormatError> {
    let range = FileRange::new(PhysicalAddress::new(0), bytes.len() as u64)?;
    let region = FileRegion::new(range, bytes)?;
    let candidate = SuperblockLocator::new().parse(SuperblockCandidateRead::Bytes(region))?;
    let SuperblockCandidate::Present(signature) = candidate else {
        return Err(FormatError::SignatureNotFound);
    };
    signature
        .parse(region)?
        .parse(region)
        .map(|(superblock, _)| superblock)
}

fn stored_checksum(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(*bytes.last_chunk().unwrap())
}

fn undefined_marker(width: u8) -> u64 {
    u64::MAX >> (u64::BITS - 8 * u32::from(width))
}

#[derive(Clone, Copy)]
struct Addresses {
    base: u64,
    extension: Option<u64>,
    eof: u64,
    root: u64,
}

impl Addresses {
    fn fixture(self, offsets: u8, lengths: u8, version: u8, flags: u8) -> v2::Superblock {
        let fixture = v2::Superblock::new(Widths::new(usize::from(offsets), usize::from(lengths)))
            .version(version)
            .consistency_flags(flags)
            .base_address(self.base)
            .eof_address(self.eof)
            .root_header_address(self.root);
        match self.extension {
            Some(extension) => fixture.extension_address(extension),
            None => fixture,
        }
    }

    fn superblock(self, widths: FormatWidths, status: SuperblockStatus) -> Superblock {
        SuperblockFields {
            status,
            offset_size: widths.offsets,
            length_size: widths.lengths,
            base_address: BaseAddress::new(self.base),
            superblock_extension_address: self.extension.map(StoredAddress::new),
            eof_address: self.eof,
            root_group_address: StoredAddress::new(self.root),
            group_leaf_node_k: None,
            group_internal_node_k: None,
            indexed_storage_internal_node_k: None,
            free_space_address: None,
            driver_info_address: None,
            checksum: None,
        }
        .build()
    }
}

const FLAGS: u8 = 0xa5;
const ADDRESSES: Addresses = Addresses {
    base: 512,
    extension: Some(64),
    eof: 4096,
    root: 96,
};
