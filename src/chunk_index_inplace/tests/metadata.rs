//! Checks metadata rejection before allocation and the read bounds of checksum verification.

use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ExtensibleArrayDataBlock;
use rstest::rstest;
use test_util::checksum;
use test_util::extensible_array::CHECKSUM_BYTES;

use super::*;

#[rstest]
#[case::header_signature(Role::Header, Damage::Signature)]
#[case::header_version(Role::Header, Damage::Version)]
#[case::header_client(Role::Header, Damage::Client)]
#[case::header_checksum(Role::Header, Damage::Checksum)]
#[case::index_signature(Role::Index, Damage::Signature)]
#[case::index_version(Role::Index, Damage::Version)]
#[case::index_client(Role::Index, Damage::Client)]
#[case::index_owner(Role::Index, Damage::Owner)]
#[case::index_checksum(Role::Index, Damage::Checksum)]
#[case::secondary_signature(Role::Secondary, Damage::Signature)]
#[case::secondary_version(Role::Secondary, Damage::Version)]
#[case::secondary_client(Role::Secondary, Damage::Client)]
#[case::secondary_owner(Role::Secondary, Damage::Owner)]
#[case::secondary_offset(Role::Secondary, Damage::Offset)]
#[case::secondary_checksum(Role::Secondary, Damage::Checksum)]
fn allocated_index_acceptance_rejects_invalid_framing_and_checksums(
    #[case] role: Role,
    #[case] damage: Damage,
) {
    let mut fixture = Fixture::new(role);
    let expected = fixture.damage(damage);
    let before = fixture.store.data.clone();
    fixture.store.reset_counters();
    let error = match Located::locate_at(
        &fixture.store,
        dataset_address(&fixture.store),
        Error::AppendUnsupported,
    ) {
        Ok(_) => panic!("invalid {role:?}/{damage:?} was accepted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, Error::Format(actual) if actual == &expected),
        "{error:?}; expected {expected:?}"
    );
    fixture.assert_unchanged(&before);
}

#[rstest]
#[case::direct_signature(Role::Direct, Damage::Signature)]
#[case::direct_version(Role::Direct, Damage::Version)]
#[case::direct_client(Role::Direct, Damage::Client)]
#[case::direct_owner(Role::Direct, Damage::Owner)]
#[case::direct_offset(Role::Direct, Damage::Offset)]
#[case::direct_checksum(Role::Direct, Damage::Checksum)]
#[case::indirect_signature(Role::Indirect, Damage::Signature)]
#[case::indirect_owner(Role::Indirect, Damage::Owner)]
#[case::indirect_offset(Role::Indirect, Damage::Offset)]
#[case::indirect_checksum(Role::Indirect, Damage::Checksum)]
#[case::paged_signature(Role::PagedPrefix, Damage::Signature)]
#[case::paged_owner(Role::PagedPrefix, Damage::Owner)]
#[case::paged_offset(Role::PagedPrefix, Damage::Offset)]
#[case::paged_checksum(Role::PagedPrefix, Damage::Checksum)]
#[case::page_checksum(Role::Page, Damage::Checksum)]
fn append_preflight_rejects_invalid_target_metadata_before_allocation(
    #[case] role: Role,
    #[case] damage: Damage,
) {
    let mut fixture = Fixture::new(role);
    let expected = fixture.damage(damage);
    let before = fixture.store.data.clone();
    let (located, datatype) = locate(&fixture.store);
    fixture.store.reset_counters();
    let error = match plan(&fixture.store, &located, &datatype) {
        Ok(_) => panic!("invalid target metadata was accepted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, Error::Format(actual) if actual == &expected),
        "{error:?}; expected {expected:?}"
    );
    fixture.assert_unchanged(&before);
    let first = located.read_element(&fixture.store, 0).unwrap().unwrap();
    assert_eq!(
        fixture.store.read_exact_at(first.addr.get(), 4).unwrap(),
        0i32.to_le_bytes()
    );
}

#[rstest]
#[case::index(Role::Index)]
#[case::secondary(Role::Secondary)]
#[case::data_block(Role::Direct)]
fn complete_index_allocations_must_fit_the_published_extent(#[case] role: Role) {
    let mut fixture = Fixture::new(role);
    let complete = fixture.complete;
    fixture.store.superblock.eof_address = complete.end().get() - 1;
    let error = match Located::locate_at(
        &fixture.store,
        dataset_address(&fixture.store),
        Error::AppendUnsupported,
    ) {
        Ok(_) => panic!("an allocation beyond EOF was accepted"),
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::InvalidSpaceOwnershipExtent {
        address, length, previous_end: 0, logical_size,
    } if address == complete.start().get() && length == complete.length() && logical_size == complete.end().get() - 1),
        "{error:?}"
    );
}

#[rstest]
#[case::index(Role::Index)]
#[case::secondary(Role::Secondary)]
#[case::data_block(Role::Direct)]
fn complete_index_allocations_must_fit_the_physical_source(#[case] role: Role) {
    let mut fixture = Fixture::new(role);
    let complete = fixture.complete;
    let available = complete.end().get() - 1;
    fixture
        .store
        .data
        .truncate(usize::try_from(available).unwrap());
    let error = match Located::locate_at(&fixture.store, fixture.dataset, Error::AppendUnsupported)
    {
        Ok(_) => panic!("a truncated index allocation was accepted"),
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::Format(FormatError::UnexpectedSourceEof {
        expected, available: actual,
    }) if expected == complete.end().get() && actual == available),
        "{error:?}"
    );
}

#[test]
fn secondary_block_verification_rejects_checksum_damage_after_prefix_parsing() {
    let mut fixture = Fixture::new(Role::Secondary);
    let array =
        super::super::metadata::read_header(&fixture.store, fixture.located.ea_addr).unwrap();
    let bytes = super::super::metadata::IndexBlockBytes::read(&fixture.store, &array).unwrap();
    let index = bytes.parse().unwrap();
    let framing = index.super_blocks().next().unwrap().unwrap().framing();
    let expected = fixture.damage(Damage::Checksum);
    let range = framing.prefix_range().unwrap();
    framing
        .parse_prefix(region(&fixture.store.data, range))
        .unwrap();
    let error = super::super::metadata::check_block(&fixture.store, framing).unwrap_err();
    assert!(
        matches!(&error, Error::Format(actual) if actual == &expected),
        "{error:?}; expected {expected:?}"
    );
}

#[test]
fn large_secondary_blocks_are_verified_in_fixed_windows() {
    let mut store = WindowProbeStore::open(build_unlimited(3, 1));
    let (located, _) = locate(&store);
    let header = usize::try_from(located.ea_addr.get()).unwrap();
    let old_index = usize::try_from(located.index_block_addr.get()).unwrap();
    let index_address = store.data.len() as u64;
    let index_length = 14 + 4 * 8 + (6 + 57) * 8 + CHECKSUM_BYTES as u64;
    let (index, secondary) = test_util::extensible_array::large_secondary_blocks(
        located.ea_addr.get(),
        index_address,
        &store.data[old_index + 14..old_index + 46],
    )
    .unwrap();
    assert_eq!(index.len() as u64, index_length);
    let secondary_length = secondary.len();
    assert!(secondary_length > 500_000);
    store.data.extend_from_slice(&index);
    store.data.extend_from_slice(&secondary);
    store.data[header + 7] = 64;
    store.data[header + 60..header + 68].copy_from_slice(&index_address.to_le_bytes());
    checksum::restamp(&mut store.data, header, 72);
    store.superblock.eof_address = store.data.len() as u64;
    store.reset_counters();
    let (located, _) = locate(&store);
    assert_eq!(located.allocation_stats.unwrap().nsuper_blks, 1);
    assert_eq!(located.allocation_stats.unwrap().ndata_blks, 0);
    assert_eq!(
        located.allocation_stats.unwrap().super_blk_size,
        secondary_length as u64
    );
    assert!(store.total_read.get() >= secondary_length);
    assert!(store.total_read.get() < secondary_length * 2);
    assert!(store.max_read.get() <= 1024);
    let end = store.data.len();
    store.data[end - 1] ^= 1;
    store.reset_counters();
    let error = match Located::locate_at(&store, dataset_address(&store), Error::AppendUnsupported)
    {
        Ok(_) => panic!("a corrupt large secondary block was accepted"),
        Err(error) => error,
    };
    let expected = test_util::bytes::u32_at(&store.data, end - CHECKSUM_BYTES);
    let computed = test_util::bytes::u32_at(&secondary, secondary_length - CHECKSUM_BYTES);
    assert!(
        matches!(error, Error::Format(FormatError::ChecksumMismatch { expected: actual, computed: actual_computed }) if actual == expected && actual_computed == computed),
        "{error:?}"
    );
    assert!(store.total_read.get() >= secondary_length);
    assert!(store.max_read.get() <= 1024);
}

#[derive(Clone, Copy, Debug)]
enum Role {
    Direct,
    Header,
    Index,
    Indirect,
    Page,
    PagedPrefix,
    Secondary,
}

#[derive(Clone, Copy, Debug)]
enum Damage {
    Checksum,
    Client,
    Offset,
    Owner,
    Signature,
    Version,
}

struct Fixture {
    store: WindowProbeStore,
    located: Located,
    dataset: u64,
    role: Role,
    parse: FileRange,
    complete: FileRange,
    block_offset: Option<u64>,
}

impl Fixture {
    fn new(role: Role) -> Self {
        let elements = match role {
            Role::Header | Role::Index => 3,
            Role::Direct => 5,
            Role::Secondary | Role::Indirect => 309,
            Role::PagedPrefix | Role::Page => 131_061,
        };
        let store = WindowProbeStore::open(build_unlimited(elements, 1));
        let dataset = dataset_address(&store);
        let (located, _) = locate(&store);
        let array = super::super::metadata::read_header(&store, located.ea_addr).unwrap();
        let bytes = super::super::metadata::IndexBlockBytes::read(&store, &array).unwrap();
        let index = bytes.parse().unwrap();
        let (parse, complete, block_offset) = match role {
            Role::Header => (array.range(), array.range(), None),
            Role::Index => {
                let range = array.index_block().unwrap().range();
                (range, range, None)
            }
            Role::Direct => {
                let reference = index.data_blocks().next().unwrap().unwrap();
                (
                    reference.parse_range(),
                    reference.range(),
                    Some(reference.slots().unwrap().start - located.idx_blk_elmts),
                )
            }
            Role::Secondary | Role::Indirect | Role::PagedPrefix | Role::Page => {
                let slot = if matches!(role, Role::PagedPrefix | Role::Page) {
                    131_060
                } else {
                    308
                };
                let reference = index
                    .super_blocks()
                    .map(Result::unwrap)
                    .find(|reference| reference.slots().unwrap().contains(&slot))
                    .unwrap();
                if matches!(role, Role::Secondary) {
                    (
                        reference.parse_range(),
                        reference.parse_range(),
                        Some(reference.slots().unwrap().start - located.idx_blk_elmts),
                    )
                } else {
                    let block = reference
                        .parse(region(&store.data, reference.parse_range()))
                        .unwrap();
                    let reference = block
                        .data_blocks()
                        .map(Result::unwrap)
                        .find(|reference| reference.slots().unwrap().contains(&slot))
                        .unwrap();
                    if matches!(role, Role::Page) {
                        let ExtensibleArrayDataBlock::Pages(pages) = reference
                            .parse(region(&store.data, reference.parse_range()))
                            .unwrap()
                        else {
                            panic!("expected a paged block")
                        };
                        let range = pages.pages().next().unwrap().unwrap().parse_range();
                        (range, reference.range(), None)
                    } else {
                        (
                            reference.parse_range(),
                            reference.range(),
                            Some(reference.slots().unwrap().start - located.idx_blk_elmts),
                        )
                    }
                }
            }
        };
        Self {
            store,
            located,
            dataset,
            role,
            parse,
            complete,
            block_offset,
        }
    }

    fn damage(&mut self, damage: Damage) -> FormatError {
        let Self {
            store,
            located,
            dataset: _,
            role,
            parse,
            complete: _,
            block_offset,
        } = self;
        let at = usize::try_from(parse.start().get()).unwrap();
        let length = usize::try_from(parse.length()).unwrap();
        let data = &mut store.data;
        let expected = match damage {
            Damage::Checksum => {
                let computed = test_util::bytes::u32_at(data, at + length - CHECKSUM_BYTES);
                data[at + length - 1] ^= 1;
                return FormatError::ChecksumMismatch {
                    expected: test_util::bytes::u32_at(data, at + length - CHECKSUM_BYTES),
                    computed,
                };
            }
            Damage::Signature => {
                data[at] = 0;
                FormatError::InvalidExtensibleArraySignature
            }
            Damage::Version => {
                data[at + 4] = 1;
                FormatError::InvalidExtensibleArrayVersion(1)
            }
            Damage::Client => {
                if matches!(*role, Role::Header) {
                    data[at + 5] = 2;
                    FormatError::InvalidExtensibleArrayClient(2)
                } else {
                    data[at + 5] = 1;
                    FormatError::InvalidExtensibleArrayBlockClient {
                        expected: 0,
                        actual: 1,
                    }
                }
            }
            Damage::Owner => {
                data[at + 6..at + 14].fill(0);
                FormatError::InvalidExtensibleArrayBlockOwner {
                    expected: located.ea_addr.get(),
                    actual: 0,
                }
            }
            Damage::Offset => {
                let width = located.ea_bits.block_offset_width().bytes();
                data[at + 14..at + 14 + width].fill(0xff);
                let mut actual = [0u8; 8];
                actual[..width].fill(0xff);
                FormatError::InvalidExtensibleArrayBlockOffset {
                    expected: block_offset.unwrap(),
                    actual: u64::from_le_bytes(actual),
                }
            }
        };
        checksum::restamp(data, at, length);
        expected
    }

    fn assert_unchanged(&self, before: &[u8]) {
        assert_eq!(self.store.data, before);
        assert!(self.store.allocations.borrow().is_empty());
        assert_eq!(self.store.appends.get(), 0);
        assert_eq!(self.store.superblock_patches.get(), 0);
        assert_eq!(self.store.syncs.get(), 0);
        assert!(self.store.retired.is_empty());
        assert!(self.store.max_read.get() <= 8192);
    }
}

fn plan(
    store: &WindowProbeStore,
    located: &Located,
    datatype: &Datatype,
) -> Result<AppendPlan, Error> {
    plan_ea_append(
        store,
        located,
        datatype,
        &[1],
        located.elem_bytes,
        None,
        true,
        LossyTailReencoding::Reject,
        &1i32.to_le_bytes(),
        1,
        FillPattern::ZERO,
    )
}

fn region(bytes: &[u8], range: FileRange) -> FileRegion<'_> {
    FileRegion::new(
        range,
        &bytes[usize::try_from(range.start().get()).unwrap()
            ..usize::try_from(range.end().get()).unwrap()],
    )
    .unwrap()
}
