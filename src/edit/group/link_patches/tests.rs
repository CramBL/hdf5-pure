use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::link_message::CharacterSet;
use test_util::link_message::HardLink;
use test_util::link_message::SymbolicLink;
use test_util::object_header::v2::HeaderFlags;
use test_util::widths::Widths;

use super::LinkPatches;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::group;
use crate::edit::object_header::OhRegion;
use crate::error::{Error, FormatError};
use crate::message_type::MessageType;
use crate::object_path::LinkNameBuf;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

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
fn link_batches_keep_region_widths_and_reject_undefined_targets(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(HeaderFlags::default(), HeaderFlags::TRACKS_CREATION_ORDER)] flags: HeaderFlags,
) {
    let widths = FormatWidths { offsets, lengths };
    let wire = Widths::new(usize::from(offsets.get()), usize::from(lengths.get()));
    let mut region = OhRegion::empty(
        widths,
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
    );
    region.push(MessageType::LINK, &HardLink::new("changed", 7).build(wire));
    let mut expected = region.with_bytes(Vec::new());
    expected.push(MessageType::LINK, &HardLink::new("changed", 29).build(wire));
    let mut patches = LinkPatches::new(&mut region, ["changed", "changed"]);
    patches.patch_next(StoredAddress::new(29)).unwrap();
    let address = StoredAddress::undefined(offsets);
    let error = patches.patch_next(address).unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::DefinedAddressIsUndefined { address: actual, offset_size }) if actual == address.get() && offset_size == offsets.get()),
        "{error:?}"
    );
    drop(patches);
    assert_eq!(&region[..], &expected[..]);
    assert_eq!(region.widths(), widths);
    let fresh = group::fresh_group_region(widths).unwrap();
    let (_, _, info) = group::find_link_info(&fresh).unwrap().unwrap();
    assert_eq!(info.fractal_heap_address, None);
    assert_eq!(info.btree_name_index_address, None);
    assert_eq!(fresh.widths(), widths);
}

#[rstest]
fn link_batches_preserve_other_fields_and_repeated_request_order(
    #[values(HeaderFlags::default(), HeaderFlags::TRACKS_CREATION_ORDER)] flags: HeaderFlags,
) {
    let mut region = mixed_region(flags, 11);
    let expected = mixed_region(flags, 29);
    let mut patches = LinkPatches::new(&mut region, ["changed", "changed"]);
    assert_eq!(patches.visited, 4);
    patches.patch_next(StoredAddress::new(19)).unwrap();
    patches.patch_next(StoredAddress::new(29)).unwrap();
    drop(patches);
    assert_eq!(*region, *expected);
    assert_eq!(region.layout(), expected.layout());
}

#[rstest]
#[case::soft(SymbolicLink::soft("changed", "/target").build(), "a group on the edited path is reached by a soft/external link")]
#[case::external(SymbolicLink::external("changed", "file.h5", "/target").build(), "a group on the edited path is reached by a soft/external link")]
#[case::truncated({ let mut body = HardLink::new("changed", 7).build(Widths::EIGHT); body.pop(); body }, "expected child link not found in parent group")]
#[case::trailing({ let mut body = HardLink::new("changed", 7).build(Widths::EIGHT); body.push(0); body }, "a link message holds bytes past its hard link address")]
#[case::invalid_soft(SymbolicLink::soft_bytes("changed", &[0xff]).build(), "expected child link not found in parent group")]
#[case::unknown_type({ let mut body = SymbolicLink::soft("changed", "/target").build(); body[2] = 65; body }, "expected child link not found in parent group")]
#[case::missing(HardLink::new("other", 7).build(Widths::EIGHT), "expected child link not found in parent group")]
fn link_batches_report_existing_target_errors_at_the_requested_operation(
    #[case] body: Vec<u8>,
    #[case] reason: &str,
    #[values(false, true)] batch: bool,
) {
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    region.push(MessageType::LINK, &body);
    region.push(
        MessageType::LINK,
        &HardLink::new("good", 1).build(Widths::EIGHT),
    );
    let names = if batch {
        vec!["good", "changed"]
    } else {
        vec!["changed"]
    };
    let mut patches = LinkPatches::new(&mut region, names);
    if batch {
        patches.patch_next(StoredAddress::new(23)).unwrap();
    }
    let error = patches.patch_next(StoredAddress::new(29)).unwrap_err();
    assert!(
        matches!(error, Error::EditUnsupported(actual) if actual == reason),
        "{error:?}"
    );
}

#[test]
fn link_batches_defer_later_framing_errors_until_the_requested_operation() {
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    region.push(
        MessageType::LINK,
        &HardLink::new("good", 1).build(Widths::EIGHT),
    );
    region.push_bytes(&[6, 0xff, 0xff, 0]);
    let mut patches = LinkPatches::new(&mut region, ["good", "missing"]);
    patches.patch_next(StoredAddress::new(23)).unwrap();
    let error = patches.patch_next(StoredAddress::new(29)).unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("malformed object header message")
        ),
        "{error:?}"
    );
}

#[test]
fn link_batches_resolve_after_deletion_and_preserve_fields_across_appends() {
    let mut region = mixed_region(HeaderFlags::default(), 11);
    region = group::remove_links_from_region(&region, [&LinkNameBuf::new("untouched").unwrap()])
        .unwrap();
    let mut expected = mixed_region(HeaderFlags::default(), 31);
    expected =
        group::remove_links_from_region(&expected, [&LinkNameBuf::new("untouched").unwrap()])
            .unwrap();
    let appended = region.layout().record(
        MessageType::LINK,
        &HardLink::new("added", 37).build(Widths::EIGHT),
    );
    expected.push_bytes(&appended);
    let mut patches = LinkPatches::new(&mut region, ["changed", "changed"]);
    patches.push_bytes(&appended);
    patches.patch_next(StoredAddress::new(23)).unwrap();
    patches.patch_next(StoredAddress::new(31)).unwrap();
    drop(patches);
    assert_eq!(*region, *expected);
}

#[test]
fn link_batches_match_lossy_names_and_preserve_symbolic_targets() {
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    region.push(
        MessageType::LINK,
        &SymbolicLink::external("skipped", "file.h5", "/target").build(),
    );
    region.push(
        MessageType::LINK,
        &HardLink::from_bytes(b"a\xff\xfez", 1).build(Widths::EIGHT),
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (_, control) = allocation::measure("link_patch_lossy_control", || vec![0u8; 64]);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert!(control.blocks >= 1, "{control}");
    let operation = || {
        LinkPatches::new(&mut region, ["a\u{fffd}\u{fffd}z"])
            .patch_next(StoredAddress::new(17))
            .unwrap();
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let ((), measured) = allocation::measure("link_patch_lossy_single", operation);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut operation = operation;
        operation();
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert_eq!(measured.blocks, 0, "{measured}");
    let mut expected = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    expected.push(
        MessageType::LINK,
        &SymbolicLink::external("skipped", "file.h5", "/target").build(),
    );
    expected.push(
        MessageType::LINK,
        &HardLink::from_bytes(b"a\xff\xfez", 17).build(Widths::EIGHT),
    );
    assert_eq!(*region, *expected);
}

#[rstest]
#[case::empty("link_patch_empty", 0)]
#[case::single("link_patch_single", 1)]
fn empty_and_single_link_patch_plans_preserve_records_and_bound_visits(
    #[case] region_name: &str,
    #[case] count: usize,
) {
    let mut region = mixed_region(HeaderFlags::default(), 11);
    let names = ["changed"];
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (_, control) = allocation::measure(&format!("{region_name}_control"), || vec![0u8; 64]);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert!(control.blocks >= 1, "{control}");
    let operation = || {
        let mut patches = LinkPatches::new(&mut region, names[..count].iter().copied());
        if count == 1 {
            patches.patch_next(StoredAddress::new(29)).unwrap();
        }
        patches.visited
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (visited, measured) = allocation::measure(region_name, operation);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let visited = {
        let mut operation = operation;
        operation()
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert_eq!(measured.blocks, 0, "{measured}");
    eprintln!("{region_name}: {visited} records visited");
    assert_eq!(visited, if count == 0 { 0 } else { 4 });
    assert_eq!(
        *region,
        *mixed_region(HeaderFlags::default(), if count == 0 { 11 } else { 29 })
    );
}

#[test]
fn an_empty_link_patch_plan_does_not_parse_malformed_records() {
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    region.push_bytes(&[6, 0xff, 0xff, 0]);
    let patches = LinkPatches::new(&mut region, []);
    assert_eq!(patches.visited, 0);
}

#[rstest]
#[case::medium("link_patch_256", 256)]
#[case::large("link_patch_512", 512)]
fn link_patch_batches_visit_each_record_once(#[case] region_name: &str, #[case] count: usize) {
    let mut region = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    let mut expected = OhRegion::empty(crate::file_writer::WIDTHS, ObjectHeaderPrefix::PLAIN);
    let names: Vec<_> = (0..count)
        .map(|index| format!("{index:04}_{}", "x".repeat(128)))
        .collect();
    for (index, name) in names.iter().enumerate() {
        region.push(
            MessageType::LINK,
            &HardLink::new(name, 1).build(Widths::EIGHT),
        );
        expected.push(
            MessageType::LINK,
            &HardLink::new(name, u64::try_from(index + 11).unwrap()).build(Widths::EIGHT),
        );
    }
    let operation = || {
        let mut patches = LinkPatches::new(&mut region, names.iter().map(String::as_str));
        for index in 0..count {
            patches
                .patch_next(StoredAddress::new(u64::try_from(index + 11).unwrap()))
                .unwrap();
        }
        patches.visited
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (visited, measured) = allocation::measure(region_name, operation);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let visited = {
        let mut operation = operation;
        operation()
    };
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        eprintln!("{region_name}: {measured}; {visited} records visited");
        assert!(
            measured.bytes >= u64::try_from(count * 8).unwrap(),
            "{measured}"
        );
        assert!(
            measured.bytes <= u64::try_from(count * 256).unwrap(),
            "{measured}"
        );
        assert!(measured.blocks <= 16, "{measured}");
    }
    eprintln!("{region_name}: {visited} records visited");
    assert_eq!(visited, count);
    assert_eq!(*region, *expected);
}

fn mixed_region(flags: HeaderFlags, address: u64) -> OhRegion {
    let mut region = OhRegion::empty(
        crate::file_writer::WIDTHS,
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
    );
    region.push(
        MessageType::LINK,
        &HardLink::new("untouched", 5).build(Widths::EIGHT),
    );
    region.push(
        MessageType::LINK,
        &SymbolicLink::soft("soft", "/target").build(),
    );
    region.push(
        MessageType::LINK,
        &SymbolicLink::external("external", "file.h5", "/target").build(),
    );
    region.push(
        MessageType::LINK,
        &HardLink::new("changed", address)
            .creation_order(7)
            .character_set(CharacterSet::UTF8)
            .build(Widths::EIGHT),
    );
    region
}
