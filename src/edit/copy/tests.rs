use rstest::rstest;
use test_util::attribute::Attribute;
use test_util::datatype;
use test_util::dense_attributes;
use test_util::widths::Widths;

use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use crate::reader::OpenedSource;
use crate::source::BytesSource;
use crate::superblock;
use crate::width::{LengthWidth, OffsetWidth};

use super::WriteEngine;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod allocation;

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
fn dense_copy_inspects_datatypes_before_short_values(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(false, true)] reference: bool,
) {
    let datatype = if reference {
        datatype::object_reference(u32::from(offsets.get()))
    } else {
        datatype::f64_le()
    };
    let message = Attribute::new("value", &datatype, &test_util::dataspace::scalar(), &[]).build();
    let bytes = dense_attributes::one_message(
        &message,
        Widths::new(usize::from(offsets.get()), usize::from(lengths.get())),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("short-attribute.h5");
    std::fs::write(&path, &bytes).unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let located = superblock::read(&source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let object = WriteEngine::read_object(
        &OpenedSource::for_test(&source, addresses),
        AccessMode::ReadOnly,
        ObjectId::new(
            addresses
                .resolve(located.superblock.root_group_address)
                .unwrap(),
        ),
        located.widths,
    );
    let Err(error) = object else {
        panic!("a short dense attribute must be rejected");
    };
    if reference {
        assert!(
            matches!(error, Error::Format(FormatError::ReferenceWidthUnsupported { offset_size }) if offset_size == offsets.get()),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(error, Error::Format(FormatError::UnexpectedEof { expected, available }) if expected == message.len() + core::mem::size_of::<f64>() && available == message.len()),
            "{error:?}"
        );
    }
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    directory.close().unwrap();
}
