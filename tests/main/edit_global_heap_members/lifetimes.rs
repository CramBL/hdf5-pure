use std::path::Path;

use hdf5_pure::{AttrValue, File};
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_format::__private::VariableLengthDescriptor;
use rstest::rstest;
use test_util::data_layout;
use test_util::dataspace;
use test_util::global_heap::{self, Object};
use test_util::object_header::{Message, MessageType};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::members::ParentImage;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

#[rstest]
#[case::shared(Lifetime::Shared)]
#[case::last_shared(Lifetime::FinalShared)]
#[case::attribute(Lifetime::Attribute)]
#[case::last_attribute(Lifetime::FinalAttribute)]
#[case::fill(Lifetime::Fill)]
#[case::last_fill(Lifetime::FinalFill)]
#[case::alias(Lifetime::Alias)]
#[case::last_alias(Lifetime::FinalAlias)]
#[case::retained_root(Lifetime::Retained)]
#[case::incomplete(Lifetime::Incomplete)]
#[case::empty(Lifetime::ZeroLength)]
#[case::last_empty(Lifetime::FinalZeroLength)]
fn member_removal_requires_the_last_supported_reference(#[case] lifetime: Lifetime) {
    let path = test_util::temp::temp_path("public-member-lifetime.h5");
    let original = lifetime.write(&path);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("value").unwrap().read_string().unwrap(),
        [lifetime.value()]
    );
    file.root().delete("value").unwrap();
    if lifetime.final_reference() {
        if lifetime.attribute() {
            file.dataset("share").unwrap().remove_attr("a").unwrap();
        } else {
            file.root().delete("share").unwrap();
        }
    }
    file.commit().unwrap();
    file.close().unwrap();
    let after = std::fs::read(&path).unwrap();
    if lifetime.final_reference() {
        original.assert_compacted(&after, &[91]);
    } else {
        original.assert_unchanged(&after);
    }
    let file = File::open(&path).unwrap();
    lifetime.assert_values(&file);
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Lifetime {
    Alias,
    Attribute,
    Fill,
    FinalAlias,
    FinalAttribute,
    FinalFill,
    FinalShared,
    FinalZeroLength,
    Incomplete,
    Retained,
    Shared,
    ZeroLength,
}

impl Lifetime {
    /// Writes the selected sharing case and returns the original complete parent image.
    ///
    /// # Panics
    ///
    /// Panics if fixture construction, writing, or parent inspection fails.
    fn write(self, path: &Path) -> ParentImage {
        let mut fixture = Fixture::new();
        let parent = u64::try_from(fixture.image.append(
            &global_heap::collection_of_size_with_padding(
                &[
                    Object::new(7, b"shared"),
                    Object {
                        index: 91,
                        reference_count: 0,
                        data: b"keep!",
                    },
                ],
                Widths::EIGHT,
                global_heap::MIN_COLLECTION_SIZE,
                0xa5,
            ),
        ))
        .unwrap();
        let reference = global_heap_fixture::reference(if self.empty() { 0 } else { 6 }, parent, 7);
        if self.empty() {
            let (descriptor, used) =
                VariableLengthDescriptor::parse(OffsetWidth::Eight, &reference).unwrap();
            assert_eq!(used, reference.len());
            assert_eq!(descriptor.element_count(), 0);
            assert_eq!(descriptor.collection_address().get(), parent);
            assert_eq!(descriptor.object_index(), 7);
        }
        let mut records = Form::CompactDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &reference,
        );
        records.push(Message::reference_count(
            if self.alias() || matches!(self, Self::Retained) {
                2
            } else {
                1
            },
        ));
        let owner = fixture.append_header(records);
        fixture.link("value", owner);
        if self.alias() {
            fixture.link("share", owner);
        } else if self.attribute() {
            let mut records = Form::CompactDataset.messages(
                &mut fixture,
                &hdf5_pure::make_i32_type(),
                &17_i32.to_le_bytes(),
            );
            records.extend(Form::CompactAttribute.messages(
                &mut fixture,
                &global_heap_fixture::vlen_string(),
                &reference,
            ));
            fixture.object("share", records);
        } else if !matches!(self, Self::Retained | Self::Incomplete) {
            let records = if matches!(self, Self::Fill | Self::FinalFill) {
                Form::FillValue
            } else {
                Form::CompactDataset
            }
            .messages(
                &mut fixture,
                &global_heap_fixture::vlen_string(),
                &reference,
            );
            fixture.object("share", records);
        }
        if matches!(self, Self::Incomplete) {
            let mut records = global_heap_fixture::dataset_messages(
                &global_heap_fixture::object_reference(),
                &dataspace::scalar(),
            );
            records.push(Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::contiguous(global_heap_fixture::UNDEFINED_ADDRESS, 0),
            ));
            fixture.object("opaque", records);
        }
        let records = Form::CompactDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(5, parent, 91),
        );
        fixture.object("keep", records);
        fixture.write(path);
        let original = ParentImage::read(&std::fs::read(path).unwrap(), parent);
        assert_eq!(original.indices(), [7, 91]);
        original
    }

    fn alias(self) -> bool {
        matches!(self, Self::Alias | Self::FinalAlias)
    }
    fn attribute(self) -> bool {
        matches!(self, Self::Attribute | Self::FinalAttribute)
    }
    fn empty(self) -> bool {
        matches!(self, Self::ZeroLength | Self::FinalZeroLength)
    }
    fn final_reference(self) -> bool {
        matches!(
            self,
            Self::FinalAlias
                | Self::FinalAttribute
                | Self::FinalFill
                | Self::FinalShared
                | Self::FinalZeroLength
        )
    }
    fn value(self) -> &'static str {
        if self.empty() { "" } else { "shared" }
    }

    fn assert_values(self, file: &File) {
        assert_eq!(
            file.dataset("keep").unwrap().read_string().unwrap(),
            ["keep!"]
        );
        let mut datasets = file.root().datasets().unwrap();
        datasets.sort();
        if self.attribute() {
            let dataset = file.dataset("share").unwrap();
            assert_eq!(dataset.read_i32().unwrap(), [17]);
            let attributes = dataset.attrs().unwrap();
            if self.final_reference() {
                assert!(attributes.is_empty());
            } else {
                assert_eq!(
                    attributes["a"],
                    AttrValue::VarLenAsciiStringArray(vec!["shared".into()])
                );
            }
            assert_eq!(datasets, ["keep", "share"]);
        } else if !self.final_reference() && !matches!(self, Self::Retained | Self::Incomplete) {
            assert_eq!(
                file.dataset("share").unwrap().read_string().unwrap(),
                [self.value()]
            );
            assert_eq!(datasets, ["keep", "share"]);
        } else if matches!(self, Self::Incomplete) {
            assert_eq!(datasets, ["keep", "opaque"]);
        } else {
            assert_eq!(datasets, ["keep"]);
        }
    }
}
