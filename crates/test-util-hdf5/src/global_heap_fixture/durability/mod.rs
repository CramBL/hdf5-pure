use std::fs;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::v2::Header;
use test_util::object_header::v2::Timestamps;
use test_util::superblock;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;

use super::Fixture;
use super::Form;
use super::members::ParentImage;
use crate::global_heap_fixture;

#[cfg(feature = "__hdf5-1.10")]
pub(crate) mod native;

/// Independent source bytes and physical addresses for heap-commit failure scenarios.
pub struct FailureSource {
    /// The three collection addresses in physical order.
    pub parents: [u64; 3],
    /// The optional physical header address of the numeric owner with two hard links.
    pub owner: Option<u64>,
    /// The physical start of the `value` dataset's original numeric payload.
    pub raw: u64,
    /// The complete original file image.
    pub original: Vec<u8>,
}

impl FailureSource {
    /// Writes persistent root-attribute, variable-length dataset, and object-reference sources.
    ///
    /// Selects PAGE when `page` is true and FSM_AGGR otherwise.
    ///
    /// # Panics
    ///
    /// Panics if construction or writing the fixture fails.
    pub fn session(path: &Path, page: bool) {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(if page {
                FileSpaceStrategy::Page
            } else {
                FileSpaceStrategy::FsmAggr
            })
            .with_persistence(true),
        );
        builder.set_attr("changed", AttrValue::VarLenAsciiString("before".into()));
        builder
            .create_dataset("labels")
            .with_vlen_strings(&["seed"]);
        builder.create_dataset("unknown").with_raw_data(
            global_heap_fixture::object_reference(),
            vec![0; usize::try_from(global_heap_fixture::object_reference().type_size()).unwrap()],
            1,
        );
        builder.write(path).unwrap();
    }

    /// Writes three parents and a numeric owner with two hard links.
    ///
    /// Declares one reference when `contradiction` is true and two otherwise. Uses `timestamp`
    /// for the `value` header's four timestamps.
    ///
    /// # Panics
    ///
    /// Panics if construction, address conversion, writing, or reading the fixture fails.
    pub fn aliased(path: &Path, contradiction: bool, timestamp: u32) -> Self {
        Self::write(path, Some((contradiction, timestamp)), true)
    }

    /// Writes three parents without an aliased owner, optionally including `numbers`.
    ///
    /// # Panics
    ///
    /// Panics if construction, address conversion, writing, or reading the fixture fails.
    pub fn later(path: &Path, numbers: bool) -> Self {
        Self::write(path, None, numbers)
    }

    fn write(path: &Path, alias: Option<(bool, u32)>, numbers: bool) -> Self {
        let mut fixture = Fixture::new();
        let parents = std::array::from_fn(|_| {
            fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")])
        });
        let references = parents
            .iter()
            .flat_map(|&parent| global_heap_fixture::reference(7, parent, 7))
            .collect::<Vec<_>>();
        let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
        let mut records = Form::ContiguousDataset.messages(
            &mut fixture,
            &hdf5_pure::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        records.extend(Form::CompactAttribute.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &references,
        ));
        if let Some((_, timestamp)) = alias {
            let header = Header::new()
                .timestamps(Timestamps {
                    access: timestamp,
                    modification: timestamp,
                    change: timestamp,
                    birth: timestamp,
                })
                .messages(records)
                .build();
            let address = u64::try_from(fixture.image.append(&header)).unwrap();
            fixture.link("value", address);
        } else {
            fixture.object("value", records);
        }
        let owner = alias.map(|(contradiction, _)| {
            let mut records = Form::CompactDataset.messages(
                &mut fixture,
                &hdf5_pure::make_i32_type(),
                &19_i32.to_le_bytes(),
            );
            records.push(Message::reference_count(if contradiction { 1 } else { 2 }));
            let owner = fixture.append_header(records);
            fixture.link("alias", owner);
            fixture.link("other", owner);
            owner
        });
        if numbers {
            let records = Form::ContiguousDataset.messages(
                &mut fixture,
                &hdf5_pure::make_i32_type(),
                &17_i32.to_le_bytes(),
            );
            fixture.object("numbers", records);
        }
        fixture.write(path);
        Self {
            parents,
            owner,
            raw,
            original: fs::read(path).unwrap(),
        }
    }

    /// Checks compaction of the first `completed` parents and unchanged bytes for the rest.
    ///
    /// # Panics
    ///
    /// Panics if a collection cannot be read, original indices differ from 7 and 91, or a
    /// compaction or unchanged-byte assertion fails.
    pub fn assert_parents(&self, image: &[u8], completed: usize) {
        for (index, &address) in self.parents.iter().enumerate() {
            let parent = ParentImage::read(&self.original, address);
            assert_eq!(parent.indices(), [7, 91]);
            if index < completed {
                parent.assert_compacted(image, &[91]);
            } else {
                parent.assert_unchanged(image);
            }
        }
    }

    /// Returns the encoded superblock size after checking the version and width prefix.
    ///
    /// # Panics
    ///
    /// Panics if `image` lacks the expected signature, version 3, or eight-byte address and length
    /// widths.
    pub fn superblock_size(image: &[u8]) -> usize {
        let widths = Widths::EIGHT;
        let Widths { offset, length } = widths;
        let prefix = [
            superblock::SIGNATURE.as_slice(),
            &[
                superblock::VERSION_3,
                u8::try_from(offset).unwrap(),
                u8::try_from(length).unwrap(),
            ],
        ]
        .concat();
        assert_eq!(image.get(..prefix.len()), Some(prefix.as_slice()));
        Superblock::new(widths)
            .version(superblock::VERSION_3)
            .build()
            .len()
    }
}
