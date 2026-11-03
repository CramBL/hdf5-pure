//! Source attributes paired with checked storage and creation metadata.

use std::collections::BTreeSet;

use hdf5_pure_format::__private::DenseAttrCreationOrder;
use hdf5_pure_format::__private::ObjectHeaderPrefix;

use crate::attribute::{AttributeMessage, StoredAttribute};
use crate::attribute_info::AttributeInfoMessage;
use crate::attribute_policy::{AttributePolicy, AttributeStorage};
use crate::error::{Error, FormatError};
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::width::OffsetWidth;

/// Attributes ordered by name and their inherited header metadata.
///
/// Creation indexes in [`header`](Self::header) follow the same order as
/// [`messages`](Self::messages).
pub(crate) struct AttributeReconstruction {
    /// Attribute messages in name order.
    pub(crate) messages: Vec<AttributeMessage>,
    /// Prefix, storage policy, and creation metadata for those messages.
    pub(crate) header: InheritedHeader,
}

impl AttributeReconstruction {
    /// Pairs source attributes with checked prefix and creation metadata.
    ///
    /// Sorts messages and their creation indexes together by name. Preserves a stored creation
    /// counter, or derives it from the highest index when Attribute Info is absent. Uses a plain
    /// prefix for version 1 headers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RepackUnsupported`] if creation metadata differs from the prefix, a
    /// tracked attribute lacks an index, or the derived counter cannot be represented. Returns
    /// [`FormatError::DuplicateAttributeName`], [`FormatError::DuplicateAttributeCreationIndex`],
    /// or [`FormatError::InvalidAttributeCreationCounter`] for inconsistent attributes.
    /// Propagates attribute-policy and Attribute Info parsing errors.
    pub(crate) fn new(
        header: &ObjectHeader,
        mut stored: Vec<StoredAttribute>,
        width: OffsetWidth,
    ) -> Result<Self, Error> {
        let prefix = header.prefix.unwrap_or(ObjectHeaderPrefix::PLAIN);
        let policy = AttributePolicy::from_prefix(prefix.attribute_phase_change)?;
        let info = header
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::ATTRIBUTE_INFO)
            .map(|message| AttributeInfoMessage::parse_with_offset_width(&message.data, width))
            .transpose()?;
        if (prefix.layout.indexes_creation_order() && !prefix.layout.tracks_creation_order())
            || info.as_ref().is_some_and(|info| {
                info.max_creation_index.is_some() != prefix.layout.tracks_creation_order()
                    || info.indexes_creation_order != prefix.layout.indexes_creation_order()
            })
        {
            return Err(Error::RepackUnsupported(
                "an object's attribute creation-order metadata does not match its header prefix"
                    .into(),
            ));
        }
        let current = if info
            .as_ref()
            .is_some_and(|info| info.fractal_heap_address.is_some())
        {
            AttributeStorage::Dense
        } else {
            AttributeStorage::Compact
        };
        stored.sort_by(|left, right| left.message.name.cmp(&right.message.name));
        for pair in stored.windows(2) {
            if pair[0].message.name == pair[1].message.name {
                return Err(
                    FormatError::DuplicateAttributeName(pair[0].message.name.clone()).into(),
                );
            }
        }
        let creation = if prefix.layout.tracks_creation_order() {
            let indices = stored
                .iter()
                .map(|attribute| {
                    attribute.creation_index.ok_or_else(|| {
                        Error::RepackUnsupported(
                            "a tracked attribute has no stored creation index".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let max = match info.as_ref().and_then(|info| info.max_creation_index) {
                Some(max) => max,
                None => match indices.iter().max() {
                    Some(index) => index.checked_add(1).ok_or_else(|| {
                        Error::RepackUnsupported(
                            "an attribute creation counter cannot be represented".into(),
                        )
                    })?,
                    None => 0,
                },
            };
            let mut seen = BTreeSet::new();
            for &index in &indices {
                if index >= max {
                    return Err(
                        FormatError::InvalidAttributeCreationCounter { index, next: max }.into(),
                    );
                }
                if !seen.insert(index) {
                    return Err(
                        FormatError::DuplicateAttributeCreationIndex(u32::from(index)).into(),
                    );
                }
            }
            DenseAttrCreationOrder::Tracked {
                indices,
                max,
                indexed: prefix.layout.indexes_creation_order(),
            }
        } else {
            DenseAttrCreationOrder::Untracked
        };
        Ok(Self {
            header: InheritedHeader {
                prefix,
                policy,
                current,
                count: stored.len(),
                creation,
            },
            messages: stored
                .into_iter()
                .map(|attribute| attribute.message)
                .collect(),
        })
    }
}

/// Source prefix properties, effective storage policy, and attribute creation metadata.
pub(crate) struct InheritedHeader {
    pub(super) prefix: ObjectHeaderPrefix,
    pub(super) policy: AttributePolicy,
    /// The source attribute storage form.
    pub(super) current: AttributeStorage,
    /// The source attribute count before reconstruction.
    pub(super) count: usize,
    /// Creation indexes paired with messages in name order, and their counter and indexing policy.
    pub(super) creation: DenseAttrCreationOrder,
}

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::{
        AttributePhaseChange, MessageRecordLayout, ObjectHeaderWriter,
    };
    use rstest::rstest;
    use test_util::object_header::v2::HeaderFlags;

    use super::*;
    use crate::access_mode::AccessMode;
    use crate::address::{StoredAddress, StoredAddressExt};
    use crate::file_writer::source::Object;
    use crate::type_builders::{self, AttrValue, DatasetBuilder};

    #[rstest]
    #[case::custom_compact(Some((32, 24)), false, 10, false)]
    #[case::custom_dense(Some((32, 24)), true, 24, true)]
    #[case::default_compact(None, false, 9, false)]
    #[case::default_dense(None, true, 5, true)]
    #[case::always_dense(Some((0, 0)), false, 1, true)]
    #[case::one_way(Some((12, 0)), true, 1, true)]
    #[case::empty(Some((32, 24)), true, 0, false)]
    fn reconstructed_storage_uses_policy_and_current_form(
        #[case] phase: Option<(u16, u16)>,
        #[case] current_dense: bool,
        #[case] count: u16,
        #[case] dense: bool,
    ) {
        let fields: Vec<_> = (0..count)
            .map(|index| (format!("a{index:03}"), Some(index)))
            .collect();
        let inherited = reconstruction(phase, current_dense, 17.max(count), &fields).unwrap();
        let mut builder = DatasetBuilder::new("data");
        builder.with_i32_data(&[7]);
        for attribute in inherited.messages {
            builder.set_attr_verbatim(attribute, None);
        }
        let object = Object::dataset(builder)
            .unwrap()
            .with_inherited(inherited.header)
            .unwrap();
        assert_eq!(object.dense().unwrap().is_some(), dense);
    }

    #[rstest]
    #[case::policy(Some((3, 4)), 17, [("a", Some(0)), ("b", Some(1))], FormatError::InvalidAttributePhaseChange { max_compact: 3, min_dense: 4 })]
    #[case::counter(None, 1, [("a", Some(0)), ("b", Some(1))], FormatError::InvalidAttributeCreationCounter { index: 1, next: 1 })]
    #[case::index(None, 17, [("a", Some(1)), ("b", Some(1))], FormatError::DuplicateAttributeCreationIndex(1))]
    #[case::name(None, 17, [("a", Some(0)), ("a", Some(1))], FormatError::DuplicateAttributeName("a".into()))]
    fn reconstruction_rejects_invalid_metadata(
        #[case] phase: Option<(u16, u16)>,
        #[case] next: u16,
        #[case] fields: [(&str, Option<u16>); 2],
        #[case] expected: FormatError,
    ) {
        let fields: Vec<_> = fields
            .into_iter()
            .map(|(name, index)| (name.into(), index))
            .collect();
        let Err(error) = reconstruction(phase, false, next, &fields) else {
            panic!("accepted invalid reconstruction metadata");
        };
        let Error::Format(error) = error else {
            panic!("expected a format error, got {error:?}");
        };
        assert_eq!(error, expected);
    }

    #[test]
    fn reconstruction_rejects_an_attribute_without_its_creation_index() {
        let Err(error) = reconstruction(None, false, 17, &[("a".into(), None)]) else {
            panic!("accepted an attribute without a creation index");
        };
        assert!(
            matches!(&error, Error::RepackUnsupported(reason) if reason == "a tracked attribute has no stored creation index"),
            "{error:?}"
        );
    }

    #[test]
    fn reconstruction_keeps_creation_indexes_paired_while_sorting_by_name() {
        let reconstruction = reconstruction(
            None,
            false,
            17,
            &[("b".into(), Some(5)), ("a".into(), Some(2))],
        )
        .unwrap();
        assert_eq!(
            reconstruction
                .messages
                .iter()
                .map(|message| message.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(
            reconstruction.header.creation,
            DenseAttrCreationOrder::Tracked {
                indices: vec![2, 5],
                max: 17,
                indexed: true
            }
        );
    }

    #[rstest]
    #[case::empty(None, Some(0))]
    #[case::after_last(Some(5), Some(6))]
    #[case::exhausted(Some(u16::MAX), None)]
    fn missing_attribute_info_uses_the_last_record_creation_index(
        #[case] index: Option<u16>,
        #[case] expected: Option<u16>,
    ) {
        let prefix = ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(
            HeaderFlags::TRACKS_CREATION_ORDER.0,
        ));
        let mut writer = ObjectHeaderWriter::with_prefix(prefix);
        let stored = index
            .map(|index| {
                let message = type_builders::build_attr_message("a", &AttrValue::I64(71)).unwrap();
                writer.add_message_record(
                    MessageType::ATTRIBUTE,
                    message.serialize(OffsetWidth::Eight, 8).unwrap(),
                    crate::message_flags::MessageFlags::NONE,
                    index,
                );
                StoredAttribute {
                    message,
                    creation_index: Some(index),
                }
            })
            .into_iter()
            .collect();
        let bytes = writer.serialize().unwrap();
        let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        let result = AttributeReconstruction::new(&header, stored, OffsetWidth::Eight);
        if let Some(expected) = expected {
            assert_eq!(
                result.unwrap().header.creation,
                DenseAttrCreationOrder::Tracked {
                    indices: index.into_iter().collect(),
                    max: expected,
                    indexed: false,
                }
            );
        } else {
            let Err(error) = result else {
                panic!("accepted an unrepresentable counter");
            };
            assert!(
                matches!(&error, Error::RepackUnsupported(reason) if reason == "an attribute creation counter cannot be represented"),
                "{error:?}"
            );
        }
    }

    #[rstest]
    #[case::untracked(HeaderFlags::default(), Some(17), false)]
    #[case::tracking_missing(HeaderFlags::TRACKS_CREATION_ORDER, None, false)]
    #[case::index_missing(HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER, Some(17), false)]
    #[case::index_without_tracking(HeaderFlags::INDEXES_CREATION_ORDER, None, true)]
    fn reconstruction_rejects_creation_metadata_that_differs_from_the_prefix(
        #[case] flags: HeaderFlags,
        #[case] max_creation_index: Option<u16>,
        #[case] indexes_creation_order: bool,
    ) {
        let prefix =
            ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0));
        let mut writer = ObjectHeaderWriter::with_prefix(prefix);
        writer.add_message(
            MessageType::ATTRIBUTE_INFO,
            AttributeInfoMessage {
                max_creation_index,
                indexes_creation_order,
                fractal_heap_address: None,
                btree_name_index_address: None,
                btree_creation_order_address: None,
            }
            .serialize(OffsetWidth::Eight)
            .unwrap(),
        );
        let bytes = writer.serialize().unwrap();
        let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        let Err(error) = AttributeReconstruction::new(&header, Vec::new(), OffsetWidth::Eight)
        else {
            panic!("accepted inconsistent creation metadata");
        };
        assert!(
            matches!(&error, Error::RepackUnsupported(reason) if reason == "an object's attribute creation-order metadata does not match its header prefix"),
            "{error:?}"
        );
    }

    fn reconstruction(
        phase: Option<(u16, u16)>,
        dense: bool,
        next: u16,
        fields: &[(String, Option<u16>)],
    ) -> Result<AttributeReconstruction, Error> {
        let prefix = ObjectHeaderPrefix {
            attribute_phase_change: phase.map(|(max_compact, min_dense)| AttributePhaseChange {
                max_compact,
                min_dense,
            }),
            ..ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(
                (HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER).0,
            ))
        };
        let mut writer = ObjectHeaderWriter::with_prefix(prefix);
        writer.add_message(
            MessageType::ATTRIBUTE_INFO,
            AttributeInfoMessage {
                max_creation_index: Some(next),
                indexes_creation_order: true,
                fractal_heap_address: dense.then_some(StoredAddress::new(1_024)),
                btree_name_index_address: None,
                btree_creation_order_address: None,
            }
            .serialize(OffsetWidth::Eight)
            .unwrap(),
        );
        let bytes = writer.serialize().unwrap();
        let header = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, 0, 8, 8).unwrap();
        AttributeReconstruction::new(
            &header,
            fields
                .iter()
                .map(|(name, creation_index)| StoredAttribute {
                    message: type_builders::build_attr_message(name, &AttrValue::I64(71)).unwrap(),
                    creation_index: *creation_index,
                })
                .collect(),
            OffsetWidth::Eight,
        )
    }
}
