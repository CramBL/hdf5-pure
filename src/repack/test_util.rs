//! Verbatim attribute fixtures and complete-message observations for private repack tests.

use std::collections::BTreeMap;
use std::path::Path;

use crate::DatatypeByteOrder;
use crate::File;
use crate::FileBuilder;
use crate::FixedPointType;
use crate::attribute::AttributeMessage;
use crate::dataspace::Dataspace;
use crate::dataspace::DataspaceType;
use crate::datatype::CharacterSet;
use crate::datatype::Datatype;
use crate::datatype::StringPadding;

/// Fixed string layouts, a null dataspace, a big-endian scalar, and a matrix.
///
/// Each owner receives the same [`AttributeMessage`] descriptions.
pub(super) struct ExoticAttributes {
    messages: [AttributeMessage; 5],
}

impl ExoticAttributes {
    fn messages() -> [AttributeMessage; 5] {
        [
            AttributeMessage {
                name: "units".into(),
                datatype: Datatype::String {
                    size: 16,
                    padding: StringPadding::NullTerminate,
                    charset: CharacterSet::Ascii,
                },
                dataspace: Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: {
                    let mut v = b"m/s".to_vec();
                    v.resize(16, 0);
                    v
                },
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            AttributeMessage {
                name: "spaced".into(),
                datatype: Datatype::String {
                    size: 8,
                    padding: StringPadding::SpacePad,
                    charset: CharacterSet::Utf8,
                },
                dataspace: Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: b"ab      ".to_vec(),
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            AttributeMessage {
                name: "nothing".into(),
                datatype: crate::make_i32_type(),
                dataspace: Dataspace {
                    space_type: DataspaceType::Null,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: vec![],
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            AttributeMessage {
                name: "big_endian".into(),
                datatype: Datatype::FixedPoint(
                    FixedPointType::new(4, DatatypeByteOrder::BigEndian, true, 0, 32).unwrap(),
                ),
                dataspace: Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: (-7i32).to_be_bytes().to_vec(),
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            AttributeMessage {
                name: "grid".into(),
                datatype: crate::make_i32_type(),
                dataspace: Dataspace {
                    space_type: DataspaceType::Simple,
                    rank: 2,
                    dimensions: vec![2, 3],
                    max_dimensions: None,
                },
                raw_data: (1i32..=6).flat_map(i32::to_le_bytes).collect(),
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
        ]
    }

    /// Constructs the fixed attribute messages.
    pub(super) fn new() -> Self {
        Self {
            messages: Self::messages(),
        }
    }

    /// Writes the messages on the root group and dataset `data`.
    ///
    /// The dataset contains one little-endian [`f64`] value equal to 1.0.
    ///
    /// # Panics
    ///
    /// Panics if writing the file fails.
    pub(super) fn write(&self, path: &Path) {
        let mut b = FileBuilder::new();
        for message in &self.messages {
            b.set_attr_verbatim(message.clone(), None);
        }
        let ds = b.create_dataset("data").with_f64_data(&[1.0]);
        for message in &self.messages {
            ds.set_attr_verbatim(message.clone(), None);
        }
        b.write(path).unwrap();
    }

    /// Compares complete messages in both files with their fixture descriptions.
    ///
    /// Checks each [`AttributeMessage`] on the root group and `data` in both `src`
    /// and `dst`, including its datatype, dataspace, raw bytes, and datatype location.
    ///
    /// # Panics
    ///
    /// Panics if either owner cannot be read, its attribute count differs, or any
    /// message differs from the fixture.
    pub(super) fn assert_repacked(&self, src: &Path, dst: &Path) {
        for owner in ["", "data"] {
            let before = messages_at(src, owner);
            let after = messages_at(dst, owner);
            assert_eq!(before.len(), self.messages.len());
            assert_eq!(after.len(), self.messages.len());
            for message in &self.messages {
                let name = &message.name;
                assert_eq!(
                    before.get(name),
                    Some(message),
                    "the source file did not record {name:?} as written"
                );
                assert_eq!(
                    after.get(name),
                    Some(message),
                    "attribute {name:?} on {owner:?} changed across the repack"
                );
            }
        }
    }
}

/// Returns parsed attribute messages keyed by name for `path`.
///
/// An empty path selects the root group. Other paths may identify a dataset or group.
///
/// # Panics
///
/// Panics if the file, owner, or attribute messages cannot be read.
pub(super) fn messages_at(file: &Path, path: &str) -> BTreeMap<String, AttributeMessage> {
    let f = File::open(file).unwrap();
    let messages = if path.is_empty() {
        f.root().attr_messages().unwrap()
    } else if let Ok(ds) = f.dataset(path) {
        ds.attr_messages().unwrap()
    } else {
        f.group(path).unwrap().attr_messages().unwrap()
    };
    messages.into_iter().map(|m| (m.name.clone(), m)).collect()
}
