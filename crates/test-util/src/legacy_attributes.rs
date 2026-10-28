//! Independent file images with compact attributes and symbol-table groups.
//!
//! Each image has a version 0 superblock, version 1 object headers, eight-byte
//! addresses and lengths, and little-endian numeric values. Attributes are stored
//! in object header messages.

use crate::attribute::Attribute;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::datatype;
use crate::datatype::Class;
use crate::image::Image;
use crate::legacy;
use crate::object_header::Message;
use crate::object_header::MessageType;

/// A fixed hierarchy with scalar or array attributes.
#[derive(Clone, Copy, Debug)]
pub enum Fixture {
    /// Attributes on the root group and the dataset `data`.
    CompactAttributes,
    /// Scalar and array attributes on the empty group `experiment`.
    MixedAttributes,
}

impl Fixture {
    /// Encodes the hierarchy and its attributes as an independent file image.
    pub fn build(self) -> Vec<u8> {
        match self {
            Self::CompactAttributes => attrs(),
            Self::MixedAttributes => mixed_attrs(),
        }
    }
}

/// Builds a contiguous [`f64`] dataset `data = [1.0, 2.0, 3.0]` with scalar attributes.
///
/// The root attribute `file_type` is `"experiment"`. The dataset attributes are
/// `description = "test dataset"`, `scale = 3.14`, and `version = 42`.
pub fn attrs() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let datatype = datatype::f64_le();
    let data = legacy::dataset_with_messages(
        &mut image,
        datatype.clone(),
        &[1.0f64, 2.0, 3.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>(),
        3,
        &[
            attribute(
                "description",
                &datatype::fixed_string(12),
                &[],
                b"test dataset",
            ),
            attribute("scale", &datatype, &[], &(314.0f64 / 100.0).to_le_bytes()),
            attribute("version", &i64_datatype(), &[], &42i64.to_le_bytes()),
        ],
    );
    let root = legacy::group_with_messages(
        &mut image,
        &[("data", data)],
        &[attribute(
            "file_type",
            &datatype::fixed_string(10),
            &[],
            b"experiment",
        )],
    );
    legacy::finish(image, root)
}

/// Builds the empty group `experiment` with scalar and array attributes.
///
/// Its attributes are `iterations = 1000`, `name = "run_001"`, and
/// `temperatures = [22.5, 23.1, 21.8]`.
pub fn mixed_attrs() -> Vec<u8> {
    let mut image = Image::starting_with(&[0; 96]);
    let experiment = legacy::group_with_messages(
        &mut image,
        &[],
        &[
            attribute("iterations", &i64_datatype(), &[], &1_000i64.to_le_bytes()),
            attribute("name", &datatype::fixed_string(7), &[], b"run_001"),
            attribute(
                "temperatures",
                &datatype::f64_le(),
                &[3],
                &[22.5f64, 23.1, 21.8]
                    .into_iter()
                    .flat_map(f64::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        ],
    );
    let root = legacy::group_with_messages(&mut image, &[("experiment", experiment)], &[]);
    legacy::finish(image, root)
}

fn attribute(name: &str, datatype: &[u8], dims: &[u64], data: &[u8]) -> Message {
    let space = dataspace::v1(dims.len() as u8, Flags::NONE, dims, None);
    let mut body = Attribute::new(name, datatype, &space, data).build();
    body.resize(body.len().next_multiple_of(legacy::MESSAGE_ALIGNMENT), 0);
    Message::new(MessageType::ATTRIBUTE, &body)
}

fn i64_datatype() -> Vec<u8> {
    let mut datatype =
        datatype::header(Class::FIXED_POINT, DATATYPE_VERSION, SIGNED_CLASS_FIELDS, 8);
    datatype.extend_from_slice(&0u16.to_le_bytes());
    datatype.extend_from_slice(&64u16.to_le_bytes());
    datatype
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.
const DATATYPE_VERSION: u8 = 1;

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0, defines
// the signed flag and little-endian byte order encoded here.
const SIGNED_CLASS_FIELDS: [u8; 3] = [0x08, 0, 0];
