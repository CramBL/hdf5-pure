//! Independent files for attribute-value reads and decoding failures.

use crate::attribute::Attribute;
use crate::bytes;
use crate::data_layout;
use crate::dataspace;
use crate::datatype::{self, Class};
use crate::global_heap::{self, Object};
use crate::image::Image;
use crate::link_message::HardLink;
use crate::object_header::v2::Header;
use crate::object_header::{Message, MessageType};
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

/// The stored representation of the selected attribute.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    /// A variable-length string whose heap member precedes an implied free tail.
    ImpliedTail,
    /// An empty variable-length string with a null heap reference.
    Null,
    /// An opaque scalar without a supported attribute-value representation.
    Opaque,
    /// A floating-point scalar stored in VAX byte order.
    VaxFloat,
}

/// A malformed heap collection or reference in a variable-length attribute.
#[derive(Clone, Copy, Debug)]
pub enum Damage {
    /// Two heap members use index 1.
    DuplicateIndex,
    /// The explicit free-space record has the wrong extent.
    FreeExtent,
    /// The declared heap collection extends beyond the file.
    HeapExtent,
    /// The global-heap collection signature is invalid.
    HeapSignature,
    /// The attribute refers to a heap object absent from the collection.
    MissingObject,
    /// The heap member extends beyond the collection.
    ObjectExtent,
    /// The reference's payload byte count exceeds the heap member's size.
    PayloadExtent,
    /// A nonempty value refers to an undefined heap address.
    UndefinedHeap,
    /// The heap object index exceeds the two-byte member-index width.
    WideIndex,
    /// A nonempty value refers to the free-space index.
    ZeroIndex,
}

/// The physical byte offset of the heap collection.
pub const HEAP: usize = 64;
/// The physical byte offset of the first heap member's payload.
pub const PAYLOAD: usize = HEAP + 32;
/// The byte length of the member before the implied free tail.
pub const PAYLOAD_LEN: usize = 4056;
/// The attribute name, including a newline and a quotation mark.
pub const ATTRIBUTE_NAME: &str = "value\n\"quoted";

/// Builds a file with the selected attribute on the root group and a dataset.
///
/// Both objects also have numeric attributes named `before` and `after`.
pub fn file(value: Value) -> Vec<u8> {
    build(value, None)
}

/// Builds the [`Value::ImpliedTail`] fixture with the selected malformed heap or reference.
pub fn damaged_file(damage: Damage) -> Vec<u8> {
    build(Value::ImpliedTail, Some(damage))
}

fn build(value: Value, damage: Option<Damage>) -> Vec<u8> {
    let payload = vec![b'x'; PAYLOAD_LEN];
    let mut heap =
        global_heap::collection_of_size(&[Object::new(1, &payload)], Widths::EIGHT, 4096);
    heap[4088..].copy_from_slice(&[7, 0, 0, 0, 0, 0, 0, 0]);
    let mut reference = global_heap::reference(4056, HEAP as u64, 1, Widths::EIGHT);
    match damage {
        None => {}
        Some(Damage::DuplicateIndex) => {
            heap = global_heap::collection_of_size(
                &[Object::new(1, b"first"), Object::new(1, b"second")],
                Widths::EIGHT,
                4096,
            );
            reference = global_heap::reference(5, HEAP as u64, 1, Widths::EIGHT);
        }
        Some(Damage::FreeExtent) => {
            heap =
                global_heap::collection_of_size(&[Object::new(1, b"first")], Widths::EIGHT, 4096);
            bytes::set_uint_at(&mut heap, 48, 8, 4048);
            reference = global_heap::reference(5, HEAP as u64, 1, Widths::EIGHT);
        }
        Some(Damage::HeapExtent) => bytes::set_uint_at(&mut heap, 8, 8, 8192),
        Some(Damage::HeapSignature) => heap[0] = b'X',
        Some(Damage::MissingObject) => bytes::set_uint_at(&mut reference, 12, 4, 2),
        Some(Damage::ObjectExtent) => bytes::set_uint_at(&mut heap, 24, 8, 4096),
        Some(Damage::PayloadExtent) => bytes::set_uint_at(&mut reference, 0, 4, 4057),
        Some(Damage::UndefinedHeap) => bytes::set_uint_at(&mut reference, 4, 8, u64::MAX),
        Some(Damage::WideIndex) => bytes::set_uint_at(&mut reference, 12, 4, 65536),
        Some(Damage::ZeroIndex) => bytes::set_uint_at(&mut reference, 12, 4, 0),
    }
    let mut image = Image::starting_with(&[0; HEAP]);
    image.append(&heap);
    let (datatype, data) = match value {
        Value::ImpliedTail => (vlen_datatype(), reference),
        Value::Null => (
            vlen_datatype(),
            global_heap::reference(0, 0, 0, Widths::EIGHT),
        ),
        Value::Opaque => (
            datatype::header(Class::OPAQUE, VERSION_ONE, [0; 3], 1),
            vec![42],
        ),
        Value::VaxFloat => {
            let mut datatype = datatype::f64_le();
            datatype[0] = VAX_FLOAT_CLASS_AND_VERSION;
            datatype[1] |= VAX_BYTE_ORDER;
            (datatype, vec![0; 8])
        }
    };
    let scalar = dataspace::scalar();
    let attrs = [
        Attribute::new("before", &datatype::f64_le(), &scalar, &7_f64.to_le_bytes()).build(),
        Attribute::new(ATTRIBUTE_NAME, &datatype, &scalar, &data).build(),
        Attribute::new("after", &datatype::f64_le(), &scalar, &9_f64.to_le_bytes()).build(),
    ];
    let messages = || {
        attrs
            .iter()
            .map(|body| Message::new(MessageType::ATTRIBUTE, body))
    };
    let dataset = image.append(
        &Header::new()
            .messages([
                Message::new(MessageType::DATATYPE, &datatype::f64_le()),
                Message::new(MessageType::DATASPACE, &scalar),
                Message::new(
                    MessageType::DATA_LAYOUT,
                    &data_layout::compact(&1_f64.to_le_bytes()),
                ),
            ])
            .messages(messages())
            .build(),
    );
    let root = image.append(
        &Header::new()
            .message(Message::new(
                MessageType::LINK,
                &HardLink::new("data", dataset as u64).build(Widths::EIGHT),
            ))
            .messages(messages())
            .build(),
    );
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .root_header_address(root as u64)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    image.build()
}

fn vlen_datatype() -> Vec<u8> {
    let mut datatype = datatype::header(Class::VARIABLE_LENGTH, VERSION_ONE, UTF8_VLEN, 16);
    let mut parent = datatype::header(Class::FIXED_POINT, VERSION_ONE, [0; 3], 1);
    parent.extend_from_slice(&[0, 0, 8, 0]);
    datatype.extend(parent);
    datatype
}

/// Datatype message version 1, defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VERSION_ONE: u8 = 1;
/// String kind and UTF-8 flags for a version 1 variable-length datatype message, defined in
/// "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const UTF8_VLEN: [u8; 3] = [1, 1, 0];
/// Floating-point class and version 3, defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VAX_FLOAT_CLASS_AND_VERSION: u8 = 0x31;
/// Bits 6 and 0 for VAX byte order in a version 3 floating-point datatype message, defined in
/// "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VAX_BYTE_ORDER: u8 = 0x41;
