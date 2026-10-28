//! Independent MATLAB struct-array images with object references and global-heap field names.
//!
//! Each image has a zero-filled 512-byte user block, a version 0 superblock,
//! symbol-table groups, version 1 object headers, and little-endian contiguous
//! datasets. These are synthetic HDF5 images with MATLAB attributes.

use crate::attribute::Attribute;
use crate::data_layout;
use crate::dataspace;
use crate::dataspace::Flags;
use crate::datatype;
use crate::datatype::Class;
use crate::global_heap;
use crate::global_heap::Object;
use crate::image::Image;
use crate::legacy;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v1::Header;
use crate::superblock::v0::Superblock;
use crate::widths::Widths;

/// Builds row, column, grid, nested, and scalar structures with MATLAB attributes.
///
/// `row` and `col` have fields `fieldA`, `fieldB`, and `fieldC`: numbers 1 through 6,
/// characters `a` through `f`, and the vector `[-6, -5, -4, -3, -2, -1]` in each
/// element. Their HDF5 field dimensions are `[6, 1]` and `[1, 6]`, respectively.
///
/// `grid` has HDF5 dimensions `[3, 2]`. Its `id` values are `0, 1, 2` and
/// `10, 11, 12` by MATLAB row, paired with tags `a` through `f`. `nested` has
/// dimensions `[2, 1]`, `fieldA` values `[1, 2]`, and scalar `inner` structures
/// whose `p` values are 0 and 100. `scalar` has direct value datasets with the
/// same fields as the first element of `row`.
///
/// Array fields store object references in column-major order. The targets are
/// linked under `#refs#`, including the nested scalar groups. Structure attributes
/// contain `MATLAB_class = "struct"` and global-heap-backed `MATLAB_fields`.
/// Value datasets have `MATLAB_class` attributes, and character datasets also
/// have `MATLAB_int_decode = 4`.
pub fn struct_array() -> Vec<u8> {
    let mut builder = Builder::new();
    let row = builder.data_array([6, 1]);
    let col = builder.data_array([1, 6]);
    let ids = (0..6)
        .map(|i| builder.target(Value::Double((i / 3 * 10 + i % 3) as f64)))
        .collect::<Vec<_>>();
    let tags = (b'a'..=b'f')
        .map(|c| builder.target(Value::Char(char::from(c))))
        .collect::<Vec<_>>();
    let grid = builder.array(&[("id", ids), ("tag", tags)], [3, 2]);
    let field_a = [1.0, 2.0].map(|value| builder.target(Value::Double(value)));
    let inner = [0.0, 100.0].map(|value| {
        let p = builder.dataset(Value::Double(value), 2);
        builder.record(p);
        let group = builder.structure(&[("p", p)]);
        builder.record(group);
        group
    });
    let nested = builder.array(
        &[("fieldA", field_a.to_vec()), ("inner", inner.to_vec())],
        [2, 1],
    );
    let a = builder.dataset(Value::Double(1.0), 1);
    let b = builder.dataset(Value::Char('a'), 1);
    let c = builder.dataset(Value::Vector, 1);
    let scalar = builder.structure(&[("fieldA", a), ("fieldB", b), ("fieldC", c)]);
    builder.finish(&[
        ("col", col),
        ("grid", grid),
        ("nested", nested),
        ("row", row),
        ("scalar", scalar),
    ])
}

/// Builds a deliberately malformed structure with equal field counts and different shapes.
///
/// `mismatched/a` has HDF5 dimensions `[3, 2]`, and `mismatched/b` has dimensions
/// `[2, 3]`. Each field refers to six distinct numeric targets containing 0
/// through 5. Their attributes, reference targets, and HDF5 storage are valid,
/// but the field shapes disagree under the MATLAB struct-array convention.
pub fn mismatched_struct_array() -> Vec<u8> {
    let mut builder = Builder::new();
    let a = (0..6)
        .map(|i| builder.target(Value::Double(i as f64)))
        .collect::<Vec<_>>();
    let b = (0..6)
        .map(|i| builder.target(Value::Double(i as f64)))
        .collect::<Vec<_>>();
    let a = builder.references(&a, [3, 2]);
    let b = builder.references(&b, [2, 3]);
    let group = builder.structure(&[("a", a), ("b", b)]);
    builder.finish(&[("mismatched", group)])
}

/// An HDF5 image with a separate list of links for the root `#refs#` group.
struct Builder {
    image: Image,
    targets: Vec<(String, u64)>,
}

impl Builder {
    fn new() -> Self {
        Self {
            image: Image::starting_with(&[0; 96]),
            targets: Vec::new(),
        }
    }

    fn data_array(&mut self, shape: [u64; 2]) -> u64 {
        let a = (1..=6)
            .map(|i| self.target(Value::Double(i as f64)))
            .collect();
        let b = (b'a'..=b'f')
            .map(|c| self.target(Value::Char(char::from(c))))
            .collect();
        let c = (0..6).map(|_| self.target(Value::Vector)).collect();
        self.array(&[("fieldA", a), ("fieldB", b), ("fieldC", c)], shape)
    }

    fn array(&mut self, fields: &[(&str, Vec<u64>)], shape: [u64; 2]) -> u64 {
        let links = fields
            .iter()
            .map(|(name, targets)| (*name, self.references(targets, shape)))
            .collect::<Vec<_>>();
        self.structure(&links)
    }

    /// Stores references to row-major `targets` in column-major order for HDF5 `shape`.
    fn references(&mut self, targets: &[u64], shape: [u64; 2]) -> u64 {
        let rows = shape[1] as usize;
        let cols = shape[0] as usize;
        assert_eq!(targets.len(), rows * cols);
        let raw = (0..cols)
            .flat_map(|col| (0..rows).flat_map(move |row| targets[row * cols + col].to_le_bytes()))
            .collect::<Vec<_>>();
        self.raw_dataset(
            &datatype::header(Class::REFERENCE, DATATYPE_VERSION, NO_FLAGS, 8),
            &raw,
            shape,
            &[],
            1,
        )
    }

    /// Writes a value dataset and records its link under `#refs#`.
    fn target(&mut self, value: Value) -> u64 {
        let address = self.dataset(value, 1);
        self.record(address);
        address
    }

    /// Records a sequential `rN` link without changing the target's header reference count.
    fn record(&mut self, address: u64) {
        self.targets
            .push((format!("r{}", self.targets.len()), address));
    }

    fn dataset(&mut self, value: Value, references: u32) -> u64 {
        let (datatype, raw, shape, class) = match value {
            Value::Char(c) => (
                u16_datatype(),
                u16::try_from(u32::from(c)).unwrap().to_le_bytes().to_vec(),
                [1, 1],
                "char",
            ),
            Value::Double(value) => (
                datatype::f64_le(),
                value.to_le_bytes().to_vec(),
                [1, 1],
                "double",
            ),
            Value::Vector => (
                datatype::f64_le(),
                [-6.0f64, -5.0, -4.0, -3.0, -2.0, -1.0]
                    .into_iter()
                    .flat_map(f64::to_le_bytes)
                    .collect(),
                [6, 1],
                "double",
            ),
        };
        let mut attrs = vec![attribute(
            "MATLAB_class",
            &datatype::fixed_string(class.len() as u32),
            &[],
            class.as_bytes(),
        )];
        if matches!(value, Value::Char(_)) {
            attrs.push(attribute(
                "MATLAB_int_decode",
                &legacy::i32_datatype(),
                &[],
                &4i32.to_le_bytes(),
            ));
        }
        self.raw_dataset(&datatype, &raw, shape, &attrs, references)
    }

    fn raw_dataset(
        &mut self,
        datatype: &[u8],
        raw: &[u8],
        shape: [u64; 2],
        attrs: &[Message],
        references: u32,
    ) -> u64 {
        let address = self.image.append_aligned(raw, legacy::MESSAGE_ALIGNMENT) as u64;
        self.image.append_aligned(
            &Header::new()
                .reference_count(references)
                .messages([
                    message(
                        MessageType::DATASPACE,
                        dataspace::v1(2, Flags::NONE, &shape, None),
                    ),
                    message(MessageType::DATATYPE, datatype.to_vec()),
                    message(
                        MessageType::DATA_LAYOUT,
                        data_layout::contiguous(address, raw.len() as u64),
                    ),
                ])
                .messages(attrs.iter().cloned())
                .build(),
            legacy::MESSAGE_ALIGNMENT,
        ) as u64
    }

    /// Writes a symbol-table group with MATLAB class and global-heap field-name attributes.
    fn structure(&mut self, links: &[(&str, u64)]) -> u64 {
        let objects = links
            .iter()
            .enumerate()
            .map(|(i, (name, _))| Object::new((i + 1) as u16, name.as_bytes()))
            .collect::<Vec<_>>();
        let heap = self.image.append_aligned(
            &global_heap::collection(&objects, Widths::EIGHT),
            legacy::MESSAGE_ALIGNMENT,
        ) as u64;
        let raw = links
            .iter()
            .enumerate()
            .flat_map(|(i, (name, _))| {
                global_heap::reference(name.len() as u32, heap, (i + 1) as u32, Widths::EIGHT)
            })
            .collect::<Vec<_>>();
        let mut datatype = datatype::header(Class::VARIABLE_LENGTH, DATATYPE_VERSION, NO_FLAGS, 16);
        datatype.extend_from_slice(&datatype::fixed_string(1));
        legacy::group_with_messages(
            &mut self.image,
            links,
            &[
                attribute("MATLAB_class", &datatype::fixed_string(6), &[], b"struct"),
                attribute("MATLAB_fields", &datatype, &[links.len() as u64], &raw),
            ],
        )
    }

    /// Writes the root and `#refs#` groups and prefixes the image with its user block.
    fn finish(mut self, links: &[(&str, u64)]) -> Vec<u8> {
        self.targets.sort();
        let targets = self
            .targets
            .iter()
            .map(|(name, address)| (name.as_str(), *address))
            .collect::<Vec<_>>();
        let refs = legacy::group_with_messages(&mut self.image, &targets, &[]);
        let root = legacy::group_with_messages(
            &mut self.image,
            &[&[("#refs#", refs)], links].concat(),
            &[],
        );
        self.image.place(
            0,
            &Superblock::new(Widths::EIGHT)
                .base_address(USERBLOCK_SIZE as u64)
                .root_group(0, root)
                .eof_address((USERBLOCK_SIZE + self.image.as_bytes().len()) as u64)
                .build(),
        );
        let mut bytes = vec![0; USERBLOCK_SIZE];
        bytes.extend_from_slice(&self.image.build());
        bytes
    }
}

#[derive(Clone, Copy)]
enum Value {
    Char(char),
    Double(f64),
    Vector,
}

fn attribute(name: &str, datatype: &[u8], shape: &[u64], raw: &[u8]) -> Message {
    message(
        MessageType::ATTRIBUTE,
        Attribute::new(
            name,
            datatype,
            &dataspace::v1(shape.len() as u8, Flags::NONE, shape, None),
            raw,
        )
        .build(),
    )
}

fn message(kind: MessageType, mut data: Vec<u8>) -> Message {
    data.resize(data.len().next_multiple_of(legacy::MESSAGE_ALIGNMENT), 0);
    Message::new(kind, &data)
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.
fn u16_datatype() -> Vec<u8> {
    let mut datatype = datatype::header(Class::FIXED_POINT, DATATYPE_VERSION, NO_FLAGS, 2);
    datatype.extend_from_slice(&0u16.to_le_bytes());
    datatype.extend_from_slice(&16u16.to_le_bytes());
    datatype
}

// Sections `subsec_fmt4_boot_super` and `subsubsec_fmt4_dataobject_hdr_msg_dtmessage`, version 4.0.
const USERBLOCK_SIZE: usize = 512;
const DATATYPE_VERSION: u8 = 1;
const NO_FLAGS: [u8; 3] = [0; 3];
