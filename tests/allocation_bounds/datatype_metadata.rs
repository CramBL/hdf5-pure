use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::VariableLengthType;
use hdf5_pure_format::__private as format;
use rstest::rstest;
use test_util::allocation;
use test_util::datatype::Class;

#[rstest]
#[case::vl_depth_16(Shape::Sequence(16))]
#[case::vl_depth_64(Shape::Sequence(64))]
#[case::vl_depth_256(Shape::Sequence(256))]
#[case::array_depth_16(Shape::Array(16))]
#[case::array_depth_64(Shape::Array(64))]
#[case::array_depth_256(Shape::Array(256))]
#[case::wide_compound(Shape::Compound(256))]
fn datatype_encoding_allocations_follow_output_growth(#[case] shape: Shape) {
    let datatype = shape.datatype();
    let names = match &datatype {
        Datatype::Compound { members, .. } => members
            .iter()
            .map(|member| member.name.as_str())
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    let region = format!("datatype_metadata_{shape:?}");
    let (control, control_metrics) =
        allocation::measure(&format!("{region}_child_buffers"), || {
            encode_with_child_buffers(shape, &names)
        });
    let (expected, copied) = control;
    let control_capacity = expected.capacity();
    assert!(control_metrics.blocks > 0, "{control_metrics}");
    assert!(
        control_metrics.bytes >= expected.len() as u64,
        "{control_metrics}"
    );
    assert_eq!(
        control_metrics.live_bytes, control_capacity as u64,
        "{control_metrics}"
    );
    assert!(
        control_metrics.peak_bytes >= control_metrics.live_bytes,
        "{control_metrics}"
    );
    let (encoded, measured) =
        allocation::measure(&region, || format::serialize_datatype(&datatype).unwrap());
    assert_eq!(encoded, expected);
    assert_eq!(encoded.len(), shape.encoded_len());
    let capacity = encoded.capacity() as u64;
    let length = encoded.len() as u64;
    // Earlier allocation requests bound potential destination growth copies.
    let growth_bound = measured.bytes - capacity;
    eprintln!(
        "{region}: output {length}, capacity {capacity}, destination growth copy bound \
         {growth_bound}, child-buffer copies {copied}"
    );
    assert!(capacity >= length);
    assert!(capacity <= 4 * length);
    assert!(measured.blocks > 0, "{measured}");
    assert!(
        measured.blocks <= 2 * u64::from(u64::BITS - capacity.leading_zeros()) + 4,
        "{measured}"
    );
    assert_eq!(measured.live_bytes, capacity, "{measured}");
    assert!(measured.peak_bytes >= capacity, "{measured}");
    assert!(measured.peak_bytes <= 2 * capacity, "{measured}");
    assert!(measured.bytes >= measured.peak_bytes, "{measured}");
    assert!(measured.bytes <= 4 * capacity, "{measured}");
    assert!(
        measured.blocks < control_metrics.blocks,
        "{measured}; {control_metrics}"
    );
    assert!(
        measured.bytes < control_metrics.bytes,
        "{measured}; {control_metrics}"
    );
    if let Shape::Sequence(depth) | Shape::Array(depth) = shape {
        assert!(copied >= length * depth as u64 / 3);
        assert!(growth_bound + length < copied);
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Array(usize),
    Compound(usize),
    Sequence(usize),
}

impl Shape {
    fn datatype(self) -> Datatype {
        match self {
            Self::Sequence(depth) => (0..depth).fold(hdf5_pure::make_i32_type(), |base, _| {
                Datatype::VariableLength(VariableLengthType::sequence(base))
            }),
            Self::Array(depth) => {
                (0..depth).fold(hdf5_pure::make_i32_type(), |base, _| Datatype::Array {
                    base_type: Box::new(base),
                    dimensions: vec![1],
                })
            }
            Self::Compound(width) => {
                let mut builder = CompoundTypeBuilder::with_size(u32::try_from(width * 4).unwrap());
                for index in 0..width {
                    builder = builder.field(
                        &format!("field{index:04}"),
                        u64::try_from(index * 4).unwrap(),
                        hdf5_pure::make_i32_type(),
                    );
                }
                builder.build().unwrap()
            }
        }
    }

    fn encoded_len(self) -> usize {
        match self {
            Self::Sequence(depth) => 8 * depth + 12,
            Self::Array(depth) => 13 * depth + 12,
            Self::Compound(width) => 8 + width * (10 + 2 + 12),
        }
    }
}

/// Returns independent datatype bytes and the total child bytes copied into parent buffers.
fn encode_with_child_buffers(shape: Shape, names: &[&str]) -> (Vec<u8>, u64) {
    let mut copied = 0;
    let encoded = match shape {
        Shape::Sequence(0) | Shape::Array(0) => {
            let mut encoded = control_header(Class::FIXED_POINT, 1, [8, 0, 0], 4);
            encoded.extend_from_slice(&[0, 0, 32, 0]);
            encoded
        }
        Shape::Sequence(depth) => {
            let child = encode_with_child_buffers(Shape::Sequence(depth - 1), &[]);
            let mut encoded = control_header(Class::VARIABLE_LENGTH, 1, [0; 3], 16);
            copied += child.1 + child.0.len() as u64;
            encoded.extend_from_slice(&child.0);
            encoded
        }
        Shape::Array(depth) => {
            let child = encode_with_child_buffers(Shape::Array(depth - 1), &[]);
            let mut encoded = control_header(Class::ARRAY, 3, [0; 3], 4);
            encoded.extend_from_slice(&[1, 1, 0, 0, 0]);
            copied += child.1 + child.0.len() as u64;
            encoded.extend_from_slice(&child.0);
            encoded
        }
        Shape::Compound(width) => {
            let count = u16::try_from(width).unwrap().to_le_bytes();
            let mut encoded = control_header(
                Class::COMPOUND,
                3,
                [count[0], count[1], 0],
                u32::try_from(width * 4).unwrap(),
            );
            assert_eq!(names.len(), width);
            for (index, name) in names.iter().enumerate() {
                encoded.extend_from_slice(name.as_bytes());
                encoded.push(0);
                encoded.extend_from_slice(&u16::try_from(index * 4).unwrap().to_le_bytes());
                let child = encode_with_child_buffers(Shape::Array(0), &[]);
                copied += child.1 + child.0.len() as u64;
                encoded.extend_from_slice(&child.0);
            }
            encoded
        }
    };
    (std::hint::black_box(encoded), copied)
}

fn control_header(class: Class, version: u8, flags: [u8; 3], size: u32) -> Vec<u8> {
    let mut header = Vec::with_capacity(8);
    header.push(class.0 | (version << 4));
    header.extend_from_slice(&flags);
    header.extend_from_slice(&size.to_le_bytes());
    header
}
