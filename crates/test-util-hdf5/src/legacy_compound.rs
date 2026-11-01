//! Compound array observations through the dataset API.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::{DatatypeByteOrder, File};
//! use test_util_hdf5::legacy_compound;
//!
//! let file = File::from_bytes(test_util::legacy_compound::file())?;
//! legacy_compound::assert_contents(&file, DatatypeByteOrder::LittleEndian);
//! # Ok::<(), hdf5_pure::Error>(())
//! ```

use hdf5_pure::CompoundField;
use hdf5_pure::CompoundType;
use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::FormatError;

/// Asserts `data` has one compound element with the complete `arr` array and its values.
///
/// # Panics
///
/// Panics if metadata, payload bytes, or decoded values differ from the fixture, or if scalar
/// field decoding accepts the array member.
pub fn assert_contents(file: &File, order: DatatypeByteOrder) {
    let dataset = file.dataset("data").unwrap();
    let datatype = dataset.datatype().unwrap();
    assert_eq!(datatype, array_compound(order));
    assert_eq!(datatype.type_size(), 16);
    let Datatype::Compound { members, .. } = &datatype else {
        panic!("expected compound, got {datatype:?}");
    };
    assert_eq!(members[0].datatype.type_size(), 16);
    assert_eq!(dataset.shape().unwrap(), [1]);
    let raw = dataset.read_raw().unwrap();
    let expected: Vec<_> = [11i32, 22, 33, 44]
        .into_iter()
        .flat_map(|value| match order {
            DatatypeByteOrder::LittleEndian => value.to_le_bytes(),
            DatatypeByteOrder::BigEndian => value.to_be_bytes(),
            DatatypeByteOrder::Vax => panic!("fixture requires integer byte order"),
        })
        .collect();
    assert_eq!(raw, expected);
    assert_eq!(
        dataset.read_compound::<ArrayRecord>().unwrap(),
        [ArrayRecord([11, 22, 33, 44])]
    );
    assert_eq!(
        i32::decode_field(&members[0].datatype, &raw).unwrap_err(),
        FormatError::CompoundFieldTypeMismatch(String::new()),
    );
}

#[derive(Debug, PartialEq)]
struct ArrayRecord([i32; 4]);

impl CompoundType for ArrayRecord {
    fn datatype() -> Result<Datatype, FormatError> {
        Ok(array_compound(DatatypeByteOrder::LittleEndian))
    }

    fn encode(&self, output: &mut Vec<u8>) {
        output.extend(self.0.iter().flat_map(|value| value.to_le_bytes()));
    }

    fn decode(datatype: &Datatype, bytes: &[u8]) -> Result<Self, FormatError> {
        let Datatype::Compound { members, .. } = datatype else {
            return Err(FormatError::CompoundFieldTypeMismatch("arr".into()));
        };
        let [member] = members.as_slice() else {
            return Err(FormatError::CompoundFieldMissing("arr".into()));
        };
        let Datatype::Array {
            base_type,
            dimensions,
        } = &member.datatype
        else {
            return Err(FormatError::CompoundFieldTypeMismatch("arr".into()));
        };
        if dimensions != &[4] || member.byte_offset != 0 {
            return Err(FormatError::CompoundFieldTypeMismatch("arr".into()));
        }
        let bytes: &[u8; 16] =
            bytes
                .try_into()
                .map_err(|_error| FormatError::DataSizeMismatch {
                    expected: 16,
                    actual: bytes.len(),
                })?;
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| i32::decode_field(base_type, chunk))
            .collect::<Result<_, _>>()?;
        let values: [i32; 4] =
            values
                .try_into()
                .map_err(|_error| FormatError::DataSizeMismatch {
                    expected: 16,
                    actual: bytes.len(),
                })?;
        Ok(Self(values))
    }
}

fn array_compound(order: DatatypeByteOrder) -> Datatype {
    CompoundTypeBuilder::with_size(16)
        .field(
            "arr",
            0,
            Datatype::Array {
                base_type: Box::new(Datatype::FixedPoint(
                    FixedPointType::new(4, order, true, 0, 32).unwrap(),
                )),
                dimensions: vec![4],
            },
        )
        .build()
        .unwrap()
}
