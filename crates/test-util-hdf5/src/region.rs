//! Reference files and unit-stride selections for dataset region crosschecks.

use std::fmt::Debug;
use std::path::Path;

use hdf5::dataset::Layout;
use hdf5::datatype::ByteOrder;
use hdf5::types::{
    FixedAscii, FixedUnicode, FloatSize, TypeDescriptor, VarLenAscii, VarLenUnicode,
};
use hdf5::{H5Type, Hyperslab, SliceOrIndex};
use ndarray::IxDyn;

use crate::file;

/// String encodings used in reference region files.
#[derive(Clone, Copy, Debug)]
pub enum StringKind {
    /// ASCII text in a fixed 16-byte element.
    FixedAscii,
    /// UTF-8 text in a fixed 16-byte element.
    FixedUnicode,
    /// ASCII text in a variable-length element.
    VariableAscii,
    /// UTF-8 text in a variable-length element.
    VariableUnicode,
}

impl StringKind {
    /// Returns the text written at the linear index.
    pub fn label(self, index: u8) -> String {
        match self {
            Self::FixedAscii | Self::VariableAscii => format!("label-{index}"),
            Self::FixedUnicode | Self::VariableUnicode => format!("étage-{index}"),
        }
    }

    /// Writes a reference string dataset and reads the selected text through the C library.
    ///
    /// The dataset is named `data`. For fixed-length chunked strings, the writer applies shuffle,
    /// deflate, and Fletcher32 filters. Variable-length chunked strings are unfiltered.
    ///
    /// # Panics
    ///
    /// Panics if creating, writing, or reading the reference file fails.
    pub fn generate(
        self,
        path: &Path,
        layout: Layout,
        shape: &[usize],
        start: &[usize],
        count: &[usize],
    ) -> Vec<String> {
        match self {
            Self::FixedAscii => write_strings(path, layout, shape, start, count, |text| {
                FixedAscii::<16>::from_ascii(text).unwrap()
            }),
            Self::FixedUnicode => write_strings(path, layout, shape, start, count, |text| {
                text.replace("label", "étage")
                    .parse::<FixedUnicode<16>>()
                    .unwrap()
            }),
            Self::VariableAscii => write_strings(path, layout, shape, start, count, |text| {
                VarLenAscii::from_ascii(text).unwrap()
            }),
            Self::VariableUnicode => write_strings(path, layout, shape, start, count, |text| {
                text.replace("label", "étage")
                    .parse::<VarLenUnicode>()
                    .unwrap()
            }),
        }
    }
}

/// Writes a reference numeric dataset named `data` with consecutive values starting at zero.
///
/// For chunked datasets, the writer applies shuffle, deflate, and Fletcher32 filters.
///
/// # Panics
///
/// Panics if the element count exceeds 256, a value cannot be converted to `T`, or writing the
/// reference file fails.
pub fn write_numeric<T: H5Type + TryFrom<u8>>(path: &Path, layout: Layout, shape: &[usize])
where
    T::Error: Debug,
{
    let values: Vec<T> = (0..shape.iter().product())
        .map(|index| T::try_from(u8::try_from(index).unwrap()).unwrap())
        .collect();
    write_dataset(path, layout, shape, &values);
}

fn write_strings<T: H5Type + AsRef<str>>(
    path: &Path,
    layout: Layout,
    shape: &[usize],
    start: &[usize],
    count: &[usize],
    parse: impl Fn(&str) -> T,
) -> Vec<String> {
    let values: Vec<_> = (0..shape.iter().product())
        .map(|index| parse(&format!("label-{index}")))
        .collect();
    write_dataset(path, layout, shape, &values);
    hdf5::File::open(path)
        .unwrap()
        .dataset("data")
        .unwrap()
        .read_slice::<T, _, IxDyn>(selection(start, count))
        .unwrap()
        .iter()
        .map(|value| value.as_ref().to_owned())
        .collect()
}

fn write_dataset<T: H5Type>(path: &Path, layout: Layout, shape: &[usize], values: &[T]) {
    let reference = file::libhdf5_create_v18(path);
    let builder = reference.new_dataset::<T>();
    let builder = if layout == Layout::Chunked {
        let builder = builder.chunk(
            shape
                .iter()
                .zip([2, 3, 4])
                .map(|(_, edge)| edge)
                .collect::<Vec<_>>(),
        );
        match T::type_descriptor() {
            TypeDescriptor::VarLenAscii | TypeDescriptor::VarLenUnicode => builder,
            _ => builder.shuffle().deflate(4).fletcher32(),
        }
    } else {
        builder.layout(layout)
    };
    let dataset = builder.shape(shape).create("data").unwrap();
    dataset.write_raw(values).unwrap();
    drop(dataset);
    reference.close().unwrap();
}

/// Encodes numeric values in the reference datatype's byte order and element width.
///
/// # Panics
///
/// Panics if the datatype descriptor cannot be read or is not an integer or floating-point type.
pub fn numeric_bytes(datatype: &hdf5::Datatype, values: &[u8]) -> Vec<u8> {
    let descriptor = datatype.to_descriptor().unwrap();
    values
        .iter()
        .flat_map(|&value| {
            let mut bytes = match descriptor {
                TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => {
                    u64::from(value).to_le_bytes()[..datatype.size()].to_vec()
                }
                TypeDescriptor::Float(FloatSize::U4) => f32::from(value).to_le_bytes().to_vec(),
                TypeDescriptor::Float(FloatSize::U8) => f64::from(value).to_le_bytes().to_vec(),
                _ => panic!("expected a numeric reference datatype"),
            };
            if datatype.byte_order() == ByteOrder::BigEndian {
                bytes.reverse();
            }
            bytes
        })
        .collect()
}

/// Returns a unit-stride [`Hyperslab`] with the supplied starting coordinates and element counts.
///
/// Coordinates are paired with counts in slice order, up to the shorter slice's length.
pub fn selection(start: &[usize], count: &[usize]) -> Hyperslab {
    start
        .iter()
        .zip(count)
        .map(|(&start, &count)| SliceOrIndex::SliceCount {
            start,
            step: 1,
            count,
            block: 1,
        })
        .collect::<Vec<_>>()
        .into()
}
