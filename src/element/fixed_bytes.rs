//! Fixed byte-array elements with matching HDF5 array datatypes.

use crate::convert::Narrow;
use crate::datatype::{self, Datatype};
use crate::edit::AppendBuilder;
use crate::error::{Error, FormatError};
use crate::reader::Dataset;
use crate::type_builders::{self, DatasetBuilder};

use super::{ElementDecoder, H5Element, sealed};

impl<const N: usize> sealed::Sealed for [u8; N] {
    fn borrowed_encoding(data: &[Self]) -> Option<Result<(&[u8], Datatype), FormatError>> {
        Some(datatype::<N>().map(|datatype| (data.as_flattened(), datatype)))
    }

    fn convert_scalar(raw: &[u8], datatype: &Datatype) -> Result<Self, Error> {
        ByteArrayDecoder::<N>::parse_decoder(datatype)?;
        raw.try_into().map_err(|_error| {
            FormatError::DataSizeMismatch {
                expected: N,
                actual: raw.len(),
            }
            .into()
        })
    }
}

impl<const N: usize> H5Element for [u8; N] {
    fn read_from(dataset: &Dataset) -> Result<Vec<Self>, Error> {
        dataset.read_whole_decoded::<Self, ByteArrayDecoder<N>>()
    }

    fn write_into(builder: &mut DatasetBuilder, data: &[Self]) {
        match datatype::<N>() {
            Ok(datatype) => {
                builder.with_raw_data(datatype, data.as_flattened().to_vec(), data.len() as u64);
            }
            Err(error) => {
                builder.with_raw_data(type_builders::make_u8_type(), Vec::new(), data.len() as u64);
                builder.data = Some(Err(Box::new(error)));
            }
        }
    }

    fn append_into(builder: &mut AppendBuilder, data: &[Self]) {
        match datatype::<N>() {
            Ok(datatype) => builder.set_dt(datatype),
            Err(error) => {
                builder.set_error(error);
                return;
            }
        }
        builder.append_raw(data.as_flattened());
    }

    fn fill_bytes(self) -> Vec<u8> {
        self.to_vec()
    }

    fn convert_raw(raw: &[u8], datatype: &Datatype) -> Result<Vec<Self>, Error> {
        let decoder = ByteArrayDecoder::<N>::parse_decoder(datatype)?;
        let mut output = Vec::new();
        decoder.append_values(raw, &mut output)?;
        Ok(output)
    }
}

/// A decoder for one-dimensional arrays of `N` full-width unsigned bytes.
pub(super) struct ByteArrayDecoder<const N: usize>;

impl<const N: usize> ElementDecoder<[u8; N]> for ByteArrayDecoder<N> {
    fn parse_decoder(stored: &Datatype) -> Result<Self, Error> {
        datatype::<N>()?;
        let compatible = match stored {
            Datatype::Array {
                base_type,
                dimensions,
            } => {
                dimensions.as_slice() == [N.narrow::<u32>()?]
                    && matches!(base_type.as_ref(), Datatype::FixedPoint(base)
                        if !base.layout().signed() && base.size().get() == 1
                            && base.layout().bit_offset() == 0 && base.layout().bit_precision() == 8)
            }
            _ => false,
        };
        if !compatible {
            return Err(FormatError::TypeMismatch {
                expected: "one-dimensional array of full-width unsigned bytes with matching width",
                actual: datatype::datatype_class_name(stored),
            }
            .into());
        }
        Ok(Self)
    }

    fn decoded_width(datatype: &Datatype) -> u32 {
        datatype.type_size()
    }

    fn append_values(&self, raw: &[u8], output: &mut Vec<[u8; N]>) -> Result<(), Error> {
        let (elements, remainder) = raw.as_chunks::<N>();
        if !remainder.is_empty() {
            return Err(FormatError::DataSizeMismatch {
                expected: raw.len() - remainder.len(),
                actual: raw.len(),
            }
            .into());
        }
        output.extend_from_slice(elements);
        Ok(())
    }
}

/// Constructs the array datatype for one element of `N` unsigned bytes.
///
/// # Errors
///
/// Returns [`FormatError::ZeroSizedDatatype`] if `N` is zero, or
/// [`FormatError::ValueTooLargeForRepresentation`] if `N` exceeds [`u32::MAX`].
fn datatype<const N: usize>() -> Result<Datatype, FormatError> {
    let width: u32 = N.narrow()?;
    if width == 0 {
        return Err(FormatError::ZeroSizedDatatype { class: ARRAY_CLASS });
    }
    Ok(Datatype::Array {
        base_type: Box::new(type_builders::make_u8_type()),
        dimensions: vec![width],
    })
}

/// The array class code in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const ARRAY_CLASS: u8 = 10;
