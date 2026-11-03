//! Fill encodings retained for dataset preparation and checked element bytes.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::datatype::Datatype;
#[cfg(feature = "std")]
use crate::element::H5Element;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// A fill element and its typed or source-derived storage encoding, pending preparation.
#[derive(Clone, Debug)]
pub(crate) struct StagedFill {
    encoding: FillEncoding,
    bytes: Vec<u8>,
}

impl StagedFill {
    /// Encodes one element and retains its write datatype or type-construction error.
    #[cfg(feature = "std")]
    pub(crate) fn typed<T: H5Element>(value: T) -> Self {
        Self {
            encoding: FillEncoding::Typed(T::datatype()),
            bytes: T::fill_bytes(value),
        }
    }

    /// Retains raw fill bytes with the resolved datatype of their source dataset.
    #[cfg(feature = "std")]
    pub(crate) fn source(datatype: Datatype, bytes: Vec<u8>) -> Self {
        Self {
            encoding: FillEncoding::Source(datatype),
            bytes,
        }
    }
}

#[derive(Clone, Debug)]
enum FillEncoding {
    Source(Datatype),
    Typed(Result<Datatype, FormatError>),
}

/// One complete fill element with an encoding compatible with its dataset datatype.
///
/// The parser checks the staged encoding and width before retaining the bytes for emission.
pub(crate) struct PreparedFill {
    bytes: Vec<u8>,
}

impl PreparedFill {
    /// Parses a fill against the dataset's checked datatype and nonzero element width.
    ///
    /// Accepts either byte order for the full-width unsigned-byte base of a typed array fill.
    /// Requires equal datatypes for other typed fills and source-derived fills.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FillValueSizeMismatch`] if the payload is not one element wide,
    /// or [`FormatError::FillValueDatatypeMismatch`] if its storage encoding is incompatible.
    /// Propagates a staged type-construction error before checking the payload.
    pub(super) fn parse(
        staged: StagedFill,
        datatype: &Datatype,
        element_size: NonZeroUsize,
    ) -> Result<Self, FormatError> {
        let StagedFill { encoding, bytes } = staged;
        let (source, typed) = match encoding {
            FillEncoding::Source(datatype) => (datatype, false),
            FillEncoding::Typed(datatype) => (datatype?, true),
        };
        if bytes.len() != element_size.get() {
            return Err(FormatError::FillValueSizeMismatch {
                expected: element_size.get(),
                actual: bytes.len(),
            });
        }
        if source != *datatype && !(typed && Self::byte_array_matches(&source, datatype)) {
            return Err(FormatError::FillValueDatatypeMismatch);
        }
        Ok(Self { bytes })
    }

    /// Converts variable-length fill references from eight-byte heap addresses to `offsets`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`narrow_variable_length_references`][narrow] for invalid element
    /// layouts, buffer ranges, or heap addresses.
    ///
    /// [narrow]: hdf5_pure_format::__private::narrow_variable_length_references
    pub(crate) fn narrow_variable_length(
        &mut self,
        datatype: &Datatype,
        offsets: OffsetWidth,
    ) -> Result<(), FormatError> {
        hdf5_pure_format::__private::narrow_variable_length_references(
            offsets,
            datatype,
            &mut self.bytes,
            None,
        )
    }

    /// Returns the complete encoded fill element.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn byte_array_matches(source: &Datatype, datatype: &Datatype) -> bool {
        match (source, datatype) {
            (
                Datatype::Array {
                    dimensions: source, ..
                },
                Datatype::Array {
                    dimensions,
                    base_type,
                },
            ) => {
                source == dimensions
                    && matches!(base_type.as_ref(), Datatype::FixedPoint(base)
                        if !base.layout().signed() && base.size().get() == 1
                            && base.layout().bit_offset() == 0 && base.layout().bit_precision() == 8)
            }
            _ => false,
        }
    }
}
#[cfg(test)]
mod tests {
    use core::num::NonZeroUsize;

    use rstest::rstest;
    use test_util::datatype::Class;

    use crate::datatype::{CharacterSet, Datatype, StringPadding};
    use crate::error::FormatError;
    use crate::type_builders;

    use super::{FillEncoding, PreparedFill, StagedFill};

    #[rstest]
    #[case::space_padding(StringPadding::SpacePad, CharacterSet::Ascii)]
    #[case::terminated_padding(StringPadding::NullTerminate, CharacterSet::Ascii)]
    #[case::utf8(StringPadding::NullPad, CharacterSet::Utf8)]
    fn source_fill_preserves_its_encoding(
        #[case] padding: StringPadding,
        #[case] charset: CharacterSet,
    ) {
        let datatype = Datatype::String {
            size: 4,
            padding,
            charset,
        };
        let bytes = b"a\0b ".to_vec();
        let staged = StagedFill {
            encoding: FillEncoding::Source(datatype.clone()),
            bytes: bytes.clone(),
        };
        let prepared =
            PreparedFill::parse(staged, &datatype, NonZeroUsize::new(4).unwrap()).unwrap();
        assert_eq!(prepared.as_bytes(), bytes);
    }

    #[test]
    fn source_fill_rejects_a_changed_destination_datatype() {
        let staged = StagedFill {
            encoding: FillEncoding::Source(type_builders::make_f32_type()),
            bytes: vec![0; 4],
        };
        let result = PreparedFill::parse(
            staged,
            &type_builders::make_i32_type(),
            NonZeroUsize::new(4).unwrap(),
        );
        let Err(error) = result else {
            panic!("expected an incompatible source fill");
        };
        assert_eq!(error, FormatError::FillValueDatatypeMismatch);
    }

    #[rstest]
    #[case::zero_ascii(FormatError::ZeroFixedStringWidth)]
    #[case::zero_array(FormatError::ZeroSizedDatatype { class: Class::ARRAY.0 })]
    fn staged_encoding_errors_precede_width_checks(#[case] expected: FormatError) {
        let staged = StagedFill {
            encoding: FillEncoding::Typed(Err(expected.clone())),
            bytes: Vec::new(),
        };
        let result = PreparedFill::parse(
            staged,
            &type_builders::make_i32_type(),
            NonZeroUsize::new(4).unwrap(),
        );
        let Err(error) = result else {
            panic!("expected an encoding error");
        };
        assert_eq!(error, expected);
    }
}
