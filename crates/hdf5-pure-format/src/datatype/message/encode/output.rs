//! Field positions and common headers for writing a prepared datatype.

use core::slice;

use hdf5_pure_core::__private;

use super::prepared::PreparedDatatype;
use super::source::{Body, Description};
use crate::bytes;
use crate::codec::Encode;
use crate::datatype::header::{DATATYPE_HEADER_SIZE, DATATYPE_VERSION_SHIFT, REFERENCE_CLASS};
use crate::datatype::message::DatatypeVersion;

impl Encode for PreparedDatatype<'_> {
    type Error = bytes::UnexpectedEof;

    fn encoded_len(&self) -> usize {
        self.encoded_size()
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let available = output.len();
        let body = output
            .get_mut(..self.encoded_size())
            .ok_or_else(|| bytes::UnexpectedEof::new(self.encoded_size(), available))?;
        self.write_body(body);
        Ok(self.encoded_size())
    }
}

impl Description<'_> {
    /// Reserves the header position before writing the properties that determine its contents.
    pub(super) fn write(&self, output: &mut Output<'_, impl FnMut(usize, &[u8])>) -> NodeHeader {
        let position = output.advance(DATATYPE_HEADER_SIZE);
        let version = self.version();
        let header = match self.body() {
            Body::Atomic(properties) => properties.write(output, version),
            Body::Reference { size, kind } => NodeHeader {
                class: REFERENCE_CLASS,
                version: DatatypeVersion::One,
                flags: [kind, 0, 0],
                size,
            },
            Body::Container(properties) => properties.write(output, version),
            #[expect(
                clippy::unreachable,
                reason = "the immutable source has a supported encoding"
            )]
            Body::Unsupported(_) => {
                // Preparation rejects unsupported descriptions before retaining the source.
                unreachable!("a prepared datatype has a supported description")
            }
        };
        (output.write_field)(position, &header.to_bytes());
        header
    }
}

/// The fields of the common header, including the selected version and stored element size.
pub(in crate::datatype::message) struct NodeHeader {
    pub(in crate::datatype::message) class: u8,
    pub(in crate::datatype::message) version: DatatypeVersion,
    /// The three class bit-field bytes.
    pub(in crate::datatype::message) flags: [u8; 3],
    /// The stored element size in bytes.
    pub(in crate::datatype::message) size: u32,
}

impl NodeHeader {
    /// Packs the class into the low four bits and the version into the high four bits of byte zero.
    ///
    /// "The Datatype Message" of the [format specification, version 4.0][spec] defines the header.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
    fn to_bytes(&self) -> [u8; DATATYPE_HEADER_SIZE] {
        let &Self {
            class,
            version,
            flags: [first, second, third],
            size,
        } = self;
        let [low, middle_low, middle_high, high] = size.to_le_bytes();
        [
            class | (version.get() << DATATYPE_VERSION_SHIFT),
            first,
            second,
            third,
            low,
            middle_low,
            middle_high,
            high,
        ]
    }
}

/// The field writer, current byte position, and sizes selected during preparation.
pub(in crate::datatype::message) struct Output<'a, WriteField> {
    write_field: WriteField,
    position: usize,
    descriptor_size: u32,
    computed_sizes: slice::Iter<'a, u32>,
}

impl<WriteField: FnMut(usize, &[u8])> Output<'_, WriteField> {
    /// Passes each field and its byte position to `write_field`, with headers after their properties.
    pub(super) fn emit(prepared: &PreparedDatatype<'_>, write_field: WriteField) {
        let (description, offset_width, computed_sizes) = prepared.emission_parts();
        let mut output = Output {
            write_field,
            position: 0,
            descriptor_size: __private::variable_length_descriptor_size(offset_width),
            computed_sizes: computed_sizes.iter(),
        };
        description.write(&mut output);
        debug_assert_eq!(output.position, prepared.encoded_size());
        debug_assert!(output.computed_sizes.as_slice().is_empty());
    }

    /// Returns the variable-length descriptor size at the destination offset width.
    pub(super) fn descriptor_size(&self) -> u32 {
        self.descriptor_size
    }

    /// Passes `bytes` to the field writer at the current position and advances by their length.
    pub(in crate::datatype::message) fn write(&mut self, bytes: &[u8]) {
        let position = self.advance(bytes.len());
        (self.write_field)(position, bytes);
    }

    fn advance(&mut self, len: usize) -> usize {
        let position = self.position;
        self.position += len;
        position
    }

    /// Returns the next prepared element size in child-before-parent order.
    ///
    /// # Panics
    ///
    /// Panics if the writer has consumed every prepared size.
    #[expect(
        clippy::expect_used,
        reason = "preparation records computed sizes in the same child-before-parent order"
    )]
    pub(super) fn next_size(&mut self) -> u32 {
        *self.computed_sizes.next().expect("a prepared size exists")
    }
}
