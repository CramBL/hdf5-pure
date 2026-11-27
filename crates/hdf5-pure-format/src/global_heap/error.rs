use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::LengthWidth;

/// An error encoding a global heap collection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlobalHeapCollectionError {
    /// Arithmetic, host-size conversion, or buffer reservation failed.
    Format(FormatError),
    /// The size of the collection does not fit a length field of `length_width` bytes.
    TooLarge {
        /// The size in bytes of the collection, padding included.
        size: u64,
        /// The width of the collection's length fields.
        length_width: LengthWidth,
    },
    /// The caller passed more objects than [`GLOBAL_HEAP_MAX_OBJECTS`](super::GLOBAL_HEAP_MAX_OBJECTS).
    TooManyObjects {
        /// The number of objects the caller passed.
        count: usize,
    },
}

impl From<FormatError> for GlobalHeapCollectionError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<GlobalHeapCollectionError> for FormatError {
    fn from(error: GlobalHeapCollectionError) -> Self {
        match error {
            GlobalHeapCollectionError::Format(error) => error,
            GlobalHeapCollectionError::TooLarge { size, length_width } => Self::LengthTooLarge {
                length: size,
                length_size: length_width.get(),
            },
            GlobalHeapCollectionError::TooManyObjects { count } => {
                Self::ValueTooLargeForRepresentation {
                    value: count.to_u64(),
                    target: "u16",
                }
            }
        }
    }
}
