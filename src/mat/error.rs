//! Error type for MATLAB v7.3 serde (de)serialization.

use core::fmt;

use crate::error::{Error as Hdf5Error, FormatError};

/// Errors that can occur when (de)serializing `.mat` v7.3 files.
///
/// Marked `#[non_exhaustive]`: reading MATLAB's MCOS opaque classes is an
/// ongoing effort (`datetime`, `categorical`, `table`, `containers.Map`,
/// `dictionary`, …), and each newly decoded — or newly refused — class can
/// introduce a more specific error variant. Keeping the enum open lets those
/// additions land without a breaking change, so downstream `match`es must
/// include a wildcard arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum MatError {
    /// Underlying HDF5 I/O or format error.
    Hdf5(Hdf5Error),
    /// Underlying HDF5 format parse error.
    Format(FormatError),
    /// I/O error when reading or writing a file path.
    Io(std::io::Error),
    /// Top-level must be a struct with named fields (each field becomes a MATLAB variable).
    RootMustBeStruct,
    /// The requested Rust type has no MATLAB v7.3 encoding in this crate.
    UnsupportedType(&'static str),
    /// A sequence contained elements of different primitive types.
    MixedSequenceElementTypes,
    /// A dataset's on-disk shape didn't match the Rust type.
    ShapeMismatch {
        /// The Rust side's expectation.
        expected: String,
        /// What the file contained.
        actual: String,
    },
    /// A field of a struct array has different dimensions from the array's first field.
    ///
    /// Every field holds one reference per array element, so all fields share the array's
    /// dimensions.
    StructArrayFieldShape {
        /// The field whose dimensions differ.
        field: String,
        /// The array's dimensions, as the file stores them.
        expected: Vec<u64>,
        /// The field's dimensions, as the file stores them.
        actual: Vec<u64>,
    },
    /// A required struct field was missing from the file.
    MissingField(String),
    /// A `MATLAB_class` attribute value wasn't recognized.
    UnknownClass(String),
    /// A recognized but not-yet-supported MATLAB class was encountered on read
    /// — an MCOS opaque class (`datetime`, `categorical`, `table`,
    /// `containers.Map`, `dictionary`, an enumeration, a user `classdef`, …)
    /// whose decoder is not yet implemented. Refused by name rather than
    /// misread; the modern `string` class is supported.
    UnsupportedMatlabClass(String),
    /// UTF-16 decoding of a `char` dataset failed.
    Utf16Decode(String),
    /// A [`DataProducer`](crate::mat::DataProducer) wrote the wrong number of
    /// bytes for a block. Refused rather than written: a block of the wrong size
    /// displaces every address after it, and the result would be a file that
    /// fails to open for reasons that no longer point back here.
    BlockSizeMismatch {
        /// Block index the producer was asked for.
        block: usize,
        /// Bytes it had to write, as
        /// [`Blocking::block_len`](crate::mat::Blocking::block_len) reports.
        expected: usize,
        /// Bytes it actually wrote.
        actual: usize,
    },
    /// A producer-backed dataset was asked for on a builder configured for
    /// compression. The layout needs each block's exact on-disk size before it
    /// writes anything, and a compressed block's size is not knowable without
    /// compressing it — which would buffer the data the path exists to avoid.
    CompressionUnsupportedForBlocks,
    /// [`Options::compression`](crate::mat::Options::compression) was set
    /// alongside an [`Options::libver`](crate::mat::Options::libver) too old to
    /// carry it.
    ///
    /// Compression needs chunked storage, and the chunk indices this crate
    /// writes arrived in HDF5 1.10 — while the MAT default is the 1.8 format,
    /// because MATLAB used HDF5 1.8.12 before R2021b. Refused rather than
    /// resolved either way: dropping the compression loses what the caller asked
    /// for, and raising the format produces a `.mat` file MATLAB cannot `load`.
    /// Set `libver` to [`LibVer::V110`](crate::LibVer::V110) to compress and
    /// accept the newer format.
    CompressionNeedsNewerFormat,
    /// A generic serde-originated error (from `Error::custom`).
    Custom(String),
    /// An error from the calling crate, carried whole.
    ///
    /// The builder's nesting closures
    /// ([`MatBuilder::struct_`](crate::mat::MatBuilder::struct_),
    /// [`MatBuilder::cell`](crate::mat::MatBuilder::cell),
    /// [`CellWriter::push_with`](crate::mat::CellWriter::push_with) and their
    /// siblings) and
    /// [`DataProducer::block_bytes`](crate::mat::DataProducer::block_bytes)
    /// return `Result<(), MatError>`, so a crate that emits `.mat` files as one
    /// of several formats has to put its own error type through that boundary.
    /// [`Custom`](MatError::Custom) keeps only the `Display` text; this keeps
    /// the error, so the caller's caller can still `downcast_ref` it back out
    /// of [`source`](std::error::Error::source):
    ///
    /// ```
    /// # use hdf5_pure::mat::{MatBuilder, MatError, Options};
    /// # use std::error::Error;
    /// # #[derive(Debug)]
    /// # struct EncodeError(&'static str);
    /// # impl std::fmt::Display for EncodeError {
    /// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    /// #         write!(f, "{}", self.0)
    /// #     }
    /// # }
    /// # impl Error for EncodeError {}
    /// # fn encode() -> Result<u32, EncodeError> { Err(EncodeError("no MAT encoding")) }
    /// let mut mb = MatBuilder::new(Options::default());
    /// let err = mb
    ///     .struct_("payload", |s| {
    ///         let value = encode().map_err(MatError::from_source)?;
    ///         s.write_scalar_u32("value", value)?;
    ///         Ok(())
    ///     })
    ///     .err()
    ///     .expect("the closure failed");
    ///
    /// let original = err.source().unwrap().downcast_ref::<EncodeError>().unwrap();
    /// assert_eq!(original.0, "no MAT encoding");
    /// ```
    ///
    /// `'static` is what `source` hands back. `Send + Sync` is what the crate
    /// already needs of a `MatError`: a failed producer's error waits in an
    /// `Arc<Mutex<_>>` for the finalizer to swap it back in, and that is what
    /// keeps `MatBuilder` itself `Send + Sync`.
    ///
    /// `Display` prints the inner error, which a formatter that walks the whole
    /// source chain will therefore print twice. That matches
    /// [`std::io::Error`]'s behaviour for the same case.
    Source(Box<dyn std::error::Error + Send + Sync + 'static>),
}

impl MatError {
    /// Carry an error from the calling crate whole, as [`MatError::Source`].
    ///
    /// Shaped after `std::io::Error::other`: it takes a concrete error type or
    /// an already-boxed one. Reach for it at a builder closure's edge, where
    /// `.map_err(MatError::from_source)` reads as a one-word conversion.
    ///
    /// The bound also admits a `String`, which the conversion accepts and
    /// nothing can recover: `downcast_ref` needs a type that implements
    /// `Error`, and `String` does not. A bare message belongs in
    /// [`MatError::Custom`].
    pub fn from_source<E>(source: E) -> Self
    where
        E: Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    {
        MatError::Source(source.into())
    }
}

impl fmt::Display for MatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatError::Hdf5(e) => write!(f, "HDF5 error: {e}"),
            MatError::Format(e) => write!(f, "HDF5 format error: {e}"),
            MatError::Io(e) => write!(f, "I/O error: {e}"),
            MatError::RootMustBeStruct => write!(
                f,
                "top-level value must be a struct with named fields; each field becomes a MATLAB variable"
            ),
            MatError::UnsupportedType(t) => write!(f, "unsupported Rust type for MAT v7.3: {t}"),
            MatError::MixedSequenceElementTypes => write!(
                f,
                "sequence elements have mixed primitive types; all elements of a numeric array must share a type"
            ),
            MatError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            MatError::StructArrayFieldShape {
                field,
                expected,
                actual,
            } => write!(
                f,
                "struct array field {field:?} has dimensions {actual:?}, not the array's {expected:?}"
            ),
            MatError::MissingField(name) => write!(f, "missing required field: {name}"),
            MatError::UnknownClass(c) => write!(f, "unknown MATLAB_class: {c:?}"),
            MatError::UnsupportedMatlabClass(c) => write!(
                f,
                "MATLAB class {c:?} is not yet supported for reading (modern `string` is; \
                 other MCOS opaque classes such as datetime/categorical/table are refused for now)"
            ),
            MatError::Utf16Decode(msg) => write!(f, "UTF-16 decode: {msg}"),
            MatError::BlockSizeMismatch {
                block,
                expected,
                actual,
            } => write!(
                f,
                "block producer wrote {actual} bytes for block {block}, which must carry exactly \
                 {expected}"
            ),
            MatError::CompressionUnsupportedForBlocks => write!(
                f,
                "a producer-backed dataset cannot be compressed: its blocks' on-disk sizes must be \
                 known before the file is laid out"
            ),
            MatError::CompressionNeedsNewerFormat => write!(
                f,
                "compression needs chunked storage, which needs the HDF5 1.10 format, but \
                 Options::libver asks for 1.8 so MATLAB's MAT v7.3 loader can read the file; \
                 set libver to LibVer::V110 to compress"
            ),
            MatError::Custom(msg) => write!(f, "{msg}"),
            MatError::Source(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MatError::Hdf5(e) => Some(e),
            MatError::Format(e) => Some(e),
            MatError::Io(e) => Some(e),
            MatError::Source(e) => Some(&**e),
            _ => None,
        }
    }
}

impl From<Hdf5Error> for MatError {
    fn from(e: Hdf5Error) -> Self {
        MatError::Hdf5(e)
    }
}

impl From<FormatError> for MatError {
    fn from(e: FormatError) -> Self {
        MatError::Format(e)
    }
}

impl From<std::io::Error> for MatError {
    fn from(e: std::io::Error) -> Self {
        MatError::Io(e)
    }
}

#[cfg(feature = "serde")]
impl serde::ser::Error for MatError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        MatError::Custom(msg.to_string())
    }
}

#[cfg(feature = "serde")]
impl serde::de::Error for MatError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        MatError::Custom(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[derive(Debug, PartialEq)]
    struct EmbedderError {
        code: u32,
    }

    impl fmt::Display for EmbedderError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "embedder failed with code {}", self.code)
        }
    }

    impl Error for EmbedderError {}

    #[test]
    fn a_carried_error_downcasts_back_to_its_own_type() {
        let err = MatError::from_source(EmbedderError { code: 7 });

        let source = err.source().expect("Source carries its error");
        assert_eq!(
            source.downcast_ref::<EmbedderError>(),
            Some(&EmbedderError { code: 7 }),
            "the whole point of the variant: the type survives the boundary"
        );
    }

    #[test]
    fn a_carried_error_displays_as_itself() {
        let err = MatError::from_source(EmbedderError { code: 7 });
        assert_eq!(err.to_string(), "embedder failed with code 7");
    }

    #[test]
    fn an_already_boxed_error_is_accepted_whole() {
        // `Box<dyn Error + Send + Sync>` does not itself implement `Error`, so a
        // bound of `E: Error` would refuse exactly the embedder that had already
        // erased its own type. `Into<Box<...>>` takes both.
        let boxed: Box<dyn Error + Send + Sync + 'static> = Box::new(EmbedderError { code: 7 });
        let err = MatError::from_source(boxed);

        assert!(
            err.source()
                .and_then(|s| s.downcast_ref::<EmbedderError>())
                .is_some()
        );
    }

    #[test]
    fn the_error_type_is_still_send_and_sync() {
        // A `Box<dyn Error>` without these bounds would revoke both, and the
        // first thing to break is a `MatBuilder` holding a producer's stashed
        // failure. This states the property where it lives, so the failure
        // names the error type rather than the builder three modules away.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MatError>();
    }
}
