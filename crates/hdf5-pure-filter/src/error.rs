use alloc::format;
use core::fmt;

/// Reports an invalid filter stream or codec configuration.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    /// Reports a decoded chunk whose length differs from its full chunk size.
    DataSizeMismatch {
        /// Number of bytes required by the chunk dimensions and element size.
        expected: usize,
        /// Number of bytes returned by the filter pipeline.
        actual: usize,
    },
    /// Reports malformed Deflate or Shuffle input, or a filter operation that failed.
    FilterError(alloc::string::String),
    /// Reports a Fletcher32 checksum that differs from the stored checksum.
    Fletcher32Mismatch {
        /// Checksum stored after the payload.
        expected: u32,
        /// Checksum calculated from the payload.
        computed: u32,
    },
    /// Reports malformed LZF input or output beyond the caller's size limit.
    InvalidLzfStream(&'static str),
    /// Reports invalid scale-offset parameters or input bytes.
    ScaleOffset(alloc::string::String),
    /// Reports a scale-offset element count that exceeds the platform's index width.
    ScaleOffsetValueTooLargeForPlatform {
        /// Element count read from the filter parameters.
        value: u64,
        /// Platform index type that cannot hold the value.
        target: &'static str,
    },
    /// Reports fewer encoded bytes than the ZFP chunk requires.
    #[cfg(feature = "zfp")]
    TruncatedZfpStream {
        /// Number of encoded bytes required for the chunk.
        expected: usize,
        /// Number of encoded bytes present.
        actual: usize,
    },
    /// Reports a filter identifier without an available encoder or decoder.
    UnsupportedFilter(u16),
    /// Reports a ZFP configuration the codec cannot encode.
    #[cfg(feature = "zfp")]
    UnsupportedZfp(alloc::string::String),
    /// Reports a dimension that does not fit the platform's index width.
    #[cfg(feature = "zfp")]
    ValueTooLargeForPlatform {
        /// Dimension read from the chunk shape.
        value: u64,
        /// Platform index type that cannot hold the value.
        target: &'static str,
    },
    /// Reports invalid ZFP parameters or input bytes.
    #[cfg(feature = "zfp")]
    ZfpFilter(alloc::string::String),
    /// Reports a float block whose header exceeds the fixed-rate bit budget.
    #[cfg(feature = "zfp")]
    ZfpHeaderTooLarge {
        /// Number of bits available at the configured rate.
        budget: usize,
        /// Number of bits required by the block header.
        required: usize,
    },
    /// Reports an overflow while calculating a ZFP chunk size.
    #[cfg(feature = "zfp")]
    ZfpSizeOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataSizeMismatch { expected, actual } => {
                write!(f, "data size mismatch: expected {expected}, got {actual}")
            }
            Self::FilterError(reason) => write!(f, "filter error: {reason}"),
            Self::Fletcher32Mismatch { expected, computed } => write!(
                f,
                "fletcher32 checksum mismatch: expected {expected}, computed {computed}"
            ),
            Self::InvalidLzfStream(reason) => write!(f, "lzf: {reason}"),
            Self::ScaleOffset(reason) => write!(f, "filter error: {reason}"),
            Self::ScaleOffsetValueTooLargeForPlatform { value, target } => write!(
                f,
                "scaleoffset: value {value} does not fit in {target} on this platform"
            ),
            #[cfg(feature = "zfp")]
            Self::TruncatedZfpStream { expected, actual } => {
                write!(f, "ZFP: encoded chunk needs {expected} bytes, got {actual}")
            }
            Self::UnsupportedFilter(id) => write!(f, "unsupported filter {id}"),
            #[cfg(feature = "zfp")]
            Self::UnsupportedZfp(reason) => write!(f, "unsupported ZFP configuration: {reason}"),
            #[cfg(feature = "zfp")]
            Self::ValueTooLargeForPlatform { value, target } => write!(
                f,
                "ZFP chunk dimension {value} does not fit in {target} on this platform"
            ),
            #[cfg(feature = "zfp")]
            Self::ZfpFilter(reason) => write!(f, "filter error: {reason}"),
            #[cfg(feature = "zfp")]
            Self::ZfpHeaderTooLarge { budget, required } => write!(
                f,
                "ZFP: nonzero float block needs {required} header bits, rate allows {budget} bits"
            ),
            #[cfg(feature = "zfp")]
            Self::ZfpSizeOverflow => {
                write!(f, "ZFP: chunk dimensions or encoded size overflow usize")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl From<Error> for hdf5_pure_core::FormatError {
    fn from(error: Error) -> Self {
        match error {
            Error::InvalidLzfStream(reason) => Self::FilterError(format!("lzf: {reason}")),
            Error::ScaleOffset(reason) => Self::FilterError(reason),
            Error::FilterError(reason) => Self::FilterError(reason),
            Error::UnsupportedFilter(id) => Self::UnsupportedFilter(id),
            Error::DataSizeMismatch { expected, actual } => {
                Self::DataSizeMismatch { expected, actual }
            }
            Error::Fletcher32Mismatch { expected, computed } => {
                Self::Fletcher32Mismatch { expected, computed }
            }
            Error::ScaleOffsetValueTooLargeForPlatform { value, target } => {
                Self::ValueTooLargeForPlatform { value, target }
            }
            #[cfg(feature = "zfp")]
            Error::ZfpFilter(reason) => Self::FilterError(reason),
            #[cfg(feature = "zfp")]
            Error::UnsupportedZfp(reason) => Self::UnsupportedZfp(reason),
            #[cfg(feature = "zfp")]
            Error::ValueTooLargeForPlatform { value, target } => {
                Self::ValueTooLargeForPlatform { value, target }
            }
            #[cfg(feature = "zfp")]
            Error::TruncatedZfpStream { expected, actual } => Self::FilterError(format!(
                "ZFP: encoded chunk needs {expected} bytes, got {actual}"
            )),
            #[cfg(feature = "zfp")]
            Error::ZfpSizeOverflow => {
                Self::FilterError("ZFP: chunk dimensions or encoded size overflow usize".into())
            }
            #[cfg(feature = "zfp")]
            Error::ZfpHeaderTooLarge { budget, required } => Self::FilterError(format!(
                "ZFP: nonzero float block needs {required} header bits, rate allows {budget} bits"
            )),
        }
    }
}
