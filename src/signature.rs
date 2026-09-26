//! HDF5 file signature (magic bytes) detection.

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::{BytesSource, Source};

/// The 8-byte HDF5 magic signature.
pub use hdf5_pure_format::HDF5_SIGNATURE;

/// Search a [`Source`] for the HDF5 signature, returning its byte offset.
///
/// The HDF5 spec allows the signature at offset 0, 512, 1024, 2048, 4096, …
/// (powers of two starting at 512, plus offset 0). Only the candidate 8-byte
/// windows are read, so this works against a lazy streaming source without
/// pulling the whole file.
pub fn find_signature_in<S: Source + ?Sized>(source: &S) -> Result<u64, FormatError> {
    let len = source.len();
    let mut sig = [0u8; 8];

    // Offset 0.
    if len >= 8 {
        source.read_at(0, &mut sig)?;
        if sig == HDF5_SIGNATURE {
            return Ok(0);
        }
    }

    // Powers of two starting at 512.
    let mut offset = 512u64;
    while offset + 8 <= len {
        source.read_at(offset, &mut sig)?;
        if sig == HDF5_SIGNATURE {
            return Ok(offset);
        }
        // `len` is whatever the source reports, and `Source` is public, so a
        // caller supplies it. A source that overstates its length past 2^63
        // would otherwise wrap this doubling to zero and re-read offset zero
        // forever in a release build (a debug build panics on the multiply).
        // There is no candidate offset past 2^63, so stop: the search falls
        // through to `SignatureNotFound`, which is the answer either way.
        let Some(next) = offset.checked_mul(2) else {
            break;
        };
        offset = next;
    }

    Err(FormatError::SignatureNotFound)
}

/// Search for the HDF5 signature in an in-memory buffer, returning its byte
/// offset. Thin wrapper over [`find_signature_in`] for the buffered reader path.
pub fn find_signature(data: &[u8]) -> Result<usize, FormatError> {
    find_signature_in(&BytesSource::new(data))?.to_usize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_at_offset_0() {
        let mut data = vec![0u8; 64];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(0));
    }

    #[test]
    fn signature_at_offset_512() {
        let mut data = vec![0u8; 1024];
        data[512..520].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(512));
    }

    #[test]
    fn signature_at_offset_1024() {
        let mut data = vec![0u8; 2048];
        data[1024..1032].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(1024));
    }

    #[test]
    fn signature_at_offset_2048() {
        let mut data = vec![0u8; 4096];
        data[2048..2056].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(2048));
    }

    #[test]
    fn signature_not_found() {
        let data = vec![0u8; 8192];
        assert_eq!(find_signature(&data), Err(FormatError::SignatureNotFound));
    }

    #[test]
    fn signature_not_found_empty() {
        assert_eq!(find_signature(&[]), Err(FormatError::SignatureNotFound));
    }

    #[test]
    fn signature_not_found_too_short() {
        assert_eq!(
            find_signature(&[0x89, b'H', b'D']),
            Err(FormatError::SignatureNotFound)
        );
    }

    #[test]
    fn signature_at_non_power_of_two_not_found() {
        // Signature at offset 100 should NOT be found
        let mut data = vec![0u8; 1024];
        data[100..108].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Err(FormatError::SignatureNotFound));
    }

    #[test]
    fn signature_prefers_earliest() {
        // Signature at both 0 and 512, should return 0
        let mut data = vec![0u8; 1024];
        data[..8].copy_from_slice(&HDF5_SIGNATURE);
        data[512..520].copy_from_slice(&HDF5_SIGNATURE);
        assert_eq!(find_signature(&data), Ok(0));
    }

    /// A source that reports a length it cannot serve is searched a bounded
    /// number of times and then given up on.
    ///
    /// `Source` is public, so `len` is a number a caller supplies rather than
    /// one a file measured. The candidate offsets double, and doubling past
    /// 2^63 used to wrap to zero: a release build then re-read offset zero
    /// forever, and a debug build panicked on the multiply. The read counter is
    /// what separates the fix from that — the search has to *stop*, not merely
    /// return the same verdict, and a source that keeps answering cannot be
    /// told from one that is being asked forever any other way.
    #[test]
    fn a_source_that_overstates_its_length_is_searched_a_bounded_number_of_times() {
        use core::cell::Cell;

        struct Overstating {
            reads: Cell<u32>,
        }

        impl Source for Overstating {
            fn len(&self) -> u64 {
                u64::MAX
            }

            fn read_at(&self, _offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
                let n = self.reads.get();
                self.reads.set(n + 1);
                // Fail rather than hang if the search does not terminate, so
                // the defect this test covers surfaces as a failure.
                if n > 1_000 {
                    return Err(FormatError::SignatureNotFound);
                }
                buf.fill(0);
                Ok(())
            }
        }

        let source = Overstating {
            reads: Cell::new(0),
        };
        assert_eq!(
            find_signature_in(&source),
            Err(FormatError::SignatureNotFound)
        );
        // Offset 0 plus the 55 powers of two from 512 to 2^63.
        assert_eq!(
            source.reads.get(),
            56,
            "the signature search did not stop at the last representable offset"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn signature_found_over_a_streaming_source() {
        use crate::source::ReadSeekSource;
        // The signature at 512 is found by reading only the 8-byte candidate
        // windows from a lazy Read+Seek source — never the whole buffer.
        let mut data = vec![0u8; 1024];
        data[512..520].copy_from_slice(&HDF5_SIGNATURE);
        let src = ReadSeekSource::new(std::io::Cursor::new(data)).unwrap();
        assert_eq!(find_signature_in(&src), Ok(512));
    }
}
