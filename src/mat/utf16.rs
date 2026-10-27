//! UTF-16LE encode/decode helpers for MATLAB `char` datasets.
//!
//! MATLAB's `char` class is a UTF-16 array. On disk it's stored as an HDF5
//! `uint16` dataset (shape `[1, N]` for a row of N code units) and the
//! `MATLAB_class` attribute is `"char"`.

use super::error::MatError;

/// Encode a Rust `&str` to a `Vec<u16>` of UTF-16 code units.
///
/// The caller can then pass the slice to `with_u16_data` to build a MATLAB
/// char dataset.
pub fn encode_utf16(s: &str) -> Vec<u16> {
    // The UTF-8 byte length is an exact upper bound on the UTF-16 unit count
    // (every byte yields at most one code unit), so a single reservation sizes
    // the buffer without ever over-allocating and `extend` never reallocates.
    let mut units = Vec::with_capacity(s.len());
    units.extend(s.encode_utf16());
    units
}

/// Decode a slice of UTF-16 code units (as `u16`) into a Rust `String`.
pub fn decode_utf16(units: &[u16]) -> Result<String, MatError> {
    String::from_utf16(units).map_err(|e| MatError::Utf16Decode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_bmp_roundtrip() {
        // U+1F4A1 (💡) requires a UTF-16 surrogate pair.
        let s = "tip: 💡";
        let units = encode_utf16(s);
        assert!(units.len() > s.chars().count());
        assert_eq!(decode_utf16(&units).unwrap(), s);
    }

    #[test]
    fn bad_utf16_is_error() {
        // Lone high surrogate is invalid UTF-16.
        let bad = [0xD83Du16];
        assert!(matches!(decode_utf16(&bad), Err(MatError::Utf16Decode(_))));
    }
}
