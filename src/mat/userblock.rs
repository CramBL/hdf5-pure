//! MATLAB v7.3 userblock (first 512 bytes of a `.mat` file).
//!
//! Layout:
//! - `[0..124]`: description text, ASCII, space-padded
//! - `[124..126]`: version tag `0x0200` (little-endian) → `[0x00, 0x02]`
//! - `[126..128]`: endian indicator `"IM"` (little-endian)
//! - `[128..512]`: zero-filled padding; HDF5 superblock follows at `[512..]`

/// MATLAB userblock length as a `usize`, for sizing the buffer that holds it.
/// Declared here rather than cast from [`USERBLOCK_SIZE`] so no 32-bit target
/// sees a narrowing conversion.
const USERBLOCK_LEN: usize = 512;

/// MATLAB userblock size (bytes). Always 512 for v7.3.
pub const USERBLOCK_SIZE: u64 = USERBLOCK_LEN as u64;

/// Default description text written in the first 124 bytes.
pub const DEFAULT_DESCRIPTION: &str = "MATLAB 7.3 MAT-file, Platform: hdf5-pure (Rust)";

/// Writes the MATLAB v7.3 header into the first 128 bytes of `file_bytes`.
///
/// Leaves bytes from offset 128 onward unchanged.
///
/// # Panics
///
/// Panics if `file_bytes` is shorter than 512 bytes.
pub fn write_header(file_bytes: &mut [u8], description: &str) {
    assert!(
        file_bytes.len() as u64 >= USERBLOCK_SIZE,
        "userblock requires at least 512 bytes, got {}",
        file_bytes.len()
    );

    // 0..124: description, space-padded, ASCII only.
    for b in file_bytes[..124].iter_mut() {
        *b = b' ';
    }
    let bytes = description.as_bytes();
    let n = bytes.len().min(124);
    for (dst, src) in file_bytes[..n].iter_mut().zip(bytes[..n].iter()) {
        // Replace non-ASCII with '?' to keep the header valid ASCII.
        *dst = if src.is_ascii() && *src != 0 {
            *src
        } else {
            b'?'
        };
    }

    // 124..126: version = 0x0200 (little-endian → bytes 0x00, 0x02).
    file_bytes[124] = 0x00;
    file_bytes[125] = 0x02;

    // 126..128: endian indicator "IM" (little-endian).
    file_bytes[126] = b'I';
    file_bytes[127] = b'M';

    // 128..512: leave as-is (zeros).
}

/// The whole 512-byte userblock, ready to be handed to
/// [`FileBuilder::with_userblock_content`](crate::FileBuilder::with_userblock_content)
/// so the writer emits it ahead of the HDF5 content.
///
/// Nothing in a v7.3 userblock depends on the file that follows it, which is why
/// it can be produced up front instead of patched in afterwards — and that in turn
/// is what lets a `.mat` be written to a sink that cannot seek.
pub fn header_block(description: &str) -> [u8; USERBLOCK_LEN] {
    let mut block = [0u8; USERBLOCK_LEN];
    write_header(&mut block, description);
    block
}

/// Verify the bytes look like a MATLAB v7.3 userblock. Returns `Ok(())` on
/// success.
pub fn verify_header(file_bytes: &[u8]) -> Result<(), &'static str> {
    if (file_bytes.len() as u64) < USERBLOCK_SIZE {
        return Err("file too short for MAT v7.3 userblock");
    }
    if !file_bytes[..6].eq_ignore_ascii_case(b"MATLAB") {
        return Err("missing `MATLAB` signature at byte 0");
    }
    if file_bytes[126] != b'I' || file_bytes[127] != b'M' {
        return Err("missing `IM` little-endian indicator at byte 126");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_header_passes_verify() {
        let mut buf = [0u8; 1024];
        write_header(&mut buf, DEFAULT_DESCRIPTION);
        verify_header(&buf).unwrap();

        assert_eq!(&buf[..6], b"MATLAB");
        assert_eq!(&buf[124..128], &[0x00, 0x02, b'I', b'M']);
        assert_eq!(buf[200], 0, "bytes past 128 must stay zero");
    }

    #[test]
    fn truncated_description_fills_first_124_bytes() {
        let mut buf = [0u8; 512];
        let long = format!("MATLAB 7.3 MAT-file, {}", "x".repeat(500));
        write_header(&mut buf, &long);
        verify_header(&buf).unwrap();
        // First 124 bytes are filled with the (truncated) description.
        assert_eq!(&buf[..6], b"MATLAB");
        assert_eq!(buf[123], b'x');
        // Version tag at 124..126, then 'I','M'.
        assert_eq!(buf[124], 0x00);
        assert_eq!(buf[125], 0x02);
    }

    #[test]
    fn verify_rejects_wrong_magic() {
        let buf = [0u8; 512];
        assert!(verify_header(&buf).is_err());
    }
}
