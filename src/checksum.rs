//! HDF5 metadata checksum: Jenkins lookup3 `hashlittle`.
//!
//! HDF5 uses Bob Jenkins' lookup3 hash (not CRC32C) for all metadata
//! checksums in superblocks, object headers, B-tree nodes, etc.

/// Compute the Jenkins lookup3 checksum of a byte slice.
///
/// This is the `hashlittle` function from Bob Jenkins' lookup3.c,
/// matching the `H5_checksum_lookup3` function in the HDF5 C library.
pub fn jenkins_lookup3(data: &[u8]) -> u32 {
    hashlittle(data, 0)
}

/// Verify the four-byte Jenkins checksum HDF5 stores at the end of a metadata
/// structure: `block` is the whole structure, checksum included, and the
/// checksum covers everything before it.
///
/// This is a no-op unless the `checksum` feature is enabled, so a caller needs
/// no `cfg` of its own — it passes the bytes it already holds and the
/// validation compiles away.
pub(crate) fn verify_trailing(block: &[u8]) -> Result<(), crate::error::FormatError> {
    #[cfg(not(feature = "checksum"))]
    {
        let _ = block;
        Ok(())
    }
    #[cfg(feature = "checksum")]
    {
        // A structure shorter than its own checksum field is truncated, not
        // merely wrong: report the shortfall rather than indexing past the end.
        let Some((body, stored)) = block.split_last_chunk::<4>() else {
            return Err(crate::error::FormatError::UnexpectedEof {
                expected: 4,
                available: block.len(),
            });
        };
        let stored = u32::from_le_bytes(*stored);
        let computed = jenkins_lookup3(body);
        if computed != stored {
            return Err(crate::error::FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }
        Ok(())
    }
}

/// Stamp the trailing checksum of the structure occupying `file[at..at + len]`,
/// the inverse of [`verify_trailing`].
///
/// Test-only: the writers build a structure into a buffer and append its
/// checksum as the last step, so only a hand-built fixture needs to patch one
/// in afterwards. A fixture that skips it is not a file any HDF5 library reads,
/// and `len` states where the structure ends independently of the reader, so a
/// disagreement about its extent fails the checksum rather than passing
/// unnoticed.
#[cfg(test)]
pub(crate) fn stamp_trailing(file: &mut [u8], at: usize, len: usize) {
    let cks = jenkins_lookup3(&file[at..at + len - 4]);
    file[at + len - 4..at + len].copy_from_slice(&cks.to_le_bytes());
}

fn rot(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 4);
    *b = b.wrapping_add(*a);
}

fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 14));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 11));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 25));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 16));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 4));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 14));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 24));
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn hashlittle(data: &[u8], initval: u32) -> u32 {
    let length = data.len();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "lookup3 seeds the mix with length mod 2^32 by definition; matching the \
                  reference (uint32_t) cast keeps checksums identical to the HDF5 C library"
    )]
    let mut a: u32 = 0xdeadbeefu32
        .wrapping_add(length as u32)
        .wrapping_add(initval);
    let mut b: u32 = a;
    let mut c: u32 = a;

    let mut offset = 0;
    let mut remaining = length;

    // Process 12-byte blocks
    while remaining > 12 {
        a = a.wrapping_add(read_u32_le(data, offset));
        b = b.wrapping_add(read_u32_le(data, offset + 4));
        c = c.wrapping_add(read_u32_le(data, offset + 8));
        mix(&mut a, &mut b, &mut c);
        offset += 12;
        remaining -= 12;
    }

    // Handle the last few bytes (switch fall-through pattern)
    let tail = &data[offset..];
    // Using the little-endian byte reading approach from hashlittle
    match remaining {
        12 => {
            a = a.wrapping_add(read_u32_le(tail, 0));
            b = b.wrapping_add(read_u32_le(tail, 4));
            c = c.wrapping_add(read_u32_le(tail, 8));
        }
        11 => {
            c = c.wrapping_add((tail[10] as u32) << 16);
            c = c.wrapping_add((tail[9] as u32) << 8);
            c = c.wrapping_add(tail[8] as u32);
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        10 => {
            c = c.wrapping_add((tail[9] as u32) << 8);
            c = c.wrapping_add(tail[8] as u32);
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        9 => {
            c = c.wrapping_add(tail[8] as u32);
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        8 => {
            b = b.wrapping_add(read_u32_le(tail, 4));
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        7 => {
            b = b.wrapping_add((tail[6] as u32) << 16);
            b = b.wrapping_add((tail[5] as u32) << 8);
            b = b.wrapping_add(tail[4] as u32);
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        6 => {
            b = b.wrapping_add((tail[5] as u32) << 8);
            b = b.wrapping_add(tail[4] as u32);
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        5 => {
            b = b.wrapping_add(tail[4] as u32);
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        4 => {
            a = a.wrapping_add(read_u32_le(tail, 0));
        }
        3 => {
            a = a.wrapping_add((tail[2] as u32) << 16);
            a = a.wrapping_add((tail[1] as u32) << 8);
            a = a.wrapping_add(tail[0] as u32);
        }
        2 => {
            a = a.wrapping_add((tail[1] as u32) << 8);
            a = a.wrapping_add(tail[0] as u32);
        }
        1 => {
            a = a.wrapping_add(tail[0] as u32);
        }
        0 => return c,
        _ => unreachable!(),
    }

    final_mix(&mut a, &mut b, &mut c);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        // Empty input should return the initial state after no mixing
        let h = jenkins_lookup3(b"");
        // Just verify it doesn't panic and returns something deterministic
        assert_eq!(h, jenkins_lookup3(b""));
    }

    #[test]
    fn known_values() {
        // Test with known input to ensure consistency
        let h1 = jenkins_lookup3(b"hello");
        let h2 = jenkins_lookup3(b"hello");
        assert_eq!(h1, h2);

        // Different inputs should give different outputs
        let h3 = jenkins_lookup3(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn twelve_byte_boundary() {
        // Exactly 12 bytes
        let h = jenkins_lookup3(b"abcdefghijkl");
        assert_eq!(h, jenkins_lookup3(b"abcdefghijkl"));
    }

    #[test]
    fn longer_than_12() {
        let h = jenkins_lookup3(b"abcdefghijklmnop");
        assert_eq!(h, jenkins_lookup3(b"abcdefghijklmnop"));
    }

    #[test]
    fn all_tail_lengths() {
        // Test every tail length from 1 to 12
        for len in 1..=12 {
            let data: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let h1 = jenkins_lookup3(&data);
            let h2 = jenkins_lookup3(&data);
            assert_eq!(h1, h2, "failed for length {len}");
        }
    }

    #[test]
    fn verify_against_hdf5_file() {
        // Verify against a real HDF5 file checksum
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/v2_groups.h5");
        // Superblock v3: checksum at offset 44, covers bytes 0..44
        let stored =
            u32::from_le_bytes([file_data[44], file_data[45], file_data[46], file_data[47]]);
        let computed = jenkins_lookup3(&file_data[0..44]);
        assert_eq!(
            computed, stored,
            "Jenkins lookup3 should match HDF5 superblock checksum"
        );
    }
}
