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

    // Process 12-byte blocks
    while data.len() - offset > 12 {
        a = a.wrapping_add(read_u32_le(data, offset));
        b = b.wrapping_add(read_u32_le(data, offset + 4));
        c = c.wrapping_add(read_u32_le(data, offset + 8));
        mix(&mut a, &mut b, &mut c);
        offset += 12;
    }

    // The tail is 1 to 12 bytes, added little-endian into a, b and c in turn --
    // the sum the reference's fall-through switch forms. An empty input has no
    // tail, and the reference returns `c` without the final mix.
    let tail = &data[offset..];
    if tail.is_empty() {
        return c;
    }
    let mut words = [0u32; 3];
    for (word, chunk) in words.iter_mut().zip(tail.chunks(4)) {
        for (i, &byte) in chunk.iter().enumerate() {
            *word |= u32::from(byte) << (8 * i);
        }
    }
    a = a.wrapping_add(words[0]);
    b = b.wrapping_add(words[1]);
    c = c.wrapping_add(words[2]);

    final_mix(&mut a, &mut b, &mut c);
    c
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0, 0xdead_beef)]
    #[case(1, 0x04ec_883b)]
    #[case(2, 0x022b_c7fb)]
    #[case(3, 0x96c2_fd10)]
    #[case(4, 0x46f8_eec9)]
    #[case(5, 0xb217_35c1)]
    #[case(6, 0x2779_a0dd)]
    #[case(7, 0x098f_be6f)]
    #[case(8, 0x9afd_b1b3)]
    #[case(9, 0xdd6e_13f5)]
    #[case(10, 0x3a58_a365)]
    #[case(11, 0x7529_9b1b)]
    #[case(12, 0x3891_9c27)]
    #[case(13, 0xe1f0_cbae)]
    #[case(14, 0x49ef_0328)]
    fn each_input_length_through_one_block_and_its_tail_hashes_to_a_fixed_value(
        #[case] len: usize,
        #[case] expected: u32,
    ) {
        const PROBE: [u8; 14] = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92];

        assert_eq!(jenkins_lookup3(&PROBE[..len]), expected);
    }

    #[rstest]
    #[case(0, 0x1777_0551)]
    #[case(1, 0xcd62_8161)]
    fn the_self_test_vector_of_lookup3_c_hashes_to_its_published_value(
        #[case] initval: u32,
        #[case] expected: u32,
    ) {
        assert_eq!(
            hashlittle(b"Four score and seven years ago", initval),
            expected
        );
    }

    #[test]
    fn a_files_stored_superblock_checksum_equals_jenkins_lookup3() {
        let file_data: &[u8] = include_bytes!("../tests/data/unattributed/v2_groups.h5");
        // Superblock v3: checksum at offset 44, covers bytes 0..44
        let stored =
            u32::from_le_bytes([file_data[44], file_data[45], file_data[46], file_data[47]]);
        assert_eq!(jenkins_lookup3(&file_data[0..44]), stored);
    }
}
