//! HDF5 metadata checksum: Jenkins lookup3 `hashlittle`.
//!
//! HDF5 uses Bob Jenkins' lookup3 hash (not CRC32C) for all metadata
//! checksums in superblocks, object headers, B-tree nodes, etc.

use crate::error;

/// Computes a Jenkins lookup3 checksum from successive byte slices.
///
/// The caller declares the total input length at construction and supplies that many bytes
/// through [`update`](Self::update) before calling [`finish`](Self::finish). The checksum
/// equals [`jenkins_lookup3`] applied to the concatenated input.
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::__private as format;
/// use format::JenkinsLookup3;
///
/// let first = b"first ";
/// let second = b"second";
/// let total_len = (first.len() + second.len()) as u64;
/// let mut checksum = JenkinsLookup3::new(total_len);
/// checksum.update(first)?;
/// checksum.update(second)?;
/// assert_eq!(checksum.finish()?, format::jenkins_lookup3(b"first second"));
/// # Ok::<(), hdf5_pure_format::FormatError>(())
/// ```
pub struct JenkinsLookup3 {
    a: u32,
    b: u32,
    c: u32,
    remaining: u64,
    pending: [u8; LOOKUP3_BLOCK_LEN],
    pending_len: usize,
}

impl JenkinsLookup3 {
    /// Initializes the checksum for an input of `total_len` bytes.
    ///
    /// The total length includes only the bytes to hash.
    /// For a trailing checksum, it excludes the checksum field.
    // With an initial value of zero, `H5_checksum_lookup3` in H5checksum.c,
    // HDF5 2.2.0, seeds all three words with 0xdeadbeef plus the input length
    // modulo 2^32 and mixes 12-byte blocks except for the final block.
    pub fn new(total_len: u64) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "lookup3 seeds each state word with the input length modulo 2^32"
        )]
        let seed = LOOKUP3_SEED.wrapping_add(total_len as u32);
        Self {
            a: seed,
            b: seed,
            c: seed,
            remaining: total_len,
            pending: [0; LOOKUP3_BLOCK_LEN],
            pending_len: 0,
        }
    }

    /// Hashes the next input bytes, retaining at most one 12-byte block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`](error::FormatError::Internal) if the cumulative
    /// input length would exceed the declared total. On error, the checksum state is unchanged.
    pub fn update(&mut self, mut data: &[u8]) -> Result<(), error::FormatError> {
        let Self {
            a,
            b,
            c,
            remaining,
            pending,
            pending_len,
        } = self;
        if data.len() as u64 > *remaining {
            return Err(error::FormatError::Internal(
                "checksum input exceeds its declared length".into(),
            ));
        }
        while !data.is_empty() {
            let len = (LOOKUP3_BLOCK_LEN - *pending_len).min(data.len());
            pending[*pending_len..*pending_len + len].copy_from_slice(&data[..len]);
            *pending_len += len;
            *remaining -= len as u64;
            data = &data[len..];
            if *pending_len == LOOKUP3_BLOCK_LEN && *remaining != 0 {
                *a = a.wrapping_add(read_u32_le::<false>(pending, 0, 0));
                *b = b.wrapping_add(read_u32_le::<false>(pending, 4, 0));
                *c = c.wrapping_add(read_u32_le::<false>(pending, 8, 0));
                mix(a, b, c);
                *pending_len = 0;
            }
        }
        Ok(())
    }

    /// Computes the checksum after hashing the declared number of bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`](error::FormatError::Internal) if the caller has
    /// supplied fewer bytes than the declared total.
    pub fn finish(self) -> Result<u32, error::FormatError> {
        let Self {
            mut a,
            mut b,
            mut c,
            remaining,
            pending,
            pending_len,
        } = self;
        if remaining != 0 {
            return Err(error::FormatError::Internal(
                "checksum input is shorter than its declared length".into(),
            ));
        }
        if pending_len == 0 {
            return Ok(c);
        }
        let mut words = [0u32; 3];
        for (word, bytes) in words.iter_mut().zip(pending[..pending_len].chunks(4)) {
            for (i, byte) in bytes.iter().enumerate() {
                *word |= u32::from(*byte) << (8 * i);
            }
        }
        a = a.wrapping_add(words[0]);
        b = b.wrapping_add(words[1]);
        c = c.wrapping_add(words[2]);
        final_mix(&mut a, &mut b, &mut c);
        Ok(c)
    }
}

const LOOKUP3_BLOCK_LEN: usize = 12;
const LOOKUP3_SEED: u32 = 0xdead_beef;

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
/// no `cfg` of its own  -  it passes the bytes it already holds and the
/// validation compiles away.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`](crate::FormatError::UnexpectedEof) if `block` is
/// shorter than four bytes, or [`FormatError::ChecksumMismatch`](crate::FormatError::ChecksumMismatch)
/// if the stored checksum differs from the computed one. Neither error occurs when the
/// `checksum` feature is disabled.
pub fn verify_trailing(block: &[u8]) -> Result<(), crate::error::FormatError> {
    #[cfg(not(feature = "checksum"))]
    {
        let _ = block;
        Ok(())
    }
    #[cfg(feature = "checksum")]
    {
        verify_trailing_required(block)
    }
}

/// Validates a trailing checksum regardless of the `checksum` feature.
///
/// Hashes every byte before the four-byte checksum field.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`](crate::FormatError::UnexpectedEof) if `block` is
/// shorter than four bytes, or [`FormatError::ChecksumMismatch`](crate::FormatError::ChecksumMismatch)
/// if the stored checksum differs from the computed one.
pub(crate) fn verify_trailing_required(block: &[u8]) -> Result<(), crate::error::FormatError> {
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

/// Checks an embedded checksum field.
///
/// With the `checksum` feature, hashes the whole block with the checksum bytes zeroed and
/// compares the result with the stored checksum.
///
/// The stored checksum field is required even when the `checksum` feature is disabled.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`](crate::FormatError::UnexpectedEof) if the checksum
/// field is incomplete, or [`FormatError::ChecksumMismatch`](crate::FormatError::ChecksumMismatch)
/// if checksum validation is enabled and the stored checksum differs from the computed one.
pub(crate) fn verify_zeroed(
    block: &[u8],
    checksum_at: usize,
) -> Result<(), crate::error::FormatError> {
    let stored = crate::bytes::Fields::new(block, checksum_at).u32()?;
    #[cfg(feature = "checksum")]
    {
        let computed = hashlittle_zeroed::<true>(block, 0, checksum_at);
        if stored != computed {
            return Err(crate::error::FormatError::ChecksumMismatch {
                expected: stored,
                computed,
            });
        }
    }
    #[cfg(not(feature = "checksum"))]
    {
        let _ = stored;
    }
    Ok(())
}

/// Validates an embedded checksum regardless of the `checksum` feature.
///
/// Hashes the complete block with the checksum field zeroed.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`](crate::FormatError::UnexpectedEof) if the checksum
/// field is incomplete, or [`FormatError::ChecksumMismatch`](crate::FormatError::ChecksumMismatch)
/// if the stored checksum differs from the computed one.
pub(crate) fn verify_zeroed_required(
    block: &[u8],
    checksum_at: usize,
) -> Result<(), crate::error::FormatError> {
    let stored = crate::bytes::Fields::new(block, checksum_at).u32()?;
    let computed = hashlittle_zeroed::<true>(block, 0, checksum_at);
    if stored != computed {
        return Err(crate::error::FormatError::ChecksumMismatch {
            expected: stored,
            computed,
        });
    }
    Ok(())
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

fn read_u32_le<const ZERO: bool>(data: &[u8], offset: usize, zero_at: usize) -> u32 {
    u32::from_le_bytes([
        byte_at::<ZERO>(data, offset, zero_at),
        byte_at::<ZERO>(data, offset + 1, zero_at),
        byte_at::<ZERO>(data, offset + 2, zero_at),
        byte_at::<ZERO>(data, offset + 3, zero_at),
    ])
}

fn byte_at<const ZERO: bool>(data: &[u8], offset: usize, zero_at: usize) -> u8 {
    if ZERO && offset >= zero_at && offset - zero_at < 4 {
        0
    } else {
        data[offset]
    }
}

fn hashlittle(data: &[u8], initval: u32) -> u32 {
    hashlittle_zeroed::<false>(data, initval, 0)
}

// `H5HF__cache_dblock_verify_chksum` in H5HFcache.c, HDF5 2.2.0, zeros the embedded
// checksum field while hashing the entire decoded direct block.
fn hashlittle_zeroed<const ZERO: bool>(data: &[u8], initval: u32, zero_at: usize) -> u32 {
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
        a = a.wrapping_add(read_u32_le::<ZERO>(data, offset, zero_at));
        b = b.wrapping_add(read_u32_le::<ZERO>(data, offset + 4, zero_at));
        c = c.wrapping_add(read_u32_le::<ZERO>(data, offset + 8, zero_at));
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
    for (word_index, (word, chunk)) in words.iter_mut().zip(tail.chunks(4)).enumerate() {
        for i in 0..chunk.len() {
            *word |=
                u32::from(byte_at::<ZERO>(data, offset + word_index * 4 + i, zero_at)) << (8 * i);
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
        #[values(1, 7, 8, 11, 12, 13, 64)] window_len: usize,
    ) {
        const PROBE: [u8; 14] = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92];

        assert_eq!(jenkins_lookup3(&PROBE[..len]), expected);
        let mut checksum = JenkinsLookup3::new(len as u64);
        checksum.update(&[]).unwrap();
        for window in PROBE[..len].chunks(window_len) {
            checksum.update(window).unwrap();
        }
        checksum.update(&[]).unwrap();
        assert_eq!(checksum.finish().unwrap(), expected);
    }

    #[rstest]
    #[case(0, 0xdead_beef)]
    #[case(4_294_967_296, 0xdead_beef)]
    #[case(4_294_967_310, 0xdead_befd)]
    #[case(u64::MAX, 0xdead_beee)]
    fn incremental_seed_adds_the_low_32_bits_of_the_total_length(
        #[case] len: u64,
        #[case] seed: u32,
    ) {
        let checksum = JenkinsLookup3::new(len);
        assert_eq!((checksum.a, checksum.b, checksum.c), (seed, seed, seed));
        assert_eq!(checksum.remaining, len);
    }

    #[test]
    fn incremental_updates_preserve_a_remaining_length_of_four_gib() {
        let mut checksum = JenkinsLookup3::new(4_294_967_310);
        checksum.update(&[0; 14]).unwrap();
        assert_eq!(checksum.remaining, 4_294_967_296);
        assert_eq!(checksum.pending_len, 2);
    }

    #[rstest]
    #[case(24)]
    #[case(25)]
    #[case(36)]
    #[case(37)]
    #[case(64)]
    #[case(65)]
    fn incremental_windows_match_one_shot_across_multiple_blocks(
        #[case] len: usize,
        #[values(1, 8, 12, 13, 16, 65)] window_len: usize,
    ) {
        let data = core::array::from_fn::<_, 65, _>(|i| u8::try_from(i * 3).unwrap());
        let mut checksum = JenkinsLookup3::new(len as u64);
        for window in data[..len].chunks(window_len) {
            checksum.update(window).unwrap();
        }
        assert_eq!(checksum.finish().unwrap(), jenkins_lookup3(&data[..len]));
    }

    #[rstest]
    fn incremental_splits_match_the_published_vector(
        #[values(0, 1, 4, 11, 12, 13, 24, 28, 29)] split_at: usize,
    ) {
        let data = b"Four score and seven years ago";
        let mut checksum = JenkinsLookup3::new(data.len() as u64);
        checksum.update(&data[..split_at]).unwrap();
        checksum.update(&data[split_at..]).unwrap();
        assert_eq!(checksum.finish().unwrap(), 0x1777_0551);
    }

    #[rstest]
    #[case(0, b"".as_slice())]
    #[case(3, b"ab".as_slice())]
    fn incremental_update_rejects_excess_input_without_changing_state(
        #[case] total_len: u64,
        #[case] prefix: &[u8],
    ) {
        let mut checksum = JenkinsLookup3::new(total_len);
        checksum.update(prefix).unwrap();
        let state = (checksum.a, checksum.b, checksum.c, checksum.remaining);
        let pending = checksum.pending;
        let pending_len = checksum.pending_len;
        assert_eq!(
            checksum.update(b"xx").unwrap_err(),
            error::FormatError::Internal("checksum input exceeds its declared length".into())
        );
        assert_eq!(
            (checksum.a, checksum.b, checksum.c, checksum.remaining),
            state
        );
        assert_eq!(checksum.pending, pending);
        assert_eq!(checksum.pending_len, pending_len);
        checksum
            .update(&b"abc"[prefix.len()..usize::try_from(total_len).unwrap()])
            .unwrap();
        assert_eq!(
            checksum.finish().unwrap(),
            jenkins_lookup3(&b"abc"[..usize::try_from(total_len).unwrap()])
        );
    }

    #[rstest]
    #[case(1, b"".as_slice())]
    #[case(14, b"twelve bytes".as_slice())]
    fn incremental_finish_rejects_incomplete_input(#[case] total_len: u64, #[case] data: &[u8]) {
        let mut checksum = JenkinsLookup3::new(total_len);
        checksum.update(data).unwrap();
        assert_eq!(
            checksum.finish().unwrap_err(),
            error::FormatError::Internal(
                "checksum input is shorter than its declared length".into()
            )
        );
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
        let file_data: &[u8] = include_bytes!("../../../tests/data/unattributed/v2_groups.h5");
        // Superblock v3: checksum at offset 44, covers bytes 0..44
        let stored =
            u32::from_le_bytes([file_data[44], file_data[45], file_data[46], file_data[47]]);
        assert_eq!(jenkins_lookup3(&file_data[0..44]), stored);
    }
}
