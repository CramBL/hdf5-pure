/// The checksum HDF5 stamps on the end of a checksummed metadata structure:
/// Bob Jenkins' lookup3 `hashlittle` seeded with zero, as `H5_checksum_lookup3`
/// in `H5checksum.c`, release 2.0.0, computes it.
pub fn lookup3(bytes: &[u8]) -> u32 {
    // The reference seeds the mix with `(uint32_t) length`, so truncating a
    // longer input is part of the checksum it computes.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the reference computes the seed from the length modulo 2^32"
    )]
    let mut a = 0xDEAD_BEEFu32.wrapping_add(bytes.len() as u32);
    let mut b = a;
    let mut c = a;

    // Every whole block but the last, so that the final mix always runs over
    // real input, as the reference's `while (length > 12)` leaves it.
    let mut rest = bytes;
    while rest.len() > BLOCK {
        a = a.wrapping_add(word(rest, 0));
        b = b.wrapping_add(word(rest, 1));
        c = c.wrapping_add(word(rest, 2));
        mix(&mut a, &mut b, &mut c);
        rest = &rest[BLOCK..];
    }

    // No input is the one case with nothing left to fold in, and the reference
    // returns the seed for it.
    if rest.is_empty() {
        return c;
    }
    a = a.wrapping_add(word(rest, 0));
    b = b.wrapping_add(word(rest, 1));
    c = c.wrapping_add(word(rest, 2));
    final_mix(&mut a, &mut b, &mut c);
    c
}

/// Appends the four-byte checksum of everything already in `bytes`, which is
/// how every checksummed structure in the format ends.
pub fn append(bytes: &mut Vec<u8>) {
    let checksum = lookup3(bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
}

/// Recomputes the four-byte checksum that ends the structure occupying
/// `file[at..at + len]`, for a caller that has edited the bytes it covers.
#[track_caller]
pub fn restamp(file: &mut [u8], at: usize, len: usize) {
    let checksum = lookup3(crate::bytes::slice_at(file, at, len - CHECKSUM));
    crate::bytes::set_slice_at(file, at + len - CHECKSUM, &checksum.to_le_bytes());
}

/// The `index`th little-endian word of `block`, zero-extended where `block` is
/// a tail too short to hold it, which is how the reference's fall-through
/// switch accumulates the last one to twelve bytes.
fn word(block: &[u8], index: usize) -> u32 {
    block
        .iter()
        .skip(index * 4)
        .take(4)
        .enumerate()
        .fold(0, |word, (shift, &byte)| {
            word | (u32::from(byte) << (8 * shift))
        })
}

/// lookup3's `mix` macro, line for line.
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}

/// lookup3's `final` macro, line for line.
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(14));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(11));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(25));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(16));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(4));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(14));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(24));
}

// Bytes of input one round of `mix` consumes: three four-byte words.
const BLOCK: usize = 12;

// The width of the checksum field itself.
const CHECKSUM: usize = 4;

#[cfg(test)]
mod tests {
    use crate::checksum;

    #[test]
    fn matches_the_published_self_test_vector() {
        // `lookup3.c`'s own `driver5` prints this for a zero seed, and the
        // twelve- and thirteen-byte results pin the boundary between the
        // block loop and the tail.
        assert_eq!(
            checksum::lookup3(b"Four score and seven years ago"),
            0x1777_0551
        );
        assert_eq!(checksum::lookup3(b""), 0xDEAD_BEEF);

        const PROBE: [u8; 13] = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85];
        assert_eq!(checksum::lookup3(&PROBE[..12]), 0x3891_9C27);
        assert_eq!(checksum::lookup3(&PROBE), 0xE1F0_CBAE);
    }

    #[test]
    fn restamps_what_it_appends() {
        let mut structure = b"OHDR".to_vec();
        checksum::append(&mut structure);
        let appended = structure.clone();

        structure[1] = b'X';
        checksum::restamp(&mut structure, 0, 8);
        assert_ne!(structure, appended);
        assert_eq!(
            checksum::lookup3(&structure[..4]).to_le_bytes(),
            structure[4..]
        );
    }
}
