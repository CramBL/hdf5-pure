#[track_caller]
pub fn u8_at(bytes: &[u8], at: usize) -> u8 {
    u8::from_le_bytes(field(bytes, at))
}

#[track_caller]
pub fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(field(bytes, at))
}

#[track_caller]
pub fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(field(bytes, at))
}

#[track_caller]
pub fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(field(bytes, at))
}

/// The little-endian unsigned integer `width` bytes wide at `at`, for the
/// offset and length fields whose width a file's superblock chooses.
#[track_caller]
pub fn uint_at(bytes: &[u8], at: usize, width: usize) -> u64 {
    assert_width(width);
    slice_at(bytes, at, width)
        .iter()
        .rev()
        .fold(0, |value, &byte| (value << 8) | u64::from(byte))
}

#[track_caller]
pub fn slice_at(bytes: &[u8], at: usize, len: usize) -> &[u8] {
    let available = bytes.len();
    bytes
        .get(at..)
        .and_then(|rest| rest.get(..len))
        .unwrap_or_else(|| {
            panic!("a {len}-byte field at offset {at} runs past the end of {available} bytes")
        })
}

#[track_caller]
pub fn set_u8_at(bytes: &mut [u8], at: usize, value: u8) {
    set_slice_at(bytes, at, &value.to_le_bytes());
}

#[track_caller]
pub fn set_slice_at(bytes: &mut [u8], at: usize, values: &[u8]) {
    let available = bytes.len();
    bytes
        .get_mut(at..)
        .and_then(|rest| rest.get_mut(..values.len()))
        .unwrap_or_else(|| {
            panic!(
                "a {}-byte field at offset {at} runs past the end of {available} bytes",
                values.len()
            )
        })
        .copy_from_slice(values);
}

/// Appends `value` as a little-endian unsigned integer `width` bytes wide.
#[track_caller]
pub fn push_uint(bytes: &mut Vec<u8>, value: u64, width: usize) {
    bytes.extend_from_slice(&uint_field(value, width)[..width]);
}

/// Appends a `width`-byte address field, left undefined where there is no
/// address.
#[track_caller]
pub fn push_address(bytes: &mut Vec<u8>, address: Option<u64>, width: usize) {
    match address {
        Some(address) => push_uint(bytes, address, width),
        None => push_undefined_address(bytes, width),
    }
}

/// Appends a `width`-byte address field left undefined.
#[track_caller]
pub fn push_undefined_address(bytes: &mut Vec<u8>, width: usize) {
    assert_width(width);
    bytes.extend_from_slice(&[UNDEFINED_ADDRESS_BYTE; 8][..width]);
}

pub fn find_signature(bytes: &[u8], signature: &[u8; 4]) -> Option<usize> {
    bytes.windows(signature.len()).position(|w| w == signature)
}

pub fn signature_offsets(bytes: &[u8], signature: &[u8; 4]) -> Vec<usize> {
    bytes
        .windows(signature.len())
        .enumerate()
        .filter(|&(_, w)| w == signature)
        .map(|(at, _)| at)
        .collect()
}

#[track_caller]
pub fn sole_signature(bytes: &[u8], signature: &[u8; 4]) -> usize {
    let offsets = signature_offsets(bytes, signature);
    let &[sole] = &offsets[..] else {
        panic!(
            "expected one {} in the file, found {}",
            String::from_utf8_lossy(signature),
            offsets.len()
        );
    };
    sole
}

#[track_caller]
fn field<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    // `slice_at` returns `N` bytes or panics before this.
    slice_at(bytes, at, N)
        .try_into()
        .expect("a slice of the requested length")
}

// The whole eight bytes, of which a caller takes the low `width` as the field.
#[track_caller]
fn uint_field(value: u64, width: usize) -> [u8; 8] {
    assert_width(width);
    assert!(
        width == 8 || value >> (width * 8) == 0,
        "{value:#x} does not fit a {width}-byte field"
    );
    value.to_le_bytes()
}

#[track_caller]
fn assert_width(width: usize) {
    assert!(
        (1..=8).contains(&width),
        "a format field is one to eight bytes wide, not {width}"
    );
}

// Every bit of an address field is set when the address is undefined: section
// `subsec_fmt4_boot_super`, version 4.0.
const UNDEFINED_ADDRESS_BYTE: u8 = 0xFF;

#[cfg(test)]
mod tests {
    use crate::bytes;

    const FIELDS: [u8; 15] = [
        0xAB, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12, 0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
    ];

    #[test]
    fn reads_and_writes_a_field_of_each_width() {
        assert_eq!(bytes::u8_at(&FIELDS, 0), 0xAB);
        assert_eq!(bytes::u16_at(&FIELDS, 1), 0x1234);
        assert_eq!(bytes::u32_at(&FIELDS, 3), 0x1234_5678);
        assert_eq!(bytes::u64_at(&FIELDS, 7), 0x1234_5678_9ABC_DEF0);
        assert_eq!(bytes::slice_at(&FIELDS, 1, 2), &FIELDS[1..3]);

        let mut written = [0u8; 15];
        bytes::set_u8_at(&mut written, 0, 0xAB);
        bytes::set_slice_at(&mut written, 1, &FIELDS[1..]);
        assert_eq!(written, FIELDS);
    }

    #[test]
    fn reads_and_writes_a_field_of_a_chosen_width() {
        let mut written = Vec::new();
        bytes::push_uint(&mut written, 0x1234, 2);
        bytes::push_uint(&mut written, 0x1234_5678, 4);
        bytes::push_undefined_address(&mut written, 2);
        assert_eq!(written, [0x34, 0x12, 0x78, 0x56, 0x34, 0x12, 0xFF, 0xFF]);
        assert_eq!(bytes::uint_at(&written, 2, 4), 0x1234_5678);
    }

    #[test]
    #[should_panic(expected = "runs past the end of 15 bytes")]
    fn refuses_a_field_that_does_not_fit() {
        bytes::u32_at(&FIELDS, 12);
    }

    #[test]
    #[should_panic(expected = "0x1234 does not fit a 1-byte field")]
    fn refuses_a_value_too_wide_for_its_field() {
        bytes::push_uint(&mut Vec::new(), 0x1234, 1);
    }

    #[test]
    fn finds_every_occurrence_of_a_signature() {
        let file = b"__EAHD__EAHD";
        assert_eq!(bytes::find_signature(file, b"EAHD"), Some(2));
        assert_eq!(bytes::signature_offsets(file, b"EAHD"), vec![2, 8]);
        assert_eq!(bytes::find_signature(file, b"FRHP"), None);
        assert_eq!(bytes::sole_signature(b"__EAHD", b"EAHD"), 2);
    }

    #[test]
    #[should_panic(expected = "expected one EAHD in the file, found 2")]
    fn refuses_a_signature_that_is_not_sole() {
        bytes::sole_signature(b"EAHDEAHD", b"EAHD");
    }
}
