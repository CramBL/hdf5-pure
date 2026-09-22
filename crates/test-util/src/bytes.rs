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
    #[should_panic(expected = "runs past the end of 15 bytes")]
    fn refuses_a_field_that_does_not_fit() {
        bytes::u32_at(&FIELDS, 12);
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
