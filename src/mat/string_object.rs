//! `mxOPAQUE_CLASS` machinery for MATLAB `string` objects.
//!
//! MATLAB's modern `string` class is encoded by reference into `#refs#` and
//! `#subsystem#/MCOS`. This module implements the byte-level layout that
//! MATLAB itself produces (reverse-engineered against fixtures in the beve
//! test suite). Treat the constants and offsets in this file as load-bearing:
//! changing them is observable to MATLAB on the read side.
//!
//! The layout has three pieces:
//! 1. A `uint64` "saveobj" payload per string scalar/array (lives in `#refs#`).
//!    Encodes a version, the dimensions, the per-string lengths, and packed
//!    UTF-16 code units.
//! 2. A `uint32` metadata array on the parent dataset. The general form is
//!    `[MCOS_MAGIC=0xDD000000, ndims, dims…, object_ids…, class_id]`; for the
//!    scalar string this writer emits it is the 6 elements
//!    `[0xDD000000, 2, 1, 1, object_id, 1]`. Carries the
//!    `MATLAB_class="string"` and `MATLAB_object_decode=3` attributes.
//! 3. A `#subsystem#/MCOS` reference dataset that points to a `FileWrapper__`
//!    metadata blob, a canonical empty placeholder, every per-string saveobj
//!    payload, an alias `int32` blob, and two "unknown template" cell arrays.
//!    Carries the `MATLAB_class="FileWrapper__"` and
//!    `MATLAB_object_decode=3` attributes.

use crate::mat::error::MatError;

/// Magic constant in the metadata header on every MATLAB string-object
/// dataset.
pub const MCOS_MAGIC_NUMBER: u32 = 0xDD00_0000;

/// Version word at the start of `FileWrapper__` blobs.
pub const FILEWRAPPER_VERSION: u32 = 4;

/// Version word at the start of the saveobj payload.
pub const MATLAB_STRING_SAVEOBJ_VERSION: u64 = 1;

/// Value of `MATLAB_object_decode` for opaque-class datasets.
pub const MATLAB_OBJECT_DECODE_OPAQUE: i32 = 3;

/// `MATLAB_class` value on the parent dataset of a string object.
pub const MATLAB_CLASS_STRING: &str = "string";

/// `MATLAB_class` value on the `#subsystem#/MCOS` reference dataset.
pub const MATLAB_CLASS_FILEWRAPPER: &str = "FileWrapper__";

/// 6-element metadata array on a string-object parent dataset.
pub fn create_string_object_metadata(object_id: u32) -> [u32; 6] {
    [MCOS_MAGIC_NUMBER, 2, 1, 1, object_id, 1]
}

/// Build the saveobj payload (a `uint64` array) for one or more string
/// values arranged in `matlab_dims` shape.
///
/// Layout: `[VERSION, ndims, dim0, dim1, ..., len0, len1, ..., utf16 packed
/// as u64]`.
pub fn encode_string_saveobj_payload(
    values: &[String],
    matlab_dims: &[usize],
) -> Result<Vec<u64>, MatError> {
    let expected = product(matlab_dims).ok_or(MatError::Custom(
        "string saveobj dims overflow usize".to_owned(),
    ))?;
    if expected != values.len() {
        return Err(MatError::Custom(format!(
            "string payload length {} does not match product of dims {}",
            values.len(),
            expected
        )));
    }

    // UTF-16 unit count is bounded above by UTF-8 byte count for every string,
    // so a single up-front sum sizes both the units scratch and the final
    // payload exactly.
    let max_utf16_units: usize = values.iter().map(|s| s.len()).sum();
    let header_count = 2 + matlab_dims.len() + values.len();

    let mut payload = Vec::with_capacity(header_count + max_utf16_units.div_ceil(4));
    payload.push(MATLAB_STRING_SAVEOBJ_VERSION);
    payload.push(matlab_dims.len() as u64);
    payload.extend(matlab_dims.iter().map(|&dim| dim as u64));

    let mut units: Vec<u16> = Vec::with_capacity(max_utf16_units);
    for value in values {
        let before = units.len();
        units.extend(value.encode_utf16());
        payload.push((units.len() - before) as u64);
    }
    // Pad to a multiple of 4 u16s so we can pack 4 per u64 word.
    units.resize(units.len().next_multiple_of(4), 0);

    for chunk in units.as_chunks::<4>().0 {
        let v = (chunk[0] as u64)
            | ((chunk[1] as u64) << 16)
            | ((chunk[2] as u64) << 32)
            | ((chunk[3] as u64) << 48);
        payload.push(v);
    }
    Ok(payload)
}

/// Build the `FileWrapper__` metadata blob. Stored as a `uint8` dataset and
/// referenced first from `#subsystem#/MCOS`.
pub fn build_string_filewrapper_metadata(object_count: usize) -> Vec<u8> {
    let mut names_bytes = b"any\0string\0".to_vec();
    while !names_bytes.len().is_multiple_of(8) {
        names_bytes.push(0);
    }

    let mut region_offsets = [0u32; 8];
    let mut regions = Vec::new();

    let version = u32_bytes(&[FILEWRAPPER_VERSION]);
    let num_names = u32_bytes(&[2]);
    let class_id_metadata = u32_bytes(&[0, 0, 0, 0, 0, 2, 0, 0]);

    let mut saveobj_metadata = Vec::with_capacity(2 + object_count * 4);
    saveobj_metadata.extend_from_slice(&[0, 0]);
    for idx in 0..object_count {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "object index written into a 4-byte on-disk saveobj field; MAT files never hold more than u32::MAX objects"
        )]
        saveobj_metadata.extend_from_slice(&[1, 1, 1, idx as u32]);
    }
    let saveobj_metadata = u32_bytes(&saveobj_metadata);

    let mut object_id_metadata = Vec::with_capacity(6 + object_count * 6);
    object_id_metadata.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "object count bounds a 4-byte on-disk object-id field; MAT files never hold more than u32::MAX objects"
    )]
    for id in 1..=object_count as u32 {
        object_id_metadata.extend_from_slice(&[1, 0, 0, id, 0, id]);
    }
    let object_id_metadata = u32_bytes(&object_id_metadata);

    let nobj_metadata = u32_bytes(&[0, 0]);

    let mut dynprop_metadata = Vec::with_capacity(2 + object_count * 2);
    dynprop_metadata.extend_from_slice(&[0, 0]);
    for _ in 0..object_count {
        dynprop_metadata.extend_from_slice(&[0, 0]);
    }
    let dynprop_metadata = u32_bytes(&dynprop_metadata);

    let region6 = Vec::new();
    let region7 = vec![0u8; 8];

    // Accumulate each metadata region's length into the 4-byte filewrapper
    // offsets. Every metadata region is far smaller than u32::MAX.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "region lengths accumulated into 4-byte on-disk filewrapper offsets; \
                  metadata regions are far smaller than u32::MAX"
    )]
    {
        let mut offset = 40u32 + names_bytes.len() as u32;
        region_offsets[0] = offset;
        offset += class_id_metadata.len() as u32;
        region_offsets[1] = offset;
        offset += saveobj_metadata.len() as u32;
        region_offsets[2] = offset;
        offset += object_id_metadata.len() as u32;
        region_offsets[3] = offset;
        offset += nobj_metadata.len() as u32;
        region_offsets[4] = offset;
        offset += dynprop_metadata.len() as u32;
        region_offsets[5] = offset;
        offset += region6.len() as u32;
        region_offsets[6] = offset;
        offset += region7.len() as u32;
        region_offsets[7] = offset;
    }

    regions.extend_from_slice(&version);
    regions.extend_from_slice(&num_names);
    regions.extend_from_slice(&u32_bytes(&region_offsets));
    regions.extend_from_slice(&names_bytes);
    regions.extend_from_slice(&class_id_metadata);
    regions.extend_from_slice(&saveobj_metadata);
    regions.extend_from_slice(&object_id_metadata);
    regions.extend_from_slice(&nobj_metadata);
    regions.extend_from_slice(&dynprop_metadata);
    regions.extend_from_slice(&region6);
    regions.extend_from_slice(&region7);
    regions
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn product(extents: &[usize]) -> Option<usize> {
    extents
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_layout_matches_matlab() {
        assert_eq!(
            create_string_object_metadata(7),
            [MCOS_MAGIC_NUMBER, 2, 1, 1, 7, 1]
        );
    }

    #[test]
    fn saveobj_roundtrips_one_string() {
        let payload = encode_string_saveobj_payload(&["hello".to_owned()], &[1, 1]).unwrap();
        // [VERSION=1, ndims=2, dim0=1, dim1=1, len0=5, then UTF-16 packed]
        assert_eq!(&payload[..5], &[1, 2, 1, 1, 5]);
    }

    #[test]
    fn saveobj_validates_dim_product() {
        let err =
            encode_string_saveobj_payload(&["a".to_owned(), "b".to_owned()], &[3, 1]).unwrap_err();
        let MatError::Custom(message) = err else {
            panic!("expected Custom, got {err:?}");
        };
        assert_eq!(
            message,
            "string payload length 2 does not match product of dims 3"
        );
    }
}
